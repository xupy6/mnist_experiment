"""
手写数字识别GUI应用
支持用户在画布上手写数字，使用训练好的CNN模型进行识别
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import os
import glob

from models.cnn import CNN
from config.config import Config

class HandwritingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST手写数字识别系统")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        # 画布尺寸
        self.canvas_width = 280
        self.canvas_height = 280
        
        # 绘图相关
        self.last_x = None
        self.last_y = None
        self.line_width = 15
        
        # 模型相关
        self.model = None
        self.model_path = None
        self.model_info = None
        
        # 创建界面
        self.create_widgets()
        
        # 自动加载最佳模型
        self.load_best_model()
    
    def create_widgets(self):
        """创建GUI组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(main_frame, text="MNIST手写数字识别", 
                                font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # 左侧：画布区域
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, padx=10, pady=10)
        
        # 画布标签
        canvas_label = ttk.Label(left_frame, text="请在下方画布上手写数字 (0-9)", 
                               font=("Arial", 10))
        canvas_label.grid(row=0, column=0, pady=5)
        
        # 创建画布
        self.canvas = tk.Canvas(left_frame, width=self.canvas_width, 
                               height=self.canvas_height, bg="white", 
                               cursor="pencil")
        self.canvas.grid(row=1, column=0, pady=5)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # 按钮区域
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=2, column=0, pady=10)
        
        # 识别按钮
        recognize_btn = ttk.Button(button_frame, text="识别", 
                                   command=self.recognize, width=12)
        recognize_btn.grid(row=0, column=0, padx=5)
        
        # 清除按钮
        clear_btn = ttk.Button(button_frame, text="清除", 
                              command=self.clear_canvas, width=12)
        clear_btn.grid(row=0, column=1, padx=5)
        
        # 保存图片按钮
        save_btn = ttk.Button(button_frame, text="保存图片", 
                             command=self.save_image, width=12)
        save_btn.grid(row=0, column=2, padx=5)
        
        # 右侧：结果显示区域
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, padx=10, pady=10, sticky=(tk.N))
        
        # 模型选择区域
        model_frame = ttk.LabelFrame(right_frame, text="模型选择", padding="10")
        model_frame.grid(row=0, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # 模型信息显示
        self.model_info_label = ttk.Label(model_frame, 
                                         text="未加载模型", 
                                         font=("Arial", 9))
        self.model_info_label.grid(row=0, column=0, pady=5)
        
        # 模型选择下拉框
        model_select_frame = ttk.Frame(model_frame)
        model_select_frame.grid(row=1, column=0, pady=5)
        
        ttk.Label(model_select_frame, text="选择模型:").grid(row=0, column=0, padx=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_select_frame, 
                                       textvariable=self.model_var,
                                       width=25, state="readonly")
        self.model_combo.grid(row=0, column=1, padx=5)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_select)
        
        # 刷新模型列表按钮
        refresh_btn = ttk.Button(model_select_frame, text="刷新", 
                                command=self.refresh_model_list, width=8)
        refresh_btn.grid(row=0, column=2, padx=5)
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(right_frame, text="识别结果", padding="10")
        result_frame.grid(row=1, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # 预测结果
        self.result_label = ttk.Label(result_frame, 
                                     text="等待识别...", 
                                     font=("Arial", 24, "bold"),
                                     foreground="blue")
        self.result_label.grid(row=0, column=0, pady=20)
        
        # 置信度显示
        self.confidence_label = ttk.Label(result_frame, 
                                         text="", 
                                         font=("Arial", 10))
        self.confidence_label.grid(row=1, column=0, pady=5)
        
        # 概率分布显示
        prob_frame = ttk.Frame(result_frame)
        prob_frame.grid(row=2, column=0, pady=10)
        
        ttk.Label(prob_frame, text="各数字概率:", font=("Arial", 9, "bold")).grid(row=0, column=0, columnspan=2)
        
        self.prob_labels = []
        for i in range(10):
            row = i // 5
            col = i % 5
            label = ttk.Label(prob_frame, text=f"{i}: 0.0%", 
                             font=("Arial", 8), width=10)
            label.grid(row=row+1, column=col, padx=2, pady=2)
            self.prob_labels.append(label)
        
        # 处理后的图像预览
        preview_frame = ttk.LabelFrame(right_frame, text="处理后图像预览", padding="10")
        preview_frame.grid(row=2, column=0, pady=10)
        
        self.preview_label = ttk.Label(preview_frame, text="(28x28)")
        self.preview_label.grid(row=0, column=0)
        
        # 状态栏
        self.status_label = ttk.Label(main_frame, text="就绪", 
                                     relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=2, column=0, columnspan=2, 
                              sticky=(tk.W, tk.E), pady=5)
        
        # 初始化模型列表
        self.refresh_model_list()
    
    def start_draw(self, event):
        """开始绘制"""
        self.last_x = event.x
        self.last_y = event.y
    
    def draw(self, event):
        """绘制线条"""
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, 
                                   event.x, event.y,
                                   width=self.line_width,
                                   capstyle=tk.ROUND,
                                   joinstyle=tk.ROUND,
                                   fill="black")
        self.last_x = event.x
        self.last_y = event.y
    
    def stop_draw(self, event):
        """停止绘制"""
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        """清除画布"""
        self.canvas.delete("all")
        self.result_label.config(text="等待识别...", foreground="blue")
        self.confidence_label.config(text="")
        for label in self.prob_labels:
            label.config(text=f"{self.prob_labels.index(label)}: 0.0%")
        self.preview_label.config(image="")
        self.status_label.config(text="画布已清除")
    
    def save_image(self):
        """保存画布图像"""
        if not self.canvas.find_all():
            messagebox.showwarning("警告", "画布为空，无法保存！")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if filename:
            # 获取画布内容
            self.canvas.postscript(file=filename.replace('.png', '.eps'))
            # 转换为PNG
            img = Image.open(filename.replace('.png', '.eps'))
            img.save(filename, 'PNG')
            os.remove(filename.replace('.png', '.eps'))
            self.status_label.config(text=f"图像已保存: {filename}")
    
    def refresh_model_list(self):
        """刷新模型列表"""
        model_files = glob.glob(os.path.join(Config.MODELS_DIR, "*.pth"))
        model_names = [os.path.basename(f) for f in model_files]
        
        if not model_names:
            self.model_combo['values'] = ["无可用模型"]
            self.status_label.config(text="未找到训练好的模型，请先运行训练脚本")
        else:
            self.model_combo['values'] = model_names
            if not self.model_var.get() and model_names:
                self.model_var.set(model_names[0])
                self.load_model_by_name(model_names[0])
    
    def on_model_select(self, event=None):
        """模型选择事件"""
        model_name = self.model_var.get()
        if model_name and model_name != "无可用模型":
            self.load_model_by_name(model_name)
    
    def load_model_by_name(self, model_name):
        """根据模型名称加载模型"""
        model_path = os.path.join(Config.MODELS_DIR, model_name)
        self.load_model(model_path)
    
    def load_best_model(self):
        """加载最佳模型（3层Adam，实验结果显示最佳）"""
        # 尝试加载3层Adam模型（根据实验结果，这是最佳配置）
        best_model_path = os.path.join(Config.MODELS_DIR, "cnn_3layers_Adam_best.pth")
        if os.path.exists(best_model_path):
            self.load_model(best_model_path)
            self.model_var.set("cnn_3layers_Adam_best.pth")
        else:
            # 如果没有，加载第一个可用模型
            self.refresh_model_list()
    
    def load_model(self, model_path):
        """加载模型"""
        try:
            # 从文件名解析模型参数
            filename = os.path.basename(model_path)
            parts = filename.replace('.pth', '').split('_')
            
            num_layers = None
            optimizer_name = None
            
            for part in parts:
                if 'layers' in part:
                    num_layers = int(part.replace('layers', ''))
                elif part in ['SGD', 'RMSprop', 'Adam']:
                    optimizer_name = part
            
            if num_layers is None:
                # 默认使用3层
                num_layers = 3
            
            # 创建模型
            self.model = CNN(num_layers=num_layers).to(Config.DEVICE)
            
            # 加载权重
            self.model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            self.model.eval()
            
            self.model_path = model_path
            self.model_info = f"{num_layers}层CNN - {optimizer_name or '未知优化器'}"
            
            self.model_info_label.config(text=f"已加载: {self.model_info}")
            self.status_label.config(text=f"模型加载成功: {filename}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {str(e)}")
            self.status_label.config(text=f"模型加载失败: {str(e)}")
    
    def canvas_to_image(self):
        """将画布内容转换为PIL图像"""
        # 获取画布尺寸
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            # 如果画布还没有渲染，使用默认尺寸
            width = self.canvas_width
            height = self.canvas_height
        
        # 创建PIL图像
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # 获取画布上的所有项目
        items = self.canvas.find_all()
        
        # 绘制所有线条
        for item in items:
            item_type = self.canvas.type(item)
            if item_type == 'line':
                coords = self.canvas.coords(item)
                if len(coords) >= 4:
                    # 获取线条颜色和宽度
                    color = self.canvas.itemcget(item, 'fill')
                    try:
                        width_val = float(self.canvas.itemcget(item, 'width'))
                    except:
                        width_val = self.line_width
                    
                    # 绘制线条
                    for i in range(0, len(coords)-2, 2):
                        if i+3 < len(coords):
                            draw.line([coords[i], coords[i+1], coords[i+2], coords[i+3]], 
                                     fill='black', width=int(width_val))
        
        # 转换为灰度图
        img = img.convert('L')
        
        # 调整大小到28x28
        img = img.resize((28, 28), Image.LANCZOS)
        
        return img
    
    def preprocess_image(self, img):
        """预处理图像，使其符合MNIST格式"""
        # 转换为numpy数组
        img_array = np.array(img, dtype=np.float32)
        
        # 反色（MNIST是白底黑字，画布是黑底白字）
        img_array = 255 - img_array
        
        # 归一化到[0, 1]
        img_array = img_array / 255.0
        
        # MNIST归一化参数
        mean = 0.1307
        std = 0.3081
        img_array = (img_array - mean) / std
        
        # 转换为tensor并添加batch和channel维度
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        
        return img_tensor.to(Config.DEVICE)
    
    def recognize(self):
        """识别手写数字"""
        if not self.canvas.find_all():
            messagebox.showwarning("警告", "请先在画布上书写数字！")
            return
        
        if self.model is None:
            messagebox.showerror("错误", "未加载模型，请先选择模型！")
            return
        
        try:
            self.status_label.config(text="正在识别...")
            self.root.update()
            
            # 将画布转换为图像
            img = self.canvas_to_image()
            
            # 显示预览
            preview_img = img.resize((84, 84), Image.NEAREST)  # 放大3倍显示
            preview_photo = ImageTk.PhotoImage(preview_img)
            self.preview_label.config(image=preview_photo)
            self.preview_label.image = preview_photo  # 保持引用
            
            # 预处理
            img_tensor = self.preprocess_image(img)
            
            # 预测
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # 显示结果
            predicted_num = predicted.item()
            confidence_val = confidence.item() * 100
            
            self.result_label.config(text=str(predicted_num), 
                                   foreground="green" if confidence_val > 50 else "orange")
            self.confidence_label.config(
                text=f"置信度: {confidence_val:.2f}%"
            )
            
            # 显示所有数字的概率
            probs = probabilities[0].cpu().numpy() * 100
            for i, label in enumerate(self.prob_labels):
                prob_val = probs[i]
                color = "green" if i == predicted_num else "black"
                label.config(text=f"{i}: {prob_val:.1f}%", foreground=color)
            
            self.status_label.config(text=f"识别完成: 数字 {predicted_num} (置信度: {confidence_val:.2f}%)")
            
        except Exception as e:
            messagebox.showerror("错误", f"识别失败: {str(e)}")
            self.status_label.config(text=f"识别失败: {str(e)}")

def main():
    """主函数"""
    root = tk.Tk()
    app = HandwritingApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()

