"""
结果分析和可视化脚本
生成实验报告和可视化图表
"""
import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

from config.config import Config

def load_latest_results():
    """加载最新的实验结果"""
    results_files = glob.glob(os.path.join(Config.RESULTS_DIR, 'experiment_results_*.json'))
    if not results_files:
        raise FileNotFoundError("未找到实验结果文件，请先运行 experiment.py")
    
    latest_file = max(results_files, key=os.path.getctime)
    print(f"加载实验结果: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results, latest_file

def plot_layer_comparison(results):
    """绘制不同层数对模型精度的影响"""
    layer_experiments = results['layer_experiments']
    
    # 提取数据
    layers = []
    best_test_accs = []
    final_test_accs = []
    histories = []
    
    for key in sorted(layer_experiments.keys(), key=lambda x: int(x.replace('layers', ''))):
        exp = layer_experiments[key]
        layers.append(exp['num_layers'])
        best_test_accs.append(exp['best_test_acc'])
        final_test_accs.append(exp['final_test_acc'])
        histories.append(exp['history'])
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('实验1: 不同层数对模型精度的影响', fontsize=16, fontweight='bold')
    
    # 子图1: 最佳测试准确率对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar([f'{l}层' for l in layers], best_test_accs, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    ax1.set_ylabel('最佳测试准确率 (%)', fontsize=12)
    ax1.set_xlabel('CNN层数', fontsize=12)
    ax1.set_title('不同层数的最佳测试准确率', fontsize=13, fontweight='bold')
    ax1.set_ylim([min(best_test_accs) - 2, max(best_test_accs) + 2])
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, acc in zip(bars1, best_test_accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 子图2: 训练过程曲线 - 测试准确率
    ax2 = axes[0, 1]
    epochs = range(1, len(histories[0]['test_acc']) + 1)
    for i, (layer, history) in enumerate(zip(layers, histories)):
        ax2.plot(epochs, history['test_acc'], 
                marker='o', label=f'{layer}层', linewidth=2, markersize=4)
    ax2.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax2.set_ylabel('测试准确率 (%)', fontsize=12)
    ax2.set_title('测试准确率随训练轮数的变化', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # 子图3: 训练过程曲线 - 训练损失
    ax3 = axes[1, 0]
    for i, (layer, history) in enumerate(zip(layers, histories)):
        ax3.plot(epochs, history['train_loss'], 
                marker='s', label=f'{layer}层', linewidth=2, markersize=4)
    ax3.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax3.set_ylabel('训练损失', fontsize=12)
    ax3.set_title('训练损失随训练轮数的变化', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # 子图4: 训练过程曲线 - 测试损失
    ax4 = axes[1, 1]
    for i, (layer, history) in enumerate(zip(layers, histories)):
        ax4.plot(epochs, history['test_loss'], 
                marker='^', label=f'{layer}层', linewidth=2, markersize=4)
    ax4.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax4.set_ylabel('测试损失', fontsize=12)
    ax4.set_title('测试损失随训练轮数的变化', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(Config.RESULTS_DIR, 'layer_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"层数对比图表已保存: {output_path}")
    
    return fig

def plot_optimizer_comparison(results):
    """绘制不同优化器对模型精度的影响"""
    optimizer_experiments = results['optimizer_experiments']
    
    # 提取数据
    optimizers = []
    best_test_accs = []
    final_test_accs = []
    histories = []
    
    for opt_name in ['SGD', 'RMSprop', 'Adam']:
        if opt_name in optimizer_experiments:
            exp = optimizer_experiments[opt_name]
            optimizers.append(opt_name)
            best_test_accs.append(exp['best_test_acc'])
            final_test_accs.append(exp['final_test_acc'])
            histories.append(exp['history'])
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('实验2: 不同优化器对模型精度的影响', fontsize=16, fontweight='bold')
    
    # 子图1: 最佳测试准确率对比
    ax1 = axes[0, 0]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars1 = ax1.bar(optimizers, best_test_accs, color=colors[:len(optimizers)])
    ax1.set_ylabel('最佳测试准确率 (%)', fontsize=12)
    ax1.set_xlabel('优化器类型', fontsize=12)
    ax1.set_title('不同优化器的最佳测试准确率', fontsize=13, fontweight='bold')
    ax1.set_ylim([min(best_test_accs) - 2, max(best_test_accs) + 2])
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, acc in zip(bars1, best_test_accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 子图2: 训练过程曲线 - 测试准确率
    ax2 = axes[0, 1]
    epochs = range(1, len(histories[0]['test_acc']) + 1)
    for i, (opt, history) in enumerate(zip(optimizers, histories)):
        ax2.plot(epochs, history['test_acc'], 
                marker='o', label=opt, linewidth=2, markersize=4, color=colors[i])
    ax2.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax2.set_ylabel('测试准确率 (%)', fontsize=12)
    ax2.set_title('测试准确率随训练轮数的变化', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # 子图3: 训练过程曲线 - 训练损失
    ax3 = axes[1, 0]
    for i, (opt, history) in enumerate(zip(optimizers, histories)):
        ax3.plot(epochs, history['train_loss'], 
                marker='s', label=opt, linewidth=2, markersize=4, color=colors[i])
    ax3.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax3.set_ylabel('训练损失', fontsize=12)
    ax3.set_title('训练损失随训练轮数的变化', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # 子图4: 训练过程曲线 - 测试损失
    ax4 = axes[1, 1]
    for i, (opt, history) in enumerate(zip(optimizers, histories)):
        ax4.plot(epochs, history['test_loss'], 
                marker='^', label=opt, linewidth=2, markersize=4, color=colors[i])
    ax4.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax4.set_ylabel('测试损失', fontsize=12)
    ax4.set_title('测试损失随训练轮数的变化', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(Config.RESULTS_DIR, 'optimizer_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"优化器对比图表已保存: {output_path}")
    
    return fig

def generate_report(results, results_file):
    """生成实验报告"""
    report_path = os.path.join(Config.RESULTS_DIR, 'experiment_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MNIST数字识别实验报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 实验配置
        f.write("实验配置:\n")
        f.write("-" * 80 + "\n")
        config = results['config']
        f.write(f"训练轮数: {config['epochs']}\n")
        f.write(f"批次大小: {config['batch_size']}\n")
        f.write(f"学习率: {config['learning_rate']}\n")
        f.write(f"设备: {config['device']}\n")
        f.write("\n")
        
        # 实验1: 层数分析
        f.write("=" * 80 + "\n")
        f.write("实验1: 不同层数对模型精度的影响\n")
        f.write("=" * 80 + "\n\n")
        f.write("使用优化器: Adam\n\n")
        
        layer_experiments = results['layer_experiments']
        f.write(f"{'层数':<10} {'最佳测试准确率':<20} {'最终测试准确率':<20}\n")
        f.write("-" * 80 + "\n")
        
        for key in sorted(layer_experiments.keys(), key=lambda x: int(x.replace('layers', ''))):
            exp = layer_experiments[key]
            f.write(f"{exp['num_layers']:<10} {exp['best_test_acc']:<20.2f} {exp['final_test_acc']:<20.2f}\n")
        
        f.write("\n分析:\n")
        best_layer = max(layer_experiments.items(), 
                        key=lambda x: x[1]['best_test_acc'])
        worst_layer = min(layer_experiments.items(), 
                         key=lambda x: x[1]['best_test_acc'])
        f.write(f"- 最佳层数: {best_layer[1]['num_layers']}层 (准确率: {best_layer[1]['best_test_acc']:.2f}%)\n")
        f.write(f"- 最差层数: {worst_layer[1]['num_layers']}层 (准确率: {worst_layer[1]['best_test_acc']:.2f}%)\n")
        f.write(f"- 准确率差异: {best_layer[1]['best_test_acc'] - worst_layer[1]['best_test_acc']:.2f}%\n")
        f.write("\n")
        
        # 实验2: 优化器分析
        f.write("=" * 80 + "\n")
        f.write("实验2: 不同优化器对模型精度的影响\n")
        f.write("=" * 80 + "\n\n")
        f.write("使用CNN层数: 3层\n\n")
        
        optimizer_experiments = results['optimizer_experiments']
        f.write(f"{'优化器':<15} {'最佳测试准确率':<20} {'最终测试准确率':<20} {'学习率':<15}\n")
        f.write("-" * 80 + "\n")
        
        for opt_name in ['SGD', 'RMSprop', 'Adam']:
            if opt_name in optimizer_experiments:
                exp = optimizer_experiments[opt_name]
                lr = exp['config'].get('lr', 'N/A')
                f.write(f"{opt_name:<15} {exp['best_test_acc']:<20.2f} {exp['final_test_acc']:<20.2f} {lr:<15}\n")
        
        f.write("\n分析:\n")
        best_opt = max(optimizer_experiments.items(), 
                      key=lambda x: x[1]['best_test_acc'])
        worst_opt = min(optimizer_experiments.items(), 
                       key=lambda x: x[1]['best_test_acc'])
        f.write(f"- 最佳优化器: {best_opt[0]} (准确率: {best_opt[1]['best_test_acc']:.2f}%)\n")
        f.write(f"- 最差优化器: {worst_opt[0]} (准确率: {worst_opt[1]['best_test_acc']:.2f}%)\n")
        f.write(f"- 准确率差异: {best_opt[1]['best_test_acc'] - worst_opt[1]['best_test_acc']:.2f}%\n")
        f.write("\n")
        
        # 结论
        f.write("=" * 80 + "\n")
        f.write("实验结论\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. 层数影响分析:\n")
        f.write("   - 通过对比不同层数的CNN模型，可以观察到层数对模型性能的影响。\n")
        f.write("   - 通常，适当的层数可以提升模型性能，但过多的层数可能导致过拟合。\n")
        f.write("\n")
        f.write("2. 优化器影响分析:\n")
        f.write("   - 不同优化器在收敛速度和最终精度上存在差异。\n")
        f.write("   - Adam优化器通常具有较好的自适应学习率特性。\n")
        f.write("   - SGD优化器需要合适的学习率和动量设置。\n")
        f.write("\n")
        f.write(f"实验结果文件: {results_file}\n")
        f.write("=" * 80 + "\n")
    
    print(f"实验报告已保存: {report_path}")

def main():
    """主函数"""
    print("=" * 80)
    print("开始分析实验结果...")
    print("=" * 80)
    
    # 加载实验结果
    results, results_file = load_latest_results()
    
    # 生成可视化图表
    print("\n生成可视化图表...")
    plot_layer_comparison(results)
    plot_optimizer_comparison(results)
    
    # 生成实验报告
    print("\n生成实验报告...")
    generate_report(results, results_file)
    
    print("\n" + "=" * 80)
    print("结果分析完成！")
    print("=" * 80)
    print(f"所有结果保存在: {Config.RESULTS_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()

