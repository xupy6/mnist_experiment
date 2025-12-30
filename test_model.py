"""
测试脚本：验证模型是否能正常工作
"""
import torch
from models.cnn import CNN
from config.config import Config

def test_model():
    """测试不同层数的CNN模型"""
    print("测试CNN模型...")
    print("=" * 50)
    
    device = Config.DEVICE
    print(f"使用设备: {device}")
    
    # 测试不同层数
    for num_layers in [2, 3, 4, 5]:
        print(f"\n测试 {num_layers} 层CNN模型...")
        try:
            model = CNN(num_layers=num_layers).to(device)
            
            # 创建随机输入 (batch_size=1, channels=1, height=28, width=28)
            test_input = torch.randn(1, 1, 28, 28).to(device)
            
            # 前向传播
            model.eval()
            with torch.no_grad():
                output = model(test_input)
            
            print(f"  输入形状: {test_input.shape}")
            print(f"  输出形状: {output.shape}")
            print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
            
            # 验证输出形状
            assert output.shape == (1, 10), f"输出形状错误: {output.shape}"
            print(f"  模型测试通过！")
            
        except Exception as e:
            print(f"  模型测试失败: {e}")
            raise
    
    print("\n" + "=" * 50)
    print("所有模型测试通过！")
    print("=" * 50)

if __name__ == '__main__':
    test_model()

