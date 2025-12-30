"""
数据集加载工具
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(data_root='./data', batch_size=64, num_workers=0):
    """
    获取MNIST数据集的训练和测试数据加载器
    
    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 数据加载的进程数
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 数据预处理：转换为张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])
    
    # 加载训练集
    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=False,  # 数据已存在，不下载
        transform=transform
    )
    
    # 加载测试集
    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=False,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

