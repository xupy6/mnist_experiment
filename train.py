"""
训练脚本：支持不同优化器的训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json
from datetime import datetime

from models.cnn import CNN
from utils.dataset import get_mnist_loaders
from config.config import Config

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def train_model(num_layers, optimizer_name, optimizer_config, epochs=None, save_model=False):
    """
    训练模型
    
    Args:
        num_layers: CNN层数
        optimizer_name: 优化器名称
        optimizer_config: 优化器配置字典
        epochs: 训练轮数，如果为None则使用Config中的默认值
        save_model: 是否保存模型
    
    Returns:
        history: 训练历史记录（损失和准确率）
    """
    if epochs is None:
        epochs = Config.EPOCHS
    
    # 创建结果目录
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    
    # 加载数据
    train_loader, test_loader = get_mnist_loaders(
        Config.DATA_ROOT,
        Config.BATCH_SIZE,
        Config.NUM_WORKERS
    )
    
    # 创建模型
    model = CNN(num_layers=num_layers).to(Config.DEVICE)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建优化器
    if optimizer_config['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=optimizer_config['lr'],
            momentum=optimizer_config.get('momentum', 0.9)
        )
    elif optimizer_config['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=optimizer_config['lr'],
            alpha=optimizer_config.get('alpha', 0.99)
        )
    elif optimizer_config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_config['optimizer']}")
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    print(f"\n开始训练: {num_layers}层CNN, 优化器: {optimizer_name}")
    print(f"设备: {Config.DEVICE}")
    print(f"训练轮数: {epochs}")
    print("-" * 50)
    
    # 训练循环
    best_test_acc = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, Config.DEVICE)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if save_model:
                model_path = os.path.join(
                    Config.MODELS_DIR,
                    f'cnn_{num_layers}layers_{optimizer_name}_best.pth'
                )
                torch.save(model.state_dict(), model_path)
    
    print(f"\n训练完成! 最佳测试准确率: {best_test_acc:.2f}%")
    print("=" * 50)
    
    return history

if __name__ == '__main__':
    # 示例：训练一个3层CNN，使用Adam优化器
    history = train_model(
        num_layers=3,
        optimizer_name='Adam',
        optimizer_config=Config.OPTIMIZERS['Adam'],
        epochs=5,
        save_model=True
    )

