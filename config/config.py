"""
配置文件：定义实验参数
"""
import torch

class Config:
    # 数据相关
    DATA_ROOT = './data'
    BATCH_SIZE = 64
    NUM_WORKERS = 0  # Windows系统建议设为0
    
    # 训练相关
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 实验相关
    # 不同层数的CNN配置（卷积层数量）
    CNN_LAYERS = [2, 3, 4, 5]  # 测试2层、3层、4层、5层CNN
    
    # 不同优化器配置
    OPTIMIZERS = {
        'SGD': {
            'optimizer': 'SGD',
            'lr': 0.01,
            'momentum': 0.9
        },
        'RMSprop': {
            'optimizer': 'RMSprop',
            'lr': 0.001,
            'alpha': 0.99
        },
        'Adam': {
            'optimizer': 'Adam',
            'lr': 0.001,
            'betas': (0.9, 0.999)
        }
    }
    
    # 结果保存路径
    RESULTS_DIR = './results'
    MODELS_DIR = './checkpoints'

