"""
卷积神经网络模型定义
支持不同层数的CNN模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    卷积神经网络模型
    可以通过num_layers参数控制卷积层的数量
    """
    def __init__(self, num_layers=3):
        """
        Args:
            num_layers: 卷积层的数量（2-5层）
        """
        super(CNN, self).__init__()
        self.num_layers = num_layers
        
        # 第一层卷积（固定）
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 根据层数动态添加卷积层
        if num_layers >= 2:
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
        
        if num_layers >= 3:
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
        
        if num_layers >= 4:
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(128)
        
        if num_layers >= 5:
            self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm2d(256)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 计算全连接层的输入维度
        # 每次池化：28x28 -> 14x14 -> 7x7 -> 3x3
        # 2层：conv1池化(28->14), conv2池化(14->7) -> 64 * 7 * 7
        # 3层：conv1池化(28->14), conv2池化(14->7), conv3池化(7->3) -> 128 * 3 * 3
        # 4层：conv1池化(28->14), conv2池化(14->7), conv3池化(7->3), conv4无池化 -> 128 * 3 * 3
        # 5层：conv1池化(28->14), conv2池化(14->7), conv3池化(7->3), conv4无池化, conv5无池化 -> 256 * 3 * 3
        if num_layers == 2:
            fc_input_size = 64 * 7 * 7    # 经过2次池化: 28->14->7
        elif num_layers == 3:
            fc_input_size = 128 * 3 * 3   # 经过3次池化: 28->14->7->3
        elif num_layers == 4:
            fc_input_size = 128 * 3 * 3   # 经过3次池化: 28->14->7->3
        elif num_layers == 5:
            fc_input_size = 256 * 3 * 3   # 经过3次池化: 28->14->7->3
        else:
            fc_input_size = 32 * 14 * 14  # 默认情况: 经过1次池化
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 10)  # 10个类别（0-9）
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 第一层
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 第二层
        if self.num_layers >= 2:
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 第三层
        if self.num_layers >= 3:
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 第四层
        if self.num_layers >= 4:
            x = F.relu(self.bn4(self.conv4(x)))
        
        # 第五层
        if self.num_layers >= 5:
            x = F.relu(self.bn5(self.conv5(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

