# MNIST数字识别实验

本项目使用Python和PyTorch实现MNIST数据集的数字识别实验，分析卷积神经网络层数和不同优化器对模型精度的影响。

## 项目结构

```
mnist_experiment/
├── config/              # 配置文件
│   └── config.py        # 实验参数配置
├── data/                # 数据目录
│   └── MNIST/           # MNIST数据集
├── models/              # 模型定义
│   └── cnn.py           # CNN模型（支持不同层数）
├── utils/               # 工具函数
│   ├── __init__.py
│   └── dataset.py      # 数据集加载
├── train.py            # 训练脚本
├── experiment.py       # 主实验脚本（运行所有实验）
├── analyze_results.py  # 结果分析和可视化
├── requirements.txt    # 依赖包
├── handwriting_app.py  # 识别手写数字可运行程序
└── README.md           # 本文件
```

## 实验内容

### 实验1: 分析不同层数对模型精度的影响
- 测试2层、3层、4层、5层CNN模型
- 使用Adam优化器进行对比
- 记录训练和测试的损失、准确率

### 实验2: 分析不同优化器对模型精度的影响
- 测试SGD、RMSprop、Adam三种优化器
- 使用3层CNN模型进行对比
- 记录训练和测试的损失、准确率

## 环境要求

- Python 3.7+
- PyTorch 1.9.0+
- 其他依赖见 `requirements.txt`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 运行所有实验

运行主实验脚本，会自动执行所有实验：

```bash
python experiment.py
```

这将：
- 运行实验1：测试不同层数的CNN（2、3、4、5层）
- 运行实验2：测试不同优化器（SGD、RMSprop、Adam）
- 保存所有实验结果到 `results/` 目录
- 保存最佳模型到 `checkpoints/` 目录

### 2. 分析实验结果

运行分析脚本，生成可视化图表和实验报告：

```bash
python analyze_results.py
```

这将生成：
- `results/layer_comparison.png` - 层数对比图表
- `results/optimizer_comparison.png` - 优化器对比图表
- `results/experiment_report.txt` - 实验报告文本

### 3. 单独训练模型

如果需要单独训练某个模型，可以直接使用 `train.py`：

```python
from train import train_model
from config.config import Config

# 训练3层CNN，使用Adam优化器
history = train_model(
    num_layers=3,
    optimizer_name='Adam',
    optimizer_config=Config.OPTIMIZERS['Adam'],
    epochs=10,
    save_model=True
)
```

## 配置说明

在 `config/config.py` 中可以修改实验参数：

- `EPOCHS`: 训练轮数（默认10）
- `BATCH_SIZE`: 批次大小（默认64）
- `LEARNING_RATE`: 学习率（默认0.001）
- `CNN_LAYERS`: 要测试的CNN层数列表
- `OPTIMIZERS`: 优化器配置字典

## 结果说明

实验结果保存在 `results/` 目录下：

1. **JSON文件**: `experiment_results_YYYYMMDD_HHMMSS.json`
   - 包含所有实验的详细数据（损失、准确率等）

2. **可视化图表**:
   - `layer_comparison.png`: 不同层数的对比图表
   - `optimizer_comparison.png`: 不同优化器的对比图表

3. **实验报告**: `experiment_report.txt`
   - 包含实验结果摘要和分析

4. **模型文件**: `checkpoints/` 目录
   - 保存每个实验的最佳模型

## 注意事项

1. 数据已放在 `data/MNIST/` 目录下，无需下载
2. Windows系统建议将 `NUM_WORKERS` 设为0
3. 如果使用GPU，会自动检测并使用CUDA
4. 实验可能需要较长时间，请耐心等待

## 实验预期结果

- **层数影响**: 通常3-4层CNN在MNIST数据集上表现较好，过多层数可能导致过拟合
- **优化器影响**: Adam优化器通常收敛更快且精度较高，SGD需要合适的学习率设置

## 作者
@xupy6

