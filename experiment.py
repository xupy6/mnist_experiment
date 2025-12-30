"""
主实验脚本：运行所有层数和优化器的实验
"""
import os
import json
from datetime import datetime
import torch

from train import train_model
from config.config import Config

def run_all_experiments():
    """
    运行所有实验：
    1. 不同层数的CNN（2, 3, 4, 5层）
    2. 不同优化器（SGD, RMSprop, Adam）
    """
    # 创建结果目录
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    
    # 存储所有实验结果
    all_results = {
        'layer_experiments': {},  # 层数实验
        'optimizer_experiments': {},  # 优化器实验
        'config': {
            'epochs': Config.EPOCHS,
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'device': str(Config.DEVICE)
        }
    }
    
    print("=" * 80)
    print("MNIST数字识别实验")
    print("=" * 80)
    print(f"设备: {Config.DEVICE}")
    print(f"训练轮数: {Config.EPOCHS}")
    print(f"批次大小: {Config.BATCH_SIZE}")
    print("=" * 80)
    
    # 实验1：分析不同层数对模型精度的影响（使用Adam优化器）
    print("\n" + "=" * 80)
    print("实验1: 分析不同层数对模型精度的影响")
    print("=" * 80)
    
    layer_results = {}
    for num_layers in Config.CNN_LAYERS:
        print(f"\n{'='*80}")
        print(f"训练 {num_layers} 层CNN模型")
        print(f"{'='*80}")
        
        # 使用Adam优化器进行层数对比实验
        history = train_model(
            num_layers=num_layers,
            optimizer_name='Adam',
            optimizer_config=Config.OPTIMIZERS['Adam'],
            epochs=Config.EPOCHS,
            save_model=True
        )
        
        # 记录结果
        layer_results[f'{num_layers}layers'] = {
            'num_layers': num_layers,
            'optimizer': 'Adam',
            'best_test_acc': max(history['test_acc']),
            'final_test_acc': history['test_acc'][-1],
            'best_train_acc': max(history['train_acc']),
            'final_train_acc': history['train_acc'][-1],
            'history': history
        }
        
        print(f"\n{num_layers}层CNN - 最佳测试准确率: {max(history['test_acc']):.2f}%")
    
    all_results['layer_experiments'] = layer_results
    
    # 实验2：分析不同优化器对模型精度的影响（使用3层CNN）
    print("\n" + "=" * 80)
    print("实验2: 分析不同优化器对模型精度的影响")
    print("=" * 80)
    
    optimizer_results = {}
    for opt_name, opt_config in Config.OPTIMIZERS.items():
        print(f"\n{'='*80}")
        print(f"训练使用 {opt_name} 优化器的模型")
        print(f"{'='*80}")
        
        # 使用3层CNN进行优化器对比实验
        history = train_model(
            num_layers=3,
            optimizer_name=opt_name,
            optimizer_config=opt_config,
            epochs=Config.EPOCHS,
            save_model=True
        )
        
        # 记录结果
        optimizer_results[opt_name] = {
            'optimizer': opt_name,
            'num_layers': 3,
            'best_test_acc': max(history['test_acc']),
            'final_test_acc': history['test_acc'][-1],
            'best_train_acc': max(history['train_acc']),
            'final_train_acc': history['train_acc'][-1],
            'history': history,
            'config': opt_config
        }
        
        print(f"\n{opt_name}优化器 - 最佳测试准确率: {max(history['test_acc']):.2f}%")
    
    all_results['optimizer_experiments'] = optimizer_results
    
    # 保存实验结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(Config.RESULTS_DIR, f'experiment_results_{timestamp}.json')
    
    # 将numpy类型转换为Python原生类型以便JSON序列化
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (float, int)):
            return obj
        elif hasattr(obj, 'item'):  # numpy/torch scalar
            return obj.item()
        else:
            return str(obj)
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("所有实验完成！")
    print("=" * 80)
    print(f"实验结果已保存到: {results_file}")
    print("\n实验结果摘要:")
    print("-" * 80)
    
    # 打印层数实验结果摘要
    print("\n实验1: 不同层数对模型精度的影响 (使用Adam优化器)")
    print("-" * 80)
    for layers, result in sorted(layer_results.items(), key=lambda x: int(x[0].replace('layers', ''))):
        print(f"{result['num_layers']}层CNN: 最佳测试准确率 = {result['best_test_acc']:.2f}%")
    
    # 打印优化器实验结果摘要
    print("\n实验2: 不同优化器对模型精度的影响 (使用3层CNN)")
    print("-" * 80)
    for opt_name, result in optimizer_results.items():
        print(f"{opt_name}: 最佳测试准确率 = {result['best_test_acc']:.2f}%")
    
    print("\n" + "=" * 80)
    print("请运行 analyze_results.py 来生成详细的分析报告和可视化图表")
    print("=" * 80)
    
    return all_results

if __name__ == '__main__':
    run_all_experiments()

