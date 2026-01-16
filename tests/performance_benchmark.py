#!/usr/bin/env python3
"""
批量特征提取性能基准测试
详细测试不同批量大小下的性能表现和内存使用情况
"""

import os
import sys
import time
import psutil
import numpy as np
import cv2
import torch
from typing import List
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_serialize.feature_extractor import DINOv3FeatureExtractor

def create_test_images(num_images: int = 100) -> List[np.ndarray]:
    """创建测试图片"""
    images = []
    for i in range(num_images):
        # 创建不同颜色的测试图片
        color = np.random.randint(0, 255, size=3)
        image = np.full((256, 256, 3), color, dtype=np.uint8)
        # 添加随机噪声使图片有所区别
        noise = np.random.randint(0, 50, size=(256, 256, 3))
        image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
        images.append(image)
    return images

def measure_memory_usage():
    """测量当前内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def benchmark_batch_sizes():
    """测试不同批量大小的性能"""
    print("=== 批量大小性能基准测试 ===")
    
    # 创建测试数据
    test_images = create_test_images(50)
    batch_sizes = [1, 4, 8, 16, 32, 64]
    
    results = {
        'batch_size': [],
        'time': [],
        'memory_peak': [],
        'throughput': []
    }
    
    for batch_size in batch_sizes:
        print(f"\n测试批量大小: {batch_size}")
        
        # 重置模型以确保公平比较
        extractor = DINOv3FeatureExtractor(batch_size=batch_size)
        
        # 测量初始内存
        initial_memory = measure_memory_usage()
        peak_memory = initial_memory
        
        # 执行特征提取
        start_time = time.time()
        
        try:
            features = extractor.extract_features_batch(test_images)
            execution_time = time.time() - start_time
            
            # 测量峰值内存
            current_memory = measure_memory_usage()
            peak_memory = max(peak_memory, current_memory)
            
            # 计算吞吐量（图片/秒）
            throughput = len(test_images) / execution_time
            
            results['batch_size'].append(batch_size)
            results['time'].append(execution_time)
            results['memory_peak'].append(peak_memory - initial_memory)
            results['throughput'].append(throughput)
            
            print(f"  执行时间: {execution_time:.4f}s")
            print(f"  内存增长: {peak_memory - initial_memory:.2f}MB")
            print(f"  吞吐量: {throughput:.2f} 图片/秒")
            
        except Exception as e:
            print(f"  错误: {e}")
            continue
    
    return results

def compare_individual_vs_batch():
    """比较逐个处理与批量处理的性能"""
    print("\n=== 逐个处理 vs 批量处理性能对比 ===")
    
    test_images = create_test_images(20)
    
    # 逐个处理
    extractor_individual = DINOv3FeatureExtractor(batch_size=1)
    
    initial_memory = measure_memory_usage()
    start_time = time.time()
    
    individual_features = []
    for image in test_images:
        feature = extractor_individual.extract_features(image)
        individual_features.append(feature.cpu().numpy())
    
    individual_time = time.time() - start_time
    individual_memory = measure_memory_usage() - initial_memory
    
    print(f"逐个处理:")
    print(f"  时间: {individual_time:.4f}s")
    print(f"  内存增长: {individual_memory:.2f}MB")
    
    # 批量处理（不同批量大小）
    for batch_size in [4, 8, 16, 32]:
        extractor_batch = DINOv3FeatureExtractor(batch_size=batch_size)
        
        initial_memory = measure_memory_usage()
        start_time = time.time()
        
        batch_features = extractor_batch.extract_features_batch(test_images)
        
        batch_time = time.time() - start_time
        batch_memory = measure_memory_usage() - initial_memory
        
        speedup = individual_time / batch_time
        
        print(f"批量处理 (batch_size={batch_size}):")
        print(f"  时间: {batch_time:.4f}s")
        print(f"  内存增长: {batch_memory:.2f}MB")
        print(f"  性能提升: {speedup:.2f}x")
        print(f"  内存效率: {individual_memory/batch_memory:.2f}x")

def test_scalability():
    """测试可扩展性"""
    print("\n=== 可扩展性测试 ===")
    
    # 测试不同数据量的处理时间
    image_counts = [10, 25, 50, 100]
    batch_size = 16
    
    for count in image_counts:
        test_images = create_test_images(count)
        extractor = DINOv3FeatureExtractor(batch_size=batch_size)
        
        start_time = time.time()
        features = extractor.extract_features_batch(test_images)
        execution_time = time.time() - start_time
        
        throughput = count / execution_time
        print(f"图片数量: {count}, 时间: {execution_time:.4f}s, 吞吐量: {throughput:.2f} 图片/秒")

def generate_performance_report(results):
    """生成性能报告"""
    print("\n=== 性能总结报告 ===")
    
    # 找到最佳批量大小
    best_idx = np.argmax(results['throughput'])
    best_batch_size = results['batch_size'][best_idx]
    best_throughput = results['throughput'][best_idx]
    
    print(f"最佳批量大小: {best_batch_size}")
    print(f"最高吞吐量: {best_throughput:.2f} 图片/秒")
    
    # 计算性能提升倍数
    baseline_throughput = results['throughput'][0]  # batch_size=1
    speedup = best_throughput / baseline_throughput
    print(f"相比逐个处理的性能提升: {speedup:.2f}x")
    
    # 内存效率分析
    baseline_memory = results['memory_peak'][0]
    best_memory = results['memory_peak'][best_idx]
    memory_efficiency = baseline_memory / best_memory if best_memory > 0 else 1.0
    print(f"相比逐个处理的内存效率: {memory_efficiency:.2f}x")

def main():
    """运行性能基准测试"""
    print("开始批量特征提取性能基准测试...\n")
    
    try:
        # 运行基准测试
        results = benchmark_batch_sizes()
        
        # 比较逐个处理与批量处理
        compare_individual_vs_batch()
        
        # 测试可扩展性
        test_scalability()
        
        # 生成性能报告
        if results['batch_size']:
            generate_performance_report(results)
        
        print("\n🎉 性能基准测试完成！")
        
    except Exception as e:
        print(f"❌ 基准测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()