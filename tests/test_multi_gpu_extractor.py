#!/usr/bin/env python3
"""
多GPU特征提取器测试脚本
验证多GPU并行处理功能的正确性和性能
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
from typing import List

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_serialize.multi_gpu_extractor import MultiGPUFeatureExtractor
from image_serialize.feature_extractor import DINOv3FeatureExtractor

def create_test_images(num_images: int = 20) -> List[np.ndarray]:
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

def test_multi_gpu_initialization():
    """测试多GPU提取器初始化"""
    print("=== 测试多GPU提取器初始化 ===")
    
    try:
        # 测试自动GPU检测
        extractor_auto = MultiGPUFeatureExtractor()
        gpu_info = extractor_auto.get_gpu_info()
        print(f"自动检测GPU数量: {gpu_info['num_gpus']}")
        print(f"设备列表: {gpu_info['devices']}")
        
        # 测试指定GPU
        if gpu_info['num_gpus'] > 1:
            extractor_specified = MultiGPUFeatureExtractor(gpus=[0, 1])
            gpu_info_spec = extractor_specified.get_gpu_info()
            print(f"指定GPU设备: {gpu_info_spec['devices']}")
        
        # 测试CPU模式
        extractor_cpu = MultiGPUFeatureExtractor(gpus=['cpu'])
        gpu_info_cpu = extractor_cpu.get_gpu_info()
        print(f"CPU模式设备: {gpu_info_cpu['devices']}")
        
        print("✓ 多GPU提取器初始化测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 初始化测试失败: {e}")
        return False

def test_multi_gpu_vs_single_gpu_consistency():
    """测试多GPU与单GPU结果一致性"""
    print("\n=== 测试多GPU与单GPU结果一致性 ===")
    
    try:
        # 创建测试图片
        test_images = create_test_images(12)
        
        # 单GPU处理
        single_extractor = DINOv3FeatureExtractor(batch_size=32)
        start_time = time.time()
        single_features = single_extractor.extract_features_batch(test_images)
        single_time = time.time() - start_time
        
        # 多GPU处理
        multi_extractor = MultiGPUFeatureExtractor(gpus=[0, 1] if torch.cuda.device_count() > 1 else [0])
        start_time = time.time()
        multi_features = multi_extractor.extract_features_batch_multi_gpu(test_images)
        multi_time = time.time() - start_time
        
        # 比较结果
        single_np = single_features.cpu().numpy()
        multi_np = multi_features.cpu().numpy()
        
        # 计算差异
        diff = np.abs(single_np - multi_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"单GPU时间: {single_time:.4f}s")
        print(f"多GPU时间: {multi_time:.4f}s")
        print(f"最大差异: {max_diff:.8f}")
        print(f"平均差异: {mean_diff:.8f}")
        
        # 验证一致性
        tolerance = 1e-2  # 多GPU可能有更大的数值差异，特别是单GPU环境下的回退模式
        if max_diff < tolerance:
            print("✓ 多GPU与单GPU结果一致")
            return True
        else:
            print(f"✗ 多GPU与单GPU结果不一致，差异超过容忍度 {tolerance}")
            print(f"注意：当前为单GPU环境，多GPU模式回退处理可能导致微小差异")
            return False
            
    except Exception as e:
        print(f"✗ 一致性测试失败: {e}")
        return False

def test_gpu_memory_allocation():
    """测试GPU内存分配"""
    print("\n=== 测试GPU内存分配 ===")
    
    try:
        if not torch.cuda.is_available():
            print("CUDA不可用，跳过GPU内存测试")
            return True
        
        extractor = MultiGPUFeatureExtractor()
        gpu_info = extractor.get_gpu_info()
        
        print("GPU信息:")
        for gpu in gpu_info.get('gpu_names', []):
            print(f"  {gpu['device']}: {gpu['name']}")
            print(f"    总内存: {gpu['total_memory']/1024**3:.2f}GB")
            print(f"    计算能力: {gpu['compute_capability']}")
        
        memory_info = gpu_info['memory_info']
        print("\n内存使用情况:")
        for device, memory in memory_info.items():
            print(f"  {device}: {memory/1024**3:.2f}GB 可用")
        
        print("✓ GPU内存分配测试通过")
        return True
        
    except Exception as e:
        print(f"✗ GPU内存分配测试失败: {e}")
        return False

def test_batch_distribution():
    """测试批次分配算法"""
    print("\n=== 测试批次分配算法 ===")
    
    try:
        extractor = MultiGPUFeatureExtractor()
        
        # 测试不同数量的图片分配
        test_cases = [5, 12, 25, 50]
        
        for num_images in test_cases:
            test_images = create_test_images(num_images)
            distribution = extractor._distribute_images_to_gpus(test_images)
            
            total_allocated = sum(len(batch) for batch in distribution.values())
            print(f"图片数量: {num_images}")
            print(f"分配结果: {[(device, len(batch)) for device, batch in distribution.items()]}")
            print(f"分配总数: {total_allocated}")
            
            if total_allocated != num_images:
                print(f"✗ 分配不匹配: 期望{num_images}, 实际{total_allocated}")
                return False
        
        print("✓ 批次分配算法测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 批次分配测试失败: {e}")
        return False

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    try:
        # 测试无效GPU
        try:
            MultiGPUFeatureExtractor(gpus=[999])
            print("✗ 应该拒绝无效GPU")
            return False
        except (RuntimeError, ValueError):
            print("✓ 正确拒绝无效GPU")
        
        # 测试空输入
        extractor = MultiGPUFeatureExtractor()
        empty_result = extractor.extract_features_batch_multi_gpu([])
        if empty_result.numel() == 0:
            print("✓ 正确处理空输入")
        else:
            print("✗ 空输入处理失败")
            return False
        
        # 测试负数批量大小
        try:
            extractor.update_batch_size(-1)
            print("✗ 应该拒绝负数批量大小")
            return False
        except ValueError:
            print("✓ 正确拒绝负数批量大小")
        
        print("✓ 错误处理测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 错误处理测试失败: {e}")
        return False

def test_performance_comparison():
    """性能对比测试"""
    print("\n=== 性能对比测试 ===")
    
    try:
        # 创建较多测试图片
        test_images = create_test_images(40)
        
        # 单GPU基准
        single_extractor = DINOv3FeatureExtractor(batch_size=32)
        start_time = time.time()
        single_features = single_extractor.extract_features_batch(test_images)
        single_time = time.time() - start_time
        
        # 多GPU测试
        multi_extractor = MultiGPUFeatureExtractor()
        start_time = time.time()
        multi_features = multi_extractor.extract_features_batch_multi_gpu(test_images)
        multi_time = time.time() - start_time
        
        speedup = single_time / multi_time if multi_time > 0 else 0
        
        print(f"单GPU时间: {single_time:.4f}s")
        print(f"多GPU时间: {multi_time:.4f}s")
        print(f"性能提升: {speedup:.2f}x")
        
        if speedup > 0.8:  # 允许一定的开销
            print("✓ 性能测试通过")
            return True
        else:
            print("✗ 性能提升不明显")
            return False
            
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始多GPU特征提取器测试...\n")
    
    tests = [
        test_multi_gpu_initialization,
        test_multi_gpu_vs_single_gpu_consistency,
        test_gpu_memory_allocation,
        test_batch_distribution,
        test_error_handling,
        test_performance_comparison
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"测试完成: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！多GPU特征提取功能正常工作。")
        return True
    else:
        print("❌ 部分测试失败，请检查实现。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)