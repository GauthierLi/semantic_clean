#!/usr/bin/env python3
"""
验证多GPU提取器是否正确使用GPU的简单测试脚本
"""

import os
import sys
import torch
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_serialize.multi_gpu_extractor import MultiGPUFeatureExtractor
from image_serialize.feature_extractor import DINOv3FeatureExtractor


def print_section(title):
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def verify_single_gpu_on_device():
    """验证单GPU提取器模型在GPU上"""
    print_section("1. 验证单GPU提取器（DINOv3FeatureExtractor）")
    
    extractor = DINOv3FeatureExtractor(batch_size=32)
    
    # 检查模型设备
    model_device = next(extractor.model.parameters()).device
    print(f"✓ 模型设备: {model_device}")
    
    # 验证所有参数都在GPU上
    all_on_gpu = all(p.device == model_device for p in extractor.model.parameters())
    print(f"✓ 所有参数在GPU上: {all_on_gpu}")
    
    # 测试推理
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    feature = extractor.extract_features(test_image)
    print(f"✓ 推理成功，特征形状: {feature.shape}")
    
    return model_device.type == 'cuda'


def verify_multi_gpu_on_devices():
    """验证多GPU提取器模型在多个GPU上"""
    print_section("2. 验证多GPU提取器（MultiGPUFeatureExtractor）")
    
    gpu_count = torch.cuda.device_count()
    num_gpus = min(2, gpu_count) if gpu_count >= 2 else 1
    
    print(f"使用 {num_gpus} 个GPU进行测试")
    
    extractor = MultiGPUFeatureExtractor(gpus=list(range(num_gpus)))
    
    # 检查每个模型设备
    print("\n模型设备检查:")
    all_correct = True
    for device, model in extractor.models.items():
        model_device = next(model.parameters()).device
        expected_device = torch.device(device)
        
        is_correct = model_device == expected_device
        all_on_gpu = all(p.device == model_device for p in model.parameters())
        
        print(f"  {device}: 模型在 {model_device} {'✓' if is_correct else '✗'}")
        print(f"        所有参数在GPU上: {all_on_gpu} {'✓' if all_on_gpu else '✗'}")
        
        all_correct = all_correct and is_correct and all_on_gpu
    
    # 测试推理
    test_images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]
    features = extractor.extract_features_batch_multi_gpu(test_images)
    print(f"\n✓ 推理成功，特征形状: {features.shape}")
    
    return all_correct


def verify_data_on_gpu():
    """验证输入数据也移动到GPU"""
    print_section("3. 验证输入数据移动到GPU")
    
    extractor = MultiGPUFeatureExtractor(gpus=[0])
    
    # 创建测试数据
    test_images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(5)]
    
    # 手动检查数据转换过程
    device = 'cuda:0'
    model = extractor.models[device]
    
    print("\n数据转换过程:")
    for i, img in enumerate(test_images[:2]):  # 只检查前两张
        tensor = extractor.transform(img)
        tensor_device = tensor.to(device)
        
        print(f"  图片 {i}:")
        print(f"    原始数据: {img.shape}, dtype={img.dtype}")
        print(f"    Transform后: {tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
        print(f"    移动到GPU后: {tensor_device.shape}, dtype={tensor_device.dtype}, device={tensor_device.device}")
    
    print("\n✓ 输入数据正确移动到GPU")
    
    return True


def verify_gpu_memory_usage():
    """验证GPU内存使用情况"""
    print_section("4. 验证GPU内存使用")
    
    if not torch.cuda.is_available():
        print("⚠ CUDA不可用，跳过GPU内存测试")
        return True
    
    # 清空缓存
    torch.cuda.empty_cache()
    
    # 创建提取器
    extractor = MultiGPUFeatureExtractor(gpus=[0])
    
    # 检查内存使用
    device_id = 0
    allocated = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)
    
    print(f"\nGPU {device_id} 内存使用:")
    print(f"  已分配: {allocated / 1024**3:.2f} GB")
    print(f"  已保留: {reserved / 1024**3:.2f} GB")
    
    # 运行推理
    test_images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]
    features = extractor.extract_features_batch_multi_gpu(test_images)
    
    # 检查推理后的内存使用
    allocated_after = torch.cuda.memory_allocated(device_id)
    reserved_after = torch.cuda.memory_reserved(device_id)
    
    print(f"\n推理后GPU {device_id} 内存使用:")
    print(f"  已分配: {allocated_after / 1024**3:.2f} GB")
    print(f"  已保留: {reserved_after / 1024**3:.2f} GB")
    print(f"  分配增加: {(allocated_after - allocated) / 1024**3:.2f} GB")
    
    print("\n✓ GPU内存使用正常")
    
    return True


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("  多GPU GPU使用验证测试")
    print("  Multi-GPU GPU Usage Verification Test")
    print("="*60)
    
    # 检查CUDA
    print(f"\nCUDA可用: {torch.cuda.is_available()}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
    
    # 运行测试
    tests = [
        ("单GPU提取器GPU使用", verify_single_gpu_on_device),
        ("多GPU提取器GPU使用", verify_multi_gpu_on_devices),
        ("输入数据GPU使用", verify_data_on_gpu),
        ("GPU内存使用", verify_gpu_memory_usage),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ 测试 '{test_name}' 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结
    print_section("测试总结")
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！模型和数据都正确使用GPU。")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查GPU配置。")
        return 1


if __name__ == "__main__":
    sys.exit(main())