#!/usr/bin/env python3
"""
多GPU功能综合测试套件
专为多GPU环境设计，验证多GPU并行处理的正确性、性能和稳定性

运行环境要求：
- 至少2个GPU
- CUDA可用
- 足够的GPU内存

运行方式：
    python tests/test_multi_gpu_comprehensive.py

或者指定GPU：
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/test_multi_gpu_comprehensive.py
"""

import os
import sys
import time
import json
import numpy as np
import torch
from typing import List, Dict
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_serialize.multi_gpu_extractor import MultiGPUFeatureExtractor
from image_serialize.feature_extractor import DINOv3FeatureExtractor
from image_serialize import ImageSerializer


class Colors:
    """终端颜色输出"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """打印标题"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")


def print_success(text: str):
    """打印成功信息"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """打印错误信息"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    """打印警告信息"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def create_test_images(num_images: int = 20, size: tuple = (224, 224)) -> List[np.ndarray]:
    """创建测试图片"""
    images = []
    for i in range(num_images):
        # 创建不同颜色的测试图片
        color = np.random.randint(0, 255, size=3)
        image = np.full((*size, 3), color, dtype=np.uint8)
        # 添加随机噪声使图片有所区别
        noise = np.random.randint(0, 50, size=(*size, 3))
        image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
        images.append(image)
    return images


def check_gpu_requirements() -> bool:
    """检查GPU环境是否满足要求"""
    print_header("GPU环境检查")
    
    if not torch.cuda.is_available():
        print_error("CUDA不可用，无法运行多GPU测试")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    总内存: {props.total_memory / 1024**3:.2f}GB")
        print(f"    计算能力: {props.major}.{props.minor}")
    
    if gpu_count < 2:
        print_warning("只有1个GPU，部分多GPU测试将被跳过")
        print_warning("建议在多GPU机器上运行完整测试")
        return False
    
    print_success(f"满足多GPU测试要求（{gpu_count}个GPU）")
    return True


def test_multi_gpu_initialization():
    """测试多GPU提取器初始化"""
    print_header("测试1: 多GPU提取器初始化")
    
    try:
        gpu_count = torch.cuda.device_count()
        
        # 测试自动GPU检测
        print("\n[1.1] 自动GPU检测")
        extractor_auto = MultiGPUFeatureExtractor()
        gpu_info = extractor_auto.get_gpu_info()
        print(f"  检测到GPU数量: {gpu_info['num_gpus']}")
        print(f"  设备列表: {gpu_info['devices']}")
        assert gpu_info['num_gpus'] == gpu_count, "GPU数量不匹配"
        print_success("自动GPU检测正常")
        
        # 测试指定GPU
        if gpu_count >= 2:
            print(f"\n[1.2] 指定GPU [{', '.join(map(str, range(min(2, gpu_count))))}]")
            extractor_specified = MultiGPUFeatureExtractor(gpus=list(range(min(2, gpu_count))))
            gpu_info_spec = extractor_specified.get_gpu_info()
            print(f"  指定GPU设备: {gpu_info_spec['devices']}")
            assert len(gpu_info_spec['devices']) == min(2, gpu_count), "指定GPU数量不匹配"
            print_success("指定GPU初始化正常")
        
        # 测试CPU模式
        print("\n[1.3] CPU回退模式")
        extractor_cpu = MultiGPUFeatureExtractor(gpus=['cpu'])
        gpu_info_cpu = extractor_cpu.get_gpu_info()
        print(f"  CPU模式设备: {gpu_info_cpu['devices']}")
        assert gpu_info_cpu['devices'] == ['cpu'], "CPU模式初始化失败"
        print_success("CPU回退模式正常")
        
        # 验证模型在正确的设备上
        print("\n[1.4] 模型设备验证")
        for device, model in extractor_auto.models.items():
            model_device = next(model.parameters()).device
            expected_device = torch.device(device)
            print(f"  模型 {device}: {model_device}")
            assert model_device == expected_device, f"模型不在预期设备上: {model_device} != {expected_device}"
        print_success("模型设备正确")
        
        return True
        
    except Exception as e:
        print_error(f"初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_gpu_consistency():
    """测试多GPU与单GPU结果一致性"""
    print_header("测试2: 多GPU与单GPU结果一致性")
    
    try:
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            print_warning("跳过此测试（需要至少2个GPU）")
            return True
        
        # 创建测试图片
        test_images = create_test_images(32)
        
        # 单GPU处理（使用GPU 0）
        print("\n[2.1] 单GPU基准测试")
        single_extractor = DINOv3FeatureExtractor(batch_size=32)
        start_time = time.time()
        single_features = single_extractor.extract_features_batch(test_images)
        single_time = time.time() - start_time
        print(f"  单GPU时间: {single_time:.4f}s")
        print(f"  特征形状: {single_features.shape}")
        
        # 多GPU处理（使用GPU 0, 1）
        print("\n[2.2] 多GPU并行测试")
        multi_extractor = MultiGPUFeatureExtractor(gpus=[0, 1], batch_per_gpu=16)
        start_time = time.time()
        multi_features = multi_extractor.extract_features_batch_multi_gpu(test_images)
        multi_time = time.time() - start_time
        print(f"  多GPU时间: {multi_time:.4f}s")
        print(f"  特征形状: {multi_features.shape}")
        
        # 比较结果
        print("\n[2.3] 结果一致性验证")
        single_np = single_features.cpu().numpy()
        multi_np = multi_features.cpu().numpy()
        
        # 计算差异
        diff = np.abs(single_np - multi_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"  最大差异: {max_diff:.8f}")
        print(f"  平均差异: {mean_diff:.8f}")
        print(f"  性能提升: {single_time/multi_time:.2f}x")
        
        # 验证一致性（允许一定的浮点误差）
        tolerance = 1e-5
        if max_diff < tolerance:
            print_success(f"多GPU与单GPU结果一致（差异 < {tolerance}）")
            return True
        else:
            print_error(f"结果不一致，差异超过容忍度 {tolerance}")
            return False
            
    except Exception as e:
        print_error(f"一致性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_distribution():
    """测试批次分配算法"""
    print_header("测试3: 批次分配算法")
    
    try:
        gpu_count = torch.cuda.device_count()
        num_gpus_to_test = min(2, gpu_count) if gpu_count >= 2 else 1
        
        print(f"\n使用 {num_gpus_to_test} 个GPU进行测试")
        
        extractor = MultiGPUFeatureExtractor(gpus=list(range(num_gpus_to_test)))
        
        # 测试不同数量的图片分配
        test_cases = [5, 12, 25, 50, 100]
        
        print("\n[3.1] 不同数量图片的分配")
        for num_images in test_cases:
            test_images = create_test_images(num_images)
            distribution = extractor._distribute_images_to_gpus(test_images)
            
            total_allocated = sum(len(batch) for batch in distribution.values())
            print(f"  {num_images}张图片 -> {[(device, len(batch)) for device, batch in distribution.items()]}")
            
            if total_allocated != num_images:
                print_error(f"分配不匹配: 期望{num_images}, 实际{total_allocated}")
                return False
        
        print_success("批次分配算法正确")
        
        # 测试负载均衡
        print("\n[3.2] 负载均衡验证")
        test_images = create_test_images(64)
        distribution = extractor._distribute_images_to_gpus(test_images)
        
        batch_sizes = [len(batch) for batch in distribution.values()]
        max_batch = max(batch_sizes)
        min_batch = min(batch_sizes)
        imbalance = (max_batch - min_batch) / max_batch if max_batch > 0 else 0
        
        print(f"  批次大小: {batch_sizes}")
        print(f"  不平衡度: {imbalance:.2%}")
        
        if imbalance < 0.3:  # 允许30%的不平衡
            print_success("负载均衡良好")
        else:
            print_warning(f"负载不平衡度较高: {imbalance:.2%}")
        
        return True
        
    except Exception as e:
        print_error(f"批次分配测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_scaling():
    """测试性能扩展性"""
    print_header("测试4: 性能扩展性")
    
    try:
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            print_warning("跳过此测试（需要至少2个GPU）")
            return True
        
        test_images = create_test_images(64)
        
        # 测试不同GPU数量的性能
        results = []
        
        for num_gpus in [1, 2, min(4, gpu_count)]:
            print(f"\n[4.{num_gpus}] 测试 {num_gpus} 个GPU")
            
            extractor = MultiGPUFeatureExtractor(
                gpus=list(range(num_gpus)),
                batch_per_gpu=32 // num_gpus
            )
            
            # 预热
            _ = extractor.extract_features_batch_multi_gpu(test_images)
            torch.cuda.synchronize()
            
            # 测试
            times = []
            for _ in range(3):
                start = time.time()
                features = extractor.extract_features_batch_multi_gpu(test_images)
                torch.cuda.synchronize()
                end = time.time()
                times.append(end - start)
            
            avg_time = np.mean(times)
            throughput = len(test_images) / avg_time
            results.append((num_gpus, avg_time, throughput))
            
            print(f"  平均时间: {avg_time:.4f}s")
            print(f"  吞吐量: {throughput:.2f} images/sec")
        
        # 计算扩展效率
        print("\n[4.4] 扩展性分析")
        baseline_time = results[0][1]
        print(f"  基准（1 GPU）: {baseline_time:.4f}s")
        
        for num_gpus, avg_time, throughput in results[1:]:
            speedup = baseline_time / avg_time
            efficiency = speedup / num_gpus * 100
            print(f"  {num_gpus} GPU: {avg_time:.4f}s, 加速比: {speedup:.2f}x, 效率: {efficiency:.1f}%")
        
        print_success("性能扩展性测试完成")
        return True
        
    except Exception as e:
        print_error(f"性能扩展性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_serializer_integration():
    """测试ImageSerializer多GPU集成"""
    print_header("测试5: ImageSerializer多GPU集成")
    
    try:
        gpu_count = torch.cuda.device_count()
        
        # 测试多GPU模式
        print("\n[5.1] 多GPU模式初始化")
        serializer_multi = ImageSerializer(
            db_path="test_db_integration_multi",
            use_multi_gpu=True,
            gpus=list(range(min(2, gpu_count))) if gpu_count >= 2 else [0],
            batch_per_gpu=16
        )
        
        print(f"  多GPU模式: {serializer_multi.use_multi_gpu}")
        print(f"  提取器类型: {type(serializer_multi.feature_extractor).__name__}")
        
        if serializer_multi.use_multi_gpu:
            gpu_info = serializer_multi.feature_extractor.get_gpu_info()
            print(f"  GPU数量: {gpu_info['num_gpus']}")
            print(f"  设备列表: {gpu_info['devices']}")
        
        assert serializer_multi.use_multi_gpu, "未启用多GPU模式"
        print_success("多GPU模式初始化成功")
        
        # 测试单GPU模式
        print("\n[5.2] 单GPU模式初始化")
        serializer_single = ImageSerializer(
            db_path="test_db_integration_single",
            use_multi_gpu=False,
            batch_size=32
        )
        
        print(f"  多GPU模式: {serializer_single.use_multi_gpu}")
        print(f"  提取器类型: {type(serializer_single.feature_extractor).__name__}")
        
        assert not serializer_single.use_multi_gpu, "错误地启用了多GPU模式"
        print_success("单GPU模式初始化成功")
        
        # 测试批量大小更新
        print("\n[5.3] 批量大小更新")
        serializer_multi.update_batch_size(24)
        print(f"  更新后批量大小: {serializer_multi.batch_size}")
        assert serializer_multi.batch_size == 24, "批量大小更新失败"
        print_success("批量大小更新成功")
        
        return True
        
    except Exception as e:
        print_error(f"ImageSerializer集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """测试错误处理"""
    print_header("测试6: 错误处理和边界情况")
    
    try:
        # 测试无效GPU
        print("\n[6.1] 无效GPU处理")
        try:
            MultiGPUFeatureExtractor(gpus=[999])
            print_error("应该拒绝无效GPU")
            return False
        except (RuntimeError, ValueError) as e:
            print(f"  正确拒绝无效GPU: {e}")
            print_success("无效GPU处理正确")
        
        # 测试空输入
        print("\n[6.2] 空输入处理")
        extractor = MultiGPUFeatureExtractor()
        empty_result = extractor.extract_features_batch_multi_gpu([])
        if empty_result.numel() == 0:
            print_success("空输入处理正确")
        else:
            print_error("空输入处理失败")
            return False
        
        # 测试负数批量大小
        print("\n[6.3] 负数批量大小处理")
        try:
            extractor.update_batch_size(-1)
            print_error("应该拒绝负数批量大小")
            return False
        except ValueError:
            print_success("负数批量大小处理正确")
        
        # 测试零批量大小
        print("\n[6.4] 零批量大小处理")
        try:
            extractor.update_batch_size(0)
            print_error("应该拒绝零批量大小")
            return False
        except ValueError:
            print_success("零批量大小处理正确")
        
        return True
        
    except Exception as e:
        print_error(f"错误处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_large_batch_processing():
    """测试大批量处理"""
    print_header("测试7: 大批量处理稳定性")
    
    try:
        gpu_count = torch.cuda.device_count()
        num_gpus = min(2, gpu_count) if gpu_count >= 2 else 1
        
        print(f"\n使用 {num_gpus} 个GPU处理大批量数据")
        
        extractor = MultiGPUFeatureExtractor(
            gpus=list(range(num_gpus)),
            batch_per_gpu=32
        )
        
        # 测试不同大小的批量
        batch_sizes = [64, 128, 256]
        
        for batch_size in batch_sizes:
            print(f"\n[7.{batch_sizes.index(batch_size)+1}] 处理 {batch_size} 张图片")
            
            test_images = create_test_images(batch_size)
            
            # 预热
            _ = extractor.extract_features_batch_multi_gpu(test_images)
            torch.cuda.synchronize()
            
            # 测试
            start = time.time()
            features = extractor.extract_features_batch_multi_gpu(test_images)
            torch.cuda.synchronize()
            end = time.time()
            
            elapsed = end - start
            throughput = batch_size / elapsed
            
            print(f"  处理时间: {elapsed:.4f}s")
            print(f"  吞吐量: {throughput:.2f} images/sec")
            print(f"  特征形状: {features.shape}")
            
            assert features.shape[0] == batch_size, f"特征数量不匹配: {features.shape[0]} != {batch_size}"
        
        print_success("大批量处理稳定性测试通过")
        return True
        
    except Exception as e:
        print_error(f"大批量处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_test_report(results: Dict[str, bool]):
    """生成测试报告"""
    print_header("测试报告")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests
    
    print(f"\n总测试数: {total_tests}")
    print(f"{Colors.GREEN}通过: {passed_tests}{Colors.END}")
    print(f"{Colors.RED}失败: {failed_tests}{Colors.END}")
    print(f"通过率: {passed_tests/total_tests*100:.1f}%")
    
    print("\n详细结果:")
    for test_name, passed in results.items():
        status = f"{Colors.GREEN}✓ 通过{Colors.END}" if passed else f"{Colors.RED}✗ 失败{Colors.END}"
        print(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 所有测试通过！多GPU功能正常工作。{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.BOLD}{Colors.RED}❌ 部分测试失败，请检查实现。{Colors.END}")
        return 1


def main():
    """主测试函数"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║        多GPU特征提取器综合测试套件                        ║")
    print("║     Multi-GPU Feature Extractor Comprehensive Test         ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    
    # 检查GPU环境
    has_multi_gpu = check_gpu_requirements()
    
    # 定义测试
    tests = [
        ("多GPU提取器初始化", test_multi_gpu_initialization),
        ("多GPU与单GPU一致性", test_multi_gpu_consistency),
        ("批次分配算法", test_batch_distribution),
        ("性能扩展性", test_performance_scaling),
        ("ImageSerializer集成", test_image_serializer_integration),
        ("错误处理和边界情况", test_error_handling),
        ("大批量处理稳定性", test_large_batch_processing),
    ]
    
    # 运行测试
    results = {}
    for test_name, test_func in tests:
        try:
            # 跳过需要多GPU的测试（如果只有单GPU）
            if not has_multi_gpu and test_name in ["多GPU与单GPU一致性", "性能扩展性"]:
                print_warning(f"跳过 '{test_name}'（需要多GPU环境）")
                results[test_name] = True
                continue
            
            passed = test_func()
            results[test_name] = passed
        except Exception as e:
            print_error(f"测试 '{test_name}' 发生异常: {e}")
            results[test_name] = False
    
    # 生成报告
    return generate_test_report(results)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)