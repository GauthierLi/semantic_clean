#!/usr/bin/env python3
"""
测试多GPU特征提取器集成功能
"""

import os
import sys
import json
import numpy as np
import cv2

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_serialize import ImageSerializer

def test_multi_gpu_initialization():
    """测试多GPU模式初始化"""
    print("=== 测试多GPU模式初始化 ===")
    
    try:
        # 测试强制使用多GPU模式
        serializer = ImageSerializer(
            db_path="test_db_multi_gpu",
            use_multi_gpu=True,
            batch_per_gpu=4
        )
        
        print(f"多GPU模式状态: {serializer.use_multi_gpu}")
        print(f"特征提取器类型: {type(serializer.feature_extractor).__name__}")
        
        if serializer.use_multi_gpu:
            gpu_info = serializer.feature_extractor.get_gpu_info()
            print(f"GPU信息: {gpu_info}")
        
        return True
        
    except Exception as e:
        print(f"多GPU初始化测试失败: {e}")
        return False

def test_single_gpu_initialization():
    """测试单GPU模式初始化"""
    print("\n=== 测试单GPU模式初始化 ===")
    
    try:
        # 测试强制使用单GPU模式
        serializer = ImageSerializer(
            db_path="test_db_single_gpu",
            use_multi_gpu=False,
            batch_size=8
        )
        
        print(f"多GPU模式状态: {serializer.use_multi_gpu}")
        print(f"特征提取器类型: {type(serializer.feature_extractor).__name__}")
        
        return True
        
    except Exception as e:
        print(f"单GPU初始化测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始多GPU集成功能测试...")
    
    tests = [
        test_multi_gpu_initialization,
        test_single_gpu_initialization,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"测试异常: {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✅ 基础测试通过！")
        result = 0
    else:
        print("❌ 部分测试失败")
        result = 1
    
    return result

if __name__ == "__main__":
    sys.exit(main())
