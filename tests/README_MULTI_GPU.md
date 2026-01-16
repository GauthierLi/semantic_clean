# 多GPU功能测试指南

## 概述

本项目支持多GPU并行特征提取，可以显著提升大规模图像处理的速度。本文档说明如何测试和使用多GPU功能。

## 环境要求

### 硬件要求
- **最少2个GPU**（推荐4个或更多）
- 每个GPU至少8GB显存
- CUDA兼容的NVIDIA GPU

### 软件要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+
- 项目依赖（见 `requirements.txt`）

## 测试文件说明

### 1. `test_multi_gpu_extractor.py`
基础多GPU提取器测试，包括：
- 多GPU初始化
- GPU设备检测
- 批次分配算法
- 错误处理
- 性能对比

**运行方式：**
```bash
python tests/test_multi_gpu_extractor.py
```

### 2. `test_multi_gpu_integration.py`
ImageSerializer多GPU集成测试，包括：
- 多GPU模式初始化
- 单GPU模式初始化
- 批量大小更新

**运行方式：**
```bash
python tests/test_multi_gpu_integration.py
```

### 3. `test_multi_gpu_comprehensive.py` ⭐ 推荐
综合测试套件，专为多GPU环境设计，包括：
- GPU环境检查
- 多GPU提取器初始化
- 多GPU与单GPU结果一致性
- 批次分配算法
- 性能扩展性测试
- ImageSerializer集成
- 错误处理和边界情况
- 大批量处理稳定性

**运行方式：**
```bash
# 使用所有可用GPU
python tests/test_multi_gpu_comprehensive.py

# 指定使用的GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/test_multi_gpu_comprehensive.py
```

## 在多GPU机器上运行测试

### 步骤1：检查GPU环境

```bash
# 检查GPU数量和状态
nvidia-smi

# 检查PyTorch能否识别GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 步骤2：运行基础测试

```bash
# 在单GPU机器上运行（会跳过部分测试）
python tests/test_multi_gpu_extractor.py

# 在多GPU机器上运行（完整测试）
CUDA_VISIBLE_DEVICES=0,1 python tests/test_multi_gpu_extractor.py
```

### 步骤3：运行综合测试

```bash
# 使用所有可用GPU
python tests/test_multi_gpu_comprehensive.py

# 指定GPU数量
CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/test_multi_gpu_comprehensive.py

# 使用特定GPU
CUDA_VISIBLE_DEVICES=0,2,4,6 python tests/test_multi_gpu_comprehensive.py
```

### 步骤4：性能基准测试

```bash
# 运行性能基准测试
python tests/performance_benchmark.py

# 批量特征提取测试
python tests/test_batch_feature_extraction.py
```

## 测试输出说明

### 成功的测试输出示例

```
╔═══════════════════════════════════════════════════════════╗
║        多GPU特征提取器综合测试套件                        ║
║     Multi-GPU Feature Extractor Comprehensive Test         ║
╚═══════════════════════════════════════════════════════════╝

============================================================
GPU环境检查
============================================================
检测到 4 个GPU
  GPU 0: NVIDIA GeForce RTX 3090
    总内存: 24.00GB
    计算能力: 8.6
  GPU 1: NVIDIA GeForce RTX 3090
    总内存: 24.00GB
    计算能力: 8.6
  GPU 2: NVIDIA GeForce RTX 3090
    总内存: 24.00GB
    计算能力: 8.6
  GPU 3: NVIDIA GeForce RTX 3090
    总内存: 24.00GB
    计算能力: 8.6
✓ 满足多GPU测试要求（4个GPU）

============================================================
测试报告
============================================================
总测试数: 7
通过: 7
失败: 0
通过率: 100.0%

🎉 所有测试通过！多GPU功能正常工作。
```

### 性能扩展性示例

```
============================================================
测试4: 性能扩展性
============================================================

[4.1] 测试 1 个GPU
  平均时间: 4.5234s
  吞吐量: 14.14 images/sec

[4.2] 测试 2 个GPU
  平均时间: 2.3456s
  吞吐量: 27.28 images/sec

[4.3] 测试 4 个GPU
  平均时间: 1.2345s
  吞吐量: 51.84 images/sec

[4.4] 扩展性分析
  基准（1 GPU）: 4.5234s
  2 GPU: 2.3456s, 加速比: 1.93x, 效率: 96.5%
  4 GPU: 1.2345s, 加速比: 3.66x, 效率: 91.5%
```

## 常见问题

### Q1: 测试提示"只有1个GPU，部分多GPU测试将被跳过"

**原因：** 当前机器只有1个GPU，部分需要多GPU的测试被跳过。

**解决：** 在多GPU机器上运行测试，或使用 `CUDA_VISIBLE_DEVICES` 模拟多GPU环境（如果有多张物理GPU）。

### Q2: 测试失败，提示"CUDA out of memory"

**原因：** GPU内存不足，无法加载模型或处理大批量数据。

**解决：**
1. 减少每GPU的批量大小（`batch_per_gpu`）
2. 关闭其他占用GPU的程序
3. 使用更小的模型或降低输入图像分辨率

### Q3: 多GPU性能提升不明显

**可能原因：**
1. 数据传输开销过大
2. 批量大小太小
3. GPU之间负载不均衡

**解决：**
1. 增加批量大小
2. 检查批次分配算法
3. 使用NVLink或PCIe 4.0+的高速GPU互联

### Q4: 如何在代码中启用多GPU模式？

**示例代码：**
```python
from src.image_serialize import ImageSerializer

# 方式1：自动检测所有GPU
serializer = ImageSerializer(
    db_path="./my_db",
    use_multi_gpu=True  # 启用多GPU模式
)

# 方式2：指定GPU
serializer = ImageSerializer(
    db_path="./my_db",
    use_multi_gpu=True,
    gpus=[0, 1, 2, 3],  # 使用GPU 0-3
    batch_per_gpu=32     # 每个GPU的批量大小
)

# 方式3：禁用多GPU（使用单GPU）
serializer = ImageSerializer(
    db_path="./my_db",
    use_multi_gpu=False
)
```

## 性能优化建议

### 1. 批量大小调优

```python
# 根据GPU内存调整批量大小
# 小显存（8GB）: batch_per_gpu=8-16
# 中等显存（16GB）: batch_per_gpu=16-32
# 大显存（24GB+）: batch_per_gpu=32-64
```

### 2. GPU选择

```bash
# 选择性能最好的GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/test_multi_gpu_comprehensive.py

# 避免使用被其他程序占用的GPU
```

### 3. 数据预处理

- 使用NVMe SSD存储图像
- 预先加载和缓存图像数据
- 使用多线程数据加载

## 联系和支持

如有问题或建议，请：
1. 查看项目文档
2. 检查GPU驱动和CUDA版本
3. 运行测试并查看详细日志
4. 提交Issue到项目仓库

## 更新日志

### v1.0.0 (2026-01-15)
- 初始版本
- 支持多GPU并行特征提取
- 完整的测试套件
- 性能优化和错误处理