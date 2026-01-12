import os
import cv2
import torch
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Union, Optional
import logging

from torchvision import transforms
from sklearn.decomposition import PCA

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiGPUFeatureExtractor:
    """多GPU批量特征提取器，支持在多个GPU上并行推理"""
    
    def __init__(self, model_path=None, gpus=None, batch_per_gpu=32):
        """初始化多GPU特征提取器
        
        Args:
            model_path: 模型路径
            gpus: GPU设备列表，如[0, 1, 2, 3]或['cuda:0', 'cuda:1']，None表示自动检测
            batch_per_gpu: 每个GPU的批量大小
        """
        self.device_list = self._validate_and_setup_gpus(gpus)
        self.batch_per_gpu = batch_per_gpu
        self.models = self._load_models_on_gpus(model_path)
        self.transform = self._make_transform()
        self.num_gpus = len(self.device_list)
        
        logger.info(f"MultiGPUFeatureExtractor initialized with {self.num_gpus} GPUs: {self.device_list}")
        logger.info(f"Batch size per GPU: {batch_per_gpu}")

    def _validate_and_setup_gpus(self, gpus):
        """验证和设置GPU设备"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return ['cpu']
        
        if gpus is None:
            # 自动检测可用GPU
            available_gpus = [i for i in range(torch.cuda.device_count())]
            logger.info(f"Auto-detected {len(available_gpus)} GPUs: {available_gpus}")
        else:
            # 验证指定的GPU
            available_gpus = []
            for gpu in gpus:
                if isinstance(gpu, int):
                    device_id = gpu
                elif isinstance(gpu, str) and gpu.startswith('cuda:'):
                    device_id = int(gpu.split(':')[1])
                elif gpu == 'cpu':
                    available_gpus = ['cpu']
                    break
                else:
                    raise ValueError(f"Invalid GPU format: {gpu}")
                
                if device_id < torch.cuda.device_count():
                    available_gpus.append(device_id)
                else:
                    logger.warning(f"GPU {device_id} not available, skipping")
        
        if not available_gpus:
            raise RuntimeError("No available GPUs found")
        
        # 转换为设备字符串格式
        if available_gpus == ['cpu']:
            return ['cpu']
        return [f"cuda:{i}" for i in available_gpus]

    def _load_models_on_gpus(self, model_path):
        """在每个GPU上加载模型"""
        models = {}
        
        for device in self.device_list:
            try:
                model = self._load_single_model(model_path)
                
                if device != 'cpu':
                    model = model.to(device)
                    # 设置模型为eval模式
                    model.eval()
                
                models[device] = model
                logger.info(f"Model loaded successfully on {device}")
                
            except Exception as e:
                logger.error(f"Failed to load model on {device}: {e}")
                raise
        
        return models

    def _load_single_model(self, model_path=None):
        """加载单个DINOv3模型"""
        dino3_path = os.path.join(os.getcwd(), 'DINOv3')
        if not os.path.exists(dino3_path):
            result = subprocess.run(
                ['git', 'clone', 'git@github.com:facebookresearch/dinov3.git', 'DINOv3'],
                capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to pull DINOv3 model: {result.stderr.decode()}")
        
        if model_path is None:
            model_path = os.path.join(os.getcwd(), 'dinov3_vitb16_pretrain.pth')
            
        return torch.hub.load(dino3_path, 'dinov3_vitb16', source='local', weights=model_path)

    def _make_transform(self, resize_size=256):
        """创建图像变换管道"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])

    def _get_gpu_memory_info(self):
        """获取各GPU内存使用情况"""
        memory_info = {}
        
        for device in self.device_list:
            if device == 'cpu':
                memory_info[device] = 8 * 1024**3  # 假设CPU有8GB可用内存
            else:
                device_id = int(device.split(':')[1])
                total_memory = torch.cuda.get_device_properties(device_id).total_memory
                allocated_memory = torch.cuda.memory_allocated(device_id)
                free_memory = total_memory - allocated_memory
                memory_info[device] = free_memory
        
        return memory_info

    def _distribute_images_to_gpus(self, images: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
        """智能分配图片到不同GPU"""
        if len(images) <= self.batch_per_gpu:
            # 如果图片数量少，只使用第一个GPU
            return {self.device_list[0]: images}
        
        # 获取各GPU内存使用情况
        gpu_memory_info = self._get_gpu_memory_info()
        
        # 计算分配比例
        total_memory = sum(gpu_memory_info.values())
        allocations = {}
        
        for device, memory in gpu_memory_info.items():
            ratio = memory / total_memory
            batch_count = max(1, int(len(images) * ratio))
            allocations[device] = batch_count
        
        # 确保所有图片都被分配
        allocated_sum = sum(allocations.values())
        if allocated_sum < len(images):
            remaining = len(images) - allocated_sum
            # 将剩余的图片分配给内存最多的GPU
            best_device = max(gpu_memory_info.items(), key=lambda x: x[1])[0]
            allocations[best_device] += remaining
        
        # 分割图片
        result = {}
        start_idx = 0
        
        for device, count in allocations.items():
            end_idx = min(start_idx + count, len(images))
            result[device] = images[start_idx:end_idx]
            start_idx = end_idx
            
            if start_idx >= len(images):
                break
        
        logger.info(f"Distributed {len(images)} images across {len(result)} GPUs: {list(result.keys())}")
        return result

    def _extract_on_gpu(self, device: str, images: List[np.ndarray]) -> torch.Tensor:
        """在指定GPU上提取特征"""
        try:
            model = self.models[device]
            
            # 转换图片为tensor
            batch_tensors = []
            for img in images:
                tensor = self.transform(img)
                if device != 'cpu':
                    tensor = tensor.to(device)
                batch_tensors.append(tensor)
            
            # 批量处理
            batch_tensor = torch.stack(batch_tensors)
            
            with torch.no_grad():
                if device != 'cpu':
                    with torch.cuda.device(device):
                        features = model.forward_features(batch_tensor)
                else:
                    features = model.forward_features(batch_tensor)
            
            # 标准化特征
            normalized_features = torch.nn.functional.normalize(features['x_norm_clstoken'], dim=-1)
            
            # 将结果移回CPU
            if device != 'cpu':
                normalized_features = normalized_features.cpu()
            
            return normalized_features
            
        except Exception as e:
            logger.error(f"Error extracting features on {device}: {e}")
            raise

    def extract_features_batch_multi_gpu(self, images: List[np.ndarray], batch_size: Optional[int] = None) -> torch.Tensor:
        """多GPU并行批量特征提取
        
        Args:
            images: 图片列表
            batch_size: 总批量大小，如果为None则自动计算
            
        Returns:
            合并后的特征张量
        """
        if not images:
            return torch.empty(0, 0)
        
        if self.num_gpus == 1 or self.device_list[0] == 'cpu':
            # 单GPU或CPU模式，直接使用标准方法
            logger.info("Using single GPU/CPU mode")
            return self._extract_on_gpu(self.device_list[0], images)
        
        # 分配图片到不同GPU
        gpu_batches = self._distribute_images_to_gpus(images)
        
        # 并行处理各GPU的batch
        results = []
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            # 提交任务
            future_to_device = {
                executor.submit(self._extract_on_gpu, device, batch_images): device
                for device, batch_images in gpu_batches.items()
                if batch_images  # 只处理非空batch
            }
            
            # 收集结果
            for future in as_completed(future_to_device):
                device = future_to_device[future]
                try:
                    result = future.result()
                    results.append((device, result))
                    logger.info(f"Completed batch processing on {device}, shape: {result.shape}")
                except Exception as e:
                    logger.error(f"Failed to process batch on {device}: {e}")
                    # 尝试在CPU上重新处理
                    try:
                        batch_images = gpu_batches[device]
                        cpu_result = self._extract_on_gpu('cpu', batch_images)
                        results.append(('cpu', cpu_result))
                        logger.info(f"Fallback to CPU for batch from {device}")
                    except Exception as fallback_error:
                        logger.error(f"CPU fallback also failed: {fallback_error}")
                        continue
        
        if not results:
            raise RuntimeError("All GPU processing failed")
        
        # 按原始顺序合并结果
        sorted_results = sorted(results, key=lambda x: list(gpu_batches.keys()).index(x[0]))
        final_features = torch.cat([result for _, result in sorted_results], dim=0)
        
        logger.info(f"Multi-GPU processing completed. Final feature shape: {final_features.shape}")
        return final_features

    def get_gpu_info(self) -> Dict:
        """获取GPU信息"""
        info = {
            'num_gpus': self.num_gpus,
            'devices': self.device_list,
            'memory_info': self._get_gpu_memory_info(),
            'batch_per_gpu': self.batch_per_gpu
        }
        
        # 添加GPU详细信息
        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['gpu_names'] = []
            for device in self.device_list:
                if device != 'cpu':
                    device_id = int(device.split(':')[1])
                    props = torch.cuda.get_device_properties(device_id)
                    info['gpu_names'].append({
                        'device': device,
                        'name': props.name,
                        'total_memory': props.total_memory,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
        else:
            info['cuda_available'] = False
        
        return info

    def update_batch_size(self, new_batch_size: int):
        """更新每GPU批量大小"""
        if new_batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {new_batch_size}")
        
        self.batch_per_gpu = new_batch_size
        logger.info(f"Updated batch size per GPU to {new_batch_size}")
