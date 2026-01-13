import os
import cv2
import json
import numpy as np
import torch

from tqdm import tqdm
from typing import Dict, List, Optional
from .chroma_manager import ChromaDBManager
from .feature_extractor import DINOv3FeatureExtractor
from .multi_gpu_extractor import MultiGPUFeatureExtractor

class ImageSerializer:
    def __init__(self, db_path: str = "chroma_db", model_path: str = None, batch_size: int = 8, 
                 use_multi_gpu: bool = True, gpus: List[str] = None, batch_per_gpu: int = 8):
        """Initialize image serializer with multi-GPU support
        
        Args:
            db_path: ChromaDB数据库路径
            model_path: 模型路径
            batch_size: 总体批量大小（向后兼容）
            use_multi_gpu: 是否使用多GPU模式
            gpus: GPU设备列表，如[0, 1, 2]或None表示自动检测
            batch_per_gpu: 每个GPU的批量大小
        """
        if use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            try:
                self.feature_extractor = MultiGPUFeatureExtractor(
                    model_path=model_path, 
                    gpus=gpus, 
                    batch_per_gpu=batch_per_gpu
                )
                self.use_multi_gpu = True
                print(f"Multi-GPU mode enabled with {torch.cuda.device_count()} GPUs")
            except Exception as e:
                print(f"Failed to initialize multi-GPU mode: {e}")
                print("Falling back to single-GPU mode")
                self.feature_extractor = DINOv3FeatureExtractor(model_path, batch_size)
                self.use_multi_gpu = False
        else:
            self.feature_extractor = DINOv3FeatureExtractor(model_path, batch_size)
            self.use_multi_gpu = False
            if not torch.cuda.is_available():
                print("CUDA not available, using CPU mode")
            elif torch.cuda.device_count() <= 1:
                print("Single GPU detected, using single-GPU mode")
        
        self.db_manager = ChromaDBManager(db_path)
        self.batch_size = batch_size  # 统一的批量大小

    def __call__(self, image: np.ndarray, metadata: Dict):
        """Process and store single image"""
        feature = self.feature_extractor.extract_features(image)
        self.db_manager.store_features(
            features=feature.cpu().numpy(),
            metadata=metadata
        )
        return feature

    def load_from_json(self, json_path: str, batch_size: int = None):
        """使用批量处理优化JSON数据加载，支持多GPU并行处理
        
        Args:
            json_path: JSON文件路径
            batch_size: 批量大小，如果为None则使用实例默认值
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if self.use_multi_gpu:
            # 多GPU模式：使用更大的批量进行并行处理
            effective_batch_size = batch_size * torch.cuda.device_count()
            print(f"Using multi-GPU mode with effective batch size: {effective_batch_size}")
        else:
            effective_batch_size = batch_size
            print(f"Using single-GPU/CPU mode with batch size: {effective_batch_size}")
        
        # 分批处理：特征提取 -> 数据库存储
        for i in tqdm(range(0, len(data), effective_batch_size), desc="Processing batches"):
            batch_data = data[i:i+effective_batch_size]
            valid_images = []
            valid_metadata = []
            valid_ids = []
            
            # 收集本批次有效数据
            for item in batch_data:
                if not os.path.exists(item['image_path']):
                    continue
                    
                image = cv2.imread(item['image_path'])
                if image is None:
                    continue
                    
                valid_images.append(image)
                # 将类别列表转换为标准化的布尔字段
                category_list = [str(c) for c in item.get('category', [])]
                metadata_entry = {
                    'image_path': item['image_path'],
                }
                # 为每个类别创建独立的布尔字段
                for category in category_list:
                    metadata_entry[f'is_{category}'] = True
                
                valid_metadata.append(metadata_entry)
                valid_ids.append(item.get('id', str(len(valid_ids))))
            
            # 批量特征提取和存储（一次性完成）
            if valid_images:
                try:
                    if self.use_multi_gpu:
                        # 多GPU模式使用专门的批量处理方法
                        try:
                            batch_features = self.feature_extractor.extract_features_batch_multi_gpu(valid_images)
                        except Exception as gpu_error:
                            print(f"Multi-GPU processing failed: {gpu_error}")
                            print("Attempting fallback to single-GPU mode...")
                            # 回退到单GPU模式
                            fallback_extractor = DINOv3FeatureExtractor(
                                model_path=getattr(self.feature_extractor, 'models', {}).get(self.feature_extractor.device_list[0] if hasattr(self.feature_extractor, 'device_list') else 'cpu', None),
                                batch_size=self.batch_size
                            )
                            batch_features = fallback_extractor.extract_features_batch(valid_images)
                            print("Successfully processed with fallback single-GPU mode")
                    else:
                        # 单GPU模式使用标准方法
                        batch_features = self.feature_extractor.extract_features_batch(valid_images)
                        
                    if batch_features.numel() > 0:  # 确保特征不为空
                        self.db_manager.store_features(
                            features=batch_features.cpu().numpy(),
                            metadata=valid_metadata,
                            ids=valid_ids
                        )
                except Exception as e:
                    print(f"Warning: Failed to process batch {i//effective_batch_size}: {e}")
                    # 尝试逐个处理图片，避免整个批次失败
                    print("Attempting individual image processing...")
                    for idx, (img, meta, img_id) in enumerate(zip(valid_images, valid_metadata, valid_ids)):
                        try:
                            single_feature = self.feature_extractor.extract_features(img)
                            if single_feature.numel() > 0:
                                self.db_manager.store_features(
                                    features=single_feature.cpu().numpy(),
                                    metadata=[meta],
                                    ids=[img_id]
                                )
                        except Exception as single_error:
                            print(f"Failed to process individual image {img_id}: {single_error}")
                            continue
                    continue
    
    def update_batch_size(self, new_batch_size: int):
        """动态更新批量大小配置"""
        if new_batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {new_batch_size}")
        self.batch_size = new_batch_size
        
        # 根据当前使用的特征提取器类型调用相应的更新方法
        if self.use_multi_gpu:
            # 多GPU模式：更新每GPU的批量大小
            if hasattr(self.feature_extractor, 'update_batch_size'):
                self.feature_extractor.update_batch_size(new_batch_size)
            else:
                print("Warning: MultiGPUFeatureExtractor does not support batch size update")
        else:
            # 单GPU模式：更新批量大小
            self.feature_extractor.update_batch_size(new_batch_size)
        
        print(f"Updated batch size to {new_batch_size} (mode: {'multi-GPU' if self.use_multi_gpu else 'single-GPU/CPU'})")

    def search_by_image(self, image: np.ndarray, n_results: int = 5) -> List[Dict]:
        """Search similar images by image query"""
        query_feature = self.feature_extractor.extract_features(image)
        return self.db_manager.query_by_feature(
            query_feature=query_feature.cpu().numpy(),
            n_results=n_results
        )

    def search_by_category(self, category: str, n_results: int = 5) -> List[Dict]:
        """Search images by category"""
        return self.db_manager.query_by_category(
            category=category,
            n_results=n_results
        )

    def get_categories(self) -> List[str]:
        """Get list of available categories"""
        return self.db_manager.get_all_categories()