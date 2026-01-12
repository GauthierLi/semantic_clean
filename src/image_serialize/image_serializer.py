import os
import cv2
import json
import numpy as np

from tqdm import tqdm
from typing import Dict, List, Optional
from .chroma_manager import ChromaDBManager
from .feature_extractor import DINOv3FeatureExtractor

class ImageSerializer:
    def __init__(self, db_path: str = "chroma_db", model_path: str = None, batch_size: int = 8):
        """Initialize image serializer with configurable batch size
        
        Args:
            batch_size: 统一的批量处理大小，控制特征提取和数据库存储
        """
        self.feature_extractor = DINOv3FeatureExtractor(model_path, batch_size)
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
        """使用批量处理优化JSON数据加载，特征提取和存储使用统一的batch_size
        
        Args:
            json_path: JSON文件路径
            batch_size: 批量大小，如果为None则使用实例默认值
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 分批处理：特征提取 -> 数据库存储
        for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
            batch_data = data[i:i+batch_size]
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
                    batch_features = self.feature_extractor.extract_features_batch(valid_images)
                    if batch_features.numel() > 0:  # 确保特征不为空
                        self.db_manager.store_features(
                            features=batch_features.cpu().numpy(),
                            metadata=valid_metadata,
                            ids=valid_ids
                        )
                except Exception as e:
                    print(f"Warning: Failed to process batch {i//batch_size}: {e}")
                    continue
    
    def update_batch_size(self, new_batch_size: int):
        """动态更新批量大小配置"""
        if new_batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {new_batch_size}")
        self.batch_size = new_batch_size
        self.feature_extractor.update_batch_size(new_batch_size)

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