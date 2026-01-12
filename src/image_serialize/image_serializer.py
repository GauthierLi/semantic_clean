import os
import cv2
import json
import numpy as np

from tqdm import tqdm
from typing import Dict, List, Optional
from .chroma_manager import ChromaDBManager
from .feature_extractor import DINOv3FeatureExtractor

class ImageSerializer:
    def __init__(self, db_path: str = "chroma_db", model_path: str = None):
        """Initialize image serializer with feature extractor and ChromaDB"""
        self.feature_extractor = DINOv3FeatureExtractor(model_path)
        self.db_manager = ChromaDBManager(db_path)

    def __call__(self, image: np.ndarray, metadata: Dict):
        """Process and store single image"""
        feature = self.feature_extractor.extract_features(image)
        self.db_manager.store_features(
            features=feature.cpu().numpy(),
            metadata=metadata
        )
        return feature

    def load_from_json(self, json_path: str):
        """Load data from subtype.py generated JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        features = []
        metadatas = []
        ids = []
        
        for item in tqdm(data[:100]):
            if not os.path.exists(item['image_path']):
                continue
                
            image = cv2.imread(item['image_path'])
            if image is None:
                continue
                
            feature = self.feature_extractor.extract_features(image)
            features.append(feature.cpu().view(-1).numpy())
            # 将类别列表转换为标准化的布尔字段
            category_list = [str(c) for c in item.get('category', [])]
            metadata_entry = {
                'image_path': item['image_path'],
            }
            # 为每个类别创建独立的布尔字段
            for category in category_list:
                metadata_entry[f'is_{category}'] = True
            
            metadatas.append(metadata_entry)
            ids.append(item.get('id', str(len(ids))))

        if features:
            self.db_manager.store_features(
                features=np.stack(features),
                metadata=metadatas,
                ids=ids
            )

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