import json
import chromadb
import numpy as np
import os

from typing import List, Dict, Optional

class ChromaDBManager:
    def __init__(self, db_path: str = "chroma_db"):
        """Initialize ChromaDB client and collection"""
        # 禁用 ChromaDB 遥测功能，避免 SSL 连接错误
        os.environ['ANONYMIZED_TELEMETRY'] = 'False'
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="image_features")

    def store_features(
        self,
        features: np.ndarray,
        metadata: Dict,
        ids: Optional[List[str]] = None
    ):
        """Store image features with metadata in ChromaDB"""
        if ids is None:
            ids = [str(i) for i in range(len(features))]
         
        self.collection.add(
            embeddings=features.tolist(),
            metadatas=metadata,
            ids=ids
        )

    def query_by_feature(
        self,
        query_feature: np.ndarray,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """Query similar images by feature vector"""
        results = self.collection.query(
            query_embeddings=query_feature.tolist(),
            n_results=n_results,
            where=where
        )
        return results
    
    def query_by_feature_batch(
        self,
        query_features: List[np.ndarray],
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """批量查询相似图像，优化百万级样本性能
        
        Args:
            query_features: 特征向量列表
            n_results: 返回的最近邻数量
            where: 过滤条件
            
        Returns:
            查询结果列表
        """
        if not query_features:
            return []
        
        # 将numpy数组转换为列表
        embeddings_list = [f.tolist() for f in query_features]
        
        # 批量查询
        results = self.collection.query(
            query_embeddings=embeddings_list,
            n_results=n_results,
            where=where
        )
        
        return results

    def query_by_category(
        self,
        category: str,
        include_embeddings: bool = False
    ) -> List[Dict]:
        """Query images by category using boolean field"""
        include_list = ['metadatas']
        if include_embeddings:
            include_list.append('embeddings')
        
        # 使用get()方法的where参数直接过滤，返回所有匹配的样本
        return self.collection.get(
            where={f"is_{category}": True},
            include=include_list
        )
    
    def get_class_features(self, category: str) -> np.ndarray:
        """获取指定类别的所有特征向量"""
        result = self.query_by_category(category, include_embeddings=True)
        
        # 检查embeddings是否存在且不为空
        if result.get('embeddings') is None or len(result['embeddings']) == 0:
            return np.array([])
        
        return np.array(result['embeddings'])
    
    def get_class_statistics(self, category: str) -> Dict:
        """获取指定类别的统计信息"""
        features = self.get_class_features(category)
        
        if len(features) == 0:
            return {
                'count': 0,
                'mean': None,
                'std': None,
                'mean_intra_distance': 0.0
            }
        
        # 获取特征的形状信息
        if features.ndim == 1:
            # 如果是1D数组，说明只有一个特征向量
            count = 1
            features_reshaped = features.reshape(1, -1)
        else:
            count = features.shape[0]
            features_reshaped = features
        
        mean_feature = np.mean(features_reshaped, axis=0)
        
        # 计算类内平均距离
        if count > 1:
            distances = []
            for feature in features_reshaped:
                dist = np.linalg.norm(feature - mean_feature)
                distances.append(dist)
            mean_intra_distance = np.mean(distances)
        else:
            mean_intra_distance = 0.0
        
        return {
            'count': count,
            'mean': mean_feature,
            'features': features_reshaped,
            'mean_intra_distance': mean_intra_distance
        }

    def get_all_categories(self) -> List[str]:
        """Get list of all unique categories in the database"""
        items = self.collection.get()
        categories = set()
        for meta in items["metadatas"]:
            # 从所有is_前缀的字段中收集类别
            for key, value in meta.items():
                if key.startswith('is_') and value is True:
                    category = key[3:]  # 去掉'is_'前缀
                    categories.add(category)
        return list(categories)