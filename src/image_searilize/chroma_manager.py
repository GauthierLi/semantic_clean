import json
import chromadb
from typing import List, Dict, Optional
import numpy as np

class ChromaDBManager:
    def __init__(self, db_path: str = "chroma_db"):
        """Initialize ChromaDB client and collection"""
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

    def query_by_category(
        self,
        category: str,
    ) -> List[Dict]:
        """Query images by category using boolean field"""
        # 使用get()方法的where参数直接过滤，返回所有匹配的样本
        return self.collection.get(
            where={f"is_{category}": True},
            include=['metadatas']
        )

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