import cv2
import numpy as np
import logging

from typing import Dict, List, Optional
from src.image_serialize import ChromaDBManager

logger = logging.getLogger(__name__)


class LabelValidator:
    def __init__(self, db_manager: ChromaDBManager):
        """接收已初始化的ChromaDBManager实例"""
        self.db_manager = db_manager
        self.k = 20  # kNN参数
        self.weights = {'w1': 1.0, 'w2': 0.5, 'w3': 0.5}
        self.thresholds = {'high': 0.4, 'low': -0.4}
        
        # 类统计信息缓存，避免重复查询数据库
        self._class_stats_cache: Dict[str, Dict] = {}
    
    def validate_label(self, image_path: str, label: str) -> Dict:
        """验证单个图像标签，返回评分和决策"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return self._create_error_result(image_path, label, "无法读取图像文件")
            
            # 从数据库获取特征（这里需要扩展ImageSerializer来支持直接路径提取）
            # 暂时使用占位符，后续在DataCleaner中实现特征提取
            query_feature = None  # 这个将在DataCleaner中提供
            
            if query_feature is None:
                return self._create_error_result(image_path, label, "特征提取失败")
            
            # 计算各项指标
            p = self._compute_knn_consistency(query_feature, label)
            d_min_norm = self._compute_nearest_same_class_distance(query_feature, label)
            d_mu_norm = self._compute_class_mean_distance(query_feature, label)
            
            # 计算置信度评分
            score = self._compute_confidence_score(p, d_min_norm, d_mu_norm)
            
            # 做出决策
            if score >= self.thresholds['high']:
                decision = "accept"
            elif score <= self.thresholds['low']:
                decision = "reject"
            else:
                decision = "review"
            
            return {
                'score': float(score),
                'decision': decision,
                'metrics': {
                    'knn_consistency': float(p),
                    'nearest_distance_normalized': float(d_min_norm),
                    'class_distance_normalized': float(d_mu_norm)
                }
            }
            
        except Exception as e:
            return self._create_error_result(image_path, label, f"验证过程中发生错误: {str(e)}")
    
    def _compute_knn_consistency(self, query_feature: np.ndarray, label: str) -> float:
        """计算kNN标签一致性比率p，复用ChromaDBManager的查询功能"""
        try:
            # 查询k个最近邻
            results = self.db_manager.query_by_feature(
                query_feature=query_feature,
                n_results=self.k
            )
            
            if not results['metadatas'] or not results['metadatas'][0]:
                return 0.0
            
            # 计算同标签的邻居比例
            same_label_count = 0
            for metadata in results['metadatas'][0]:
                # 检查是否有对应的类别字段
                if metadata.get(f'is_{label}', False):
                    same_label_count += 1
            
            return same_label_count / len(results['metadatas'][0])
            
        except Exception:
            return 0.0
    
    def preload_class_statistics(self, categories: List[str]):
        """预加载指定类别的统计信息到缓存
        
        Args:
            categories: 需要预加载的类别列表
        """
        if not categories:
            return
        
        logger.info(f"开始预加载 {len(categories)} 个类别的统计信息")
        
        for category in categories:
            if category not in self._class_stats_cache:
                try:
                    stats = self.db_manager.get_class_statistics(category)
                    self._class_stats_cache[category] = stats
                    logger.debug(f"已缓存类别 {category} 的统计信息，样本数: {stats['count']}")
                except Exception as e:
                    logger.warning(f"预加载类别 {category} 统计信息失败: {e}")
        
        logger.info(f"类统计信息预加载完成，缓存包含 {len(self._class_stats_cache)} 个类别")
    
    def batch_compute_knn_consistency(
        self, 
        query_features: List[np.ndarray], 
        labels: List[str]
    ) -> List[float]:
        """批量计算kNN标签一致性比率，优化百万级样本性能
        
        Args:
            query_features: 特征向量列表
            labels: 对应的标签列表
            
        Returns:
            一致性比率列表
        """
        if not query_features or not labels:
            return []
        
        if len(query_features) != len(labels):
            raise ValueError("特征向量数量与标签数量不匹配")
        
        # 按类别分组
        category_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in category_to_indices:
                category_to_indices[label] = []
            category_to_indices[label].append(idx)
        
        # 初始化结果列表
        results = [0.0] * len(query_features)
        
        # 对每个类别批量查询
        for category, indices in category_to_indices.items():
            # 提取该类别的所有查询特征
            category_features = [query_features[i] for i in indices]
            
            try:
                # 批量查询kNN
                batch_results = self.db_manager.query_by_feature_batch(
                    query_features=category_features,
                    n_results=self.k
                )
                
                # 计算一致性
                # batch_results['metadatas'][i] 是第i个查询的k个最近邻的元数据列表
                for i in range(len(batch_results['metadatas'])):
                    if not batch_results['metadatas'][i] or len(batch_results['metadatas'][i]) == 0:
                        results[indices[i]] = 0.0
                        continue
                    
                    same_label_count = 0
                    for metadata in batch_results['metadatas'][i]:
                        if metadata.get(f'is_{category}', False):
                            same_label_count += 1
                    
                    results[indices[i]] = same_label_count / len(batch_results['metadatas'][i])
                    
            except Exception as e:
                logger.error(f"批量计算kNN一致性失败，类别 {category}: {e}")
                for idx in indices:
                    results[idx] = 0.0
        
        return results
    
    def batch_compute_class_mean_distance(
        self,
        query_features: List[np.ndarray],
        labels: List[str]
    ) -> List[float]:
        """批量计算类均值距离，优化百万级样本性能
        
        Args:
            query_features: 特征向量列表
            labels: 对应的标签列表
            
        Returns:
            归一化距离列表
        """
        if not query_features or not labels:
            return []
        
        if len(query_features) != len(labels):
            raise ValueError("特征向量数量与标签数量不匹配")
        
        results = []
        
        for query_feature, label in zip(query_features, labels):
            try:
                # 优先从缓存读取
                if label in self._class_stats_cache:
                    class_stats = self._class_stats_cache[label]
                else:
                    # 缓存未命中时查询数据库并缓存
                    class_stats = self.db_manager.get_class_statistics(label)
                    self._class_stats_cache[label] = class_stats
                
                if class_stats['count'] == 0:
                    results.append(1.0)
                elif class_stats['count'] < 2:
                    results.append(1.0)
                else:
                    class_mean = class_stats['mean']
                    mean_intra_distance = class_stats['mean_intra_distance']
                    
                    # 计算查询特征到类均值的距离
                    distance_to_mean = np.linalg.norm(query_feature - class_mean)
                    
                    # 归一化距离
                    if mean_intra_distance > 0:
                        normalized_distance = distance_to_mean / mean_intra_distance
                        results.append(min(normalized_distance, 3.0))
                    else:
                        results.append(1.0)
                        
            except Exception as e:
                logger.error(f"计算类均值距离失败，标签 {label}: {e}")
                results.append(1.0)
        
        return results
    
    def batch_compute_nearest_same_class_distance(
        self,
        query_features: List[np.ndarray],
        labels: List[str]
    ) -> List[float]:
        """批量计算最近同类距离，优化百万级样本性能
        
        Args:
            query_features: 特征向量列表
            labels: 对应的标签列表
            
        Returns:
            归一化距离列表
        """
        if not query_features or not labels:
            return []
        
        if len(query_features) != len(labels):
            raise ValueError("特征向量数量与标签数量不匹配")
        
        # 按类别分组
        category_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in category_to_indices:
                category_to_indices[label] = []
            category_to_indices[label].append(idx)
        
        # 初始化结果列表
        results = [1.0] * len(query_features)
        
        # 对每个类别批量查询
        for category, indices in category_to_indices.items():
            # 提取该类别的所有查询特征
            category_features = [query_features[i] for i in indices]
            
            try:
                # 批量查询同类别的最近邻
                batch_results = self.db_manager.query_by_feature_batch(
                    query_features=category_features,
                    n_results=1,
                    where={f'is_{category}': True}
                )
                
                # 计算距离
                # batch_results['distances'][i][j] 是第i个查询的第j个最近邻的距离
                for i in range(len(batch_results['distances'])):
                    if not batch_results['metadatas'][i] or len(batch_results['metadatas'][i]) == 0 or not batch_results['distances'][i] or len(batch_results['distances'][i]) == 0:
                        results[indices[i]] = 1.0
                        continue
                    
                    min_distance = batch_results['distances'][i][0]
                    
                    # 使用固定阈值进行归一化
                    normalized_distance = min(min_distance / 0.5, 1.0)
                    results[indices[i]] = normalized_distance
                    
            except Exception as e:
                logger.error(f"批量计算最近同类距离失败，类别 {category}: {e}")
                for idx in indices:
                    results[idx] = 1.0
        
        return results
    
    def _compute_class_mean_distance(self, query_feature: np.ndarray, label: str) -> float:
        """计算类均值距离d_mu，优先使用缓存数据"""
        try:
            # 优先从缓存读取
            if label in self._class_stats_cache:
                class_stats = self._class_stats_cache[label]
            else:
                # 缓存未命中时查询数据库并缓存
                class_stats = self.db_manager.get_class_statistics(label)
                self._class_stats_cache[label] = class_stats
            
            if class_stats['count'] == 0:
                return 1.0  # 如果没有同类数据，返回最大距离
            
            if class_stats['count'] < 2:
                return 1.0  # 如果只有一个样本，无法计算有意义的距离
            
            class_mean = class_stats['mean']
            mean_intra_distance = class_stats['mean_intra_distance']
            
            # 计算查询特征到类均值的距离
            distance_to_mean = np.linalg.norm(query_feature - class_mean)
            
            # 归一化距离
            if mean_intra_distance > 0:
                normalized_distance = distance_to_mean / mean_intra_distance
                # 限制在合理范围内
                return min(normalized_distance, 3.0)
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"计算类均值距离时发生错误: {str(e)}")
            return 1.0
    
    def clear_cache(self):
        """清除类统计信息缓存"""
        self._class_stats_cache.clear()
        logger.info("类统计信息缓存已清除")
    
    def _compute_nearest_same_class_distance(self, query_feature: np.ndarray, label: str) -> float:
        """计算最近同类距离d_min，利用ChromaDB的精确查询"""
        try:
            # 查询同类别的最近邻
            results = self.db_manager.query_by_feature(
                query_feature=query_feature,
                n_results=1,
                where={f'is_{label}': True}
            )
            
            if not results['metadatas'][0] or not results['distances'][0]:
                return 1.0
            
            min_distance = results['distances'][0][0]
            
            # 计算同类数据的平均最近邻距离用于归一化
            class_data = self.db_manager.query_by_category(label)
            if len(class_data['metadatas']) < 2:
                return min_distance
            
            # 简化实现：使用固定阈值进行归一化
            # 在实际实现中，应该计算类内统计信息
            normalized_distance = min(min_distance / 0.5, 1.0)  # 假设0.5为典型类内距离
            
            return normalized_distance
            
        except Exception:
            return 1.0
    
    def _compute_confidence_score(self, p: float, d_min_norm: float, d_mu_norm: float) -> float:
        """
        S = w1 * p - w2 * d_min_norm - w3 * d_mu_norm
        推荐权重：w1=1.0, w2=0.5, w3=0.5
        """
        return (self.weights['w1'] * p - 
                self.weights['w2'] * d_min_norm - 
                self.weights['w3'] * d_mu_norm)
    
    def _create_error_result(self, image_path: str, label: str, error_message: str) -> Dict:
        """创建错误结果"""
        return {
            'score': -1.0,
            'decision': 'reject',
            'error': error_message,
            'metrics': {
                'knn_consistency': 0.0,
                'nearest_distance_normalized': 1.0,
                'class_distance_normalized': 1.0
            }
        }
