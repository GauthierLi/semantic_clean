import cv2
import numpy as np

from typing import Dict, List, Optional
from src.image_serialize import ChromaDBManager


class LabelValidator:
    def __init__(self, db_manager: ChromaDBManager):
        """接收已初始化的ChromaDBManager实例"""
        self.db_manager = db_manager
        self.k = 20  # kNN参数
        self.weights = {'w1': 1.0, 'w2': 0.5, 'w3': 0.5}
        self.thresholds = {'high': 0.4, 'low': -0.4}
    
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
    
    def _compute_class_mean_distance(self, query_feature: np.ndarray, label: str) -> float:
        """计算类均值距离d_mu，基于数据库中的类统计信息"""
        try:
            # 获取类统计信息
            class_stats = self.db_manager.get_class_statistics(label)
            
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
            print(f"计算类均值距离时发生错误: {str(e)}")
            return 1.0
    
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
