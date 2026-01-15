import os
import cv2
import json
import numpy as np
import logging
import torch
from tqdm import tqdm

from typing import Dict, List
from src.image_serialize import ImageSerializer
from .label_validator import LabelValidator

logger = logging.getLogger(__name__)


class DataCleaner:
    def __init__(self, db_path: str, validate_categories: List[str] = None):
        """复用image_searilize中的组件
        
        Args:
            db_path: 数据库路径
            validate_categories: 需要验证的类别列表，为None或空列表时验证所有类别
        """
        self.image_serializer = ImageSerializer(db_path=db_path)
        # 通过image_serializer访问已有的feature_extractor和db_manager
        self.feature_extractor = self.image_serializer.feature_extractor
        self.db_manager = self.image_serializer.db_manager
        self.label_validator = LabelValidator(self.db_manager)
        # 配置需要验证的类别，为空时验证所有类别
        self.validate_categories = validate_categories if validate_categories is not None else []
        
        # 验证配置参数类型
        if not isinstance(self.validate_categories, list):
            raise ValueError("validate_categories must be a list of strings")
        for cat in self.validate_categories:
            if not isinstance(cat, str):
                raise ValueError("All items in validate_categories must be strings")
    
    def clean_target_data(self, target_json: str, output_json: str, batch_size: int = 50) -> List[Dict]:
        """清洗目标数据并返回结果，支持批处理以优化内存使用"""
        if not os.path.exists(target_json):
            raise FileNotFoundError(f"Target file {target_json} does not exist")
        
        # 加载目标数据
        with open(target_json, 'r') as f:
            target_data = json.load(f)
        
        if not isinstance(target_data, list):
            raise ValueError("Target JSON file should contain a list of image data")
        
        total_count = len(target_data)
        results = []
        processed_count = 0
        
        # 分批处理数据，使用 tqdm 显示进度
        num_batches = (total_count + batch_size - 1) // batch_size
        for batch_start in tqdm(range(0, total_count, batch_size), 
                                total=num_batches, 
                                desc=f"处理 {total_count} 个图像",
                                unit="batch"):
            batch_end = min(batch_start + batch_size, total_count)
            batch_data = target_data[batch_start:batch_end]
            
            batch_results = self._process_batch(batch_data)
            results.extend(batch_results)
            processed_count += len(batch_results)
            
            # 释放内存
            del batch_data
            del batch_results
        
        # 保存结果
        self._save_results(results, output_json)
        
        print(f"数据清洗完成！处理了 {processed_count} 个图像，结果保存到 {output_json}")
        return results
    
    def _extract_features_batch(self, batch_data: List[Dict]) -> Dict[str, torch.Tensor]:
        """批量提取特征
        
        Args:
            batch_data: 批次数据列表
            
        Returns:
            字典，key为image_id，value为特征tensor
        """
        images = []
        image_ids = []
        
        # 收集所有有效图像
        for item in batch_data:
            image_path = item.get('image_path', '')
            if not image_path or not os.path.exists(image_path):
                continue
                
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            images.append(image)
            image_ids.append(item.get('id', ''))
        
        if not images:
            return {}
        
        # 批量提取特征
        try:
            if hasattr(self.feature_extractor, 'extract_features_batch_multi_gpu'):
                # 多GPU模式
                batch_features = self.feature_extractor.extract_features_batch_multi_gpu(images)
            else:
                # 单GPU模式
                batch_features = self.feature_extractor.extract_features_batch(images)
            
            # 构建特征字典
            features_dict = {}
            for img_id, feature in zip(image_ids, batch_features):
                features_dict[img_id] = feature
            
            return features_dict
            
        except Exception as e:
            logger.error(f"批量特征提取失败: {e}")
            # 回退到单张处理
            features_dict = {}
            for img_id, image in zip(image_ids, images):
                try:
                    feature = self.feature_extractor.extract_features(image)
                    features_dict[img_id] = feature
                except Exception as single_error:
                    logger.error(f"单张图像特征提取失败 {img_id}: {single_error}")
                    features_dict[img_id] = None
            
            return features_dict
    
    def _process_batch(self, batch_data: List[Dict]) -> List[Dict]:
        """处理一个批次的数据"""
        # 先批量提取特征
        features_dict = self._extract_features_batch(batch_data)
        
        # 收集本批次所有需要验证的类别并预加载统计信息
        all_categories = set()
        for item in batch_data:
            categories = item.get('category', [])
            if self.validate_categories:
                # 只收集配置的验证类别
                categories_to_validate = [cat for cat in categories if cat in self.validate_categories]
            else:
                categories_to_validate = categories
            all_categories.update(categories_to_validate)
        
        # 预加载类统计信息以优化性能
        if all_categories:
            self.label_validator.preload_class_statistics(list(all_categories))
        
        batch_results = []
        
        for item in batch_data:
            try:
                # 使用预提取的特征
                result = self.process_single_image(item, features_dict)
                batch_results.append(result)
                    
            except Exception as e:
                print(f"处理图像时发生错误: {item.get('id', 'unknown')}, 错误: {str(e)}")
                # 创建错误结果
                error_result = {
                    'image_id': item.get('id', 'unknown'),
                    'image_path': item.get('image_path', ''),
                    'decision': 'reject',
                    'error': str(e)
                }
                batch_results.append(error_result)
        
        return batch_results
    
    def process_single_image(self, image_data: Dict, features_dict: Dict[str, torch.Tensor] = None) -> Dict:
        """处理单个图像的清洗逻辑，支持使用预提取的特征
        
        Args:
            image_data: 图像数据
            features_dict: 预提取的特征字典（可选）
        """
        image_id = image_data.get('id', '')
        image_path = image_data.get('image_path', '')
        categories = image_data.get('category', [])
        
        if not image_path:
            logger.error(f"缺少图像路径 - Image ID: {image_id}, Path: {image_path}, Error: '缺少图像路径'")
            return {
                'image_id': image_id,
                'image_path': image_path,
                'decision': 'drop',
                'error': '缺少图像路径'
            }
        
        if not os.path.exists(image_path):
            logger.error(f"图像文件不存在 - Image ID: {image_id}, Path: {image_path}, Error: '图像文件不存在'")
            return {
                'image_id': image_id,
                'image_path': image_path,
                'decision': 'drop',
                'error': '图像文件不存在'
            }
        
        # 提取特征
        try:
            # 优先使用预提取的特征
            if features_dict and image_id in features_dict:
                query_feature = features_dict[image_id]
                if query_feature is None:
                    raise ValueError("特征提取失败")
            else:
                # 回退到单张提取
                image = cv2.imread(image_path)
                if image is None:
                    return {
                        'image_id': image_id,
                        'image_path': image_path,
                        'decision': 'reject',
                        'error': '无法读取图像文件'
                    }
                
                query_feature = self.feature_extractor.extract_features(image)
            
            query_feature = query_feature.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"特征提取失败 - Image ID: {image_id}, Path: {image_path}, Error: {str(e)}")
            return {
                'image_id': image_id,
                'image_path': image_path,
                'decision': 'reject',
                'error': f'特征提取失败: {str(e)}'
            }
        
        # 过滤需要验证的类别
        if self.validate_categories:
            # 只验证配置列表中的类别（如果存在于图像类别中）
            categories_to_validate = [cat for cat in categories if cat in self.validate_categories]
            if not categories_to_validate:
                return {
                    'image_id': image_id,
                    'image_path': image_path,
                    'decision': 'accept',
                    'total_categories': len(categories),
                    'validated_categories': 0,
                    'categories': [],
                    'score': 0.0,
                    'error': f'图像类别 {[categories]} 与配置的验证类别 {self.validate_categories} 无匹配'
                }
        else:
            # 验证所有类别
            categories_to_validate = categories
        
        if not categories_to_validate:
            return {
                'image_id': image_id,
                'image_path': image_path,
                'decision': 'review',
                'total_categories': len(categories),
                'validated_categories': 0,
                'categories': [],
                'score': 0.0,
                'error': '没有指定类别'
            }
        
        # 对指定类别进行验证
        category_results = []
        final_decision = "accept"  # 默认接受
        overall_score = 0.0

        for category in categories_to_validate:
            validation_result = self._validate_label_with_feature(query_feature, category)
            category_results.append({
                'category': category,
                'decision': validation_result['decision'],
                'score': validation_result['score'],
                'metrics': validation_result.get('metrics', {}),
                'error': validation_result.get('error')
            })
            
            # 综合决策逻辑：任一类别被reject则整体reject，任一类别被review且无reject则整体review
            if validation_result['decision'] == 'reject':
                final_decision = 'reject'
            elif validation_result['decision'] == 'review' and final_decision != 'reject':
                final_decision = 'review'
            
            overall_score += validation_result['score']

        # 计算平均分数
        if category_results:
            overall_score = overall_score / len(category_results)

        return {
            'image_id': image_id,
            'image_path': image_path,
            'decision': final_decision,
            'score': overall_score,
            'categories': category_results,  # 包含验证类别的详细结果
            'total_categories': len(categories),  # 原始类别总数
            'validated_categories': len(categories_to_validate),  # 实际验证的类别数
            'error': None if all(not r.get('error') for r in category_results) else "部分类别验证失败"
        }
    
    def _validate_label_with_feature(self, query_feature: np.ndarray, label: str) -> Dict:
        """使用已提取的特征进行标签验证"""
        try:
            # 计算各项指标
            p = self.label_validator._compute_knn_consistency(query_feature, label)
            d_min_norm = self.label_validator._compute_nearest_same_class_distance(query_feature, label)
            d_mu_norm = self.label_validator._compute_class_mean_distance(query_feature, label)
            
            # 计算置信度评分
            score = self.label_validator._compute_confidence_score(p, d_min_norm, d_mu_norm)
            
            # 做出决策
            if score >= self.label_validator.thresholds['high']:
                decision = "accept"
            elif score <= self.label_validator.thresholds['low']:
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
            logger.error(f"标签验证失败 - Label: {label}, Query Feature Shape: {query_feature.shape}, Error: {str(e)}")
            return {
                'score': -1.0,
                'decision': 'reject',
                'error': f"验证过程中发生错误: {str(e)}",
                'metrics': {
                    'knn_consistency': 0.0,
                    'nearest_distance_normalized': 1.0,
                    'class_distance_normalized': 1.0
                }
            }
    
    def _save_results(self, results: List[Dict], output_json: str):
        """保存清洗结果到JSON文件"""
        # 确保输出目录存在
        output_dir = os.path.dirname(output_json)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def get_statistics(self, results: List[Dict]) -> Dict:
        """获取清洗结果的统计信息"""
        total = len(results)
        accept_count = sum(1 for r in results if r.get('decision') == 'accept')
        reject_count = sum(1 for r in results if r.get('decision') == 'reject')
        review_count = sum(1 for r in results if r.get('decision') == 'review')
        error_count = sum(1 for r in results if r.get('error'))
        
        return {
            'total': total,
            'accept': accept_count,
            'reject': reject_count,
            'review': review_count,
            'error': error_count,
            'accept_rate': accept_count / total if total > 0 else 0,
            'reject_rate': reject_count / total if total > 0 else 0,
            'review_rate': review_count / total if total > 0 else 0
        }