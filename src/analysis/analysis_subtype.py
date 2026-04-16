#!/usr/bin/env python3
"""
Subtype清洗结果分析工具

该脚本用于分析数据清洗模块生成的JSON结果，生成详细的清洗报告。
支持统计accept/reject/review比例，按task_id聚合分析，并生成分类报告。
"""

import argparse
import json
import os
import csv
from typing import Dict, List, Set


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='分析subtype清洗结果，生成统计报告和分类文件'
    )
    parser.add_argument(
        '--target',
        required=True,
        help='清洗后的JSON文件路径'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='输出目录路径'
    )
    return parser.parse_args()


def load_large_json(file_path: str) -> List[Dict]:
    """
    加载JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        数据字典列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    print(f"正在加载数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON文件应包含一个数据字典列表")
    
    print(f"加载完成，共 {len(data)} 个样本")
    return data


def calculate_statistics(data: List[Dict]) -> Dict[str, int]:
    """
    计算accept/reject/review的比例
    
    Args:
        data: 数据字典列表
        
    Returns:
        包含统计信息的字典
    """
    total = len(data)
    accept_count = sum(1 for item in data if item.get('decision') == 'accept')
    reject_count = sum(1 for item in data if item.get('decision') == 'reject')
    review_count = sum(1 for item in data if item.get('decision') == 'review')
    
    print(f"\n=== 整体统计 ===")
    print(f"总样本数: {total}")
    print(f"接受样本: {accept_count} ({accept_count/total*100:.2f}%)" if total > 0 else "接受样本: 0 (0.00%)")
    print(f"拒绝样本: {reject_count} ({reject_count/total*100:.2f}%)" if total > 0 else "拒绝样本: 0 (0.00%)")
    print(f"废弃样本: {review_count} ({review_count/total*100:.2f}%)" if total > 0 else "废弃样本: 0 (0.00%)")
    
    return {
        'total': total,
        'accept': accept_count,
        'reject': reject_count,
        'review': review_count
    }


def extract_task_id(image_id: str) -> str:
    """
    从image_id中提取task_id（最后一个'_'后的部分）
    
    Args:
        image_id: 图像ID
        
    Returns:
        task_id
    """
    if not image_id:
        return ''
    return image_id.split('_')[-1]


def aggregate_rejected_data(data: List[Dict]) -> Dict[str, Set[str]]:
    """
    聚合拒绝样本的数据，按task_id分组
    
    Args:
        data: 数据字典列表
        
    Returns:
        字典，key为task_id，value为拒绝类别的集合
    """
    rejected_tasks = {}
    
    for item in data:
        if item.get('decision') != 'reject':
            continue
        
        image_id = item.get('image_id', '')
        task_id = extract_task_id(image_id)
        
        if not task_id:
            continue
        
        # 获取所有decision为reject的category
        rejected_categories = set()
        for cat in item.get('categories', []):
            if cat.get('decision') == 'reject':
                category_name = cat.get('category', '')
                if category_name:
                    rejected_categories.add(category_name)
        
        # 聚合到task_id
        if task_id not in rejected_tasks:
            rejected_tasks[task_id] = set()
        rejected_tasks[task_id].update(rejected_categories)
    
    return rejected_tasks


def generate_rejected_csv(rejected_tasks: Dict[str, Set[str]], output_path: str):
    """
    生成rejected.csv文件
    
    Args:
        rejected_tasks: 拒绝任务字典
        output_path: 输出文件路径
    """
    print(f"\n正在生成 {output_path}...")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['task_id', 'rejected_categories'])
        
        # 按task_id排序输出
        for task_id in sorted(rejected_tasks.keys()):
            categories = rejected_tasks[task_id]
            # 将set转换为逗号分隔的字符串
            categories_str = ','.join(sorted(categories))
            writer.writerow([task_id, categories_str])
    
    print(f"已生成 {output_path}，共 {len(rejected_tasks)} 个task")


def generate_task_lists(data: List[Dict], output_dir: str):
    """
    生成accept.txt和reject.txt
    
    Args:
        data: 数据字典列表
        output_dir: 输出目录
        
    Returns:
        (accept_tasks, reject_tasks) 元组
    """
    print(f"\n正在生成task列表...")
    
    accept_tasks = set()
    reject_tasks = set()
    
    for item in data:
        image_id = item.get('image_id', '')
        task_id = extract_task_id(image_id)
        decision = item.get('decision')
        
        if not task_id:
            continue
        
        if decision == 'accept':
            accept_tasks.add(task_id)
        elif decision == 'reject':
            reject_tasks.add(task_id)
    
    # 写入accept.txt
    accept_path = os.path.join(output_dir, 'accept.txt')
    with open(accept_path, 'w', encoding='utf-8') as f:
        for task_id in sorted(accept_tasks):
            f.write(f"{task_id}\n")
    print(f"已生成 {accept_path}，共 {len(accept_tasks)} 个task")
    
    # 写入reject.txt
    reject_path = os.path.join(output_dir, 'reject.txt')
    with open(reject_path, 'w', encoding='utf-8') as f:
        for task_id in sorted(reject_tasks):
            f.write(f"{task_id}\n")
    print(f"已生成 {reject_path}，共 {len(reject_tasks)} 个task")
    
    return accept_tasks, reject_tasks


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    print(f"输出目录: {args.output}")
    
    try:
        # 加载数据
        data = load_large_json(args.target)
        
        # 1. 统计整体决策比例
        calculate_statistics(data)
        
        # 2. 聚合拒绝数据
        print("\n=== 聚合拒绝样本 ===")
        rejected_tasks = aggregate_rejected_data(data)
        print(f"找到 {len(rejected_tasks)} 个被拒绝的task")
        
        # 3. 生成rejected.csv
        rejected_csv_path = os.path.join(args.output, 'rejected.csv')
        generate_rejected_csv(rejected_tasks, rejected_csv_path)
        
        # 4. 生成accept.txt和reject.txt
        generate_task_lists(data, args.output)
        
        print(f"\n{'='*50}")
        print(f"分析完成！结果已保存到 {args.output}")
        print(f"{'='*50}")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败 - {e}")
        return 1
    except ValueError as e:
        print(f"错误: 数据格式错误 - {e}")
        return 1
    except Exception as e:
        print(f"错误: 发生未知错误 - {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())