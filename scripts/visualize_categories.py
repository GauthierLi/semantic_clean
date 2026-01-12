#!/usr/bin/env python3
"""
CV2可视化脚本：根据test_output.json中的category信息在图片上显示标签
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse

# 颜色映射，每个类别对应不同颜色
COLOR_MAP = {
    "TRUCK": (0, 255, 0),     # 绿色
    "CAR": (255, 0, 0),       # 蓝色
    "PEDESTRIAN": (0, 0, 255), # 红色
    "CYCLIST": (255, 255, 0), # 青色
    "BUS": (255, 0, 255),     # 紫色
    "MOTORCYCLE": (0, 255, 255), # 黄色
    "OTHER": (128, 128, 128)  # 灰色
}

def get_category_color(category_name):
    """获取类别对应的颜色"""
    return COLOR_MAP.get(category_name, (128, 128, 128))  # 默认为灰色

def visualize_images_with_categories(json_file, image_dir, output_dir, max_images=None):
    """
    可视化图片并显示category信息
    
    Args:
        json_file: test_output.json文件路径
        image_dir: 图片目录路径
        output_dir: 输出目录路径
        max_images: 最大处理图片数量（None表示处理所有）
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载JSON数据
    print(f"Loading JSON data from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries from JSON")
    
    # 处理每张图片
    processed_count = 0
    missing_images = []
    
    for item in data:
        if max_images and processed_count >= max_images:
            break
            
        image_id = item.get('id', '')
        categories = item.get('category', [])
        
        if not image_id:
            continue
            
        # 构建图片路径
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        
        if not os.path.exists(image_path):
            missing_images.append(image_id)
            continue
        
        # 读取图片
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                missing_images.append(image_id)
                continue
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            continue
        
        # 创建拷贝用于绘制
        image_with_text = image.copy()
        height, width = image.shape[:2]
        
        # 添加类别信息到图片
        if categories:
            # 在图片顶部添加类别信息
            text_y = 30
            for i, category in enumerate(categories):
                color = get_category_color(category)
                
                # 绘制类别标签背景
                text = f"{category}"
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # 绘制半透明背景
                overlay = image_with_text.copy()
                cv2.rectangle(overlay, (10, text_y - text_height - 5), 
                            (10 + text_width + 10, text_y + 5), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, image_with_text, 0.4, 0, image_with_text)
                
                # 绘制文本
                cv2.putText(image_with_text, text, (15, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                text_y += 20
        
        # 添加图片ID信息
        cv2.putText(image_with_text, f"ID: {image_id}", (15, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 保存处理后的图片
        output_path = os.path.join(output_dir, f"visualized_{image_id}.jpg")
        cv2.imwrite(output_path, image_with_text)
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} images...")
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Missing images: {len(missing_images)}")
    
    if missing_images:
        print("Missing image IDs (first 10):")
        for img_id in missing_images[:10]:
            print(f"  - {img_id}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
    
    return processed_count, missing_images

def create_summary_image(image_dir, output_dir):
    """创建类别分布摘要图片"""
    category_counts = {}
    
    # 统计每个类别的图片数量
    for image_file in os.listdir(image_dir):
        if image_file.startswith("visualized_") and image_file.endswith(".jpg"):
            # 从文件名提取原始ID
            orig_id = image_file.replace("visualized_", "").replace(".jpg", "")
            
            # 从JSON中查找对应的类别
            json_file = os.path.join(os.path.dirname(__file__), "../test_output.json")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                if item.get('id') == orig_id:
                    categories = item.get('category', [])
                    for category in categories:
                        category_counts[category] = category_counts.get(category, 0) + 1
                    break
    
    # 创建摘要图片
    if category_counts:
        summary_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # 添加标题
        cv2.putText(summary_image, "Category Distribution Summary", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 添加类别统计
        y_offset = 100
        for category, count in category_counts.items():
            color = get_category_color(category)
            text = f"{category}: {count} images"
            cv2.putText(summary_image, text, (50, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 40
        
        # 保存摘要图片
        summary_path = os.path.join(output_dir, "category_summary.jpg")
        cv2.imwrite(summary_path, summary_image)
        print(f"Summary image saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize category information on images using CV2')
    parser.add_argument('--json_file', default='../test_output.json', 
                       help='Path to test_output.json file')
    parser.add_argument('--image_dir', default='~/data/subtype/train_images_dir', 
                       help='Path to image directory')
    parser.add_argument('--output_dir', default='./visualized_images', 
                       help='Output directory for visualized images')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process (default: all)')
    
    args = parser.parse_args()
    
    # 展开路径中的 ~
    image_dir = os.path.expanduser(args.image_dir)
    json_file = os.path.expanduser(args.json_file)
    output_dir = os.path.expanduser(args.output_dir)
    
    # 检查文件是否存在
    if not os.path.exists(json_file):
        print(f"Error: JSON file {json_file} does not exist")
        return
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory {image_dir} does not exist")
        return
    
    print(f"JSON file: {json_file}")
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max images: {args.max_images if args.max_images else 'All'}")
    print("-" * 50)
    
    # 执行可视化
    processed_count, missing_images = visualize_images_with_categories(
        json_file, image_dir, output_dir, args.max_images
    )
    
    # 创建摘要
    create_summary_image(output_dir, output_dir)
    
    print(f"\nVisualization completed! Output saved to: {output_dir}")

if __name__ == "__main__":
    main()