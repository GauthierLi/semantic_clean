#!/usr/bin/env python3
"""
交互式筛选系统后端服务
提供RESTful API接口用于图片筛选和状态管理
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from flask import Flask, request, jsonify, send_file, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 全局变量存储当前工作状态
current_data = {
    'original_file': None,
    'copy_file': None,
    'review_samples': [],
    'categories': set(),
    'loaded_at': None
}

# 配置
ALLOWED_EXTENSIONS = {'json'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
PER_PAGE = 20  # 每页显示数量

def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否合法"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_file_copy(original_path: str) -> str:
    """创建文件副本，使用固定文件名"""
    original = Path(original_path)
    copy_name = f"{original.stem}_modified{original.suffix}"
    copy_path = original.parent / copy_name
    
    try:
        shutil.copy2(original_path, copy_path)
        logger.info(f"Created file copy: {copy_path}")
        return str(copy_path)
    except Exception as e:
        logger.error(f"Failed to create file copy: {e}")
        raise

def load_and_process_json(file_path: str) -> Dict[str, Any]:
    """加载JSON文件并提取review样本"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
        
        review_samples = []
        categories = set()
        
        for item in data:
            if item.get('decision') == 'review':
                review_samples.append(item)
                
                # 提取所有类别
                if 'categories' in item and isinstance(item['categories'], list):
                    for cat_info in item['categories']:
                        if isinstance(cat_info, dict) and 'category' in cat_info:
                            categories.add(cat_info['category'])
        
        logger.info(f"Loaded {len(data)} total items, {len(review_samples)} review samples")
        logger.info(f"Found categories: {sorted(categories)}")
        
        return {
            'review_samples': review_samples,
            'categories': sorted(list(categories)),
            'total_count': len(data),
            'review_count': len(review_samples)
        }
        
    except Exception as e:
        logger.error(f"Error processing JSON file: {e}")
        raise

def filter_samples_by_category(samples: List[Dict], category: str) -> List[Dict]:
    """根据类别筛选样本"""
    filtered = []
    for sample in samples:
        if 'categories' in sample and isinstance(sample['categories'], list):
            for cat_info in sample['categories']:
                if (isinstance(cat_info, dict) and 
                    cat_info.get('category') == category):
                    filtered.append(sample)
                    break
    return filtered

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/load_review_data', methods=['POST'])
def load_review_data():
    """加载edge_cases_result.json文件，创建副本，返回review样本"""
    try:
        data = request.get_json()
        if not data or 'file_path' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing file_path parameter'
            }), 400
        
        file_path = data['file_path']
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': f'File not found: {file_path}'
            }), 404
        
        # 检查文件扩展名
        if not allowed_file(file_path):
            return jsonify({
                'success': False,
                'error': 'Only JSON files are allowed'
            }), 400
        
        # 处理JSON文件
        processed_data = load_and_process_json(file_path)
        
        # 创建文件副本
        copy_path = create_file_copy(file_path)
        
        # 更新全局状态
        current_data.update({
            'original_file': file_path,
            'copy_file': copy_path,
            'review_samples': processed_data['review_samples'],
            'categories': processed_data['categories'],
            'loaded_at': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'review_samples': processed_data['review_samples'],
            'categories': processed_data['categories'],
            'total_count': processed_data['total_count'],
            'review_count': processed_data['review_count'],
            'output_path': copy_path,
            'loaded_at': current_data['loaded_at']
        })
        
    except Exception as e:
        logger.error(f"Error in load_review_data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/image/<path:image_path>')
def serve_image(image_path):
    """提供图片文件访问服务"""
    try:
        logger.info(f"Original image path from URL: {image_path}")
        
        # 安全处理路径
        if '..' in image_path:
            return jsonify({'error': 'Invalid path: path traversal not allowed'}), 400
        
        # 如果路径不是绝对路径，添加开头的斜杠
        # 因为Flask路由会移除URL中的开头的斜杠
        if not image_path.startswith('/'):
            final_path = '/' + image_path
        else:
            final_path = image_path
        
        logger.info(f"Final image path: {final_path}")
        
        if not os.path.exists(final_path):
            logger.error(f"Image not found: {final_path}")
            return jsonify({'error': f'Image not found: {final_path}'}), 404
        
        # 检查文件是否为图片
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        file_ext = Path(final_path).suffix.lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'File type not allowed'}), 400
        
        return send_file(final_path)
        
    except Exception as e:
        logger.error(f"Error serving image {image_path}: {e}")
        return jsonify({'error': 'Failed to serve image'}), 500

@app.route('/api/filter_by_category', methods=['POST'])
def filter_by_category():
    """根据类别筛选review样本"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        category = data.get('category')
        page = int(data.get('page', 1))
        per_page = int(data.get('per_page', 20))
        
        if not current_data['review_samples']:
            return jsonify({
                'success': False,
                'error': 'No data loaded. Please load review data first.'
            }), 400
        
        # 筛选样本
        if category == 'all':
            filtered_samples = current_data['review_samples']
        else:
            filtered_samples = filter_samples_by_category(
                current_data['review_samples'], category
            )
        
        # 分页
        total_count = len(filtered_samples)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_samples = filtered_samples[start_idx:end_idx]
        
        # 处理图片路径，确保可访问
        for sample in page_samples:
            if 'image_path' in sample:
                # 这里可能需要根据实际情况调整路径
                sample['display_path'] = f"/api/image/{sample['image_path']}"
        
        return jsonify({
            'success': True,
            'samples': page_samples,
            'total_count': total_count,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_count + per_page - 1) // per_page
        })
        
    except Exception as e:
        logger.error(f"Error in filter_by_category: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/get_status', methods=['GET'])
def get_status():
    """获取当前加载状态"""
    return jsonify({
        'success': True,
        'status': {
            'loaded': current_data['loaded_at'] is not None,
            'original_file': current_data['original_file'],
            'copy_file': current_data['copy_file'],
            'categories': list(current_data['categories']),  # 转换set为list
            'review_count': len(current_data['review_samples']),
            'loaded_at': current_data['loaded_at']
        }
    })

def update_sample_decision(sample: Dict, category: str, decision: str) -> bool:
    """更新单个样本中指定类别的决策"""
    if 'categories' not in sample or not isinstance(sample['categories'], list):
        return False
    
    for cat_info in sample['categories']:
        if (isinstance(cat_info, dict) and 
            cat_info.get('category') == category):
            cat_info['decision'] = decision
            return True
    
    return False

def update_overall_decision(sample: Dict) -> str:
    """根据类别决策更新样本的整体决策"""
    if 'categories' not in sample or not isinstance(sample['categories'], list):
        return 'reject'
    
    final_decision = 'accept'
    
    for cat_info in sample['categories']:
        if isinstance(cat_info, dict):
            cat_decision = cat_info.get('decision', 'reject')
            if cat_decision == 'reject':
                final_decision = 'reject'
                break
            elif cat_decision == 'review' and final_decision != 'reject':
                final_decision = 'review'
    
    sample['decision'] = final_decision
    return final_decision

@app.route('/api/update_decisions', methods=['POST'])
def update_decisions():
    """更新选定图片的决策状态"""
    try:
        data = request.get_json()
        if not data or 'updates' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing updates parameter'
            }), 400
        
        updates = data['updates']
        selection_mode = data.get('selection_mode', 'positive')
        
        if not current_data['copy_file']:
            return jsonify({
                'success': False,
                'error': 'No data loaded. Please load review data first.'
            }), 400
        
        # 读取当前副本数据
        try:
            with open(current_data['copy_file'], 'r', encoding='utf-8') as f:
                copy_data = json.load(f)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to read copy file: {e}'
            }), 500
        
        updated_count = 0
        
        # 处理每个更新
        for update in updates:
            image_id = update.get('image_id')
            category = update.get('category')
            decision = update.get('decision')
            
            if not all([image_id, category, decision]):
                logger.warning(f"Invalid update entry: {update}")
                continue
            
            # 在副本数据中找到对应的样本
            for sample in copy_data:
                if sample.get('image_id') == image_id:
                    # 更新指定类别的决策
                    if update_sample_decision(sample, category, decision):
                        # 更新整体决策
                        update_overall_decision(sample)
                        updated_count += 1
                    break
        
        # 保存更新后的副本数据
        try:
            with open(current_data['copy_file'], 'w', encoding='utf-8') as f:
                json.dump(copy_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to save copy file: {e}'
            }), 500
        
        # 同时更新内存中的review_samples
        for update in updates:
            image_id = update.get('image_id')
            category = update.get('category')
            decision = update.get('decision')
            
            for sample in current_data['review_samples']:
                if sample.get('image_id') == image_id:
                    update_sample_decision(sample, category, decision)
                    update_overall_decision(sample)
                    break
        
        logger.info(f"Updated decisions for {updated_count} samples")
        
        return jsonify({
            'success': True,
            'updated_count': updated_count,
            'selection_mode': selection_mode
        })
        
    except Exception as e:
        logger.error(f"Error in update_decisions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
def save_current_state():
    """保存当前状态到文件"""
    try:
        if not current_data['copy_file']:
            return jsonify({
                'success': False,
                'error': 'No data loaded. Please load review data first.'
            }), 400
        
        # 读取当前副本数据
        with open(current_data['copy_file'], 'r', encoding='utf-8') as f:
            copy_data = json.load(f)
        
        # 确保所有样本的外层decision都是最新的
        updated_count = 0
        for sample in copy_data:
            if 'categories' in sample and isinstance(sample['categories'], list):
                old_decision = sample.get('decision')
                new_decision = update_overall_decision(sample)
                if old_decision != new_decision:
                    updated_count += 1
        
        # 保存最终数据
        with open(current_data['copy_file'], 'w', encoding='utf-8') as f:
            json.dump(copy_data, f, indent=2, ensure_ascii=False)
        
        # 统计信息
        total_count = len(copy_data)
        accept_count = sum(1 for s in copy_data if s.get('decision') == 'accept')
        reject_count = sum(1 for s in copy_data if s.get('decision') == 'reject')
        review_count = sum(1 for s in copy_data if s.get('decision') == 'review')
        
        logger.info(f"Saved changes: updated {updated_count} overall decisions")
        
        return jsonify({
            'success': True,
            'updated_count': updated_count,
            'statistics': {
                'total': total_count,
                'accept': accept_count,
                'reject': reject_count,
                'review': review_count
            },
            'output_path': current_data['copy_file']
        })
        
    except Exception as e:
        logger.error(f"Error in save_current_state: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def apply_empty_selection_logic(copy_data, selection_mode, current_category):
    """处理空选择情况的逻辑"""
    updated_count = 0
    
    for sample in copy_data:
        if sample.get('decision') == 'review':
            # 找到对应类别的决策
            category_found = False
            if 'categories' in sample and isinstance(sample['categories'], list):
                for cat_info in sample['categories']:
                    if (isinstance(cat_info, dict) and 
                        cat_info.get('category') == current_category):
                        # 根据选择模式设置决策
                        if selection_mode == 'positive':
                            # 正选模式：未选择 = reject
                            if cat_info.get('decision') != 'reject':
                                cat_info['decision'] = 'reject'
                                updated_count += 1
                        else:
                            # 反选模式：未选择 = accept
                            if cat_info.get('decision') != 'accept':
                                cat_info['decision'] = 'accept'
                                updated_count += 1
                        category_found = True
                        break
            
            if category_found:
                # 更新整体决策
                update_overall_decision(sample)
                if sample.get('decision') != 'review':  # 只有当整体决策真的改变时才计数
                    updated_count += 1
    
    return updated_count

@app.route('/api/save_changes', methods=['POST'])
def save_changes():
    """保存所有更改，更新外层decision状态"""
    try:
        data = request.get_json()
        
        if not current_data['copy_file']:
            return jsonify({
                'success': False,
                'error': 'No data loaded. Please load review data first.'
            }), 400
        
        # 如果没有提供更新数据，直接保存当前状态
        if not data:
            return save_current_state()
        
        # 处理选择状态
        selection_mode = data.get('selection_mode', 'positive')
        current_category = data.get('current_category', 'all')
        selected_images = set(data.get('selected_images', []))
        updates = data.get('updates', [])
        
        if not current_data['copy_file']:
            return jsonify({
                'success': False,
                'error': 'No data loaded. Please load review data first.'
            }), 400
        
        # 读取当前副本数据
        try:
            with open(current_data['copy_file'], 'r', encoding='utf-8') as f:
                copy_data = json.load(f)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to read copy file: {e}'
            }), 500
        
        total_updated = 0
        
        # 如果没有选择任何图片，应用空选择逻辑
        if len(selected_images) == 0 and len(updates) == 0 and current_category != 'all':
            logger.info(f"No images selected, applying empty selection logic for {selection_mode} mode")
            empty_update_count = apply_empty_selection_logic(copy_data, selection_mode, current_category)
            total_updated += empty_update_count
            logger.info(f"Empty selection logic updated {empty_update_count} samples")
        else:
            # 处理每个更新
            for update in updates:
                image_id = update.get('image_id')
                category = update.get('category')
                decision = update.get('decision')
                
                if not all([image_id, category, decision]):
                    logger.warning(f"Invalid update entry: {update}")
                    continue
                
                # 在副本数据中找到对应的样本
                for sample in copy_data:
                    if sample.get('image_id') == image_id:
                        # 更新指定类别的决策
                        if update_sample_decision(sample, category, decision):
                            # 更新整体决策
                            update_overall_decision(sample)
                            total_updated += 1
                        break
        
        # 保存更新后的副本数据
        try:
            with open(current_data['copy_file'], 'w', encoding='utf-8') as f:
                json.dump(copy_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to save copy file: {e}'
            }), 500
        
        logger.info(f"Updated decisions for {total_updated} samples")
        
        return jsonify({
            'success': True,
            'updated_count': total_updated,
            'selection_mode': selection_mode,
            'empty_selection_applied': len(selected_images) == 0 and len(updates) == 0 and current_category != 'all'
        })
        
    except Exception as e:
        logger.error(f"Error in save_changes: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/download_result/<path:filename>')
def download_result(filename):
    """下载修改后的文件副本"""
    try:
        if not current_data['copy_file']:
            return jsonify({'error': 'No file available for download'}), 400
        
        # 安全处理文件名
        safe_filename = secure_filename(filename)
        copy_file_path = Path(current_data['copy_file'])
        
        # 检查文件是否存在
        if not copy_file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        # 确保请求的文件名与实际文件匹配
        if safe_filename not in copy_file_path.name:
            return jsonify({'error': 'Invalid filename'}), 400
        
        return send_file(
            str(copy_file_path),
            as_attachment=True,
            download_name=copy_file_path.name,
            mimetype='application/json'
        )
        
    except Exception as e:
        logger.error(f"Error in download_result: {e}")
        return jsonify({
            'error': 'Failed to download file'
        }), 500

@app.route('/api/get_file_info', methods=['GET'])
def get_file_info():
    """获取当前处理文件的信息"""
    try:
        if not current_data['copy_file']:
            return jsonify({
                'success': False,
                'error': 'No file loaded'
            }), 400
        
        copy_file_path = Path(current_data['copy_file'])
        
        if not copy_file_path.exists():
            return jsonify({
                'success': False,
                'error': 'Copy file not found'
            }), 404
        
        # 获取文件信息
        file_stat = copy_file_path.stat()
        
        return jsonify({
            'success': True,
            'file_info': {
                'name': copy_file_path.name,
                'path': str(copy_file_path),
                'size': file_stat.st_size,
                'modified_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'created_time': datetime.fromtimestamp(file_stat.st_ctime).isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_file_info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    """文件过大错误处理"""
    return jsonify({
        'success': False,
        'error': 'File too large'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """404错误处理"""
    return jsonify({
        'success': False,
        'error': 'Resource not found'
    }), 404

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.errorhandler(500)
def internal_error(e):
    """500错误处理"""
    logger.error(f"Internal server error: {e}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    import os
    
    logger.info("Starting Interactive Visual Filter Backend Server")
    logger.info(f"Flask app running in {app.config.get('ENV', 'development')} mode")
    
    # 从环境变量获取端口，默认8023
    port = int(os.environ.get('SERVER_PORT', '8023'))
    logger.info(f"Using port: {port}")
    
    # 开发模式运行
    app.run(
        host='0.0.0.0',
        port=port,
        debug=True,
        threaded=True
    )