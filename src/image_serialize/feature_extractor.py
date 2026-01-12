import os
import cv2
import torch
import subprocess
import numpy as np

from torchvision import transforms
from sklearn.decomposition import PCA

class DINOv3FeatureExtractor:
    def __init__(self, model_path=None, batch_size=32):
        self.model = self._load_model(model_path)
        self.transform = self._make_transform()
        self.batch_size = batch_size  # 统一的批量大小超参数

    def _make_transform(self, resize_size=256):
        """Create image transformation pipeline"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])

    def _load_model(self, model_path=None):
        """Load DINOv3 model"""
        dino3_path = os.path.join(os.getcwd(), 'DINOv3')
        if not os.path.exists(dino3_path):
            result = subprocess.run(
                ['git', 'clone', 'git@github.com:facebookresearch/dinov3.git', 'DINOv3'],
                capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to pull DINOv3 model: {result.stderr.decode()}")
        
        if model_path is None:
            model_path = os.path.join(os.getcwd(), 'dinov3_vitb16_pretrain.pth')
            
        return torch.hub.load(dino3_path, 'dinov3_vitb16', source='local', weights=model_path)

    def extract_features_batch(self, images, batch_size=None):
        """批量提取特征，支持多张图片同时处理
        
        Args:
            images: 图片列表
            batch_size: 批量大小，如果为None则使用实例默认值
        """
        if not images:
            return torch.empty(0, 0)
            
        if batch_size is None:
            batch_size = self.batch_size
            
        # 批量大小验证
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
            
        # 如果图片数量小于等于批量大小，直接处理
        if len(images) <= batch_size:
            batch_tensors = torch.stack([self.transform(img) for img in images])
            with torch.no_grad():
                features = self.model.forward_features(batch_tensors)
            return torch.nn.functional.normalize(features['x_norm_clstoken'], dim=-1)
        
        # 否则分批处理并合并结果
        all_features = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            try:
                batch_tensors = torch.stack([self.transform(img) for img in batch_images])
                with torch.no_grad():
                    batch_features = self.model.forward_features(batch_tensors)
                normalized_features = torch.nn.functional.normalize(batch_features['x_norm_clstoken'], dim=-1)
                all_features.append(normalized_features)
            except Exception as e:
                print(f"Warning: Failed to process batch {i//batch_size}: {e}")
                continue
        
        if not all_features:
            return torch.empty(0, 0)
            
        return torch.cat(all_features, dim=0)

    def extract_features(self, image: np.ndarray):
        """Extract features from image using DINOv3"""
        # 使用批量方法保持向后兼容性
        batch_features = self.extract_features_batch([image])
        if batch_features.numel() == 0:
            return torch.empty(0)
        return batch_features[0]

    def update_batch_size(self, new_batch_size: int):
        """动态更新批量大小"""
        if new_batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {new_batch_size}")
        self.batch_size = new_batch_size

    def _debug_visualize(self, image, features):
        """Debug visualization of features"""
        ori_H, ori_W, _ = image.shape
        patch_feat = features['x_norm_patchtokens'].squeeze()
        
        pca = PCA(n_components=3)
        patch_rgb = pca.fit_transform(patch_feat.cpu().numpy())
        patch_rgb = (patch_rgb - patch_rgb.min()) / (patch_rgb.max() - patch_rgb.min()) * 255
        
        HW, _ = patch_rgb.shape
        W = int(np.sqrt(HW))
        H = W
        patch_rgb = patch_rgb.reshape(H, W, 3).astype(np.uint8)
        patch_rgb = cv2.resize(patch_rgb, (ori_W, ori_H))
        show_im = np.concatenate((patch_rgb, image), axis=1)
        cv2.imwrite("patch_rgb.png", show_im)