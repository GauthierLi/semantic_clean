import os
import cv2
import torch
import subprocess
import numpy as np

from torchvision import transforms
from sklearn.decomposition import PCA

class DINOv3FeatureExtractor:
    def __init__(self, model_path=None):
        self.model = self._load_model(model_path)
        self.transform = self._make_transform()

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

    def extract_features(self, image: np.ndarray):
        """Extract features from image using DINOv3"""
        tensor_image = self.transform(image)[None]
        with torch.no_grad():
            features = self.model.forward_features(tensor_image)
        
        if os.getenv("DEBUG"):
            self._debug_visualize(image, features)
            
        return torch.nn.functional.normalize(features['x_norm_clstoken'], dim=-1)

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