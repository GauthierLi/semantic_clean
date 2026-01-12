import os
import json
from typing import Dict, List
import numpy as np
import cv2

def load_images_from_dir(directory: str, extensions: List[str] = ['.jpg', '.png', '.jpeg']):
    """Load all images from directory with given extensions"""
    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append({
                        'image': img,
                        'path': img_path,
                        'filename': file
                    })
    return images

def save_metadata(metadata: Dict, output_path: str):
    """Save metadata to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_metadata(input_path: str) -> Dict:
    """Load metadata from JSON file"""
    with open(input_path, 'r') as f:
        return json.load(f)

def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize feature vectors to unit length"""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / norms

def create_category_mapping(categories: List[str]) -> Dict[str, int]:
    """Create mapping from category names to indices"""
    return {category: idx for idx, category in enumerate(sorted(set(categories)))}