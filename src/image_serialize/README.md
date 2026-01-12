# TARGET
我想使用chroma来实现一个持久化图像序列化功能，
1. chromadb主要储存/scripts/dataset_transforms/subtype.py生成出来的json里内容。
2. 以及使用dinov3序列化出来的features；
3. 并且chromadb在图搜的时候也是依赖这个feature。
4. 并且要求储存下来的数据库可以实现指定类别查找。
5. 最好使用面向对象编程，单独实现一个类使用__call__来实现图像序列化的功能。
6. 不同的功能用不同的python文件实现，公共的工具单独写一个utils.py中实现，要求高内聚低耦合。

# REFERENCE
下面是图像序列化的参考代码：
```python
import torch
from torchvision.transforms import v

def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

def pull_dinov3_model():
    dino3_path = f'{os.getcwd()}/DINOv3'
    if not os.path.exists(dino3_path):
        result = subprocess.run(
            ['git', 'clone', 'git@github.com:facebookresearch/dinov3.git', 'DINOv3'],
            capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to pull DINOv3 model: {result.stderr.decode()}")
    dinov3_vitb16 = torch.hub.load(dino3_path, 'dinov3_vitb16', source='local',
                weights="/home/liweikang02/data/dinov3_ckpt/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    return dinov3_vitb16

def feat_extract(image: np.ndarray, model: torch.nn.Module, transform):
    ori_H, ori_W, _ = image.shape
    tensor_image = transform(image)[None]
    with torch.no_grad():
        features = model.forward_features(tensor_image)
    if os.getenv("DEBUG"):
        for k in features:
            if isinstance(features[k], torch.Tensor):
                print(f"{k} shape is {features[k].shape}")
    
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
    im_feat =features['x_norm_clstoken'] 
    save_feat = torch.nn.functional.normalize(im_feat, dim=-1)
    return save_feat 
```
