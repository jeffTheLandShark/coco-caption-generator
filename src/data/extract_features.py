import os
import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import *

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_encoder():
    """
    Load ResNet50 encoder (classifier removed).
    Call once and reuse across extract_features / extract_features_from_image.

    Returns:
        encoder : nn.Sequential on the appropriate device
        device  : torch.device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    encoder = nn.Sequential(*list(resnet.children())[:-1])
    encoder.eval().to(device)
    return encoder, device

def load_images():
    """
    Load images from dataset.

    Returns:
        Tensor (N, 3, H, W)
    """
    image_dir = RAW_DIR + "images/"
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
    print(f"[Load] Found {len(image_files):,} images")

    tensors = []
    for fname in tqdm(image_files, desc="Loading images"):
        path = image_dir + fname
        img = Image.open(path).convert("RGB")
        tensors.append(TRANSFORM(img))

    return torch.stack(tensors) # (N, 3, 224, 224)

def extract_features(images, encoder=None, device=None):
    """
    Extract CNN features from a batch of images.

    Args:
        images          : Tensor (N, 3, H, W)
        encoder, device : from load_encoder(); loaded fresh if not provided

    Returns:
        Tensor (N, 2048)
    """
    if encoder is None:
        encoder, device = load_encoder()

    print(f"[Extract] Using device: {device}")

    all_features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Extracting"):
            batch = images[i : i + BATCH_SIZE].to(device)
            feats = encoder(batch).squeeze(-1).squeeze(-1)  # (B, 2048)
            all_features.append(feats.cpu())

    return torch.cat(all_features, dim=0)  # (N, 2048)


def extract_features_from_image(image_path: str, encoder=None, device=None) -> torch.Tensor:
    """
    Extract CNN features from a single image outside the dataset.

    Args:
        image_path      : Path to any .jpg/.png image file
        encoder, device : from load_encoder(); loaded fresh if not provided

    Returns:
        Tensor (2048,)
    """
    if encoder is None:
        encoder, device = load_encoder()

    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        features = encoder(tensor).squeeze()  # (2048,)

    return features.cpu()


def main():
    """
    End-to-end feature extraction:
    - load images
    - run CNN
    - save features
    """
    os.makedirs(FEATURE_DIR, exist_ok=True)

    # 1. Load
    images = load_images()
    print(f"[Main] Loaded tensor shape: {images.shape}")

    # 2. Extract
    features = extract_features(images)
    print(f"[Main] Feature tensor shape: {features.shape}")

    # 3. Save
    torch.save(features, FEATURES_FILE)
    print(f"[Mean] Features saved -> {FEATURES_FILE}")

    # Verification
    print(f"\n[Main] Sample feture vector:")
    print(f" shape   : {features[0].shape}")
    print(f" min/max : {features[0].min():.4f} / {features[0].max():.4f}")


if __name__ == "__main__":
    main()