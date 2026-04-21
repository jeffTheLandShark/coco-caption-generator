import os
import torch

from src.utils.config import *


def load_images():
    """
    Load images from dataset.

    Returns:
        Tensor (N, 3, H, W)
    """
    raise NotImplementedError


def extract_features(images):
    """
    Extract CNN features (e.g., ResNet).

    Returns:
        Tensor (N, FEATURE_DIM)
    """
    raise NotImplementedError


def main():
    """
    End-to-end feature extraction:
    - load images
    - run CNN
    - save features
    """
    os.makedirs(FEATURE_DIR, exist_ok=True)

    # TODO: implement pipeline
    raise NotImplementedError


if __name__ == "__main__":
    main()