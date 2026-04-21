import torch

from src.utils.config import *


def load_model():
    """
    Load trained model.

    Returns:
        nn.Module
    """
    raise NotImplementedError


def generate_caption(model, image_feature):
    """
    Generate caption from image feature.

    Returns:
        list[str]
    """
    raise NotImplementedError


def main():
    """
    Run inference on a sample input.
    """
    # TODO: implement inference
    raise NotImplementedError


if __name__ == "__main__":
    main()