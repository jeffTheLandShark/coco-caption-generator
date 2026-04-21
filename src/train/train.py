import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.config import *


def load_dataloader():
    """
    Load dataset/dataloader.

    Returns:
        iterable of (image_features, caption_input, caption_target)
    """
    raise NotImplementedError


def build_model():
    """
    Initialize model.

    Returns:
        nn.Module
    """
    raise NotImplementedError


def train():
    """
    Training loop:
    - forward pass
    - compute loss
    - backward pass
    - optimizer step
    """
    # TODO: implement training loop
    raise NotImplementedError


if __name__ == "__main__":
    train()