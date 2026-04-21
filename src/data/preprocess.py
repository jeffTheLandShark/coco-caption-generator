import os
import json
import pickle

from src.utils.config import *


def load_captions():
    """
    Load raw COCO captions.

    Returns:
        list[str]
    """
    raise NotImplementedError


def build_vocab(captions, min_freq=VOCAB_MIN_FREQ):
    """
    Build vocabulary from captions.

    Returns:
        dict[str, int]
    """
    raise NotImplementedError


def encode_captions(captions, vocab):
    """
    Convert captions to padded integer sequences.

    Returns:
        list[list[int]]
    """
    raise NotImplementedError


def main():
    """
    End-to-end preprocessing:
    - load captions
    - build vocab
    - encode captions
    - save outputs
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # TODO: implement pipeline
    raise NotImplementedError


if __name__ == "__main__":
    main()