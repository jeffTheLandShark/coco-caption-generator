import torch

from src.utils.config import *

from src.models import ImageCaptionModel
from src.data.vocab import Vocabulary

def load_model():
    """
    Load trained model.

    Returns:
        nn.Module
    """
    vocab = Vocabulary.load(VOCAB_FILE)

    model = ImageCaptionModel(vocab_size=len(vocab))
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    model.eval()

    return model, vocab


def generate_caption(model, image_feature):
    """
    Generate caption from image feature.

    Returns:
        list[str]
    """
    vocab = Vocabulary.load(VOCAB_FILE)
    return model.generate_caption(image_feature, vocab)


def main():
    """
    Run inference on a sample input.
    """
    model, vocab = load_model()

    features = torch.load(FEATURES_FILE)

    # TODO Randomize index?
    infer_caption = generate_caption(model, features[0])

    print(f"Inference: {caption}")


if __name__ == "__main__":
    main()