import torch
import sacrebleu

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data.dataset import COCOCaptionDataset
from src.data.vocab import Vocabulary
from src.models import ImageCaptionModel
from src.utils.config import *


def clean_caption(tokens):
    # Remove <eos> and anything after it
    if "<eos>" in tokens:
        tokens = tokens[:tokens.index("<eos>")]
    return tokens


def evaluate(num_samples=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset = COCOCaptionDataset()
    vocab = Vocabulary.load(VOCAB_FILE)

    # Load model
    model = ImageCaptionModel(vocab_size=len(vocab), num_layers=NUM_LAYERS)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    references = []

    for i in range(min(num_samples, len(dataset))):
        feature, token_ids = dataset[i]

        # Generate caption
        pred_tokens = model.generate_caption(feature, vocab)
        pred_tokens = clean_caption(pred_tokens)
        pred_sentence = " ".join(pred_tokens)

        # Ground truth
        gt_sentence = vocab.decode(token_ids.tolist())

        predictions.append(pred_sentence)
        references.append([gt_sentence])  # sacrebleu expects list of refs

        # Show a few examples
        if i < 5:
            print("\nGround truth:", gt_sentence)
            print("Precision-recall:", pred_sentence)

    # Compute BLEU
    bleu = sacrebleu.corpus_bleu(predictions, list(zip(*references)))
    print(bleu.format())


if __name__ == "__main__":
    evaluate()