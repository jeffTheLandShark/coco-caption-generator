import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.config import *

from torch.utils.data import DataLoader
from src.data.dataset import COCOCaptionDataset
from src.data.vocab import Vocabulary
from src.models.model import ImageCaptionModel

# TODO Might want a collate_fn?
# TODO Look into num_workers
def load_dataloader():
    """
    Load dataset/dataloader.

    Returns:
        iterable of (image_features, caption_input, caption_target)
    """
    dataset = COCOCaptionDataset()
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True
    )
    return loader


def build_model():
    """
    Initialize model.

    Returns:
        nn.Module
    """
    model = ImageCaptionModel(vocab_size=len(vocab))
    return model


def train():
    """
    Training loop:
    - forward pass
    - compute loss
    - backward pass
    - optimizer step
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using {device}")

    vocab = Vocabulary.load(VOCAB_FILE)
    loader = load_dataloader()
    model = build_model()

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train(mode=True)
        total_loss = 0.0

        for features, captions_in, captions_out in loader:
            features = features.to(device)
            captions_in = captions_in.to(device)
            captions_out = captions_out.to(device)

            # Forward
            logits = model(features, captions_in)

            # CE loss reshape
            logits = logits.reshape(-1, logits.size(-1))
            captions_out = captions_out.reshape(-1)
            loss = criterion(logits, captions_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        torch.save(model.state_dict(), MODEL_FILE)

    print(f"Training complete. Model saved at {MODEL_FILE}")


if __name__ == "__main__":
    train()