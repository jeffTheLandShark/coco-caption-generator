import torch
import torch.nn as nn

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import *
from src.data.vocab import Vocabulary


class ImageCaptionModel(nn.Module):
    """
    LSTM caption model

    Pipeline:
    image_features → embeddings → LSTM → output → logits
    """

    def __init__(self, vocab_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.feature_proj = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.output_proj = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(
        self, image_features: torch.Tensor, captions_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image_features: (batch, feature_dim)
            captions_in: (batch, seq_len)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """

        embeddings = self.embedding(captions_in)  # (batch, seq_len, embed_dim)

        # Project image features to hidden dimension
        projected_features = self.feature_proj(image_features)  # (batch, hidden_dim)

        # Initialize LSTM hidden and cell states with projected image features
        h_0 = projected_features.unsqueeze(0)  # (1, batch, hidden_dim)
        c_0 = torch.zeros_like(h_0)  # (1, batch, hidden_dim)

        # Pass embeddings through LSTM
        hidden_states, _ = self.lstm(
            embeddings, (h_0, c_0)
        )  # (batch, seq_len, hidden_dim)

        logits = self.output_proj(hidden_states)  # (batch, seq_len, vocab_size)

        return logits

    def generate_caption(
        self, image_feature: torch.Tensor, vocab: Vocabulary, max_len: int = MAX_SEQ_LEN
    ):
        """
        Generate caption from image feature.

        Args:
            image_feature: (feature_dim,)
            vocab: Vocabulary object for decoding indices to words
            max_len: Maximum length of generated caption
        Returns:
            caption: list of words
        """
        device = next(self.parameters()).device

        # Project image feature to hidden dimension
        projected_feature = self.feature_proj(
            image_feature.unsqueeze(0).to(device)
        )  # (1, hidden_dim)

        # Initialize LSTM hidden and cell states
        h_t = projected_feature.unsqueeze(0)  # (1, 1, hidden_dim)
        c_t = torch.zeros_like(h_t)  # (1, 1, hidden_dim)

        # Start token index (assuming <start> token is at index 1)
        input_token = torch.tensor([[vocab.sos_idx]], device=device)  # (1, 1)

        caption_indices = []

        for _ in range(max_len):
            embeddings = self.embedding(input_token)  # (1, 1, embed_dim)
            output, (h_t, c_t) = self.lstm(
                embeddings, (h_t, c_t)
            )  # output: (1, 1, hidden_dim)
            logits = self.output_proj(output.squeeze(1))  # (1, vocab_size)
            predicted_idx = logits.argmax(dim=-1).item()  # Get predicted index
            caption_indices.append(predicted_idx)

            if predicted_idx == vocab.eos_idx:  # Stop if <end> token is generated
                break

            input_token = torch.tensor(
                [[predicted_idx]], device=device
            )  # Next input token

        caption = [vocab.idx2word[idx] for idx in caption_indices]
        return caption
