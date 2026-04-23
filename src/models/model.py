import torch
import torch.nn as nn

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import *


class ImageCaptionModel(nn.Module):
    """
    LSTM caption model

    Pipeline:
    image_features → embeddings → LSTM → output → logits
    """

    def __init__(self, vocab_size, num_layers=1, dropout=0.1):
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
