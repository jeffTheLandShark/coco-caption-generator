import torch
import torch.nn as nn

from src.utils.config import *


class LSTMDecoder(nn.Module):
    """
    Core LSTM sequence model.

    Responsible for:
    - incorporating image features
    - processing embedded captions
    """

    def __init__(self):
        super().__init__()

        # TODO:
        # self.lstm = nn.LSTM(...)
        # self.feature_proj = nn.Linear(FEATURE_DIM, HIDDEN_DIM)  # optional

        raise NotImplementedError

    def forward(
        self,
        image_features: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_features: (batch, feature_dim)
            embeddings: (batch, seq_len, embed_dim)

        Returns:
            hidden_states: (batch, seq_len, hidden_dim)
        """
        raise NotImplementedError