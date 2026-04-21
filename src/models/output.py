import torch
import torch.nn as nn

from src.utils.config import *


class OutputLayer(nn.Module):
    """
    Maps LSTM hidden states to vocabulary logits.
    """

    def __init__(self, vocab_size: int):
        super().__init__()

        # TODO:
        # self.fc = nn.Linear(HIDDEN_DIM, vocab_size)

        raise NotImplementedError

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        raise NotImplementedError