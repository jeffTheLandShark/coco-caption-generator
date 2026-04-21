import torch
import torch.nn as nn

from src.utils.config import *


class CaptionEmbedding(nn.Module):
    """
    Converts tokenized captions into embeddings.
    """

    def __init__(self, vocab_size: int):
        super().__init__()

        # TODO: define embedding layer
        # self.embedding = nn.Embedding(vocab_size, EMBED_DIM)

        raise NotImplementedError

    def forward(self, captions_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            captions_in: (batch, seq_len)

        Returns:
            embeddings: (batch, seq_len, embed_dim)
        """
        raise NotImplementedError