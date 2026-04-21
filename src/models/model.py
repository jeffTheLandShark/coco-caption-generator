import torch
import torch.nn as nn

from src.models.embedding import CaptionEmbedding
from src.models.lstm_decoder import LSTMDecoder
from src.models.output import OutputLayer


class ImageCaptionModel(nn.Module):
    """
    Full model wrapper.

    Pipeline:
    image_features → embeddings → LSTM → output → logits
    """

    def __init__(self, vocab_size: int):
        super().__init__()

        self.embedding = CaptionEmbedding(vocab_size)
        self.decoder = LSTMDecoder()
        self.output = OutputLayer(vocab_size)

    def forward(
        self,
        image_features: torch.Tensor,
        captions_in: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_features: (batch, feature_dim)
            captions_in: (batch, seq_len)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        raise NotImplementedError