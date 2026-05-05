import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.W = nn.Linear(feature_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, features, hidden):
        """
        features: (B, 49, 2048)
        hidden:   (B, hidden_dim)
        """

        # Expand hidden to match spatial regions
        hidden = hidden.unsqueeze(1)  # (B, 1, H)

        scores = self.v(torch.tanh(
            self.W(features) + self.U(hidden)
        ))  # (B, 49, 1)

        attention_weights = torch.softmax(scores, dim=1)  # attention weights

        context = (attention_weights * features).sum(dim=1)  # (B, 2048)

        return context, attention_weights