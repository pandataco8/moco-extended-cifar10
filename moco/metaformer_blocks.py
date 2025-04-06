import torch
import torch.nn as nn

class TokenMixingBlock(nn.Module):
    def __init__(self, dim, hidden_ratio=2.0):
        super(TokenMixingBlock, self).__init__()
        hidden_dim = int(dim * hidden_ratio)

        self.mix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.mix(x)