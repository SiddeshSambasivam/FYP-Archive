import torch
from torch.nn import TransformerDecoderLayer

from .set_encoder import SetEncoder


class NemoModel(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden: int = 512,
        num_heads: int = 8,
        num_features: int = 10,
        num_inds: int = 50,
        num_isab: int = 5,
        ln=False,
    ):
        super(NemoModel, self).__init__()

        self.enc = SetEncoder(
            dim_in, dim_hidden, num_heads, num_features, num_inds, num_isab, ln=ln
        )

        self.dec = TransformerDecoderLayer(
            dim_hidden, num_heads, dim_hidden, dim_hidden, dropout=0.1
        )

    def forward(self, X):
        return self.enc(X)
