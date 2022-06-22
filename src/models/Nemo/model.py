import logging

import torch
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from torchsummary import summary

from .set_encoder import SetEncoder

logger = logging.getLogger(__name__)


class NemoModel(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        num_classes: int,
        dim_hidden: int = 512,
        num_heads: int = 8,
        num_seed_features: int = 10,
        num_inds: int = 50,
        num_isab: int = 5,
        dropout: float = 0.1,
        ln=False,
    ):
        super(NemoModel, self).__init__()

        self.num_classes = num_classes
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.num_inds = num_inds
        self.num_isab = num_isab
        self.num_seed_features = num_seed_features
        self.dropout = dropout
        self.ln = ln

        self.enc = SetEncoder(
            self.dim_in,
            self.dim_hidden,
            self.num_heads,
            self.num_seed_features,
            self.num_inds,
            self.num_isab,
            ln=self.ln,
        )  # batch_size x num_samples x dim_hidden

        self.dec_layer = TransformerDecoderLayer(
            self.dim_hidden,
            self.num_heads,
            self.dim_hidden,
            dropout=self.dropout,
            batch_first=True,
        )
        self.dec = TransformerDecoder(self.dec_layer, num_layers=5)

        self.fc = torch.nn.Linear(
            self.dim_hidden * self.num_seed_features, self.num_classes
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X):

        enc_out = self.enc(X)

        dec_out = self.dec(enc_out, memory=enc_out)
        dec_out = torch.flatten(dec_out, 1)

        fc_out = self.fc(dec_out)

        out = self.sigmoid(fc_out)

        return out

    def summary(self, input_size: tuple, batch_size: int, device: str = "cpu"):
        """Logs the model architecture and parameters."""
        sm = summary(self, input_size=input_size, batch_size=batch_size, device=device)
        logger.log(logging.INFO, sm)

        return sm
