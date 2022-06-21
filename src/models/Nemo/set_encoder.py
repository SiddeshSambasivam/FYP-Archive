import logging
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


logger = logging.getLogger(__name__)

# Set Encoder is reused and modified from https://github.com/juho-lee/set_transformer/blob/master/models.py


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):

        super(MultiheadAttentionBlock, self).__init__()

        self.dim_V = dim_V
        self.num_heads = num_heads

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)

        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):

        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads

        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)

        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MultiheadAttentionBlock(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MultiheadAttentionBlock(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MultiheadAttentionBlock(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MultiheadAttentionBlock(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetEncoder(nn.Module):
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
        super(SetEncoder, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.num_inds = num_inds
        self.num_isab = num_isab

        self.input_layer = ISAB(dim_in, dim_hidden, num_heads, num_inds, ln=ln)

        self.layers = [("ISAB-1", self.input_layer)]

        for i in range(1, self.num_isab):
            self.layers.append(
                (
                    f"ISAB-{i+1}",
                    ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
                )
            )

        self.outl = PMA(dim_hidden, num_heads, num_features, ln=ln)
        self.layers.append(("PMA", self.outl))

        self._enc = nn.Sequential(OrderedDict(self.layers))

    def summary(self, input_size: tuple, batch_size: int, device: str = "cpu"):
        sm = summary(self, input_size=input_size, batch_size=batch_size, device=device)
        logger.log(logging.INFO, sm)

        return sm

    def forward(self, X):

        return self._enc(X)
