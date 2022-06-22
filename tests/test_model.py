import unittest
import torch

from src.models.Nemo.model import NemoModel
from src.models.Nemo.set_encoder import SetEncoder


class TestSetEncoder(unittest.TestCase):
    def test_output_shape(self):

        d_x, d_y = 2, 6
        batch_size = 5  # number of equations
        num_support_points = 500  # number of (x,y) pairs
        number_heads = 8
        number_inds = 50
        number_isab = 5
        dim_hidden = 512
        num_seed_features = 10
        ln = False

        enc_input = torch.randn(batch_size, num_support_points, (d_x + d_y))

        model = SetEncoder(
            dim_in=d_x + d_y,
            dim_hidden=dim_hidden,
            num_seed_features=num_seed_features,
            num_heads=number_heads,
            num_inds=number_inds,
            num_isab=number_isab,
            ln=ln,
        )
        model.eval()

        out = model(enc_input)

        assert out.shape == (batch_size, num_seed_features, dim_hidden)


class TestNemoModel(unittest.TestCase):
    def test_output_shape(self):

        d_x, d_y = 2, 6
        batch_size = 5  # number of equations
        num_support_points = 500  # number of (x,y) pairs
        num_classes = 12

        number_heads = 8
        number_inds = 50
        number_isab = 5
        dim_hidden = 512
        num_seed_features = 10
        ln = False

        enc_input = torch.randn(batch_size, num_support_points, (d_x + d_y))

        model = NemoModel(
            dim_in=d_x + d_y,
            num_classes=num_classes,
            dim_hidden=dim_hidden,
            num_seed_features=num_seed_features,
            num_heads=number_heads,
            num_inds=number_inds,
            num_isab=number_isab,
            ln=ln,
        )

        model.eval()
        out = model(enc_input)  # (batch_size, num_features, dim_hidden)

        assert out.shape == (batch_size, num_classes)
