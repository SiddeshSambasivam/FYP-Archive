import unittest
import torch

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
        num_features = 10
        ln = False

        enc_input = torch.randn(batch_size, num_support_points, (d_x + d_y))

        model = SetEncoder(
            dim_in=d_x + d_y,
            dim_hidden=dim_hidden,
            num_features=num_features,
            num_heads=number_heads,
            num_inds=number_inds,
            num_isab=number_isab,
            ln=ln,
        )
        model.eval()

        out = model(enc_input)

        assert out.shape == (batch_size, num_features, dim_hidden)
