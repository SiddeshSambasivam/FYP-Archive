import unittest
import torch
from src.dataset import SymbolicOperatorDataset


class TestTorchSymbolicDataset(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.data_path = "tests/data/test.csv"
        self.dataset = SymbolicOperatorDataset(self.data_path)

    def test_dataset_len(self):

        assert len(self.dataset) == 2

    def test_dataset_getitem(self):

        eq = self.dataset[0]

        assert type(eq) == dict
        assert "inputs" in eq
        assert "labels" in eq

    def test_label_generation(self):

        eq = self.dataset[0]
        labels = eq["labels"]

        assert all(labels.detach().cpu().numpy() == [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0])

    def test_input_shape(self):

        eq = self.dataset[0]
        inputs = eq["inputs"]
        labels = eq["labels"]

        n = self.dataset.equations[0].number_of_points
        dx = self.dataset.equations[0].x.shape[1]
        dy = 1

        with self.assertRaises(AssertionError):
            assert inputs.shape == (n, dx + dy)

    def test_padding_shape(self):

        eq = self.dataset[0]
        inputs = eq["inputs"]
        num_points = eq["num_points"]

        assert inputs.shape == (num_points, self.dataset.max_variables)

    def test_torch_dataloader(self):

        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=2)
        for idx, batch in enumerate(dataloader):
            assert batch["inputs"].shape == (
                2,
                batch["num_points"][0],
                self.dataset.max_variables,
            )
