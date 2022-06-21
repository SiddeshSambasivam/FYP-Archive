import unittest
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


    def test_label_generation(self):
        
        eq = self.dataset[0]

        assert all(eq.labels.detach().cpu().numpy() == [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0])

    def test_data_generation(self):
        
        eq = self.dataset[1]

        assert eq.x is not None
        assert eq.y is not None
        assert eq.labels != []

    def test_input_shape(self):
        pass
