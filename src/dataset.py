import logging
from dataclasses import dataclass, field
import os
from typing import Callable, List, Type
from regex import F

import torch
import numpy as np
import pandas as pd
from sympy import Symbol, lambdify, sympify
from torch.utils.data import Dataset as TorchDataset
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)

AVAILABLE_OPERATORS = [
    "add",
    "sub",
    "mul",
    "div",
    "sqrt",
    "log",
    "inv",
    "sin",
    "neg",
    "pow",
    "cos",
    "arcsin",
]


@dataclass
class Equation:
    expr: str
    variables: set
    support: dict
    number_of_points: int
    ops: List[str]
    code: Callable = field(init=False, default=None)
    labels: List[int] = field(init=False, default_factory=list)
    sympy_expr: Type[Symbol] = field(init=False, default=None)

    def __post_init__(self):
        self.code = lambdify([*self.variables], expr=self.expr)
        self.sympy_expr = sympify(self.expr)
        self.x: np.ndarray = None
        self.y: np.ndarray = None

    def __repr__(self) -> str:
        return f"Equation(expr={self.expr}, number_of_points={self.number_of_points})"


def load_equations_dataframe(path: str) -> pd.DataFrame:
    """Loads the equations from the given path and returns a pandas dataframe."""

    EXPECTED_COLUMNS = ["eq", "support", "num_points", "ops"]

    df = pd.read_csv(path)
    if not all(x in df.columns for x in EXPECTED_COLUMNS):
        raise ValueError(
            "dataframe not compliant with the format. Ensure that it has eq, support and num_points as column name"
        )

    df = df[EXPECTED_COLUMNS]

    return df


def create_equation(eq: str, support: str, num_points: int, ops: str) -> Equation:
    """Returns an Equation object from the given string equation and support points."""

    supp = eval(support)
    variables = set(supp.keys())

    eq = Equation(
        expr=eq,
        variables=variables,
        ops=ops.split(','),
        support=supp,
        number_of_points=num_points,
    )

    return eq


def generate_data_pts(eq: Equation):
    """Generates data points for the given equation."""

    input_data = []
    for var in eq.variables:
        _max = eq.support[var]["max"]
        _min = eq.support[var]["min"]
        input_data.append(np.random.uniform(_min, _max, eq.number_of_points))

    x = np.stack(input_data, axis=1)

    return x


def generate_equation_data(equation: Equation, noise: float) -> Equation:
    """Generates data for the given equation."""

    x = generate_data_pts(equation)

    y = equation.code(*x.T)

    generated_noise = np.random.normal(0, noise, equation.number_of_points)
    y += generated_noise

    equation.x = x
    equation.y = y

    return equation


def generate_multiple_equation_data(
    equations: List[Equation], noise: float
) -> List[Equation]:
    """Generates data for each equation."""

    for i, equation in enumerate(equations):
        equations[i] = generate_equation_data(equation, noise)

    return equations


class SymbolicOperatorDataset(TorchDataset):
    """PyTorch dataset of equations for symbolic operator discovery
    Args:
        data_path: Path to the dataframe containing the equations.

    """

    def __init__(self, data_path: str, noise: float = 0.0, device: str = "cpu"):

        if not os.path.exists(data_path):
            raise ValueError(f"{data_path} does not exist")

        self.df = load_equations_dataframe(data_path)
        self.equations = [create_equation(*x) for x in self.df.itertuples(index=False)]

        self.multilabel_binarizer = MultiLabelBinarizer()
        self.multilabel_binarizer = self.multilabel_binarizer.fit([AVAILABLE_OPERATORS])

        self.device = device
        self.noise = noise

    def __getitem__(self, index: int) -> Equation:

        eq = self.equations[index]
        if eq.x is None:
            x = generate_data_pts(eq)
            self.equations[index].x = torch.tensor(
                x, device=self.device, requires_grad=False
            )

        if eq.y is None:
            y = eq.code(*eq.x.T)
            self.equations[index].y = y

        if len(eq.labels) == 0:
            encoded_label = list(self.multilabel_binarizer.fit_transform(eq.ops)[0])
            self.equations[index].labels = torch.tensor(
                encoded_label, device=self.device, requires_grad=False
            )

        return self.equations[index]

    def __len__(self) -> int:
        return len(self.equations)


class Dataset:
    """
    A class for dataset of symbolic equations.

    Args:
        equations: List of equations.
    """

    def __init__(self, equations: List[Equation], noise: float = 0.0) -> None:
        self.equations = equations
        self.noise = noise

    def generate_data(self):
        self.equations = generate_multiple_equation_data(self.equations, self.noise)

    @staticmethod
    def evaluate_func(eq: Equation, X: np.ndarray):
        return eq.code(*X.T)

    def __iter__(self):
        return iter(self.equations)

    def __getitem__(self, index: int) -> Equation:
        return self.equations[index]

    def __len__(self) -> int:
        return len(self.equations)
