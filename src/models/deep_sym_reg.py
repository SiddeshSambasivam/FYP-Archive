import pdb
import json
from typing import Any, Dict
from dso import DeepSymbolicRegressor
import numpy as np


def load_config(path: str = "configs/dso_config.json") -> Dict[str, Any]:
    with open("configs/dso_config.json") as file:
        config = json.load(file)

    return config


def get_dso_model(epochs: int):
    config = load_config()
    config["training"]["n_epochs"] = epochs

    model = DeepSymbolicRegressor(config)

    return model
