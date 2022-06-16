from enum import Enum
from typing import Callable, Dict, Any, Type
from time import process_time, time

import json
import click
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from .data import load_data, load_equation, get_data
from .metrics import get_pointwise_acc, Metrics
from ..models.deep_sym_reg import get_dso_model
from src.benchmark import metrics


class Benchmark:
    def __init__(self, runner, get_model, hp_name, hp) -> None:
        self.runner = runner
        self.get_model = get_model
        self.hp_name = hp_name
        self.hp = hp


def run_dso_benchmark(data_path: str, get_model, epochs: int):

    df = load_data(data_path)
    
    n_equations = 0
    n_correct_pred = 0
    compute_time = 0

    DATA_SIZE = 10000

    for idx, _ in df.iterrows():

        if idx == 1: break

        eq = load_equation(df, idx)
        x, y = get_data(eq, DATA_SIZE, 'ood')

        model = get_model(epochs)
        start = process_time()
        model.fit(x, y)
        end = process_time()

        y_pred = model.predict(x)
        duration = end - start
        compute_time += duration

        acc = get_pointwise_acc(y, y_pred, atol=1e-3, rtol=0.05)
        if acc > 0.95:
            n_correct_pred += 1
        
        n_equations += 1

    # print(f'Model: DSR')
    # print(f'Hyper-parameter: Epochs={epochs}')
    # print(f'Compute time: {test_compute_time}')
    # print(f"Accuracy: {n_correct_pred}")
    # with open(f"logs/results_DSO_epochs_{epochs}_{compute_time}_{time()}.json", "w") as file:
    print(n_correct_pred, n_equations)
        
    return {
        "model": "DSR",
        "hyper-parameter": f"Epochs: {epochs}",
        "compute time": compute_time,
        "accuracy": n_correct_pred/n_equations,
    }
    

def run_model_benchmark(data_path, model):

    get_model = None
    if model == "DSO":
        benchmark = Benchmark(
            run_dso_benchmark, get_dso_model,"epochs", [2**i for i in range(2, 8)]
        )

    data = []
    for hp in benchmark.hp[:1]:

        metrics = benchmark.runner(data_path, benchmark.get_model, hp)
        data.append(metrics)

    df = pd.DataFrame(data=data)
    writer = pd.ExcelWriter(f"logs/results_DSO.xlsx", engine="xlsxwriter")
    df.to_excel(writer, sheet_name="Sheet1")

    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]

    format1 = workbook.add_format({"num_format": "0.000"})
    worksheet.set_column("D:E", None, format1)  # Adds formatting to column D

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()    
        


@click.command()
@click.option(
    "--data-path",
    "-d",
    type=str,
    help="Path to the csv containing the equations",
    required=True,
)
@click.option(
    "--model-name",
    "-m",
    type=str,
    help="Provide one of the available models for benchmark. Available models: DSO",
    required=True,
)
def main(data_path: str, model_name) -> None:

    # if model_name not in MODEL_MAPPER:
    # raise ValueError("Invalid model name")

    run_model_benchmark(data_path, model_name)


if __name__ == "__main__":
    main()
