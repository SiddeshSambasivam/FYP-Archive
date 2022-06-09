import os
import types
from multiprocessing import Manager
from itertools import chain
from pathlib import Path

import h5py
import click
import pickle
import copyreg
import signal
import pandas as pd
import numpy as np
import sympy as sp
from sympy import lambdify

import dclasses
import generator
from tools import create_env, H5FilesCreator, code_unpickler, code_pickler


class Pipepile:
    def __init__(self, env: generator.Generator, is_timer=False):
        self.env = env
        manager = Manager()
        self.cnt = manager.list()
        self.is_timer = is_timer
        self.fun_args = ",".join(chain(list(env.variables), env.coefficients))

    def handler(self, signum, frame):
        raise TimeoutError

    def return_training_set(self, i):
        np.random.seed(i)
        while True:
            try:
                res = self.create_lambda(np.random.randint(2 ** 32 - 1))
                assert type(res) == dclasses.Equation
                return res
            except TimeoutError:
                signal.alarm(0)
                continue
            except generator.NotCorrectIndependentVariables:
                signal.alarm(0)
                continue
            except generator.UnknownSymPyOperator:
                signal.alarm(0)
                continue
            except generator.ValueErrorExpression:
                signal.alarm(0)
                continue
            except generator.ImAccomulationBounds:
                signal.alarm(0)
                continue

    def convert_lambda(self, i, variables, support) -> dclasses.Equation:
        sym = self.env.infix_to_sympy(i, self.env.variables, self.env.rewrite_functions)
        placeholder = {x: sp.Symbol(x, real=True, nonzero=True) for x in ["cm", "ca"]}
        constants_expression = sym
        consts_elemns = {}
        infix = str(constants_expression)
        eq = lambdify(self.fun_args, constants_expression, modules=["numpy"])
        res = dclasses.Equation(
            expr=infix, code=eq.__code__, coeff_dict=consts_elemns, variables=variables
        )
        return res


@click.command()
@click.option("--csv_path", default="data/benchmark")
@click.option("--dataset-name")
def converter(csv_path, dataset_name):

    dir_path = os.path.dirname(csv_path)
    config_path = os.path.join(dir_path, "dataset_configuration.json")
    env, param, config_dict = create_env(config_path)

    validation = pd.read_csv(csv_path)
    copyreg.pickle(
        types.CodeType, code_pickler, code_unpickler
    )  # Needed for serializing code objects

    folder_path = Path(f"data/validation/{dataset_name}")
    folder_path.mkdir(parents=True, exist_ok=True)

    h5_creator = H5FilesCreator(target_path=folder_path)
    env_pip = Pipepile(env, is_timer=False)

    res = []
    for idx in range(len(validation)):

        gt_expr = validation.iloc[idx]["eq"]
        gt_expr = gt_expr.replace("pow", "Pow")

        variables = list(eval(validation.iloc[idx]["support"]).keys())
        support = validation.iloc[idx]["support"]

        curr = env_pip.convert_lambda(gt_expr, variables, support)
        res.append(curr)

    print("Finishing generating set")
    h5_creator.create_single_hd5_from_eqs(("0", res))
    dataset = dclasses.DatasetDetails(
        config=config_dict,
        total_coefficients=env.coefficients,
        total_variables=list(env.variables),
        word2id=env.word2id,
        id2word=env.id2word,
        una_ops=env.una_ops,
        bin_ops=env.una_ops,
        rewrite_functions=env.rewrite_functions,
        total_number_of_eqs=len(res),
        eqs_per_hdf=len(res),
        generator_details=param,
    )

    t_hf = h5py.File(os.path.join(folder_path, "metadata.h5"), "w")
    t_hf.create_dataset("other", data=np.void(pickle.dumps(dataset)))
    t_hf.close()


if __name__ == "__main__":
    converter()