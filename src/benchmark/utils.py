from datetime import datetime
from pathlib import Path

import numpy as np


def standardize_equation(equation):
    """
    Replace x1 -> x, x2 -> y, x3 -> z in order to work with equivalent formats.
    Also, add 0 times all variables in order to make all variables appear always
    """
    if equation is None or (not isinstance(equation, str) and np.isnan(equation)):
        return None
    equation = equation.replace("x1", "x")
    equation = equation.replace("x2", "y")
    equation = equation.replace("x3", "z")
    return equation


def patch_benchmark_name(args_df):
    """For experiments before 2021-02-03, there was no key 'benchmark_name'.
    They were all using the AI Feynman dataset
    """
    output_dirs = args_df.output_dir
    output_dir = output_dirs[0]
    date = Path(output_dir).parent.name.split("_")[0]
    print(f"date: {date}")
    if datetime.strptime(date, "%Y-%m-%d") < datetime(2021, 2, 3):
        print("Patching benchmark_name!")
        args_df["benchmark_name"] = "ai_feymann"
