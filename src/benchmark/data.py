import pandas as pd
import sympy as sp
import numpy as np

from .evaluate import evaluate_func
from ..dataset.dclasses import Equation


def load_data(benchmark_name):
    df = pd.read_csv(benchmark_name)
    if not all(x in df.columns for x in ["eq", "support", "num_points"]):
        raise ValueError(
            "dataframe not compliant with the format. Ensure that it has eq, support and num_points as column name"
        )
    df = df[["eq", "support", "num_points"]]
    return df


def load_equation(benchmark_path, equation_idx):
    df = load_data(benchmark_path)
    benchmark_row = df.loc[equation_idx]
    gt_equation = benchmark_row["eq"]
    supp = eval(benchmark_row["support"])
    variables = set(supp.keys())
    eq = Equation(
        code=None,
        expr=gt_equation,
        coeff_dict=None,
        variables=variables,
        support=supp,
        valid=True,
        number_of_points=benchmark_row["num_points"],
    )
    return eq


def get_variables(equation):
    """Parse all free variables in equations and return them in
    lexicographic order"""
    expr = sp.parse_expr(equation)
    variables = expr.free_symbols
    variables = {str(v) for v in variables}
    # # Tighter sanity check: we only accept variables in ascending order
    # # to avoid silent errors with lambdify later.
    # if (variables not in [{'x'}, {'x', 'y'}, {'x', 'y', 'z'}]
    #         and variables not in [{'x1'}, {'x1', 'x2'}, {'x1', 'x2', 'x3'}]):
    #     raise ValueError(f'Unexpected set of variables: {variables}. '
    #                      f'If you want to allow this, make sure that the '
    #                      f'order of variables of the lambdify output will be '
    #                      f'correct.')

    # Make a sorted list out of variables
    # Assumption: the correct order is lexicographic (x, y, z)
    variables = sorted(variables)
    return variables


def return_order_variables(var: set):
    return sorted(list(var), key=lambda x: int(x[2:]))


def get_data(eq: Equation, number_of_points, mode):
    """
    iid_ood_mode: if set to "iid", sample uniformly from the support as given
                  by supp; if set to "ood", sample from a larger support

    """
    sym = []
    vars_list = []
    for i, var in enumerate(eq.variables):

        l, h = eq.support[var]["min"], eq.support[var]["max"]
        if mode == "iid":
            x = np.random.uniform(l, h, number_of_points)
        elif mode == "ood":
            support_length = h - l
            assert support_length > 0
            x = np.random.uniform(
                l - support_length, h + support_length, number_of_points
            )
        else:
            raise ValueError(f"Invalid iid_ood_mode: {mode}")
        sym.append(x)
        vars_list.append(vars_list)

    X = np.column_stack(sym)
    assert X.ndim == 2
    assert X.shape[0] == number_of_points
    var = return_order_variables(eq.variables)
    y = evaluate_func(eq.expr, var, X)
    # y = lambdify(var,eq.expr)(*X.T)[:,None]
    # y = evaluate_func(gt_equation, vars_list, X)

    return X, y


def get_robust_data(eq: Equation, mode):
    n_attempts_max = 100
    X, y = get_data(eq, eq.number_of_points, mode)
    for _ in range(n_attempts_max):
        to_replace = np.isnan(y).squeeze() | np.iscomplex(y).squeeze()
        if not to_replace.any():
            break

        n_to_replace = to_replace.sum()
        X[to_replace, :], y[to_replace] = get_data(eq, n_to_replace, mode)

    if to_replace.any():
        raise ValueError(
            "Could not sample valid points for equation " f"{eq.expr} supp={eq.support}"
        )

    return X, y
