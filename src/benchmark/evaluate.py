import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sympy import lambdify

# from csem_exptrack import process, utils
# from .data import get_variables


def evaluate_func(func_str, vars_list, X):
    assert X.ndim == 2
    assert len(set(vars_list)) == len(vars_list), "Duplicates in vars_list!"

    order_list = vars_list
    indeces = [int(x[2:]) - 1 for x in order_list]

    if not order_list:
        # Empty order list. Constant function predicted
        f = lambdify([], func_str)
        return f() * np.ones(X.shape[0])

    # Pad X with zero-columns, allowing for variables to appear in the equation
    # that are not in the ground-truth equation
    X_padded = np.zeros((X.shape[0], len(vars_list)))

    X_padded[:, : X.shape[1]] = X[:, : X_padded.shape[1]]
    # Subselect columns of X that corrspond to provided variables
    X_subsel = X_padded[:, indeces]

    # The positional arguments of the resulting function will correspond to
    # the order of variables in "vars_list"
    f = lambdify(vars_list, func_str)
    return f(*X_subsel.T)


def plot_durations(combined_df):
    ax = sns.stripplot("nesymres_beam_size", "duration", data=combined_df, size=3)
    ax.set_yscale("log")
    ax.set_ylabel("duration (seconds)")
    sns.despine(ax=ax)
    ax.grid(axis="y")
    ax.grid(axis="y", which="minor", alpha=0.2)
    plt.show()


# if __name__ == "__main__":
# eval_df = collect()
