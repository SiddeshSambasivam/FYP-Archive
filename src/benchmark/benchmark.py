import warnings

import hydra
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from tqdm import tqdm

from .data import get_data, get_robust_data, load_equation
from .evaluate import evaluate_equation
from .metrics import _get_acc_key, get_pointwise_acc

# TODO: Implement loading the results.json
def load_results(path: str):
    raise NotImplementedError


def evaluate_model(
    model_predict,
    benchmark_path,
    equation_idx,
    num_test_points,
    pointwise_acc_rtol,
    pointwise_acc_atol,
):
    """
    model_predict is a callable that takes an X of shape
    (n_datapoints, n_variables) and returns scalar predictions
    """

    eq = load_equation(benchmark_path, equation_idx)
    gt_equation, num_variables, supp = get_robust_data(eq)
    print(f"gt_equation: {gt_equation}")

    metrics = {
        "gt_equation": gt_equation,
        "num_variables": num_variables,
    }
    for iid_ood_mode in ["iid", "ood"]:
        X, y = get_data(
            gt_equation, num_variables, supp, num_test_points, iid_ood_mode=iid_ood_mode
        )

        y_pred = model_predict(X)
        assert y_pred.shape == (X.shape[0],)

        if np.iscomplex(y_pred).any():
            warnings.warn("Complex values found!")
            y_pred = np.real(y_pred)

        pointwise_acc = get_pointwise_acc(
            y, y_pred, rtol=pointwise_acc_rtol, atol=pointwise_acc_atol
        )
        acc_key = _get_acc_key(iid_ood_mode)
        metrics[acc_key] = pointwise_acc

        # Drop all indices where the ground truth is NaN or +-inf
        assert y.ndim == 1
        valid_idxs = ~np.isnan(y) & ~np.isinf(y)
        metrics[f"frac_valid_{iid_ood_mode}"] = valid_idxs.sum() / num_test_points
        y = y[valid_idxs]
        y_pred = y_pred[valid_idxs]
        assert y.shape[0] == valid_idxs.sum()

        # Replace NaN or infinity prediction
        replace_prediction_idxs = np.isnan(y_pred) | np.isinf(y_pred)
        metrics[f"frac_replace_{iid_ood_mode}"] = (
            replace_prediction_idxs.sum() / y.shape[0]
        )
        y_pred[replace_prediction_idxs] = 0.0

        # Add default metrics
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        metrics[f"r2_{iid_ood_mode}"] = r2
        metrics[f"mse_{iid_ood_mode}"] = mse

    return metrics


def benchmark_sklearn(
    model_path,
    benchmark_name,
    equation_idx,
    num_test_points,
    pointwise_acc_rtol,
    pointwise_acc_atol,
):
    """Evaluate sklearn model at model_path."""

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    def model_predict(X):
        return model.predict(X)

    return evaluate_model(
        model_predict,
        benchmark_name,
        equation_idx,
        num_test_points,
        pointwise_acc_rtol,
        pointwise_acc_atol,
    )


def collect_results(cfg):

    loader = load_results("results.json")
    df = loader.load_folder(hydra.utils.to_absolute_path("fin_res_v2")).T
    # f = raw_df.loc[[0,("other","test_path"),("other","name"),("other", "eq"),("other", "benchmark_name") ]]

    eqs = list(df.loc[:, [("other", "equation_idx")]].values.reshape(-1))

    res = []
    for i in range(len(df)):
        with open(df["path"].iloc[i].values[0]) as json_file:
            json_data = json.load(json_file)
            res.append(json_data)
    best_eq = [x["equation"][0] if x["equation"] else None for x in res]
    duration = [x["duration"] for x in res]
    df.loc[:, "pred_eq"] = best_eq
    df.loc[:, "duration"] = duration
    df.index = eqs

    eval_rows = []

    for idx, df_row in tqdm(df.iterrows(), desc="Evaluate equations..."):
        if df_row.pred_eq[0]:
            assert getattr(df_row, "model_path", None) is None
            metrics = evaluate_equation(
                pred_equation=df_row.pred_eq[0],
                benchmark_name=hydra.utils.to_absolute_path(
                    df_row.loc[[("other", "benchmark_path")]][0]
                ),
                equation_idx=df_row.loc[[("other", "equation_idx")]][0],
                cfg=cfg,
            )
        else:
            metrics = {}
        # else:
        #     model_path = reroute_path(df_row.model_path, df_row.output_dir,
        #         root_dirs)
        #     metrics = evaluate_sklearn(
        #         model_path=model_path,
        #         benchmark_name=df_row.benchmark_name,
        #         equation_idx=df_row.equation_idx,
        #         num_test_points=cfg.NUM_TEST_POINTS,
        #         pointwise_acc_rtol=cfg.POINTWISE_ACC_RTOL,
        #         pointwise_acc_atol=cfg.POINTWISE_ACC_ATOL
        #     )
        eval_row = df_row.to_dict()
        eval_row.update(metrics)
        eval_rows.append(eval_row)

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv("eval_df_v2.csv")

    return eval_df  # , root_dirs
