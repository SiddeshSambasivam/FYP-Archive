from typing import List
import numpy as np
import pandas as pd


class Metrics:
    """
    Equation:
    Accuracy:
    Timetaken:
    """

    def __init__(self) -> None:
        self._metrics = []

    def log_metric(self, eq, acc, dur, pred) -> None:
        self._metrics.append(
            {"eq": eq, "pred_eq": pred, "accuracy": acc, "duration": dur}
        )

    def to_csv(self, path: str) -> None:
        df = pd.DataFrame(data=self._metrics)
        writer = pd.ExcelWriter(path, engine="xlsxwriter")
        df.to_excel(writer, sheet_name="Sheet1")

        workbook = writer.book
        worksheet = writer.sheets["Sheet1"]

        format1 = workbook.add_format({"num_format": "0.000000000"})
        worksheet.set_column("E:E", None, format1)  # Adds formatting to column C

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()


def get_pointwise_acc(y_true, y_pred, rtol, atol):
    # r = roughly_equal(y_true, y_pred, rel_threshold)
    r = np.isclose(y_true, y_pred, rtol=rtol, atol=atol, equal_nan=True)
    return r.mean()


def _get_acc_key(iid_ood_mode, cfg):
    return (
        f"pointwise_acc_r{cfg.pointwise_acc_rtol:.2}_"
        f"a{cfg.pointwise_acc_atol:.2}_{iid_ood_mode}"
    )