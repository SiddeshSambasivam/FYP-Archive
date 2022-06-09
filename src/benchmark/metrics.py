import numpy as np


def get_pointwise_acc(y_true, y_pred, rtol, atol):
    # r = roughly_equal(y_true, y_pred, rel_threshold)
    r = np.isclose(y_true, y_pred, rtol=rtol, atol=atol, equal_nan=True)
    return r.mean()


def _get_acc_key(iid_ood_mode, cfg):
    return (
        f"pointwise_acc_r{cfg.pointwise_acc_rtol:.2}_"
        f"a{cfg.pointwise_acc_atol:.2}_{iid_ood_mode}"
    )
