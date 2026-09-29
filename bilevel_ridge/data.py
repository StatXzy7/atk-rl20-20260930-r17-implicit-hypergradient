"""Synthetic data with a strictly disjoint train/validation split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DataSplit:
    X: np.ndarray
    y: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    train_index: np.ndarray
    val_index: np.ndarray
    w_true: np.ndarray


def make_data(n_train=60, n_val=40, p=8, noise=0.1, seed=0, col_cond=None,
              rotate=False):
    """Linear model y = X w_true + eps with disjoint train/val rows.

    ``col_cond`` scales columns by a geometric ladder from 1 to 1/col_cond,
    making the inner Hessian ill-conditioned (kappa grows ~ col_cond^2).
    With ``rotate=True`` the scaled columns are additionally rotated by a
    random orthogonal matrix: the spectrum is the same but the ill-conditioning
    is no longer axis-aligned, so a *diagonal* preconditioner cannot capture
    it (used as a negative control in the preconditioner experiment).
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_val
    X = rng.standard_normal((n, p))
    if col_cond is not None:
        X = X * np.geomspace(1.0, 1.0 / col_cond, p)
    if rotate:
        Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
        X = X @ Q.T
    w_true = rng.standard_normal(p) * (rng.random(p) < 0.6)  # sparse-ish truth
    y = X @ w_true + noise * rng.standard_normal(n)
    perm = rng.permutation(n)
    tr, va = perm[:n_train], perm[n_train:]
    assert np.intersect1d(tr, va).size == 0, "train/val split must be disjoint"
    return DataSplit(
        X=np.array(X[tr]), y=np.array(y[tr]),
        X_val=np.array(X[va]), y_val=np.array(y[va]),
        train_index=tr, val_index=va, w_true=w_true,
    )
