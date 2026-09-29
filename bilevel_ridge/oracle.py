"""Independent dense oracle (double precision) for small problems.

Forms H = X^T X / n + diag(exp(rho)) explicitly and solves with LAPACK via
np.linalg.solve. Used ONLY as a ground-truth reference for the matrix-free
implementation; never on the main hypergradient path.
"""

from __future__ import annotations

import numpy as np

from .core import outer_loss, validate_rho, _check_xy


def dense_inner(X, y, rho):
    """Exact inner solution w* = H^{-1} X^T y / n via dense solve."""
    X, y = _check_xy(X, y, "train")
    rho = validate_rho(rho, X.shape[1])
    n = X.shape[0]
    H = X.T @ X / n + np.diag(np.exp(rho))
    return np.linalg.solve(H, X.T @ y / n)


def dense_outer_loss(X, y, X_val, y_val, rho):
    """Outer loss evaluated at the dense-oracle inner solution."""
    w = dense_inner(X, y, rho)
    return outer_loss(X_val, y_val, w)


def dense_hypergradient(X, y, X_val, y_val, rho):
    """Exact hypergradient via dense solves: grad_i = -exp(rho_i) w*_i lam*_i."""
    X, y = _check_xy(X, y, "train")
    X_val, y_val = _check_xy(X_val, y_val, "val")
    rho = validate_rho(rho, X.shape[1])
    n, m = X.shape[0], X_val.shape[0]
    H = X.T @ X / n + np.diag(np.exp(rho))
    w = np.linalg.solve(H, X.T @ y / n)
    v = X_val.T @ (X_val @ w - y_val) / m
    adj = np.linalg.solve(H, v)
    return -np.exp(rho) * w * adj


def dense_hessian(X, rho):
    """Explicit inner Hessian (diagnostics/tests only)."""
    rho = validate_rho(rho, X.shape[1])
    return X.T @ X / X.shape[0] + np.diag(np.exp(rho))
