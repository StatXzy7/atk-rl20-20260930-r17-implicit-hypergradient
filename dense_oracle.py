from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bilevel_hypergrad import (
    finite_float64_vector,
    inner_objective,
    regularization_factors,
    validate_data,
    validation_objective,
)


@dataclass
class DenseOracleResult:
    w: np.ndarray
    adjoint: np.ndarray
    gradient: np.ndarray
    train_loss: float
    val_loss: float
    hessian: np.ndarray
    min_eigenvalue: float
    condition_number: float
    inner_residual_norm: float
    adjoint_residual_norm: float


def dense_inner_system(
    x_train: np.ndarray,
    y_train: np.ndarray,
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    factors = regularization_factors(rho)
    hessian = x_train.T @ x_train / x_train.shape[0] + np.diag(factors)
    rhs = x_train.T @ y_train / x_train.shape[0]
    return hessian, rhs, factors


def dense_hypergradient(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    rho: np.ndarray,
) -> DenseOracleResult:
    x_train, y_train, x_val, y_val = validate_data(x_train, y_train, x_val, y_val)
    rho = finite_float64_vector("rho", rho, x_train.shape[1])
    hessian, rhs, factors = dense_inner_system(x_train, y_train, rho)
    w = np.linalg.solve(hessian, rhs)
    validation_grad = x_val.T @ (x_val @ w - y_val) / x_val.shape[0]
    adjoint = np.linalg.solve(hessian, validation_grad)
    gradient = -factors * w * adjoint
    eigvals = np.linalg.eigvalsh(hessian)

    return DenseOracleResult(
        w=w,
        adjoint=adjoint,
        gradient=gradient,
        train_loss=inner_objective(w, x_train, y_train, factors),
        val_loss=validation_objective(w, x_val, y_val),
        hessian=hessian,
        min_eigenvalue=float(eigvals[0]),
        condition_number=float(eigvals[-1] / eigvals[0]),
        inner_residual_norm=float(np.linalg.norm(hessian @ w - rhs)),
        adjoint_residual_norm=float(np.linalg.norm(hessian @ adjoint - validation_grad)),
    )


def central_difference_gradient(
    loss_function,
    rho: np.ndarray,
    step: float,
) -> np.ndarray:
    rho = finite_float64_vector("rho", rho)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("finite-difference step must be positive and finite")
    estimate = np.empty_like(rho)
    for index in range(rho.size):
        delta = np.zeros_like(rho)
        delta[index] = step
        plus = np.asarray(loss_function(rho + delta), dtype=np.float64)
        minus = np.asarray(loss_function(rho - delta), dtype=np.float64)
        if plus.size != 1 or minus.size != 1:
            raise ValueError("loss function must return a scalar")
        if not np.isfinite(plus.item()) or not np.isfinite(minus.item()):
            estimate[index] = np.nan
        else:
            estimate[index] = (plus.item() - minus.item()) / (2.0 * step)
    return estimate


def relative_l2_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return float(np.linalg.norm(estimate - reference))
    return float(np.linalg.norm(estimate - reference) / denominator)
