from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ConvergenceError(RuntimeError):
    def __init__(self, message: str, result: "LinearSolveResult"):
        super().__init__(message)
        self.result = result


@dataclass
class LinearSolveResult:
    x: np.ndarray
    iterations: int
    matvec_count: int
    residual_norm: float
    target_norm: float
    tolerance: float
    converged: bool
    restart_count: int = 0


def finite_float64_vector(name: str, value: np.ndarray, size: int | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"{name} must be a one-dimensional floating-point array")
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinity")
    if size is not None and arr.size != size:
        raise ValueError(f"{name} must have length {size}, got {arr.size}")
    return arr


def finite_float64_matrix(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"{name} must be a two-dimensional floating-point array")
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinity")
    return arr


def regularization_factors(rho: np.ndarray) -> np.ndarray:
    rho = finite_float64_vector("rho", rho)
    with np.errstate(over="ignore"):
        factors = np.exp(rho)
    if not np.all(np.isfinite(factors)):
        raise ValueError("exp(rho) must be finite for every feature")
    if np.any(factors <= 0.0):
        raise ValueError("exp(rho) must be strictly positive for every feature")
    return factors


def validate_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = finite_float64_matrix("x_train", x_train)
    x_val = finite_float64_matrix("x_val", x_val)
    y_train = finite_float64_vector("y_train", y_train)
    y_val = finite_float64_vector("y_val", y_val)
    n, d = x_train.shape
    m, d_val = x_val.shape
    if n <= 0 or d <= 0 or m <= 0:
        raise ValueError("training and validation data must be non-empty")
    if y_train.shape != (n,):
        raise ValueError("y_train must have one entry per training row")
    if d_val != d or y_val.shape != (m,):
        raise ValueError("validation data must have the same feature count as training data")
    return x_train, y_train, x_val, y_val


def disjoint_train_validation_split(
    n_total: int,
    n_train: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if n_total <= 1 or n_train < 1 or n_train >= n_total:
        raise ValueError("require 1 <= n_train < n_total")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_total)
    train_idx = np.sort(permutation[:n_train])
    val_idx = np.sort(permutation[n_train:])
    return train_idx, val_idx


def assert_disjoint_indices(train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    train = np.asarray(train_idx)
    val = np.asarray(val_idx)
    if np.intersect1d(train, val, assume_unique=False).size:
        raise ValueError("training and validation indices must be disjoint")


def split_by_indices(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert_disjoint_indices(train_idx, val_idx)
    return (
        x[train_idx],
        y[train_idx],
        x[val_idx],
        y[val_idx],
    )


def make_inner_matvec(x: np.ndarray, factors: np.ndarray):
    n, d = x.shape

    def matvec(vector: np.ndarray) -> np.ndarray:
        return x.T @ (x @ vector) / n + factors * vector

    return matvec


def inner_rhs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x.T @ y / x.shape[0]


def hessian_diagonal(x: np.ndarray, factors: np.ndarray) -> np.ndarray:
    return np.sum(x * x, axis=0) / x.shape[0] + factors


def pcg_solve(
    matvec,
    rhs: np.ndarray,
    diagonal: np.ndarray | None = None,
    rtol: float = 1e-10,
    atol: float = 0.0,
    maxiter: int | None = None,
    residual_period: int = 20,
    strict: bool = True,
) -> LinearSolveResult:
    rhs = finite_float64_vector("rhs", rhs)
    d = rhs.size
    if maxiter is None:
        maxiter = 10 * d + 100
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("tolerances must be non-negative")

    target_norm = float(np.linalg.norm(rhs))
    tolerance = float(rtol * target_norm + atol)
    if target_norm == 0.0:
        return LinearSolveResult(
            np.zeros(d, dtype=np.float64),
            0,
            0,
            0.0,
            0.0,
            0.0,
            True,
        )

    def precond(vector: np.ndarray) -> np.ndarray:
        if diagonal is None:
            return vector.copy()
        diag = finite_float64_vector("diagonal", diagonal, d)
        if np.any(diag <= 0.0) or not np.all(np.isfinite(diag)):
            raise ValueError("a SPD diagonal preconditioner requires positive finite entries")
        return vector / diag

    x = np.zeros(d, dtype=np.float64)
    r = rhs.copy()
    z = precond(r)
    p = z.copy()
    rz = float(r @ z)
    if not np.isfinite(rz) or rz <= 0.0:
        raise RuntimeError("preconditioner did not return a positive-definite quadratic form")

    matvec_count = 0
    restart_count = 0
    converged = False
    iterations = 0

    for iterations in range(1, maxiter + 1):
        ap = matvec(p)
        ap = finite_float64_vector("matvec output", ap, d)
        matvec_count += 1
        pap = float(p @ ap)
        if not np.isfinite(pap) or pap <= 0.0:
            raise RuntimeError("CG encountered non-positive p^T A p; the linear operator is not SPD")

        alpha = rz / pap
        x += alpha * p
        r -= alpha * ap

        if iterations % residual_period == 0 or iterations == maxiter:
            direct_residual = rhs - matvec(x)
            direct_residual = finite_float64_vector("matvec output", direct_residual, d)
            matvec_count += 1
            if np.linalg.norm(direct_residual - r) > 0.1 * max(np.linalg.norm(direct_residual), tolerance):
                r = direct_residual
                residual_norm = float(np.linalg.norm(r))
                z = precond(r)
                rz_new = float(r @ z)
                if not np.isfinite(rz_new) or rz_new <= 0.0:
                    raise RuntimeError("recurrent residual is incompatible with an SPD preconditioned system")
                p = z
                rz = rz_new
                restart_count += 1
                continue
            else:
                r = direct_residual

        residual_norm = float(np.linalg.norm(r))
        if residual_norm <= tolerance:
            direct_residual = rhs - matvec(x)
            direct_residual = finite_float64_vector("matvec output", direct_residual, d)
            matvec_count += 1
            residual_norm = float(np.linalg.norm(direct_residual))
            r = direct_residual
            if residual_norm <= tolerance:
                converged = True
                break
            z = precond(r)
            rz = float(r @ z)
            if not np.isfinite(rz) or rz <= 0.0:
                raise RuntimeError("recurrent residual is incompatible with an SPD preconditioned system")
            p = z
            restart_count += 1
            continue

        z = precond(r)
        rz_new = float(r @ z)
        if not np.isfinite(rz_new) or rz_new <= 0.0:
            raise RuntimeError("preconditioner did not return a positive-definite quadratic form")
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    if not converged:
        result = LinearSolveResult(
            x,
            iterations,
            matvec_count,
            residual_norm,
            target_norm,
            tolerance,
            False,
            restart_count,
        )
        if strict:
            raise ConvergenceError(
                f"CG did not reach residual {tolerance:.3e}; last residual {residual_norm:.3e}",
                result,
            )
        return result

    return LinearSolveResult(
        x,
        iterations,
        matvec_count,
        residual_norm,
        target_norm,
        tolerance,
        True,
        restart_count,
    )


def inner_objective(w: np.ndarray, x: np.ndarray, y: np.ndarray, factors: np.ndarray) -> float:
    w = finite_float64_vector("w", w)
    residual = x @ w - y
    return float(residual @ residual / (2.0 * x.shape[0]) + 0.5 * np.sum(factors * w * w))


def validation_objective(w: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> float:
    w = finite_float64_vector("w", w)
    residual = x_val @ w - y_val
    return float(residual @ residual / (2.0 * x_val.shape[0]))


def validation_gradient(w: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    w = finite_float64_vector("w", w)
    return x_val.T @ (x_val @ w - y_val) / x_val.shape[0]


def solve_inner(
    x: np.ndarray,
    y: np.ndarray,
    rho: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 0.0,
    maxiter: int | None = None,
    precondition: bool = True,
    strict: bool = True,
) -> LinearSolveResult:
    factors = regularization_factors(rho)
    diagonal = hessian_diagonal(x, factors) if precondition else None
    return pcg_solve(
        make_inner_matvec(x, factors),
        inner_rhs(x, y),
        diagonal,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        strict=strict,
    )


@dataclass
class HypergradientResult:
    rho: np.ndarray
    factors: np.ndarray
    w: np.ndarray
    adjoint: np.ndarray
    gradient: np.ndarray
    inner_result: LinearSolveResult
    adjoint_result: LinearSolveResult
    val_loss: float
    lambda_lower: float
    h_upper: float
    w_error_bound: float
    adjoint_error_bound: float
    adjoint_residual_error_bound: float
    gradient_bound: float


def frobenius_quadratic_upper_bound(x_val: np.ndarray) -> float:
    return float(np.sum(x_val * x_val) / x_val.shape[0])


def hypergradient(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    rho: np.ndarray,
    inner_rtol: float = 1e-10,
    inner_atol: float = 0.0,
    adjoint_rtol: float = 1e-10,
    adjoint_atol: float = 0.0,
    maxiter: int | None = None,
    precondition: bool = True,
    lambda_lower: float | None = None,
    strict: bool = True,
) -> HypergradientResult:
    x_train, y_train, x_val, y_val = validate_data(x_train, y_train, x_val, y_val)
    d = x_train.shape[1]
    rho = finite_float64_vector("rho", rho, d)
    factors = regularization_factors(rho)

    matvec = make_inner_matvec(x_train, factors)
    diagonal = hessian_diagonal(x_train, factors) if precondition else None
    inner_result = pcg_solve(
        matvec,
        inner_rhs(x_train, y_train),
        diagonal,
        rtol=inner_rtol,
        atol=inner_atol,
        maxiter=maxiter,
        strict=strict,
    )

    w = inner_result.x
    g_w = validation_gradient(w, x_val, y_val)
    adjoint_result = pcg_solve(
        matvec,
        g_w,
        diagonal,
        rtol=adjoint_rtol,
        atol=adjoint_atol,
        maxiter=maxiter,
        strict=strict,
    )

    v = adjoint_result.x
    gradient = -factors * w * v

    if lambda_lower is None:
        lambda_lower = float(np.min(factors))
    if not np.isfinite(lambda_lower) or lambda_lower <= 0.0:
        raise ValueError("lambda_lower must be a positive finite number")

    h_upper = frobenius_quadratic_upper_bound(x_val)
    w_error_bound = inner_result.residual_norm / lambda_lower
    adjoint_residual_error_bound = adjoint_result.residual_norm / lambda_lower
    adjoint_rhs_error_bound = h_upper * w_error_bound / lambda_lower
    adjoint_error_bound = adjoint_residual_error_bound + adjoint_rhs_error_bound
    max_factor = float(np.max(factors))
    weighted_w_norm = float(np.linalg.norm(factors * w))
    gradient_bound = (
        max_factor * w_error_bound * float(np.linalg.norm(v))
        + (weighted_w_norm + max_factor * w_error_bound) * adjoint_error_bound
    )

    return HypergradientResult(
        rho=rho,
        factors=factors,
        w=w,
        adjoint=v,
        gradient=gradient,
        inner_result=inner_result,
        adjoint_result=adjoint_result,
        val_loss=validation_objective(w, x_val, y_val),
        lambda_lower=float(lambda_lower),
        h_upper=float(h_upper),
        w_error_bound=float(w_error_bound),
        adjoint_error_bound=float(adjoint_error_bound),
        adjoint_residual_error_bound=float(adjoint_residual_error_bound),
        gradient_bound=float(gradient_bound),
    )


def outer_loss(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    rho: np.ndarray,
    inner_rtol: float = 1e-12,
    inner_atol: float = 0.0,
    maxiter: int | None = None,
) -> float:
    inner = solve_inner(
        x_train,
        y_train,
        rho,
        rtol=inner_rtol,
        atol=inner_atol,
        maxiter=maxiter,
        precondition=True,
        strict=True,
    )
    return validation_objective(inner.x, x_val, y_val)


def backtracking_outer_step(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    rho: np.ndarray,
    result: HypergradientResult,
    initial_step: float = 0.05,
    shrink: float = 0.5,
    armijo: float = 1e-4,
    max_trials: int = 30,
) -> tuple[np.ndarray, float, float, int]:
    if initial_step <= 0.0 or not 0.0 < shrink < 1.0 or not 0.0 <= armijo < 1.0:
        raise ValueError("invalid backtracking parameters")
    step = float(initial_step)
    direction = -result.gradient
    directional = float(result.gradient @ direction)
    if directional >= 0.0:
        raise RuntimeError("descent direction is not a descent direction")

    for trial in range(max_trials):
        candidate = result.rho + step * direction
        try:
            candidate_loss = outer_loss(
                x_train,
                y_train,
                x_val,
                y_val,
                candidate,
                inner_rtol=1e-12,
            )
        except (ConvergenceError, FloatingPointError, OverflowError, ValueError):
            candidate_loss = np.inf
        if candidate_loss <= result.val_loss + armijo * step * directional:
            return candidate, float(candidate_loss), step, trial
        step *= shrink

    raise ConvergenceError("backtracking line search failed", result.adjoint_result)
