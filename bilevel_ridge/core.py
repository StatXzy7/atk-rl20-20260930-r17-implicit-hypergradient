"""Matrix-free implicit hypergradient via preconditioned conjugate gradients.

Derivation (see docs/DERIVATION.md for the full version):

    Inner FOC:        H w* = X^T y / n,   H = X^T X / n + diag(exp(rho))
    Implicit diff.:   H (dw*/d rho_i) = -exp(rho_i) w*_i e_i
    Adjoint system:   H lam = (1/m) X_val^T (X_val w* - y_val)
    Hypergradient:    dg/d rho_i = -exp(rho_i) w*_i lam_i

Everything here is matrix-free: H is only applied through Hessian-vector
products (two matvecs with X), systems are solved by diagonally
preconditioned CG, no inverse is formed, and no computation graph of the
inner solver is unrolled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CGInfo",
    "HypergradResult",
    "RhoValidationError",
    "cg",
    "hypergradient",
    "inner_objective",
    "outer_loss",
    "solve_inner",
    "validate_rho",
]


class RhoValidationError(ValueError):
    """Raised when the log-regularization vector rho is not usable."""


def validate_rho(rho, p=None):
    """Return ``rho`` as a float64 vector, guaranteeing ``exp(rho)`` in (0, inf).

    exp(rho_i) must be strictly positive and finite in float64 arithmetic:
    rho must be finite, must not overflow exp (rho_i >~ 709.78) and must not
    underflow exp to exactly 0.0 (rho_i <~ -745.13).
    """
    rho = np.asarray(rho, dtype=np.float64)
    if rho.ndim != 1:
        raise RhoValidationError(f"rho must be a 1-D vector, got shape {rho.shape}")
    if p is not None and rho.shape[0] != p:
        raise RhoValidationError(f"rho has length {rho.shape[0]}, expected {p}")
    bad = np.nonzero(~np.isfinite(rho))[0]
    if bad.size:
        raise RhoValidationError(
            f"rho contains non-finite entries at indices {bad.tolist()}"
        )
    with np.errstate(over="ignore", under="ignore"):
        lam = np.exp(rho)
    over = np.nonzero(~np.isfinite(lam))[0]
    if over.size:
        raise RhoValidationError(
            f"exp(rho) overflowed to inf at indices {over.tolist()}; "
            "float64 exp overflows for rho_i > 709.78"
        )
    under = np.nonzero(lam <= 0.0)[0]
    if under.size:
        raise RhoValidationError(
            f"exp(rho) underflowed to 0 at indices {under.tolist()}; "
            "regularization must stay strictly positive "
            "(float64 exp underflows for rho_i < -745.13)"
        )
    return rho


def _check_xy(X, y, name):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name}: X must be 2-D, got shape {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(
            f"{name}: y must be a 1-D vector of length {X.shape[0]}, got shape {y.shape}"
        )
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise ValueError(f"{name}: X and y must be finite")
    return X, y


def inner_objective(X, y, w, rho):
    """||X w - y||^2 / (2 n) + sum_i exp(rho_i) w_i^2 / 2."""
    rho = validate_rho(rho, X.shape[1])
    n = X.shape[0]
    r = X @ w - y
    lam = np.exp(rho)
    return 0.5 * float(r @ r) / n + 0.5 * float(np.sum(lam * w * w))


def outer_loss(X_val, y_val, w):
    """||X_val w - y_val||^2 / (2 m)."""
    m = X_val.shape[0]
    r = X_val @ w - y_val
    return 0.5 * float(r @ r) / m


def make_hvp(X, lam):
    """Matrix-free Hessian-vector product v -> (X^T X / n + diag(lam)) v."""
    n = X.shape[0]

    def hvp(v):
        return X.T @ (X @ v) / n + lam * v

    return hvp


def diag_preconditioner(X, lam):
    """Jacobi preconditioner M^{-1} with M = diag(H) = colnorms(X)^2 / n + lam."""
    n = X.shape[0]
    diag = (X * X).sum(axis=0) / n + lam  # >= min(lam) > 0, always safe
    inv = 1.0 / diag
    return lambda v: inv * v


@dataclass
class CGInfo:
    iters: int
    residual: float          # true ||b - A x||_2 of the returned iterate
    rel_residual: float      # residual / ||b||_2
    converged: bool
    history: list = field(default_factory=list)


def cg(matvec, b, tol=1e-10, maxiter=None, precond=None, x0=None):
    """Preconditioned conjugate gradients for SPD systems.

    Stops when the (recursively estimated) residual satisfies
    ``||r_k||_2 <= tol * ||b||_2``. ``precond`` applies M^{-1}.
    Raises RuntimeError on breakdown (operator not SPD).
    """
    b = np.asarray(b, dtype=np.float64)
    n = b.shape[0]
    if maxiter is None:
        maxiter = max(10 * n, 200)
    if precond is None:
        precond = lambda v: v
    x = np.zeros(n, dtype=np.float64) if x0 is None else np.array(x0, dtype=np.float64)
    bnorm = float(np.linalg.norm(b))
    if bnorm == 0.0:
        return x, CGInfo(0, 0.0, 0.0, True, [0.0])
    thresh = tol * bnorm
    r = b - matvec(x)
    res = float(np.linalg.norm(r))
    history = [res]
    if res <= thresh:
        return x, CGInfo(0, res, res / bnorm, True, history)
    z = precond(r)
    p = z.copy()
    rz = float(r @ z)
    it = 0
    while res > thresh and it < maxiter:
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if not np.isfinite(pAp) or pAp <= 0.0:
            raise RuntimeError(
                f"CG breakdown at iteration {it}: p^T A p = {pAp!r}; "
                "operator is not symmetric positive definite"
            )
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        res = float(np.linalg.norm(r))
        history.append(res)
        it += 1
        z = precond(r)
        rz_new = float(r @ z)
        beta = rz_new / rz if rz != 0.0 else 0.0
        p = z + beta * p
        rz = rz_new
    return x, CGInfo(it, res, res / bnorm, res <= thresh, history)


def _solve_spd(matvec, b, tol, maxiter, precond, what):
    """CG with a posteriori residual verification and warm restarts.

    The stopping test inside CG uses the recursively updated residual, which
    can drift from the true residual in finite precision. Here the true
    residual ||b - A x||_2 is recomputed exactly (one extra HVP) and CG is
    warm-restarted if it violates the tolerance; failure raises.
    """
    bnorm = float(np.linalg.norm(b))
    x = np.zeros_like(b, dtype=np.float64)
    total_iters = 0
    history = []
    for _ in range(6):
        x, info = cg(matvec, b, tol=tol, maxiter=maxiter, precond=precond, x0=x)
        total_iters += info.iters
        history.extend(info.history[1:])
        true_res = float(np.linalg.norm(b - matvec(x)))
        if bnorm == 0.0 or true_res <= tol * bnorm:
            return x, CGInfo(total_iters, true_res, true_res / bnorm if bnorm else 0.0, True, history)
    raise RuntimeError(
        f"{what}: CG failed to reach rel. residual {tol:.1e} "
        f"(true rel. residual {true_res / bnorm:.3e} after {total_iters} iterations)"
    )


def solve_inner(X, y, rho, tol=1e-10, maxiter=None, use_precond=True):
    """Solve the inner problem H w = X^T y / n by (P)CG. Returns (w, CGInfo)."""
    X, y = _check_xy(X, y, "train")
    rho = validate_rho(rho, X.shape[1])
    lam = np.exp(rho)
    n = X.shape[0]
    hvp = make_hvp(X, lam)
    pre = diag_preconditioner(X, lam) if use_precond else None
    return _solve_spd(hvp, X.T @ y / n, tol, maxiter, pre, "inner solve")


@dataclass
class HypergradResult:
    grad: np.ndarray           # matrix-free hypergradient estimate
    w: np.ndarray              # inner solution iterate
    adjoint: np.ndarray        # adjoint system iterate
    inner: CGInfo
    adjoint_info: CGInfo
    grad_err_bound: float      # computable conservative bound on ||grad - exact||_2
    diagnostics: dict


def hypergradient(X, y, X_val, y_val, rho, tol_inner=1e-10, tol_adj=1e-10,
                  use_precond=True, maxiter=None, spd_lower=None):
    """Implicit hypergradient of the outer loss w.r.t. rho (matrix-free).

    Solves, both by diagonally preconditioned CG with separate tolerances:
      inner:   H w   = X^T y / n
      adjoint: H lam = (1/m) X_val^T (X_val w - y_val)
    and returns grad_i = -exp(rho_i) w_i lam_i.

    ``spd_lower`` is a certified lower bound on lambda_min(H); the default
    min_i exp(rho_i) is always valid because X^T X / n is PSD. A tighter
    bound (e.g. dense eigvalsh on small problems) sharpens the diagnostic.
    """
    X, y = _check_xy(X, y, "train")
    X_val, y_val = _check_xy(X_val, y_val, "val")
    if X.shape[1] != X_val.shape[1]:
        raise ValueError(
            f"train/val feature mismatch: {X.shape[1]} vs {X_val.shape[1]}"
        )
    if np.shares_memory(X, X_val) or np.shares_memory(y, y_val):
        raise ValueError(
            "train and validation arrays share memory; the two sets must be disjoint"
        )
    rho = validate_rho(rho, X.shape[1])
    lam = np.exp(rho)
    n, p = X.shape
    m = X_val.shape[0]

    hvp = make_hvp(X, lam)
    pre = diag_preconditioner(X, lam) if use_precond else None

    b_inner = X.T @ y / n
    w, info_w = _solve_spd(hvp, b_inner, tol_inner, maxiter, pre, "inner solve")

    r_val = X_val @ w - y_val
    b_adj = X_val.T @ r_val / m
    adj, info_adj = _solve_spd(hvp, b_adj, tol_adj, maxiter, pre, "adjoint solve")

    grad = -lam * w * adj

    # ---- computable conservative error bound (docs/DERIVATION.md, Sec. 4) ----
    d_min = float(lam.min()) if spd_lower is None else float(spd_lower)
    if not (np.isfinite(d_min) and d_min > 0.0):
        raise ValueError(f"spd_lower must be positive and finite, got {d_min!r}")
    lam_max = float(lam.max())
    xval_fro = float(np.linalg.norm(X_val))  # Frobenius norm >= spectral norm
    delta_w = info_w.residual / d_min                       # >= ||w - w*||_2
    gamma = xval_fro * xval_fro / (m * d_min)               # d(RHS_adj)/d(w) / d_min
    delta_lam = info_adj.residual / d_min + gamma * delta_w  # >= ||adj - lam*||_2
    W = float(np.linalg.norm(b_inner)) / d_min              # >= ||w*||_2
    L = xval_fro * (float(np.linalg.norm(r_val)) + xval_fro * delta_w) / (m * d_min)
    bound = lam_max * (delta_w * L + W * delta_lam + delta_w * delta_lam)

    diagnostics = {
        "inner_resid": info_w.residual,
        "inner_rel_resid": info_w.rel_residual,
        "inner_iters": info_w.iters,
        "adj_resid": info_adj.residual,
        "adj_rel_resid": info_adj.rel_residual,
        "adj_iters": info_adj.iters,
        "spd_lower": d_min,
        "delta_w": delta_w,
        "delta_adjoint": delta_lam,
        "w_norm_bound": W,
        "adjoint_norm_bound": L,
        "grad_err_bound": bound,
    }
    return HypergradResult(grad, w, adj, info_w, info_adj, bound, diagnostics)
