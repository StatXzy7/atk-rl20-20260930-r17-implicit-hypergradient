"""Verifiable bilevel optimization: implicit hypergradient of ridge regression.

Inner problem (training set only):
    w*(rho) = argmin_w ||X w - y||^2 / (2 n) + sum_i exp(rho_i) w_i^2 / 2
Outer problem (independent validation set only):
    g(rho)  = ||X_val w*(rho) - y_val||^2 / (2 m)

The main implementation (core.py) is matrix-free: the inner Hessian
    H = X^T X / n + diag(exp(rho))
is only ever applied through Hessian-vector products, linear systems are
solved by diagonally preconditioned conjugate gradients, no inverse is
formed, and there is no backpropagation through the inner iteration graph.
"""

from .core import (
    CGInfo,
    HypergradResult,
    RhoValidationError,
    cg,
    hypergradient,
    inner_objective,
    outer_loss,
    solve_inner,
    validate_rho,
)

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
