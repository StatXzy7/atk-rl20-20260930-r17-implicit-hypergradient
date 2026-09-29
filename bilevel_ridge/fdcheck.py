"""Central finite-difference checks of the outer gradient.

Error model (docs/DERIVATION.md, Sec. 6): the central difference
    (g(rho + h e_i) - g(rho - h e_i)) / (2 h)
has O(h^2) truncation error plus O(eps_machine / h) roundoff amplification,
so the total error is V-shaped in h: too-large h is dominated by truncation,
too-small h by floating-point cancellation.
"""

from __future__ import annotations

import numpy as np


def central_difference(loss_fn, rho, h):
    """Componentwise central difference of ``loss_fn`` at ``rho`` with step h."""
    rho = np.asarray(rho, dtype=np.float64)
    grad = np.empty_like(rho)
    for i in range(rho.size):
        e = np.zeros_like(rho)
        e[i] = h
        grad[i] = (loss_fn(rho + e) - loss_fn(rho - e)) / (2.0 * h)
    return grad


def fd_step_study(loss_fn, rho, hs, grad_exact):
    """Relative L2 error of the central difference for each step size in hs."""
    rho = np.asarray(rho, dtype=np.float64)
    grad_exact = np.asarray(grad_exact, dtype=np.float64)
    denom = max(1.0, float(np.linalg.norm(grad_exact)))
    rows = []
    for h in hs:
        g = central_difference(loss_fn, rho, float(h))
        err = float(np.linalg.norm(g - grad_exact)) / denom
        rows.append((float(h), err))
    return rows
