"""Reproducible experiments for the matrix-free implicit hypergradient.

Run from the repo root:  python3 experiments/run_experiments.py
All randomness is seeded; every table is printed to stdout.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bilevel_ridge.core import hypergradient, outer_loss, solve_inner
from bilevel_ridge.data import make_data
from bilevel_ridge.fdcheck import fd_step_study
from bilevel_ridge.oracle import dense_hessian, dense_hypergradient, dense_inner

TOL = 1e-12  # tight tolerance for "reference-quality" iterative results


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / max(1.0, np.linalg.norm(b)))


def exp1_oracle_agreement():
    print("=" * 78)
    print("Exp 1. Matrix-free hypergradient vs. dense double-precision oracle")
    print("=" * 78)
    d = make_data(seed=0)
    rho = np.linspace(-1.0, 1.0, 8)
    res = hypergradient(d.X, d.y, d.X_val, d.y_val, rho, tol_inner=TOL, tol_adj=TOL)
    g_or = dense_hypergradient(d.X, d.y, d.X_val, d.y_val, rho)
    w_or = dense_inner(d.X, d.y, rho)
    print(f"  ||w_cg - w_dense||_2 / ||w_dense||_2 = {rel_err(res.w, w_or):.3e}")
    print(f"  ||grad_mf - grad_dense||_2          = {np.linalg.norm(res.grad - g_or):.3e}")
    print(f"  computable conservative bound       = {res.grad_err_bound:.3e}"
          f"   (bound >= actual error: {res.grad_err_bound >= np.linalg.norm(res.grad - g_or)})")
    print(f"  inner rel. resid {res.inner.rel_residual:.2e} ({res.inner.iters} iters),"
          f" adjoint rel. resid {res.adjoint_info.rel_residual:.2e} ({res.adjoint_info.iters} iters)")


def exp2_fd_step_study():
    print("\n" + "=" * 78)
    print("Exp 2. Central finite differences: error vs. step size h (V-shape)")
    print("=" * 78)
    d = make_data(seed=0)
    rho = np.linspace(-1.0, 1.0, 8)
    loss_fn = lambda r: outer_loss(d.X_val, d.y_val, dense_inner(d.X, d.y, r))
    g_exact = dense_hypergradient(d.X, d.y, d.X_val, d.y_val, rho)
    hs = np.logspace(-1, -12, 23)
    print(f"  {'h':>10}   {'rel. FD error':>14}   regime")
    for h, err in fd_step_study(loss_fn, rho, hs, g_exact):
        regime = "truncation O(h^2)" if h > 1e-4 else ("roundoff O(eps/h)" if h < 1e-6 else "balanced")
        print(f"  {h:10.1e}   {err:14.3e}   {regime}")


def exp3_preconditioner():
    print("\n" + "=" * 78)
    print("Exp 3. Ill-conditioned X: CG iterations without vs. with diagonal preconditioner")
    print("=" * 78)
    print("  axis-aligned scaling (Jacobi is the right preconditioner):")
    print(f"  {'col_cond':>8} {'lam':>8} {'kappa(H)':>10} {'CG inner/adj':>14} {'PCG inner/adj':>14}")
    for cc in [1e2, 1e3, 1e4]:
        for lam0 in [1e-6, 1e-8]:
            d = make_data(n_train=200, n_val=100, p=20, seed=1, col_cond=cc)
            rho = np.full(20, np.log(lam0))
            kap = np.linalg.cond(dense_hessian(d.X, rho))
            rp = hypergradient(d.X, d.y, d.X_val, d.y_val, rho, tol_inner=1e-10, tol_adj=1e-10, use_precond=True)
            ru = hypergradient(d.X, d.y, d.X_val, d.y_val, rho, tol_inner=1e-10, tol_adj=1e-10, use_precond=False)
            print(f"  {cc:8.0e} {lam0:8.0e} {kap:10.3e} "
                  f"{str(ru.inner.iters) + '/' + str(ru.adjoint_info.iters):>14} "
                  f"{str(rp.inner.iters) + '/' + str(rp.adjoint_info.iters):>14}")
    print("  negative control: same spectrum, randomly rotated (diagonal precond cannot help):")
    print(f"  {'col_cond':>8} {'lam':>8} {'kappa(H)':>10} {'CG inner/adj':>14} {'PCG inner/adj':>14}")
    for cc in [1e2, 1e3, 1e4]:
        d = make_data(n_train=200, n_val=100, p=20, seed=1, col_cond=cc, rotate=True)
        rho = np.full(20, np.log(1e-8))
        kap = np.linalg.cond(dense_hessian(d.X, rho))
        rp = hypergradient(d.X, d.y, d.X_val, d.y_val, rho, tol_inner=1e-10, tol_adj=1e-10, use_precond=True)
        ru = hypergradient(d.X, d.y, d.X_val, d.y_val, rho, tol_inner=1e-10, tol_adj=1e-10, use_precond=False)
        print(f"  {cc:8.0e} {1e-8:8.0e} {kap:10.3e} "
              f"{str(ru.inner.iters) + '/' + str(ru.adjoint_info.iters):>14} "
              f"{str(rp.inner.iters) + '/' + str(rp.adjoint_info.iters):>14}")


def exp4_tolerance_coupling():
    print("\n" + "=" * 78)
    print("Exp 4. Inner/adjoint tolerance coupling: both residuals must be controlled")
    print("=" * 78)
    d = make_data(seed=0)
    rho = np.linspace(-1.0, 1.0, 8)
    g_or = dense_hypergradient(d.X, d.y, d.X_val, d.y_val, rho)
    tols = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    print(f"  {'tol_inner':>10} {'tol_adj':>10} {'||grad err||_2':>14} {'bound':>10} {'bound>=err':>10}")
    for t in tols:  # loose inner, tight adjoint
        r = hypergradient(d.X, d.y, d.X_val, d.y_val, rho, tol_inner=t, tol_adj=TOL)
        e = np.linalg.norm(r.grad - g_or)
        print(f"  {t:10.0e} {TOL:10.0e} {e:14.3e} {r.grad_err_bound:10.2e} {str(r.grad_err_bound >= e):>10}")
    for t in tols:  # tight inner, loose adjoint
        r = hypergradient(d.X, d.y, d.X_val, d.y_val, rho, tol_inner=TOL, tol_adj=t)
        e = np.linalg.norm(r.grad - g_or)
        print(f"  {TOL:10.0e} {t:10.0e} {e:14.3e} {r.grad_err_bound:10.2e} {str(r.grad_err_bound >= e):>10}")


def exp5_outer_updates():
    print("\n" + "=" * 78)
    print("Exp 5. Five outer gradient-descent steps on rho (Armijo, seed fixed)")
    print("=" * 78)
    d = make_data(seed=3)
    rho = np.zeros(8)
    print(f"  {'step':>4} {'outer loss':>12} {'||grad||':>10} {'eta':>7} "
          f"{'grad err vs oracle':>18} {'bound':>10} {'resid i/a':>14}")
    for k in range(6):
        r = hypergradient(d.X, d.y, d.X_val, d.y_val, rho, tol_inner=1e-10, tol_adj=1e-10)
        g = r.grad
        loss = outer_loss(d.X_val, d.y_val, r.w)
        g_or = dense_hypergradient(d.X, d.y, d.X_val, d.y_val, rho)
        eta = 1.0
        while outer_loss(d.X_val, d.y_val, dense_inner(d.X, d.y, rho - eta * g)) > loss - 1e-4 * eta * float(g @ g):
            eta *= 0.5
        print(f"  {k:4d} {loss:12.6f} {np.linalg.norm(g):10.3e} {eta:7.2f} "
              f"{np.linalg.norm(g - g_or):18.3e} {r.grad_err_bound:10.2e} "
              f"{r.inner.rel_residual:.0e}/{r.adjoint_info.rel_residual:.0e}")
        rho = rho - eta * g
    print(f"  final rho = {np.array2string(rho, precision=3, separator=', ')}")


def exp6_seeds():
    print("\n" + "=" * 78)
    print("Exp 6. Different random seeds: residuals and gradient errors (tol=1e-12)")
    print("=" * 78)
    print(f"  {'seed':>4} {'inner resid':>12} {'adj resid':>12} {'grad err':>10} "
          f"{'bound':>10} {'FD err h=1e-5':>14} {'iters i/a':>10}")
    for seed in range(5):
        d = make_data(seed=seed)
        rho = np.linspace(-1.0, 1.0, 8)
        r = hypergradient(d.X, d.y, d.X_val, d.y_val, rho, tol_inner=TOL, tol_adj=TOL)
        g_or = dense_hypergradient(d.X, d.y, d.X_val, d.y_val, rho)
        loss_fn = lambda rr: outer_loss(d.X_val, d.y_val, dense_inner(d.X, d.y, rr))
        fd_err = fd_step_study(loss_fn, rho, [1e-5], g_or)[0][1]
        print(f"  {seed:4d} {r.inner.residual:12.3e} {r.adjoint_info.residual:12.3e} "
              f"{np.linalg.norm(r.grad - g_or):10.2e} {r.grad_err_bound:10.2e} {fd_err:14.3e} "
              f"{str(r.inner.iters) + '/' + str(r.adjoint_info.iters):>10}")


if __name__ == "__main__":
    exp1_oracle_agreement()
    exp2_fd_step_study()
    exp3_preconditioner()
    exp4_tolerance_coupling()
    exp5_outer_updates()
    exp6_seeds()
