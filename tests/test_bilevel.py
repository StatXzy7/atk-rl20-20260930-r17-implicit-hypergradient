import inspect

import numpy as np
import pytest

import bilevel_ridge.core as core
from bilevel_ridge.core import (
    RhoValidationError,
    cg,
    hypergradient,
    inner_objective,
    outer_loss,
    solve_inner,
    validate_rho,
)
from bilevel_ridge.data import make_data
from bilevel_ridge.fdcheck import central_difference, fd_step_study
from bilevel_ridge.oracle import dense_hessian, dense_hypergradient, dense_inner

TOL = 1e-12


@pytest.fixture()
def data():
    return make_data(seed=0)


@pytest.fixture()
def rho():
    return np.linspace(-1.0, 1.0, 8)


# ---------- rho validation: exp(rho) must be strictly positive and finite ----------

def test_validate_rho_accepts_reasonable():
    r = validate_rho(np.array([-700.0, -1.0, 0.0, 1.0, 700.0]))
    lam = np.exp(r)
    assert np.all(np.isfinite(lam)) and np.all(lam > 0.0)


@pytest.mark.parametrize("bad", [
    np.array([np.nan, 0.0]),          # NaN
    np.array([0.0, np.inf]),          # +inf
    np.array([-np.inf, 0.0]),         # -inf
    np.array([0.0, 800.0]),           # exp overflows to inf
    np.array([0.0, -800.0]),          # exp underflows to exactly 0.0
])
def test_validate_rho_rejects_nonfinite_or_degenerate(bad):
    with pytest.raises(RhoValidationError):
        validate_rho(bad)


def test_validate_rho_rejects_bad_shape():
    with pytest.raises(RhoValidationError):
        validate_rho(np.zeros((3, 3)))
    with pytest.raises(RhoValidationError):
        validate_rho(np.zeros(5), p=4)


def test_hypergradient_rejects_invalid_rho(data):
    with pytest.raises(RhoValidationError):
        hypergradient(data.X, data.y, data.X_val, data.y_val, np.full(8, np.nan))


# ---------- train/validation separation ----------

def test_split_is_disjoint(data):
    assert np.intersect1d(data.train_index, data.val_index).size == 0
    assert not np.shares_memory(data.X, data.X_val)


def test_hypergradient_rejects_shared_memory(data):
    # identical objects: validation set IS the training set
    with pytest.raises(ValueError, match="disjoint"):
        hypergradient(data.X, data.y, data.X, data.y, np.zeros(8))
    # overlapping views of one buffer
    n = data.X.shape[0]
    big = np.vstack([data.X, data.X_val])
    with pytest.raises(ValueError, match="disjoint"):
        hypergradient(big[: n + 1], np.zeros(n + 1), big[n:], np.zeros(big.shape[0] - n),
                      np.zeros(8))


# ---------- inner solve ----------

def test_inner_solve_matches_dense(data, rho):
    w, info = solve_inner(data.X, data.y, rho, tol=TOL)
    w_dense = dense_inner(data.X, data.y, rho)
    assert info.converged and info.rel_residual <= TOL
    assert np.linalg.norm(w - w_dense) <= 1e-8 * max(1.0, np.linalg.norm(w_dense))


def test_inner_solution_satisfies_foc(data, rho):
    w, _ = solve_inner(data.X, data.y, rho, tol=TOL)
    n = data.X.shape[0]
    foc = data.X.T @ (data.X @ w - data.y) / n + np.exp(rho) * w
    assert np.linalg.norm(foc) <= 1e-9


def test_inner_objective_minimized(data, rho):
    w, _ = solve_inner(data.X, data.y, rho, tol=TOL)
    rng = np.random.default_rng(0)
    f0 = inner_objective(data.X, data.y, w, rho)
    for _ in range(5):
        f1 = inner_objective(data.X, data.y, w + 0.01 * rng.standard_normal(8), rho)
        assert f1 >= f0 - 1e-10


# ---------- hypergradient correctness ----------

def test_hypergradient_matches_dense_oracle(data, rho):
    res = hypergradient(data.X, data.y, data.X_val, data.y_val, rho,
                        tol_inner=TOL, tol_adj=TOL)
    g_or = dense_hypergradient(data.X, data.y, data.X_val, data.y_val, rho)
    assert np.linalg.norm(res.grad - g_or) <= 1e-9 * max(1.0, np.linalg.norm(g_or))


def test_hypergradient_matches_central_difference(data, rho):
    res = hypergradient(data.X, data.y, data.X_val, data.y_val, rho,
                        tol_inner=TOL, tol_adj=TOL)
    loss_fn = lambda r: outer_loss(data.X_val, data.y_val, dense_inner(data.X, data.y, r))
    g_fd = central_difference(loss_fn, rho, h=1e-5)
    assert np.linalg.norm(res.grad - g_fd) <= 1e-7 * max(1.0, np.linalg.norm(g_fd))


def test_fd_error_is_v_shaped_in_h(data, rho):
    loss_fn = lambda r: outer_loss(data.X_val, data.y_val, dense_inner(data.X, data.y, r))
    g_exact = dense_hypergradient(data.X, data.y, data.X_val, data.y_val, rho)
    rows = dict(fd_step_study(loss_fn, rho, [1e-1, 1e-5, 1e-11], g_exact))
    assert rows[1e-1] > 100 * rows[1e-5]    # too large: truncation dominated
    assert rows[1e-11] > 100 * rows[1e-5]   # too small: roundoff dominated


# ---------- inexact solves: bias and computable bound ----------

def test_error_bound_is_conservative(data, rho):
    g_or = dense_hypergradient(data.X, data.y, data.X_val, data.y_val, rho)
    for tol_inner, tol_adj in [(1e-2, TOL), (1e-6, TOL), (TOL, 1e-4), (TOL, TOL)]:
        res = hypergradient(data.X, data.y, data.X_val, data.y_val, rho,
                            tol_inner=tol_inner, tol_adj=tol_adj)
        err = np.linalg.norm(res.grad - g_or)
        assert res.grad_err_bound >= err * (1.0 - 1e-9)


def test_loose_inner_or_adjoint_tolerance_biases_gradient(data, rho):
    g_or = dense_hypergradient(data.X, data.y, data.X_val, data.y_val, rho)
    tight = hypergradient(data.X, data.y, data.X_val, data.y_val, rho,
                          tol_inner=TOL, tol_adj=TOL)
    loose_inner = hypergradient(data.X, data.y, data.X_val, data.y_val, rho,
                                tol_inner=1e-4, tol_adj=TOL)
    loose_adj = hypergradient(data.X, data.y, data.X_val, data.y_val, rho,
                              tol_inner=TOL, tol_adj=1e-4)
    e_tight = np.linalg.norm(tight.grad - g_or)
    assert np.linalg.norm(loose_inner.grad - g_or) > 1e3 * e_tight
    assert np.linalg.norm(loose_adj.grad - g_or) > 1e3 * e_tight


# ---------- preconditioner ----------

def test_preconditioner_reduces_iterations_on_illconditioned():
    d = make_data(n_train=200, n_val=100, p=20, seed=1, col_cond=1e4)
    rho = np.full(20, np.log(1e-8))
    with_pre = hypergradient(d.X, d.y, d.X_val, d.y_val, rho, tol_inner=1e-10,
                             tol_adj=1e-10, use_precond=True)
    without = hypergradient(d.X, d.y, d.X_val, d.y_val, rho, tol_inner=1e-10,
                            tol_adj=1e-10, use_precond=False)
    assert with_pre.inner.iters * 3 <= without.inner.iters
    assert with_pre.adjoint_info.iters * 3 <= without.adjoint_info.iters
    # both still agree with the oracle: iteration count is the only difference
    g_or = dense_hypergradient(d.X, d.y, d.X_val, d.y_val, rho)
    assert np.linalg.norm(with_pre.grad - g_or) < 1e-6
    assert np.linalg.norm(without.grad - g_or) < 1e-6


def test_diagonal_preconditioner_cannot_fix_rotated_spectrum():
    d = make_data(n_train=200, n_val=100, p=20, seed=1, col_cond=1e4, rotate=True)
    rho = np.full(20, np.log(1e-8))
    with_pre = solve_inner(d.X, d.y, rho, tol=1e-10, use_precond=True)[1]
    without = solve_inner(d.X, d.y, rho, tol=1e-10, use_precond=False)[1]
    assert with_pre.iters > 0.5 * without.iters  # no meaningful speedup


# ---------- CG robustness / failure diagnostics ----------

def test_cg_breakdown_on_indefinite_operator():
    with pytest.raises(RuntimeError, match="not symmetric positive definite"):
        cg(lambda v: -v, np.ones(4))


def test_cg_zero_rhs():
    x, info = cg(lambda v: 2.0 * v, np.zeros(3))
    assert info.converged and np.all(x == 0.0)


def test_shape_mismatch_raises(data):
    with pytest.raises(ValueError):
        hypergradient(data.X, data.y, data.X_val[:, :4], data.y_val, np.zeros(8))
    with pytest.raises(ValueError):
        solve_inner(data.X, data.y[:-1], np.zeros(8))


# ---------- implementation constraints ----------

def test_core_is_matrix_free_and_inverse_free():
    src = inspect.getsource(core)
    for forbidden in ("linalg.inv", "linalg.solve", "linalg.pinv", "X.T @ X"):
        assert forbidden not in src, f"core.py must not contain {forbidden!r}"


def test_one_gradient_step_decreases_outer_loss(data):
    rho = np.zeros(8)
    res = hypergradient(data.X, data.y, data.X_val, data.y_val, rho,
                        tol_inner=1e-10, tol_adj=1e-10)
    loss0 = outer_loss(data.X_val, data.y_val, res.w)
    rho_new = rho - 0.5 * res.grad
    loss1 = outer_loss(data.X_val, data.y_val, dense_inner(data.X, data.y, rho_new))
    assert loss1 < loss0
