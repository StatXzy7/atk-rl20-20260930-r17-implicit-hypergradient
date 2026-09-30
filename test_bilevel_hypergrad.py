from __future__ import annotations

import unittest

import numpy as np

from bilevel_hypergrad import (
    ConvergenceError,
    disjoint_train_validation_split,
    hessian_diagonal,
    hypergradient,
    make_inner_matvec,
    pcg_solve,
    regularization_factors,
    solve_inner,
    split_by_indices,
    validate_data,
)
from dense_oracle import central_difference_gradient, dense_hypergradient, relative_l2_error


def sample_data(seed: int = 0, n: int = 24, m: int = 12, d: int = 5):
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=d)
    x_train = rng.normal(size=(n, d))
    x_val = rng.normal(size=(m, d))
    y_train = x_train @ beta + 0.02 * rng.normal(size=n)
    y_val = x_val @ beta + 0.02 * rng.normal(size=m)
    return x_train, y_train, x_val, y_val


class ValidationTests(unittest.TestCase):
    def test_exp_rho_strictly_positive_and_finite(self):
        factors = regularization_factors(np.array([-100.0, 0.0, 100.0]))
        self.assertTrue(np.all(factors > 0.0))
        self.assertTrue(np.all(np.isfinite(factors)))

        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            regularization_factors(np.array([np.nan]))
        with self.assertRaisesRegex(ValueError, "finite"):
            regularization_factors(np.array([800.0]))

    def test_shape_and_dtype_validation(self):
        x_train, y_train, x_val, y_val = sample_data()
        with self.assertRaisesRegex(ValueError, "same feature count"):
            validate_data(x_train, y_train, x_val[:, :-1], y_val)
        with self.assertRaisesRegex(TypeError, "floating-point"):
            regularization_factors(np.array([0], dtype=np.int32))
        with self.assertRaisesRegex(ValueError, "finite"):
            regularization_factors(np.array([1000.0]))

    def test_validation_rejects_integer_data(self):
        x_train, y_train, x_val, y_val = sample_data()
        with self.assertRaisesRegex(TypeError, "floating-point"):
            validate_data(x_train.astype(np.int32), y_train, x_val, y_val)

    def test_index_split_is_disjoint(self):
        train_idx, val_idx = disjoint_train_validation_split(10, 6, seed=4)
        self.assertEqual(len(train_idx), 6)
        self.assertEqual(len(val_idx), 4)
        self.assertTrue(np.intersect1d(train_idx, val_idx).size == 0)

        x = np.arange(10, dtype=np.float64)[:, None]
        y = np.arange(10, dtype=np.float64)
        split_by_indices(x, y, train_idx, val_idx)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            split_by_indices(x, y, np.array([0, 1]), np.array([1, 2]))


class SolverTests(unittest.TestCase):
    def test_matrix_free_pcg_matches_dense_solve(self):
        x_train, y_train, _, _ = sample_data(seed=2)
        rho = np.linspace(-2.0, 1.0, x_train.shape[1])
        factors = regularization_factors(rho)
        dense_hessian = x_train.T @ x_train / x_train.shape[0] + np.diag(factors)
        rhs = x_train.T @ y_train / x_train.shape[0]
        result = pcg_solve(
            make_inner_matvec(x_train, factors),
            rhs,
            hessian_diagonal(x_train, factors),
            rtol=1e-13,
        )
        expected = np.linalg.solve(dense_hessian, rhs)
        self.assertLess(np.linalg.norm(result.x - expected) / np.linalg.norm(expected), 1e-11)
        self.assertLess(result.residual_norm, 1e-13 * np.linalg.norm(rhs))

    def test_zero_rhs_returns_zero(self):
        result = solve_inner(np.eye(3), np.zeros(3), np.zeros(3), rtol=1e-12)
        self.assertTrue(np.all(result.x == 0.0))
        self.assertTrue(result.converged)

    def test_non_convergence_is_explicit(self):
        matrix = np.diag([1.0, 1000.0])
        rhs = np.array([1.0, 1.0])
        with self.assertRaises(ConvergenceError):
            pcg_solve(lambda v: matrix @ v, rhs, rtol=1e-14, maxiter=1, strict=True)
        result = pcg_solve(lambda v: matrix @ v, rhs, rtol=1e-14, maxiter=1, strict=False)
        self.assertFalse(result.converged)
        self.assertGreater(result.residual_norm, result.tolerance)

    def test_non_spd_operator_raises(self):
        with self.assertRaisesRegex(RuntimeError, "not SPD"):
            pcg_solve(lambda v: -np.ones(2), np.ones(2), rtol=1e-12, maxiter=5)


class HypergradientTests(unittest.TestCase):
    def test_matrix_free_hypergradient_matches_dense_oracle(self):
        x_train, y_train, x_val, y_val = sample_data(seed=5, n=30, m=14, d=7)
        rho = np.array([-2.0, -1.0, -0.4, 0.0, 0.8, 1.5, 2.0])
        result = hypergradient(
            x_train,
            y_train,
            x_val,
            y_val,
            rho,
            inner_rtol=1e-12,
            adjoint_rtol=1e-12,
        )
        oracle = dense_hypergradient(x_train, y_train, x_val, y_val, rho)
        self.assertLess(relative_l2_error(result.w, oracle.w), 1e-11)
        self.assertLess(relative_l2_error(result.adjoint, oracle.adjoint), 1e-10)
        self.assertLess(relative_l2_error(result.gradient, oracle.gradient), 1e-10)

    def test_central_difference_matches_exact_gradient_near_optimal_step(self):
        x_train, y_train, x_val, y_val = sample_data(seed=9, n=20, m=10, d=4)
        rho = np.array([-1.0, -0.3, 0.2, 0.9])
        oracle = dense_hypergradient(x_train, y_train, x_val, y_val, rho)

        def loss(value: np.ndarray) -> float:
            return dense_hypergradient(x_train, y_train, x_val, y_val, value).val_loss

        good = central_difference_gradient(loss, rho, 1e-5)
        coarse = central_difference_gradient(loss, rho, 1e-1)
        tiny = central_difference_gradient(loss, rho, 1e-9)
        self.assertLess(relative_l2_error(good, oracle.gradient), 1e-8)
        self.assertGreater(relative_l2_error(coarse, oracle.gradient), 1e-4)
        self.assertGreater(relative_l2_error(tiny, oracle.gradient), 1e-7)

    def test_loose_tolerance_biases_gradient_and_bound_covers_error(self):
        x_train, y_train, x_val, y_val = sample_data(seed=11, n=24, m=12, d=6)
        rho = np.full(6, -4.0)
        oracle = dense_hypergradient(x_train, y_train, x_val, y_val, rho)
        result = hypergradient(
            x_train,
            y_train,
            x_val,
            y_val,
            rho,
            inner_rtol=1e-2,
            adjoint_rtol=1e-2,
            strict=False,
        )
        error = np.linalg.norm(result.gradient - oracle.gradient)
        self.assertGreater(error, 1e-6)
        self.assertLessEqual(error, result.gradient_bound)


if __name__ == "__main__":
    unittest.main()
