from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from bilevel_hypergrad import backtracking_outer_step, hypergradient, outer_loss
from dense_oracle import central_difference_gradient, dense_hypergradient, relative_l2_error


def make_data(seed: int, n: int = 60, m: int = 30, d: int = 12, ill_conditioned: bool = False):
    rng = np.random.default_rng(seed)
    if ill_conditioned:
        scales = np.logspace(0.0, -3.0, d)
    else:
        scales = np.logspace(0.0, -0.7, d)
    beta = rng.normal(size=d)
    x_train = rng.normal(size=(n, d)) * scales
    x_val = rng.normal(size=(m, d)) * scales
    y_train = x_train @ beta + 0.03 * rng.normal(size=n)
    y_val = x_val @ beta + 0.03 * rng.normal(size=m)
    return x_train, y_train, x_val, y_val, scales


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def render(value: object) -> str:
        if isinstance(value, float):
            if np.isnan(value):
                return "nan"
            if value == 0.0:
                return "0"
            if abs(value) < 1e-3 or abs(value) >= 1e4:
                return f"{value:.3e}"
            return f"{value:.6f}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def finite_difference_experiment():
    x_train, y_train, x_val, y_val, _ = make_data(0)
    rho = np.full(x_train.shape[1], -1.0)
    reference = dense_hypergradient(x_train, y_train, x_val, y_val, rho)

    def loss(value: np.ndarray) -> float:
        return dense_hypergradient(x_train, y_train, x_val, y_val, value).val_loss

    steps = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    rows = []
    for step in steps:
        estimate = central_difference_gradient(loss, rho, step)
        rows.append([
            step,
            relative_l2_error(estimate, reference.gradient),
            float(np.linalg.norm(estimate - reference.gradient)),
        ])
    return rows


def outer_update_experiment():
    x_train, y_train, x_val, y_val, _ = make_data(101)
    rho = np.full(x_train.shape[1], -1.2)
    rows = []

    for update in range(5):
        result = hypergradient(
            x_train,
            y_train,
            x_val,
            y_val,
            rho,
            inner_rtol=1e-11,
            adjoint_rtol=1e-11,
            precondition=True,
        )
        reference = dense_hypergradient(x_train, y_train, x_val, y_val, rho)
        new_rho, new_loss, step, trials = backtracking_outer_step(
            x_train,
            y_train,
            x_val,
            y_val,
            rho,
            result,
            initial_step=0.05,
        )
        rows.append([
            update,
            result.val_loss,
            new_loss,
            step,
            relative_l2_error(result.gradient, reference.gradient),
            result.inner_result.iterations,
            result.adjoint_result.iterations,
        ])
        rho = new_rho

    return rows


def tolerance_experiment():
    x_train, y_train, x_val, y_val, _ = make_data(202, ill_conditioned=True)
    rho = np.full(x_train.shape[1], np.log(1e-6))
    reference = dense_hypergradient(x_train, y_train, x_val, y_val, rho)
    cases = [
        (1e-3, 1e-3),
        (1e-6, 1e-8),
        (1e-8, 1e-8),
        (1e-11, 1e-11),
    ]
    rows = []
    for inner_tol, adjoint_tol in cases:
        result = hypergradient(
            x_train,
            y_train,
            x_val,
            y_val,
            rho,
            inner_rtol=inner_tol,
            adjoint_rtol=adjoint_tol,
            precondition=True,
            strict=False,
        )
        rows.append([
            inner_tol,
            adjoint_tol,
            result.inner_result.residual_norm,
            result.adjoint_result.residual_norm,
            relative_l2_error(result.gradient, reference.gradient),
            result.gradient_bound,
            result.w_error_bound,
            result.adjoint_error_bound,
        ])
    return rows


def seed_experiment():
    rows = []
    for seed in [0, 1, 2, 3, 7]:
        x_train, y_train, x_val, y_val, _ = make_data(seed)
        rho = np.full(x_train.shape[1], -1.0 + 0.05 * seed)
        result = hypergradient(
            x_train,
            y_train,
            x_val,
            y_val,
            rho,
            inner_rtol=1e-11,
            adjoint_rtol=1e-11,
        )
        reference = dense_hypergradient(x_train, y_train, x_val, y_val, rho)
        rows.append([
            seed,
            result.inner_result.residual_norm,
            result.adjoint_result.residual_norm,
            relative_l2_error(result.gradient, reference.gradient),
            result.gradient_bound,
            result.lambda_lower,
            reference.min_eigenvalue,
        ])
    return rows


def preconditioner_experiment():
    rows = []
    for seed in range(5):
        x_train, y_train, x_val, y_val, _ = make_data(
            seed,
            n=80,
            m=40,
            d=30,
            ill_conditioned=True,
        )
        rho = np.full(x_train.shape[1], np.log(1e-6))
        reference = dense_hypergradient(x_train, y_train, x_val, y_val, rho)
        for precondition in [False, True]:
            result = hypergradient(
                x_train,
                y_train,
                x_val,
                y_val,
                rho,
                inner_rtol=1e-12,
                adjoint_rtol=1e-12,
                precondition=precondition,
            )
            rows.append([
                seed,
                "diagonal" if precondition else "none",
                reference.condition_number,
                result.inner_result.iterations,
                result.adjoint_result.iterations,
                result.inner_result.residual_norm,
                result.adjoint_result.residual_norm,
                relative_l2_error(result.gradient, reference.gradient),
            ])
    return rows


def main() -> None:
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    experiments = [
        (
            "finite_difference",
            ["h", "relative_l2_error", "absolute_l2_error"],
            finite_difference_experiment(),
        ),
        (
            "outer_updates",
            ["update", "loss", "next_loss", "accepted_step", "gradient_error", "inner_iters", "adjoint_iters"],
            outer_update_experiment(),
        ),
        (
            "tolerance_linkage",
            ["inner_rtol", "adjoint_rtol", "inner_residual", "adjoint_residual", "gradient_error", "gradient_bound", "w_bound", "adjoint_bound"],
            tolerance_experiment(),
        ),
        (
            "seeds",
            ["seed", "inner_residual", "adjoint_residual", "gradient_error", "guaranteed_bound", "lambda_guaranteed", "lambda_actual"],
            seed_experiment(),
        ),
        (
            "preconditioner",
            ["seed", "preconditioner", "condition_number", "inner_iters", "adjoint_iters", "inner_residual", "adjoint_residual", "gradient_error"],
            preconditioner_experiment(),
        ),
    ]

    sections = ["# Verifiable Bilevel Optimization Experiments", ""]
    for name, headers, rows in experiments:
        write_csv(output_dir / f"{name}.csv", headers, rows)
        title = " ".join(part.capitalize() for part in name.split("_"))
        sections.extend([f"## {title}", "", markdown_table(headers, rows), ""])

    (output_dir / "report.md").write_text("\n".join(sections), encoding="utf-8")
    print((output_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
