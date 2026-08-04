"""Reproduce a compact version of the paper's transport experiment.

Run from the ``python`` directory after installing the package:

    python -m pip install -e '.[example]'
    python examples/run_transport_counterfactual.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qse_redding import (
    Parameters,
    SolverOptions,
    draw_fundamentals,
    land_rent,
    make_grid_economy,
    price_index_crs,
    solve_counterfactual_hat,
    solve_equilibrium,
    welfare_gain_finite_mobility,
)


def _as_grid(values: np.ndarray, side: int) -> np.ndarray:
    return np.asarray(values).reshape(side, side)


def main() -> None:
    parameters = Parameters(alpha=0.75, theta=4.0, epsilon=3.0, sigma=4.0)
    options = SolverOptions(tolerance=1e-8)
    grid = make_grid_economy(side=11, distance_elasticity=0.33)
    fundamentals = draw_fundamentals(grid.side**2, seed=1)
    total_labor = 153_889.0

    baseline = solve_equilibrium(
        fundamentals,
        grid.baseline_trade_costs,
        total_labor,
        parameters,
        options,
    )
    counterfactual = solve_equilibrium(
        fundamentals,
        grid.counterfactual_trade_costs,
        total_labor,
        parameters,
        options,
        initial_wages=baseline.wages,
        initial_population=baseline.population,
    )
    counterfactual_hat = solve_counterfactual_hat(
        baseline,
        grid.baseline_trade_costs,
        grid.counterfactual_trade_costs,
        parameters,
        options,
    )

    prices0 = price_index_crs(baseline, fundamentals, parameters)
    prices1 = price_index_crs(counterfactual, fundamentals, parameters)
    rents0 = land_rent(baseline, fundamentals, parameters)
    rents1 = land_rent(counterfactual, fundamentals, parameters)
    welfare = welfare_gain_finite_mobility(baseline, counterfactual, parameters)

    level_hat_population_error = np.max(
        np.abs(counterfactual_hat.population / counterfactual.population - 1.0)
    )
    print(f"Baseline population iterations: {baseline.population_iterations}")
    print(f"Counterfactual population iterations: {counterfactual.population_iterations}")
    print(f"Level-vs-hat maximum population discrepancy: {level_hat_population_error:.3e}")
    print(f"Common welfare gain: {np.exp(np.mean(np.log(welfare))):.6f}")
    print(f"Cross-location welfare dispersion: {np.ptp(welfare):.3e}")

    distance_ratio = np.mean(
        grid.counterfactual_effective_distance
        / grid.baseline_effective_distance,
        axis=1,
    )
    panels = [
        (distance_ratio, "Average effective-distance ratio"),
        (counterfactual.population / baseline.population, "Population ratio"),
        (counterfactual.wages / baseline.wages, "Wage ratio"),
        (prices1 / prices0, "Price-index ratio"),
        (rents1 / rents0, "Land-rent ratio"),
        (welfare, "Finite-mobility welfare ratio"),
    ]

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, (values, title) in zip(axes.flat, panels, strict=True):
        color_limits = {}
        if title == "Finite-mobility welfare ratio":
            center = float(np.exp(np.mean(np.log(values))))
            color_limits = {"vmin": center * 0.999, "vmax": center * 1.001}
        image = axis.imshow(
            _as_grid(values, grid.side),
            origin="lower",
            cmap="viridis",
            **color_limits,
        )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, shrink=0.8)

    output_directory = Path(__file__).resolve().parents[1] / "outputs"
    output_directory.mkdir(exist_ok=True)
    output_path = output_directory / "transport_counterfactual.png"
    figure.savefig(output_path, dpi=180)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
