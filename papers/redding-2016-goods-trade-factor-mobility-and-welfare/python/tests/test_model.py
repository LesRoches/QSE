from __future__ import annotations

import numpy as np

from qse_redding import (
    Fundamentals,
    Parameters,
    SolverOptions,
    recover_fundamentals,
    solve_counterfactual_hat,
    solve_equilibrium,
    trade_shares,
    welfare_gain_finite_mobility,
)


PARAMETERS = Parameters(alpha=0.75, theta=4.0, epsilon=3.0, sigma=4.0)
OPTIONS = SolverOptions(tolerance=1e-9, max_wage_iterations=20_000)


def symmetric_trade_costs(locations: int, foreign_cost: float = 1.5) -> np.ndarray:
    costs = np.full((locations, locations), foreign_cost)
    np.fill_diagonal(costs, 1.0)
    return costs


def test_trade_share_orientation_and_normalization() -> None:
    shares = trade_shares(
        wages=np.array([1.0, 1.2, 0.9]),
        population=np.ones(3),
        productivity=np.array([1.1, 0.8, 1.0]),
        trade_costs=symmetric_trade_costs(3),
        theta=4.0,
    )
    np.testing.assert_allclose(shares.sum(axis=0), 1.0)


def test_symmetric_crs_equilibrium() -> None:
    locations = 4
    fundamentals = Fundamentals(
        productivity=np.ones(locations),
        amenities=np.ones(locations),
        land=np.full(locations, 100.0),
    )
    equilibrium = solve_equilibrium(
        fundamentals,
        symmetric_trade_costs(locations),
        total_labor=400.0,
        parameters=PARAMETERS,
        options=OPTIONS,
    )
    np.testing.assert_allclose(equilibrium.wages, 1.0, atol=1e-10)
    np.testing.assert_allclose(equilibrium.population, 100.0, atol=1e-10)
    np.testing.assert_allclose(
        equilibrium.trade_shares @ equilibrium.income,
        equilibrium.income,
        rtol=1e-9,
    )


def test_recover_fundamentals_from_generated_equilibrium() -> None:
    productivity = np.array([0.8, 1.0, 1.2, 1.4])
    productivity /= np.exp(np.mean(np.log(productivity)))
    amenities = np.array([1.3, 0.9, 1.1, 0.8])
    amenities /= np.exp(np.mean(np.log(amenities)))
    fundamentals = Fundamentals(productivity, amenities, np.full(4, 100.0))
    costs = symmetric_trade_costs(4, foreign_cost=1.35)
    equilibrium = solve_equilibrium(
        fundamentals, costs, 1_000.0, PARAMETERS, OPTIONS
    )

    recovered, _, _, _ = recover_fundamentals(
        equilibrium.wages,
        equilibrium.population,
        fundamentals.land,
        costs,
        PARAMETERS,
        OPTIONS,
    )
    np.testing.assert_allclose(
        recovered.productivity, fundamentals.productivity, rtol=2e-7, atol=2e-7
    )
    np.testing.assert_allclose(
        recovered.amenities, fundamentals.amenities, rtol=2e-7, atol=2e-7
    )


def test_exact_hat_matches_level_counterfactual() -> None:
    fundamentals = Fundamentals(
        productivity=np.array([0.9, 1.1, 1.0, 1.2]),
        amenities=np.array([1.1, 0.9, 1.2, 0.8]),
        land=np.full(4, 100.0),
    )
    baseline_costs = symmetric_trade_costs(4, foreign_cost=1.5)
    counterfactual_costs = baseline_costs.copy()
    counterfactual_costs[0, 1] = counterfactual_costs[1, 0] = 1.15

    baseline = solve_equilibrium(
        fundamentals, baseline_costs, 1_000.0, PARAMETERS, OPTIONS
    )
    level = solve_equilibrium(
        fundamentals,
        counterfactual_costs,
        1_000.0,
        PARAMETERS,
        OPTIONS,
        initial_wages=baseline.wages,
        initial_population=baseline.population,
    )
    exact_hat = solve_counterfactual_hat(
        baseline,
        baseline_costs,
        counterfactual_costs,
        PARAMETERS,
        OPTIONS,
    )

    np.testing.assert_allclose(exact_hat.population, level.population, rtol=2e-7)
    np.testing.assert_allclose(
        exact_hat.wages / np.exp(np.mean(np.log(exact_hat.wages))),
        level.wages / np.exp(np.mean(np.log(level.wages))),
        rtol=2e-7,
    )
    np.testing.assert_allclose(exact_hat.trade_shares, level.trade_shares, rtol=2e-7)
    welfare = welfare_gain_finite_mobility(baseline, level, PARAMETERS)
    assert np.ptp(welfare) < 2e-7


def test_country_population_totals_are_preserved() -> None:
    fundamentals = Fundamentals(
        productivity=np.array([1.2, 0.8, 0.9, 1.1]),
        amenities=np.array([0.9, 1.1, 1.2, 0.8]),
        land=np.full(4, 100.0),
    )
    country_ids = np.array(["west", "west", "east", "east"])
    equilibrium = solve_equilibrium(
        fundamentals,
        symmetric_trade_costs(4, 1.4),
        total_labor=np.array([600.0, 400.0]),
        parameters=PARAMETERS,
        options=OPTIONS,
        country_ids=country_ids,
    )
    np.testing.assert_allclose(equilibrium.population[:2].sum(), 600.0)
    np.testing.assert_allclose(equilibrium.population[2:].sum(), 400.0)


def test_symmetric_irs_equilibrium() -> None:
    locations = 4
    fundamentals = Fundamentals(
        productivity=np.ones(locations),
        amenities=np.ones(locations),
        land=np.full(locations, 100.0),
    )
    equilibrium = solve_equilibrium(
        fundamentals,
        symmetric_trade_costs(locations),
        total_labor=400.0,
        parameters=PARAMETERS,
        options=OPTIONS,
        model="irs",
    )
    np.testing.assert_allclose(equilibrium.wages, 1.0, atol=1e-10)
    np.testing.assert_allclose(equilibrium.population, 100.0, atol=1e-10)
