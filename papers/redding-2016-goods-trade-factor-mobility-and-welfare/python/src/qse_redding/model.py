r"""Numerical model for Redding (2016).

Matrix convention
-----------------
Rows index origins ``i`` and columns index destinations ``n``. Therefore
``trade_shares[i, n]`` is the paper's :math:`\pi_{ni}` and every column
sums to one.

The solvers follow the author's nested fixed-point idea: conditional on
population, wages clear goods markets; conditional on those wages, population
adjusts until residential choice probabilities are consistent with population.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Model = Literal["crs", "irs"]


class ConvergenceError(RuntimeError):
    """Raised when a fixed-point iteration reaches its iteration limit."""


@dataclass(frozen=True)
class Parameters:
    """Structural parameters.

    ``theta`` is the Frechet trade elasticity in the constant-returns model.
    In the increasing-returns model it is ``sigma - 1``.
    """

    alpha: float = 0.75
    theta: float = 4.0
    epsilon: float = 3.0
    sigma: float = 4.0
    fixed_cost: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.theta <= 0.0:
            raise ValueError("theta must be positive")
        if self.epsilon <= 1.0:
            raise ValueError("epsilon must exceed one for finite expected utility")
        if self.sigma <= 1.0:
            raise ValueError("sigma must exceed one")
        if self.fixed_cost <= 0.0:
            raise ValueError("fixed_cost must be positive")


@dataclass(frozen=True)
class Fundamentals:
    productivity: FloatArray
    amenities: FloatArray
    land: FloatArray

    def __post_init__(self) -> None:
        a = _positive_vector(self.productivity, "productivity")
        b = _positive_vector(self.amenities, "amenities")
        h = _positive_vector(self.land, "land")
        if not (a.size == b.size == h.size):
            raise ValueError("fundamental vectors must have the same length")
        object.__setattr__(self, "productivity", a)
        object.__setattr__(self, "amenities", b)
        object.__setattr__(self, "land", h)


@dataclass(frozen=True)
class SolverOptions:
    tolerance: float = 1e-9
    max_wage_iterations: int = 10_000
    max_population_iterations: int = 5_000
    wage_damping: float = 0.25
    population_damping: float = 0.25

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if self.max_wage_iterations < 1 or self.max_population_iterations < 1:
            raise ValueError("iteration limits must be positive")
        if not 0.0 < self.wage_damping <= 1.0:
            raise ValueError("wage_damping must lie in (0, 1]")
        if not 0.0 < self.population_damping <= 1.0:
            raise ValueError("population_damping must lie in (0, 1]")


@dataclass(frozen=True)
class Equilibrium:
    wages: FloatArray
    population: FloatArray
    trade_shares: FloatArray
    wage_iterations: int
    population_iterations: int
    goods_market_error: float
    population_error: float

    @property
    def domestic_trade_shares(self) -> FloatArray:
        return np.diag(self.trade_shares).copy()

    @property
    def income(self) -> FloatArray:
        return self.wages * self.population


def _positive_vector(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return array.copy()


def _trade_cost_matrix(values: ArrayLike, locations: int) -> FloatArray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (locations, locations):
        raise ValueError("trade_costs must be an N by N matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix <= 0.0):
        raise ValueError("trade_costs must contain finite positive values")
    return matrix.copy()


def geometric_mean(values: ArrayLike) -> float:
    vector = _positive_vector(values, "values")
    return float(np.exp(np.mean(np.log(vector))))


def _normalize_wages(wages: FloatArray) -> FloatArray:
    return wages / geometric_mean(wages)


def _country_structure(
    locations: int,
    total_labor: float | ArrayLike,
    country_ids: ArrayLike | None,
) -> tuple[NDArray[np.int64], FloatArray, FloatArray]:
    if country_ids is None:
        ids = np.zeros(locations, dtype=np.int64)
    else:
        ids = _encode_country_ids(country_ids, locations)

    countries = int(ids.max()) + 1
    totals = np.asarray(total_labor, dtype=float)
    if totals.ndim == 0:
        if countries != 1:
            raise ValueError("provide one total_labor value per country")
        totals = totals.reshape(1)
    if totals.shape != (countries,) or np.any(totals <= 0.0):
        raise ValueError("total_labor must be positive and match the countries")

    location_totals = totals[ids]
    return ids, totals, location_totals


def _encode_country_ids(
    country_ids: ArrayLike, locations: int
) -> NDArray[np.int64]:
    raw = np.asarray(country_ids)
    if raw.shape != (locations,):
        raise ValueError("country_ids must have one entry per location")
    mapping: dict[object, int] = {}
    encoded = np.empty(locations, dtype=np.int64)
    for index, label in enumerate(raw.tolist()):
        if label not in mapping:
            mapping[label] = len(mapping)
        encoded[index] = mapping[label]
    return encoded


def _normalize_population(
    weights: FloatArray,
    country_ids: NDArray[np.int64],
    country_totals: FloatArray,
) -> FloatArray:
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise FloatingPointError("population weights became non-positive or non-finite")
    result = np.empty_like(weights)
    for country, total in enumerate(country_totals):
        mask = country_ids == country
        result[mask] = total * weights[mask] / weights[mask].sum()
    return result


def trade_shares(
    wages: ArrayLike,
    population: ArrayLike,
    productivity: ArrayLike,
    trade_costs: ArrayLike,
    theta: float,
    model: Model = "crs",
) -> FloatArray:
    """Return bilateral expenditure shares, with origins in rows.

    CRS implements paper equation (18). IRS implements equation (68), using
    ``theta = sigma - 1`` and the endogenous variety measure proportional to
    population.
    """

    w = _positive_vector(wages, "wages")
    l = _positive_vector(population, "population")
    a = _positive_vector(productivity, "productivity")
    if not (w.size == l.size == a.size):
        raise ValueError("wages, population, and productivity must align")
    d = _trade_cost_matrix(trade_costs, w.size)
    if theta <= 0.0:
        raise ValueError("theta must be positive")

    if model == "crs":
        origin_competitiveness = a * w ** (-theta)
    elif model == "irs":
        origin_competitiveness = l * a**theta * w ** (-theta)
    else:
        raise ValueError("model must be 'crs' or 'irs'")

    numerators = origin_competitiveness[:, None] * d ** (-theta)
    denominators = numerators.sum(axis=0, keepdims=True)
    if np.any(denominators <= 0.0) or not np.all(np.isfinite(denominators)):
        raise FloatingPointError("invalid trade-share denominator")
    return numerators / denominators


def _population_attractiveness(
    population: FloatArray,
    domestic_shares: FloatArray,
    fundamentals: Fundamentals,
    parameters: Parameters,
    model: Model,
) -> FloatArray:
    a = fundamentals.productivity
    b = fundamentals.amenities
    h = fundamentals.land
    alpha = parameters.alpha
    theta = parameters.theta
    epsilon = parameters.epsilon

    if model == "crs":
        return (
            b
            * (a / domestic_shares) ** (alpha * epsilon / theta)
            * (population / h) ** (-epsilon * (1.0 - alpha))
        )
    if model == "irs":
        return (
            b
            * a ** (alpha * epsilon)
            * h ** (epsilon * (1.0 - alpha))
            * domestic_shares ** (-alpha * epsilon / theta)
            * population
            ** (-(epsilon * (1.0 - alpha) - alpha * epsilon / theta))
        )
    raise ValueError("model must be 'crs' or 'irs'")


def _solve_wages_given_population(
    population: FloatArray,
    wages: FloatArray,
    fundamentals: Fundamentals,
    trade_costs: FloatArray,
    parameters: Parameters,
    options: SolverOptions,
    model: Model,
) -> tuple[FloatArray, FloatArray, int, float]:
    theta = parameters.theta
    w = _normalize_wages(wages)

    for iteration in range(1, options.max_wage_iterations + 1):
        shares = trade_shares(
            w, population, fundamentals.productivity, trade_costs, theta, model
        )
        income = w * population
        expenditure = shares @ income
        error = float(np.max(np.abs(np.log(expenditure / income))))
        if error < options.tolerance:
            return w, shares, iteration, error

        proposal = w * (expenditure / income) ** (1.0 / theta)
        w = (1.0 - options.wage_damping) * w + options.wage_damping * proposal
        w = _normalize_wages(w)

    raise ConvergenceError(
        f"wages did not converge after {options.max_wage_iterations} iterations; "
        f"last log market-clearing error={error:.3e}"
    )


def solve_equilibrium(
    fundamentals: Fundamentals,
    trade_costs: ArrayLike,
    total_labor: float | ArrayLike,
    parameters: Parameters = Parameters(),
    options: SolverOptions = SolverOptions(),
    *,
    model: Model = "crs",
    country_ids: ArrayLike | None = None,
    initial_wages: ArrayLike | None = None,
    initial_population: ArrayLike | None = None,
) -> Equilibrium:
    """Solve the regional or multi-country spatial equilibrium.

    With ``country_ids``, population is mobile only among regions sharing the
    same id. Goods trade remains global. ``total_labor`` must then provide one
    fixed labor total per country.
    """

    locations = fundamentals.productivity.size
    d = _trade_cost_matrix(trade_costs, locations)
    ids, totals, location_totals = _country_structure(
        locations, total_labor, country_ids
    )

    if initial_population is None:
        counts = np.bincount(ids)
        population = location_totals / counts[ids]
    else:
        population = _positive_vector(initial_population, "initial_population")
        if population.size != locations:
            raise ValueError("initial_population has the wrong length")
        population = _normalize_population(population, ids, totals)

    if initial_wages is None:
        wages = np.ones(locations)
    else:
        wages = _positive_vector(initial_wages, "initial_wages")
        if wages.size != locations:
            raise ValueError("initial_wages has the wrong length")
    wages = _normalize_wages(wages)

    wage_iterations_total = 0
    population_error = np.inf
    goods_error = np.inf

    for outer_iteration in range(1, options.max_population_iterations + 1):
        wages, shares, wage_iterations, goods_error = _solve_wages_given_population(
            population,
            wages,
            fundamentals,
            d,
            parameters,
            options,
            model,
        )
        wage_iterations_total += wage_iterations

        attractiveness = _population_attractiveness(
            population,
            np.diag(shares),
            fundamentals,
            parameters,
            model,
        )
        target = _normalize_population(attractiveness, ids, totals)
        population_error = float(np.max(np.abs(np.log(target / population))))
        if population_error < options.tolerance:
            return Equilibrium(
                wages=wages,
                population=population,
                trade_shares=shares,
                wage_iterations=wage_iterations_total,
                population_iterations=outer_iteration,
                goods_market_error=goods_error,
                population_error=population_error,
            )

        elasticity_scale = parameters.epsilon * (1.0 - parameters.alpha)
        proposal = population * (target / population) ** (1.0 / elasticity_scale)
        population = (
            (1.0 - options.population_damping) * population
            + options.population_damping * proposal
        )
        population = _normalize_population(population, ids, totals)

    raise ConvergenceError(
        "population did not converge after "
        f"{options.max_population_iterations} iterations; "
        f"last log population error={population_error:.3e}"
    )


def recover_fundamentals(
    wages: ArrayLike,
    population: ArrayLike,
    land: ArrayLike,
    trade_costs: ArrayLike,
    parameters: Parameters = Parameters(),
    options: SolverOptions = SolverOptions(),
    *,
    model: Model = "crs",
) -> tuple[Fundamentals, FloatArray, int, float]:
    """Recover productivity iteratively and amenities in closed form.

    This implements Appendix section 2.8 while removing the redundant outer
    amenity loop in the original MATLAB routine.
    """

    w = _positive_vector(wages, "wages")
    l = _positive_vector(population, "population")
    h = _positive_vector(land, "land")
    if not (w.size == l.size == h.size):
        raise ValueError("wages, population, and land must align")
    d = _trade_cost_matrix(trade_costs, w.size)

    productivity = np.ones(w.size)
    income = w * l
    error = np.inf
    for iteration in range(1, options.max_wage_iterations + 1):
        shares = trade_shares(
            w, l, productivity, d, parameters.theta, model=model
        )
        expenditure = shares @ income
        error = float(np.max(np.abs(np.log(expenditure / income))))
        if error < options.tolerance:
            break

        exponent = 1.0 if model == "crs" else 1.0 / parameters.theta
        proposal = productivity * (income / expenditure) ** exponent
        productivity = (
            (1.0 - options.wage_damping) * productivity
            + options.wage_damping * proposal
        )
        productivity = productivity / geometric_mean(productivity)
    else:
        raise ConvergenceError(
            "productivity inversion did not converge after "
            f"{options.max_wage_iterations} iterations; last error={error:.3e}"
        )

    domestic = np.diag(shares)
    alpha = parameters.alpha
    theta = parameters.theta
    epsilon = parameters.epsilon
    if model == "crs":
        non_amenity_attractiveness = (
            (productivity / domestic) ** (alpha * epsilon / theta)
            * (l / h) ** (-epsilon * (1.0 - alpha))
        )
    else:
        non_amenity_attractiveness = (
            productivity ** (alpha * epsilon)
            * h ** (epsilon * (1.0 - alpha))
            * domestic ** (-alpha * epsilon / theta)
            * l ** (-(epsilon * (1.0 - alpha) - alpha * epsilon / theta))
        )

    observed_share = l / l.sum()
    amenities = observed_share / non_amenity_attractiveness
    amenities = amenities / geometric_mean(amenities)
    fundamentals = Fundamentals(productivity, amenities, h)
    return fundamentals, shares, iteration, error


def solve_counterfactual_hat(
    baseline: Equilibrium,
    baseline_trade_costs: ArrayLike,
    counterfactual_trade_costs: ArrayLike,
    parameters: Parameters = Parameters(),
    options: SolverOptions = SolverOptions(),
    *,
    country_ids: ArrayLike | None = None,
) -> Equilibrium:
    """Solve CRS equations (45)-(47) with A, B, and H held fixed."""

    w0 = _positive_vector(baseline.wages, "baseline wages")
    l0 = _positive_vector(baseline.population, "baseline population")
    locations = w0.size
    d0 = _trade_cost_matrix(baseline_trade_costs, locations)
    d1 = _trade_cost_matrix(counterfactual_trade_costs, locations)
    pi0 = np.asarray(baseline.trade_shares, dtype=float)
    if pi0.shape != (locations, locations):
        raise ValueError("baseline trade shares have the wrong shape")

    if country_ids is None:
        totals_input: float | FloatArray = float(l0.sum())
    else:
        encoded = _encode_country_ids(country_ids, locations)
        totals_input = np.bincount(encoded, weights=l0)
    ids, totals, _ = _country_structure(locations, totals_input, country_ids)

    d_hat_power = (d1 / d0) ** (-parameters.theta)
    domestic0 = np.diag(pi0)
    baseline_country_shares = np.empty(locations)
    for country, total in enumerate(totals):
        mask = ids == country
        baseline_country_shares[mask] = l0[mask] / total

    w1 = w0.copy()
    l1 = l0.copy()
    total_wage_iterations = 0
    goods_error = np.inf
    population_error = np.inf

    for outer_iteration in range(1, options.max_population_iterations + 1):
        for wage_iteration in range(1, options.max_wage_iterations + 1):
            w_hat = w1 / w0
            numerators = pi0 * d_hat_power * w_hat[:, None] ** (-parameters.theta)
            pi1 = numerators / numerators.sum(axis=0, keepdims=True)
            income1 = w_hat * (l1 / l0) * baseline.income
            expenditure1 = pi1 @ income1
            goods_error = float(
                np.max(np.abs(np.log(expenditure1 / income1)))
            )
            if goods_error < options.tolerance:
                break
            proposal = w1 * (expenditure1 / income1) ** (1.0 / parameters.theta)
            w1 = (1.0 - options.wage_damping) * w1 + options.wage_damping * proposal
            w_hat = w1 / w0
            w_hat = w_hat / geometric_mean(w_hat)
            w1 = w0 * w_hat
        else:
            raise ConvergenceError(
                "counterfactual wages did not converge; "
                f"last error={goods_error:.3e}"
            )
        total_wage_iterations += wage_iteration

        domestic_hat = np.diag(pi1) / domestic0
        population_hat = l1 / l0
        attractiveness = (
            baseline_country_shares
            * domestic_hat ** (-parameters.alpha * parameters.epsilon / parameters.theta)
            * population_hat ** (-parameters.epsilon * (1.0 - parameters.alpha))
        )
        target = _normalize_population(attractiveness, ids, totals)
        population_error = float(np.max(np.abs(np.log(target / l1))))
        if population_error < options.tolerance:
            return Equilibrium(
                wages=w1,
                population=l1,
                trade_shares=pi1,
                wage_iterations=total_wage_iterations,
                population_iterations=outer_iteration,
                goods_market_error=goods_error,
                population_error=population_error,
            )

        scale = parameters.epsilon * (1.0 - parameters.alpha)
        proposal = l1 * (target / l1) ** (1.0 / scale)
        l1 = (1.0 - options.population_damping) * l1 + options.population_damping * proposal
        l1 = _normalize_population(l1, ids, totals)

    raise ConvergenceError(
        "counterfactual population did not converge; "
        f"last error={population_error:.3e}"
    )


def price_index_crs(
    equilibrium: Equilibrium,
    fundamentals: Fundamentals,
    parameters: Parameters = Parameters(),
) -> FloatArray:
    """Equation (10), using the paper's definition of gamma."""

    from math import gamma

    if parameters.theta <= parameters.sigma - 1.0:
        raise ValueError("theta must exceed sigma - 1 for a finite price index")
    gamma_constant = gamma(
        (parameters.theta + 1.0 - parameters.sigma) / parameters.theta
    ) ** (1.0 / (1.0 - parameters.sigma))
    return gamma_constant * (
        fundamentals.productivity
        * equilibrium.wages ** (-parameters.theta)
        / equilibrium.domestic_trade_shares
    ) ** (-1.0 / parameters.theta)


def price_index_irs(
    equilibrium: Equilibrium,
    fundamentals: Fundamentals,
    trade_costs: ArrayLike,
    parameters: Parameters = Parameters(),
) -> FloatArray:
    """Equation (63) for the increasing-returns model."""

    locations = equilibrium.wages.size
    d = _trade_cost_matrix(trade_costs, locations)
    sigma = parameters.sigma
    if not np.isclose(parameters.theta, sigma - 1.0):
        raise ValueError("IRS price index requires theta = sigma - 1")
    inside = np.sum(
        equilibrium.population[:, None]
        * (d * equilibrium.wages[:, None] / fundamentals.productivity[:, None])
        ** (1.0 - sigma),
        axis=0,
    )
    return (
        sigma
        / (sigma - 1.0)
        * (1.0 / (sigma * parameters.fixed_cost)) ** (1.0 / (1.0 - sigma))
        * inside ** (1.0 / (1.0 - sigma))
    )


def land_rent(
    equilibrium: Equilibrium,
    fundamentals: Fundamentals,
    parameters: Parameters = Parameters(),
) -> FloatArray:
    """Land-market clearing, equation (16)."""

    return (
        (1.0 - parameters.alpha)
        / parameters.alpha
        * equilibrium.income
        / fundamentals.land
    )


def expected_utility_crs(
    equilibrium: Equilibrium,
    fundamentals: Fundamentals,
    parameters: Parameters = Parameters(),
) -> float:
    """Expected maximum utility, equation (13), in the CRS model."""

    from math import gamma

    prices = price_index_crs(equilibrium, fundamentals, parameters)
    rents = land_rent(equilibrium, fundamentals, parameters)
    total_income_per_worker = equilibrium.wages / parameters.alpha
    deterministic_income = total_income_per_worker / (
        prices**parameters.alpha * rents ** (1.0 - parameters.alpha)
    )
    delta = gamma((parameters.epsilon - 1.0) / parameters.epsilon)
    return float(
        delta
        * np.sum(
            fundamentals.amenities * deterministic_income**parameters.epsilon
        )
        ** (1.0 / parameters.epsilon)
    )


def welfare_gain_finite_mobility(
    baseline: Equilibrium,
    counterfactual: Equilibrium,
    parameters: Parameters = Parameters(),
) -> FloatArray:
    """Equation (52); entries coincide at a converged spatial equilibrium."""

    return (
        baseline.domestic_trade_shares
        / counterfactual.domestic_trade_shares
    ) ** (parameters.alpha / parameters.theta) * (
        baseline.population / counterfactual.population
    ) ** (1.0 / parameters.epsilon + 1.0 - parameters.alpha)


def welfare_gain_immobile(
    baseline: Equilibrium,
    counterfactual: Equilibrium,
    parameters: Parameters = Parameters(),
) -> FloatArray:
    """Equation (56), the perfectly immobile (ACR) limiting case."""

    return (
        baseline.domestic_trade_shares
        / counterfactual.domestic_trade_shares
    ) ** (parameters.alpha / parameters.theta)


def welfare_gain_perfect_mobility(
    baseline: Equilibrium,
    counterfactual: Equilibrium,
    parameters: Parameters = Parameters(),
) -> FloatArray:
    """Equation (54), no idiosyncratic location-taste heterogeneity."""

    return (
        baseline.domestic_trade_shares
        / counterfactual.domestic_trade_shares
    ) ** (parameters.alpha / parameters.theta) * (
        baseline.population / counterfactual.population
    ) ** (1.0 - parameters.alpha)
