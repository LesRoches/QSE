"""Construction of the paper-style grid economy and transport corridor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .model import Fundamentals, geometric_mean

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GridEconomy:
    side: int
    baseline_trade_costs: FloatArray
    counterfactual_trade_costs: FloatArray
    baseline_effective_distance: FloatArray
    counterfactual_effective_distance: FloatArray
    treated: NDArray[np.bool_]


def _all_pairs_shortest_paths(adjacency: FloatArray) -> FloatArray:
    distances = adjacency.copy()
    for intermediate in range(distances.shape[0]):
        distances = np.minimum(
            distances,
            distances[:, intermediate, None] + distances[None, intermediate, :],
        )
    return distances


def _grid_distances(node_costs: FloatArray) -> FloatArray:
    side = node_costs.shape[0]
    locations = side * side
    adjacency = np.full((locations, locations), np.inf)
    np.fill_diagonal(adjacency, 0.0)

    directions = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for row in range(side):
        for column in range(side):
            origin = row * side + column
            for drow, dcolumn in directions:
                other_row = row + drow
                other_column = column + dcolumn
                if not (0 <= other_row < side and 0 <= other_column < side):
                    continue
                destination = other_row * side + other_column
                step_length = np.sqrt(2.0) if drow and dcolumn else 1.0
                edge_cost = (
                    0.5
                    * (node_costs[row, column] + node_costs[other_row, other_column])
                    * step_length
                )
                adjacency[origin, destination] = edge_cost

    distances = _all_pairs_shortest_paths(adjacency)
    # The iceberg convention is d_nn = 1 rather than zero.
    np.fill_diagonal(distances, 1.0)
    return distances


def make_grid_economy(
    side: int = 11,
    *,
    baseline_node_cost: float = 7.9,
    corridor_node_cost: float = 1.0,
    distance_elasticity: float = 0.33,
) -> GridEconomy:
    """Create the cross-shaped transport experiment used in the MATLAB code."""

    if side < 3 or side % 2 == 0:
        raise ValueError("side must be an odd integer of at least three")
    if baseline_node_cost <= 0.0 or corridor_node_cost <= 0.0:
        raise ValueError("node costs must be positive")
    if distance_elasticity <= 0.0:
        raise ValueError("distance_elasticity must be positive")

    baseline_nodes = np.full((side, side), baseline_node_cost)
    counterfactual_nodes = baseline_nodes.copy()
    center = side // 2
    counterfactual_nodes[center, :] = corridor_node_cost
    counterfactual_nodes[:, center] = corridor_node_cost

    baseline_distance = _grid_distances(baseline_nodes)
    counterfactual_distance = _grid_distances(counterfactual_nodes)
    treated_grid = np.zeros((side, side), dtype=bool)
    treated_grid[center, :] = True
    treated_grid[:, center] = True

    return GridEconomy(
        side=side,
        baseline_trade_costs=baseline_distance**distance_elasticity,
        counterfactual_trade_costs=counterfactual_distance**distance_elasticity,
        baseline_effective_distance=baseline_distance,
        counterfactual_effective_distance=counterfactual_distance,
        treated=treated_grid.reshape(-1),
    )


def draw_fundamentals(
    locations: int,
    *,
    seed: int = 1,
    land_per_location: float = 100.0,
) -> Fundamentals:
    """Draw independent normalized log-normal productivity and amenities."""

    if locations < 1 or land_per_location <= 0.0:
        raise ValueError("locations and land_per_location must be positive")
    generator = np.random.default_rng(seed)
    productivity = np.exp(generator.normal(size=locations))
    amenities = np.exp(generator.normal(size=locations))
    productivity /= geometric_mean(productivity)
    amenities /= geometric_mean(amenities)
    land = np.full(locations, land_per_location)
    return Fundamentals(productivity, amenities, land)
