from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


def farthest_pair(distance: np.ndarray) -> Tuple[int, int]:
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("Distance matrix must be square.")
    n = distance.shape[0]
    if n < 2:
        raise ValueError("Need at least two frames to pick endpoints.")

    idx = np.unravel_index(np.argmax(distance, axis=None), distance.shape)
    return int(idx[0]), int(idx[1])


def bidirectional_greedy(distance: np.ndarray, start_pair: Tuple[int, int]) -> np.ndarray:
    n = distance.shape[0]
    remaining = set(range(n))
    order = [start_pair[0], start_pair[1]]
    remaining.discard(start_pair[0])
    remaining.discard(start_pair[1])

    while remaining:
        best_candidate = None
        best_side = None
        best_cost = float("inf")

        left = order[0]
        right = order[-1]
        for candidate in remaining:
            left_cost = distance[candidate, left]
            right_cost = distance[right, candidate]
            if left_cost < best_cost:
                best_candidate = candidate
                best_side = "left"
                best_cost = left_cost
            if right_cost < best_cost:
                best_candidate = candidate
                best_side = "right"
                best_cost = right_cost

        if best_candidate is None or best_side is None:
            raise RuntimeError("Failed to select a candidate during greedy path construction.")

        if best_side == "left":
            order.insert(0, best_candidate)
        else:
            order.append(best_candidate)
        remaining.remove(best_candidate)

    return np.asarray(order, dtype=int)


def two_opt(
    order: Sequence[int],
    distance: np.ndarray,
    *,
    max_iter: int = 20000,
    band: int = 0,
) -> np.ndarray:
    n = len(order)
    if n < 4:
        return np.asarray(order, dtype=int)

    best_order = np.asarray(order, dtype=int)
    iteration = 0
    improved = True

    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i in range(n - 3):
            max_j = n - 2
            if band > 0:
                max_j = min(max_j, i + band)
            if max_j < i + 2:
                continue
            for j in range(i + 2, max_j + 1):
                if j == i + 1:
                    continue
                gain = _two_opt_gain(best_order, distance, i, j)
                if gain < -1e-9:
                    best_order[i + 1 : j + 1] = best_order[i + 1 : j + 1][::-1]
                    improved = True
        if not improved:
            break
    return best_order


def path_cost(order: Iterable[int], distance: np.ndarray) -> float:
    total = 0.0
    order = list(order)
    for a, b in zip(order[:-1], order[1:]):
        total += float(distance[a, b])
    return total


def _two_opt_gain(order: np.ndarray, distance: np.ndarray, i: int, j: int) -> float:
    a, b = order[i], order[i + 1]
    c, d = order[j], order[j + 1]
    current = distance[a, b] + distance[c, d]
    proposed = distance[a, c] + distance[b, d]
    return proposed - current
