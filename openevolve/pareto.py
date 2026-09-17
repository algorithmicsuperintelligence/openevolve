"""Pareto (non-dominated) selection over several objectives.

With ``database.objectives`` set, programs are compared by dominance instead of by one scalar: A is better than B if A
is at least as good on every objective and strictly better on one. Programs that neither dominates are ordered by
NSGA-II's two secondary keys, computed against the current population:

* **rank**: peel the non-dominated set (rank 0), remove it, peel again (rank 1), and so on; lower is better. A rank-k
  program is dominated only by programs of lower rank.
* **crowding distance**: within a front, the sum over objectives of the normalized gap between a program's two
  nearest neighbours on that objective; boundary programs (the best on some objective) get infinity. Larger is better,
  because it prefers programs in sparsely populated parts of the front and keeps the front spread out instead of
  letting near-duplicates accumulate.

These are the rules of Deb et al., NSGA-II (2002). No objective is weighted or ordered above another.

The database turns (rank, crowding) into one float, ``key = -rank + 0.999 * crowding / (1 + crowding)`` (infinity ->
0.999), so that every existing "sort by fitness" call can keep working: rank always dominates, crowding only orders
programs of equal rank. The key is relative to the population and is recomputed whenever the population changes.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Values = Tuple[float, ...]


def objective_values(metrics: Dict[str, Any], objectives: Sequence[str], directions: Optional[Sequence[str]] = None) -> Optional[Values]:
    """The objective vector of a metrics dict, oriented so that larger is better on every coordinate
    (``directions`` entries are "max" (default) or "min"). None if any objective is missing or not a finite number."""
    directions = list(directions or [])
    directions += ["max"] * (len(objectives) - len(directions))
    out = []
    for name, direction in zip(objectives, directions):
        v = metrics.get(name)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        v = float(v)
        if not math.isfinite(v):
            return None
        out.append(v if direction == "max" else -v)
    return tuple(out)


def dominates(a: Values, b: Values) -> bool:
    """a is at least as good as b everywhere and strictly better somewhere."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def non_dominated_sort(points: Sequence[Values]) -> List[List[int]]:
    """Fronts as lists of indices into ``points``; fronts[0] is the non-dominated set."""
    n = len(points)
    if n == 0:
        return []
    P = np.asarray(points, dtype=float)
    ge = (P[:, None, :] >= P[None, :, :]).all(axis=2)
    gt = (P[:, None, :] > P[None, :, :]).any(axis=2)
    dom = ge & gt                                   # dom[i, j]: i dominates j
    remaining = np.ones(n, dtype=bool)
    fronts: List[List[int]] = []
    while remaining.any():
        dominated_by_remaining = (dom & remaining[:, None]).any(axis=0)      # j is dominated by some remaining i
        front = np.where(remaining & ~dominated_by_remaining)[0]
        fronts.append(front.tolist())
        remaining[front] = False
    return fronts


def crowding_distance(points: Sequence[Values]) -> List[float]:
    """NSGA-II crowding distance of each point within one front (boundary points: infinity)."""
    n = len(points)
    if n == 0:
        return []
    if n <= 2:
        return [math.inf] * n
    P = np.asarray(points, dtype=float)
    d = np.zeros(n)
    for j in range(P.shape[1]):
        order = np.argsort(P[:, j], kind="stable")
        lo, hi = P[order[0], j], P[order[-1], j]
        d[order[0]] = d[order[-1]] = math.inf
        if hi > lo:
            gaps = (P[order[2:], j] - P[order[:-2], j]) / (hi - lo)
            d[order[1:-1]] += gaps
    return d.tolist()


def key_of(rank: int, crowding: float) -> float:
    c = 0.999 if math.isinf(crowding) else 0.999 * crowding / (1.0 + crowding)
    return -float(rank) + c


def pareto_keys(items: Sequence[Tuple[str, Optional[Values]]]) -> Dict[str, Tuple[int, float, float]]:
    """{id: (rank, crowding, key)} for a population; items whose values are None (missing objectives) get the rank
    below the last front and no crowding."""
    valid = [(i, v) for i, v in items if v is not None]
    out: Dict[str, Tuple[int, float, float]] = {}
    fronts = non_dominated_sort([v for _, v in valid])
    for rank, front in enumerate(fronts):
        cd = crowding_distance([valid[i][1] for i in front])
        for i, c in zip(front, cd):
            out[valid[i][0]] = (rank, c, key_of(rank, c))
    worst = len(fronts)
    for i, v in items:
        if v is None:
            out[i] = (worst, 0.0, key_of(worst, 0.0))
    return out
