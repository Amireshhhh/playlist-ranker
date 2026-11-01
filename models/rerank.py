"""Greedy constrained re-ranking used at inference time (API and evaluation)."""

from collections import Counter
from typing import List

import numpy as np


def rerank(
    scores: List[float],
    artists: List[str],
    recent_artists: set,
    top_k: int,
    fatigue: float = 0.0,
    diversity: float = 0.0,
) -> List[int]:
    """
    Greedy constrained re-ranking. Returns the indices of the top_k picks in order.

    Scores are standardised (z-scores) so the weights are in units of score standard
    deviations. At each step the candidate maximising

        z_i - fatigue * [artist_i in last 5 prefix tracks] - diversity * (#earlier picks of artist_i)

    is picked. With both weights 0 this is plain sorting by score.
    """
    n = len(scores)
    if n == 0 or top_k <= 0:
        return []
    s = np.asarray(scores, dtype=np.float64)
    std = s.std()
    z = (s - s.mean()) / std if std > 0 else np.zeros(n)

    if fatigue == 0 and diversity == 0:
        return [int(i) for i in np.argsort(-z, kind="stable")[:top_k]]

    base = z - fatigue * np.array([a in recent_artists for a in artists], dtype=np.float64)
    picked_count: Counter = Counter()
    remaining = set(range(n))
    order: List[int] = []
    for _ in range(top_k):
        best = max(remaining, key=lambda i: (base[i] - diversity * picked_count[artists[i]], -i))
        order.append(best)
        remaining.remove(best)
        picked_count[artists[best]] += 1
    return order
