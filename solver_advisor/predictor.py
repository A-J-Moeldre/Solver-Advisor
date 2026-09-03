# solver_advisor/predictor.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .features import FeatureFingerprint
from . import memory


def _score_performance(perf: Dict[str, Any]) -> float:
    """
    Convert a performance dict into a scalar score.
    Lower is better.

    Priority:
    1. runtime (seconds)
    2. iterations (fallback)
    """
    if "runtime" in perf:
        return float(perf["runtime"])
    if "iterations" in perf:
        return float(perf["iterations"])
    # If no usable metric, treat as very bad
    return float("inf")


def predict_config(
    features: FeatureFingerprint,
    k: int = 5,
    metric: str = "cosine",
) -> Dict[str, Any]:
    """
    Warm-start solver configuration prediction.

    Steps:
    1. Retrieve k nearest neighbors from memory.
    2. Score each by performance (runtime or iterations).
    3. Return the best-performing configuration.

    Returns
    -------
    config : Dict[str, Any]
        Solver configuration to warm-start the bandit.
    """
    neighbors: List[Tuple[Dict[str, Any], Dict[str, Any]]] = memory.retrieve(
        features=features,
        k=k,
        metric=metric,
    )

    if not neighbors:
        # No memory yet → return empty config (bandit will explore)
        return {}

    # Pick best-performing neighbor
    best_config = None
    best_score = float("inf")

    for config, perf in neighbors:
        score = _score_performance(perf)
        if score < best_score:
            best_score = score
            best_config = config

    return best_config or {}
