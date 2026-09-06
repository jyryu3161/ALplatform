"""Finite regression and discovery metrics for heldout docking replay."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import rankdata


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 1D array")
    return array


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    """Compute RMSE, MAE, R² and Spearman with JSON-safe undefined values.

    Empty inputs have no defined metrics. R² is undefined for fewer than two
    observations or constant targets. Spearman is undefined for fewer than two
    observations or either constant vector. Undefined values are ``None``.
    """
    truth = _finite_vector(y_true, "y_true")
    prediction = _finite_vector(y_pred, "y_pred")
    if prediction.shape != truth.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    metrics = {"n": len(truth), "rmse": None, "mae": None, "r2": None, "spearman": None}
    if not len(truth):
        return metrics
    error = truth - prediction
    metrics["rmse"] = float(np.sqrt(np.mean(error ** 2)))
    metrics["mae"] = float(np.mean(np.abs(error)))
    total = np.sum((truth - truth.mean()) ** 2)
    if len(truth) >= 2 and np.ptp(truth) > 0 and total > 0:
        metrics["r2"] = float(1 - np.sum(error ** 2) / total)
    if len(truth) >= 2 and np.ptp(truth) > 0 and np.ptp(prediction) > 0:
        metrics["spearman"] = float(np.corrcoef(rankdata(truth), rankdata(prediction))[0, 1])
    # Inputs can be finite yet overflow in squared errors at extreme scales.
    return {key: value if value is None or np.isfinite(value) else None for key, value in metrics.items()}


def discovery_metrics(
    pool_scores: Sequence[float],
    acquired_indices: Sequence[int],
    top_k: int,
    ids: Sequence[str] | None = None,
) -> dict:
    """Evaluate discovery on candidate-pool truths, excluding all holdout rows.

    ``acquired_indices`` contains unique positions in ``pool_scores``. The
    caller must supply only the eligible candidate pool, never the full library
    including a holdout. Top-k ties use lexical IDs or stable input order.
    ``top_k`` is capped by pool size. Recall is hits/k, enrichment is observed
    hit rate divided by k/pool_size, and simple regret is best observed score
    minus the pool minimum. At zero evaluations recall is zero; enrichment,
    best score and regret are undefined (``None``). Overflowed regret is also
    ``None`` so finite inputs always produce JSON-safe metrics.
    """
    truth = _finite_vector(pool_scores, "pool_scores")
    if not len(truth):
        raise ValueError("pool_scores must not be empty")
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    acquired = np.asarray(acquired_indices)
    if acquired.ndim != 1:
        raise ValueError("acquired_indices must be 1D")
    if len(acquired) and not np.issubdtype(acquired.dtype, np.integer):
        raise ValueError("acquired_indices must be integers")
    acquired = acquired.astype(np.int64)
    if np.any(acquired < 0) or np.any(acquired >= len(truth)):
        raise ValueError("acquired_indices contains positions outside the candidate pool")
    if len(np.unique(acquired)) != len(acquired):
        raise ValueError("acquired_indices must be unique; repeated queries cannot increase discovery")
    if ids is None:
        tie_key = np.arange(len(truth))
    else:
        tie_key = np.asarray(ids, dtype=str)
        if tie_key.shape != truth.shape or len(np.unique(tie_key)) != len(truth):
            raise ValueError("ids must contain one unique identity per candidate")
    k = min(int(top_k), len(truth))
    best_indices = np.lexsort((tie_key, truth))[:k]
    hits = int(np.isin(acquired, best_indices).sum())
    optimum = float(np.min(truth))
    best = float(np.min(truth[acquired])) if len(acquired) else None
    regret = best - optimum if best is not None else None
    return {
        "pool_size": len(truth),
        "n_acquired": len(acquired),
        "top_k": k,
        "top_k_hits": hits,
        "top_k_recall": hits / k,
        "enrichment_factor": (hits / len(acquired)) / (k / len(truth)) if len(acquired) else None,
        "best_score": best,
        "optimal_score": optimum,
        "simple_regret": regret if regret is None or np.isfinite(regret) else None,
    }
