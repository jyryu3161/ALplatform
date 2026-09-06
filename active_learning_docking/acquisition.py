"""Acquisition rules for minimizing docking scores in their original units."""

from __future__ import annotations

from typing import Sequence

import numpy as np


BETA_SCHEDULES = ("constant", "linear", "exponential")


def scheduled_beta(cycle: int, cycles: int, beta: float, beta_final: float | None = None,
                   schedule: str = "constant") -> float:
    """Exploration weight for adaptive cycle ``cycle`` (1-based) of ``cycles``.

    ``constant`` returns ``beta``. ``linear`` moves from ``beta`` at cycle 1 to
    ``beta_final`` at the configured last cycle; ``exponential`` follows the
    geometric path between the same endpoints and needs positive values. The
    schedule is defined over the configured cycle count, so an early stop for
    budget or pool exhaustion leaves later beta values unused.
    """
    if schedule not in BETA_SCHEDULES:
        raise ValueError(f"schedule must be one of {BETA_SCHEDULES}")
    if isinstance(cycle, bool) or not isinstance(cycle, (int, np.integer)) or cycle < 1:
        raise ValueError("cycle must be a positive integer")
    if isinstance(cycles, bool) or not isinstance(cycles, (int, np.integer)) or cycles < cycle:
        raise ValueError("cycles must be an integer no smaller than cycle")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if schedule == "constant":
        return float(beta)
    if beta_final is None or not np.isfinite(beta_final) or beta_final < 0:
        raise ValueError("beta_final must be finite and nonnegative for a scheduled beta")
    fraction = 0.0 if cycles == 1 else (cycle - 1) / (cycles - 1)
    if schedule == "linear":
        return float(beta + (beta_final - beta) * fraction)
    if beta <= 0 or beta_final <= 0:
        raise ValueError("exponential schedule requires positive beta and beta_final")
    return float(beta * (beta_final / beta) ** fraction)


def select_indices(
    predictions: Sequence[float],
    uncertainties: Sequence[float] | None,
    n: int,
    strategy: str = "lcb",
    beta: float = 1.0,
    rng: np.random.Generator | None = None,
    ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Select up to ``n`` unique local indices; lower values are always better.

    LCB minimizes ``mean - beta * standard_deviation``. Greedy minimizes the
    mean; Thompson uses one independent normal draw per compound with standard
    deviation ``beta * uncertainty``; random ignores model scores. Stochastic
    strategies require an explicit generator. This Gaussian Thompson rule is
    an acquisition heuristic, not a claim of a calibrated posterior.

    Ties use lexical compound IDs when supplied, otherwise stable input order.
    Stochastic draws follow lexical ID order too, so row permutations do not
    change selected identities under the same seed. Uncertainty must be finite,
    nonnegative and expressed in the same units as predictions.
    """
    if strategy not in {"lcb", "greedy", "random", "thompson"}:
        raise ValueError("Unknown acquisition strategy")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    mean = np.asarray(predictions, dtype=np.float64)
    std = np.zeros_like(mean) if uncertainties is None else np.asarray(uncertainties, dtype=np.float64)
    if mean.ndim != 1 or std.shape != mean.shape:
        raise ValueError("predictions and uncertainties must be matching 1D arrays")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError("predictions must be finite and uncertainties finite and nonnegative")
    if ids is None:
        tie_key = np.arange(len(mean))
    else:
        tie_key = np.asarray(ids, dtype=str)
        if tie_key.shape != mean.shape or len(np.unique(tie_key)) != len(mean):
            raise ValueError("ids must contain one unique identity per prediction")
    if n == 0 or len(mean) == 0:
        return np.empty(0, dtype=np.int64)
    if strategy == "greedy":
        values = mean
    elif strategy == "lcb":
        values = mean - beta * std
    else:
        if not isinstance(rng, np.random.Generator):
            raise ValueError("random and thompson strategies require an explicit numpy Generator")
        order = tie_key if ids is None else np.argsort(tie_key, kind="stable")
        values = np.empty_like(mean)
        if strategy == "random":
            values[order] = rng.random(len(mean))
        else:
            values[order] = rng.normal(mean[order], beta * std[order])
    if not np.isfinite(values).all():
        raise ValueError("acquisition values overflowed; check beta and input score scales")
    count = min(n, len(mean))
    if count * 4 >= len(mean):
        return np.lexsort((tie_key, values))[:count].astype(np.int64)
    # AL normally selects a tiny fraction of a large pool. Partition by value
    # in linear time, then resolve the boundary ties explicitly: argpartition
    # alone could discard the lexically preferred compound among equal scores.
    boundary_value = np.partition(values, count - 1)[count - 1]
    better = np.flatnonzero(values < boundary_value)
    boundary = np.flatnonzero(values == boundary_value)
    needed = count - len(better)
    if ids is not None:
        boundary = boundary[np.argsort(tie_key[boundary], kind="stable")[:needed]]
    else:
        boundary = boundary[:needed]
    selected = np.concatenate((better, boundary))
    return selected[np.lexsort((tie_key[selected], values[selected]))].astype(np.int64)
