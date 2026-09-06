"""Regression and closed-pool discovery metrics with independent expectations."""

import json

import numpy as np
import pytest

from active_learning_docking.evaluation import discovery_metrics, regression_metrics


def test_regression_metrics_match_hand_calculation():
    metrics = regression_metrics([1, 2, 3], [1, 2, 4])
    assert metrics["n"] == 3
    assert metrics["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["r2"] == pytest.approx(0.5)
    assert metrics["spearman"] == pytest.approx(1)


def test_spearman_uses_average_ranks_for_ties():
    assert regression_metrics([1, 1, 3], [2, 2, 1])["spearman"] == pytest.approx(-1)


@pytest.mark.parametrize("truth,prediction", [([], []), ([1], [2]), ([0.1] * 3, [1, 2, 3])])
def test_undefined_metrics_are_none_and_json_safe(truth, prediction):
    metrics = regression_metrics(truth, prediction)
    assert metrics["r2"] is None
    assert metrics["spearman"] is None
    json.dumps(metrics, allow_nan=False)
    if not truth:
        assert metrics["rmse"] is None


def test_constant_prediction_spearman_undefined_r2_defined():
    metrics = regression_metrics([1, 2, 3], [2, 2, 2])
    assert metrics["spearman"] is None
    assert metrics["r2"] == 0


@pytest.mark.parametrize("truth,prediction", [([1], []), ([float("nan")], [1]), ([[1]], [[1]])])
def test_regression_rejects_invalid_data(truth, prediction):
    with pytest.raises(ValueError):
        regression_metrics(truth, prediction)


def test_discovery_metrics_match_hand_calculation():
    metrics = discovery_metrics([-10, -8, -6, -4, -2], [0, 2], top_k=2)
    assert metrics == {
        "pool_size": 5, "n_acquired": 2, "top_k": 2, "top_k_hits": 1,
        "top_k_recall": 0.5, "enrichment_factor": 1.25,
        "best_score": -10, "optimal_score": -10, "simple_regret": 0,
    }


def test_nonoptimal_discovery_and_ties_use_ids():
    metrics = discovery_metrics([-4, -8, -8], [0, 1], top_k=1, ids=["a", "c", "b"])
    assert metrics["top_k_hits"] == 0
    assert metrics["simple_regret"] == 0
    assert discovery_metrics([-10, -8, -6], [1, 2], 1)["simple_regret"] == 2


def test_empty_acquisition_and_topk_cap():
    metrics = discovery_metrics([-8, -6], [], top_k=20)
    assert metrics["top_k"] == 2
    assert metrics["top_k_recall"] == 0
    assert metrics["enrichment_factor"] is None
    assert metrics["best_score"] is None
    assert metrics["simple_regret"] is None
    json.dumps(metrics, allow_nan=False)


def test_extreme_finite_scores_have_json_safe_regret():
    metrics = discovery_metrics([-1e308, 1e308], [1], top_k=1)
    assert metrics["simple_regret"] is None
    assert metrics["best_score"] == 1e308
    assert metrics["optimal_score"] == -1e308
    assert metrics["top_k_recall"] == 0
    assert metrics["enrichment_factor"] == 0
    json.dumps(metrics, allow_nan=False)


def test_holdout_truth_is_excluded_from_denominator():
    library_truth = np.array([-100, -9, -8, -7, -6])
    candidate_indices = [1, 2, 3, 4]
    metrics = discovery_metrics(library_truth[candidate_indices], [0], top_k=1)
    assert metrics["optimal_score"] == -9
    assert metrics["enrichment_factor"] == 4
    assert metrics["top_k_recall"] == 1


@pytest.mark.parametrize("truth,acquired,k", [
    ([], [], 1), ([-1], [1], 1), ([-1], [-1], 1), ([-1], [0, 0], 1),
    ([-1], [0.5], 1), ([-1], [True], 1), ([-1], [0], 0),
    ([float("inf")], [0], 1),
])
def test_invalid_discovery_data(truth, acquired, k):
    with pytest.raises(ValueError):
        discovery_metrics(truth, acquired, k)
