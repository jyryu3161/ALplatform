"""Selection direction, uncertainty units and explicit RNG regression tests."""

import numpy as np
import pytest

from active_learning_docking.acquisition import select_indices


def test_greedy_favors_more_negative_docking_scores():
    assert select_indices([-2, -9, -4], [100, 0, 0], 2, "greedy").tolist() == [1, 2]


def test_lcb_exploration_uses_original_score_units():
    mean = [-9, -8, -1]
    std = [0.1, 3, 0]
    assert select_indices(mean, std, 1, "lcb", beta=0).tolist() == [0]
    assert select_indices(mean, std, 1, "lcb", beta=1).tolist() == [1]
    # Appending a distant prediction must not renormalize existing candidates.
    assert select_indices(mean + [1e6], std + [0], 1, "lcb", beta=1).tolist() == [1]


def test_stable_ties_by_id_or_input_order():
    assert select_indices([-5] * 3, None, 2, ids=["c", "a", "b"]).tolist() == [1, 2]
    assert select_indices([-5] * 3, None, 2).tolist() == [0, 1]


@pytest.mark.parametrize("strategy", ["random", "thompson"])
def test_stochastic_selection_is_seeded_unique_and_id_order_invariant(strategy):
    ids = np.array(["z", "b", "a", "q"])
    mean = np.array([-4, -7, -5, -8])
    std = np.array([1, 2, 3, 2])
    first = select_indices(mean, std, 3, strategy, rng=np.random.default_rng(41), ids=ids)
    permutation = np.array([3, 0, 2, 1])
    second = select_indices(mean[permutation], std[permutation], 3, strategy,
                            rng=np.random.default_rng(41), ids=ids[permutation])
    assert ids[first].tolist() == ids[permutation][second].tolist()
    assert len(np.unique(first)) == 3
    with pytest.raises(ValueError, match="explicit numpy Generator"):
        select_indices(mean, std, 1, strategy)


def test_zero_uncertainty_thompson_is_greedy():
    assert select_indices([-2, -8, -3], [0, 0, 0], 2, "thompson",
                          rng=np.random.default_rng(4)).tolist() == [1, 2]


def test_empty_and_exhausted_pool():
    assert select_indices([], [], 5).tolist() == []
    assert select_indices([-2], [0], 0).tolist() == []
    assert select_indices([-2, -3], [0, 0], 100).tolist() == [1, 0]


@pytest.mark.parametrize("mean,std,kwargs", [
    ([float("nan")], [0], {}), ([1], [-1], {}), ([1], [float("inf")], {}),
    ([1], [1, 2], {}), ([[1]], [[0]], {}), ([1], [0], {"beta": -1}),
    ([1], [0], {"beta": float("nan")}), ([1], [0], {"strategy": "bad"}),
    ([1], [0], {"ids": ["a", "b"]}), ([1, 2], [0, 0], {"ids": ["a", "a"]}),
])
def test_invalid_acquisition_inputs(mean, std, kwargs):
    with pytest.raises(ValueError):
        select_indices(mean, std, 1, **kwargs)


@pytest.mark.parametrize("n", [-1, 0.5, True])
def test_invalid_batch_size(n):
    with pytest.raises(ValueError, match="nonnegative integer"):
        select_indices([-5], [0], n)


def test_scheduled_beta_endpoints_paths_and_validation():
    from active_learning_docking.acquisition import scheduled_beta

    assert [scheduled_beta(c, 4, 2.0, 0.0, "linear") for c in range(1, 5)] == pytest.approx([2.0, 4 / 3, 2 / 3, 0.0])
    assert [scheduled_beta(c, 4, 2.0, 0.25, "exponential") for c in range(1, 5)] == pytest.approx([2.0, 1.0, 0.5, 0.25])
    assert scheduled_beta(1, 1, 3.0, 0.0, "linear") == 3.0
    assert scheduled_beta(5, 5, 1.5) == 1.5
    with pytest.raises(ValueError, match="positive beta"):
        scheduled_beta(1, 3, 0.0, 1.0, "exponential")
    with pytest.raises(ValueError, match="beta_final"):
        scheduled_beta(1, 3, 1.0, None, "linear")
    with pytest.raises(ValueError, match="cycles"):
        scheduled_beta(4, 3, 1.0, 0.0, "linear")
    with pytest.raises(ValueError, match="schedule"):
        scheduled_beta(1, 3, 1.0, 0.0, "cosine")

