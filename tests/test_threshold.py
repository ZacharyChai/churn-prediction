import numpy as np

from app.threshold import cost_minimizing_threshold, expected_cost


def test_expected_cost_weights_false_negatives_by_the_ratio():
    y = [1, 1, 0, 0]
    proba = [0.9, 0.2, 0.6, 0.1]
    # At 0.5: one missed churner (0.2) and one false alarm (0.6).
    assert expected_cost(y, proba, 0.5, cost_ratio=3) == 1 + 3


def test_separable_scores_put_the_threshold_between_the_classes():
    y = [0, 0, 1, 1]
    proba = [0.1, 0.2, 0.8, 0.9]
    threshold = cost_minimizing_threshold(y, proba, cost_ratio=2)
    assert 0.2 < threshold <= 0.8
    assert expected_cost(y, proba, threshold, cost_ratio=2) == 0


def test_costlier_misses_never_raise_the_threshold():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(0.35 * y + rng.random(500) * 0.65, 0, 1)
    thresholds = [cost_minimizing_threshold(y, proba, r) for r in (1, 2, 5, 10)]
    assert thresholds == sorted(thresholds, reverse=True)
