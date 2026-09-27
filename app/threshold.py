# Decision-threshold selection, used by train_model.py to pick the threshold
# that ships inside model.joblib. The classifier outputs a probability; the
# threshold turns it into a yes/no call. Where it sits is a business trade-off
# between missed churners (false negatives) and retention offers spent on
# customers who were staying anyway (false positives).

import numpy as np

DEFAULT_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def expected_cost(y_true, proba, threshold, cost_ratio):
    """Total cost of calling churn at `threshold`, counting each false positive
    as 1 and each false negative as `cost_ratio`."""
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(proba) >= threshold
    false_pos = np.sum(flagged & (y_true == 0))
    false_neg = np.sum(~flagged & (y_true == 1))
    return float(false_pos + cost_ratio * false_neg)


def cost_minimizing_threshold(y_true, proba, cost_ratio, grid=DEFAULT_GRID):
    """Threshold on `grid` with the lowest expected cost. Ties go to the highest
    threshold, which flags the fewest customers for the same cost."""
    costs = np.array([expected_cost(y_true, proba, t, cost_ratio) for t in grid])
    best = np.flatnonzero(costs == costs.min())
    return float(grid[best[-1]])
