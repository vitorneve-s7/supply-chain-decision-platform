"""Ordering policies. Each takes the ReviewState of one item on a review day and returns units."""

import numpy as np

from src.inventory.simulate import ReviewState


def replenish_last_week(state: ReviewState, factor: float = 1.0) -> int:
    """Baseline: reorder what sold in the last 7 days (times `factor`, for the Part H sweep)."""
    return int(np.ceil(factor * state.past_sales[-7:].sum()))
