import numpy as np
import pandas as pd

from src.inventory.policies import order_up_to_normal, replenish_last_week
from src.inventory.simulate import ReviewState


def make_state(past_sales, on_hand=0.0, on_order=0.0, q50=1.0, q90=2.0, q95=3.0):
    return ReviewState(
        day=pd.Timestamp("2016-01-04"),
        on_hand=on_hand,
        on_order=on_order,
        forecast=pd.DataFrame({"q50": [q50] * 14, "q90": [q90] * 14, "q95": [q95] * 14}),
        past_sales=np.array(past_sales, dtype=float),
        price=2.0,
        demand_class="smooth",
    )


def test_baseline_reorders_last_seven_days():
    state = make_state([100, 1, 2, 3, 4, 5, 6, 7])  # the 100 is 8 days ago: ignored
    assert replenish_last_week(state) == 28


def test_baseline_factor_scales_and_rounds_up():
    state = make_state([1] * 7)
    assert replenish_last_week(state, factor=1.1) == 8  # 7.7 -> 8


def test_order_up_to_normal_matches_hand_calculation():
    # mean 2 per day over 12 days; past sales 0, 4, 0, 4... have std 2; lead-time variance 2
    # S = 2 * 12 + 1.645 * sqrt(12 * 2**2 + 2**2 * 2) = 24 + 1.645 * sqrt(56) = 36.3
    state = make_state([0, 4] * 28, on_hand=10, on_order=5, q50=2.0)
    assert order_up_to_normal(state, service=0.95) == 22  # ceil(36.3 - 15)


def test_order_up_to_never_negative():
    state = make_state([1] * 56, on_hand=500, q50=1.0)
    assert order_up_to_normal(state) == 0
