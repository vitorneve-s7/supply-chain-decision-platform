import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.inventory.policies import (
    order_up_to_normal,
    order_up_to_quantile,
    order_up_to_quantile_sum,
    reorder_point,
    replenish_last_week,
)
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


def test_quantile_sum_adds_daily_q95():
    state = make_state([1] * 56, on_hand=10, on_order=5, q95=3.0)
    assert order_up_to_quantile_sum(state) == 21  # 12 days x 3 = 36, minus 15


def test_quantile_spread_matches_hand_calculation():
    # q50 = 2 and q95 = 2 + 1.645, so each day's sigma is 1
    # S = 12 * 2 + 1.645 * sqrt(12 * 1**2 + 2**2 * 2) = 24 + 1.645 * sqrt(20) = 31.4
    state = make_state([1] * 56, on_hand=10, on_order=5, q50=2.0, q95=2.0 + norm.ppf(0.95))
    assert order_up_to_quantile(state, service=0.95) == 17  # ceil(31.4 - 15)


@pytest.mark.parametrize("z, expected", [(0.0, 9), (2.0, 18)])
def test_quantile_uses_the_z_of_the_item_class(z, expected):
    # z = 0: S = 24, order 9; z = 2: S = 24 + 2 * sqrt(20) = 32.9, order 18
    state = make_state([1] * 56, on_hand=10, on_order=5, q50=2.0, q95=2.0 + norm.ppf(0.95))
    assert order_up_to_quantile(state, z_by_class={"smooth": z, "lumpy": 9.9}) == expected


def test_reorder_point_skips_when_above_s():
    # s = 31.4 (as above); position 40 is above it, so no order this week
    state = make_state([1] * 56, on_hand=40, q50=2.0, q95=2.0 + norm.ppf(0.95))
    assert reorder_point(state) == 0


def test_reorder_point_orders_a_batch_up_to_s_plus_q():
    # position 15 <= s = 31.4; D = 2 * 365 = 730, h = 0.25 * price 2 = 0.5, K = 1
    # Q = sqrt(2 * 730 * 1 / 0.5) = 54.0, so S = 85.4 and the order is ceil(85.4 - 15) = 71
    state = make_state([1] * 56, on_hand=10, on_order=5, q50=2.0, q95=2.0 + norm.ppf(0.95))
    assert reorder_point(state) == 71
