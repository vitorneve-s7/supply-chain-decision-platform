"""Ordering policies. Each takes the ReviewState of one item on a review day and returns units."""

import numpy as np
from scipy.stats import norm

from src.inventory.inputs import LEAD_MAX, LEAD_MIN
from src.inventory.simulate import ReviewState

REVIEW_DAYS = 7
LEAD_MEAN = (LEAD_MIN + LEAD_MAX) / 2  # 5 days
LEAD_VAR = ((LEAD_MAX - LEAD_MIN + 1) ** 2 - 1) / 12  # variance of a uniform whole number: 2
PROTECTION = int(REVIEW_DAYS + LEAD_MEAN)  # 12 days: what one order must cover
SIGMA_DAYS = 56  # past days used for the textbook standard deviation


def replenish_last_week(state: ReviewState, factor: float = 1.0) -> int:
    """Baseline: reorder what sold in the last 7 days (times `factor`, for the Part H sweep)."""
    return int(np.ceil(factor * state.past_sales[-7:].sum()))


def order_up_to(state: ReviewState, level: float) -> int:
    """Order the gap between the order-up-to level and the inventory position."""
    position = state.on_hand + state.on_order
    return max(0, int(np.ceil(level - position)))


def order_up_to_normal(state: ReviewState, service: float = 0.95) -> int:
    """(R, S) with the textbook safety stock: forecast mean, std of past daily sales."""
    mu = state.forecast["q50"].head(PROTECTION).mean()
    sigma = np.std(state.past_sales[-SIGMA_DAYS:])
    z = norm.ppf(service)
    level = mu * PROTECTION + z * np.sqrt(PROTECTION * sigma**2 + mu**2 * LEAD_VAR)
    return order_up_to(state, level)


def order_up_to_quantile_sum(state: ReviewState) -> int:
    """(R, S) with S = the sum of the daily q95 forecasts over the protection period."""
    level = state.forecast["q95"].head(PROTECTION).sum()
    return order_up_to(state, level)


def order_up_to_quantile(
    state: ReviewState, service: float = 0.95, z_by_class: dict[str, float] | None = None
) -> int:
    """(R, S) with the spread read from the calibrated quantiles, added up like variances.

    z_by_class, if given, replaces the z of `service` with one tuned per demand class.
    """
    fc = state.forecast.head(PROTECTION)
    sigma_t = (fc["q95"] - fc["q50"]) / norm.ppf(0.95)  # daily spread from the forecast
    mu = fc["q50"].mean()
    z = z_by_class[state.demand_class] if z_by_class else norm.ppf(service)
    level = fc["q50"].sum() + z * np.sqrt((sigma_t**2).sum() + mu**2 * LEAD_VAR)
    return order_up_to(state, level)
