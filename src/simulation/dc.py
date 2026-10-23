"""DC ordering policies. Each takes the twin on a Monday and returns units per item."""

import numpy as np

from src.inventory.policies import quantile_level
from src.inventory.simulate import ReviewState

DC_PROTECTION = 14  # review 7 days + mean supplier lead time 7 days


def dc_mirror(twin, t: int) -> np.ndarray:
    """Simplest rule, for tests: order from the supplier what the store ordered today."""
    return twin.store_order.copy()


def dc_order_up_to(twin, t: int, service: float = 0.95) -> np.ndarray:
    """Weekly (R, S) on the store's forecast: cover 14 days at `service`, minus the DC position."""
    levels = np.zeros(len(twin.items))
    for i in range(len(twin.items)):
        state = ReviewState(
            day=twin.dates[t],
            on_hand=0.0,
            on_order=0.0,
            forecast=twin.forecast(i, t),
            past_sales=np.array([]),
            price=twin.price[i, t],
            demand_class=twin.classes[i],
        )
        levels[i] = quantile_level(state, service=service, protection=DC_PROTECTION)
    return np.maximum(0.0, levels - twin.dc_position())
