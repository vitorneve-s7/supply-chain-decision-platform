"""Tune the safety factor z per demand class so the quantile policy reaches its service target."""

from functools import partial

import pandas as pd

from src.inventory.kpis import WARMUP_END, scorecard
from src.inventory.policies import order_up_to_quantile
from src.inventory.simulate import simulate

CLASSES = ["smooth", "erratic", "intermittent", "lumpy"]
TUNE_END = pd.Timestamp("2016-04-24")  # tune on the validation part; the test window checks it


def tune_z(
    inputs: pd.DataFrame,
    lead_times: pd.DataFrame,
    target: float = 0.95,
    low: float = 1.0,
    high: float = 4.0,
    steps: int = 7,
) -> dict[str, float]:
    """Smallest z per demand class whose cycle service reaches `target` before TUNE_END.

    Bisection on all classes at once: each step runs one replay with a different z per class,
    and halves each class's search range. 7 steps from [1, 4] give a precision of about 0.02.
    """
    lo = dict.fromkeys(CLASSES, low)
    hi = dict.fromkeys(CLASSES, high)
    for _ in range(steps):
        mid = {k: (lo[k] + hi[k]) / 2 for k in CLASSES}
        policy = partial(order_up_to_quantile, z_by_class=mid)
        card = scorecard(simulate(inputs, policy, lead_times), WARMUP_END, TUNE_END)
        for k in CLASSES:
            if card.loc[k, "cycle_service"] >= target:
                hi[k] = mid[k]  # enough service: try a smaller z
            else:
                lo[k] = mid[k]
    return {k: round(hi[k], 2) for k in CLASSES}
