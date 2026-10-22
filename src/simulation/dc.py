"""DC ordering policies. Each takes the twin on a Monday and returns units per item."""

import numpy as np


def dc_mirror(twin, t: int) -> np.ndarray:
    """Placeholder until Part D: order from the supplier what the store ordered today."""
    return twin.store_order.copy()
