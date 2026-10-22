import math

import pandas as pd
import pytest

from src.inventory.sweep import stock_at_service


def test_stock_at_service_interpolates_along_the_curve():
    curve = pd.DataFrame({"cycle_service": [0.95, 0.80, 0.90], "stock_value": [300, 100, 200]})
    assert stock_at_service(curve, 0.85) == pytest.approx(150)  # halfway between 100 and 200


def test_stock_at_service_out_of_reach_is_nan():
    curve = pd.DataFrame({"cycle_service": [0.80, 0.90], "stock_value": [100, 200]})
    assert math.isnan(stock_at_service(curve, 0.95))
