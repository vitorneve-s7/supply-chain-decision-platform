import numpy as np
import pandas as pd
import pytest

from src.forecasting.calibration import apply_factors, find_factor


def test_find_factor_widens_a_too_narrow_interval():
    y = np.arange(1, 101)  # actuals 1..100
    mean = np.zeros(100)
    q = np.full(100, 45.0)  # covers only 45%
    assert find_factor(y, mean, q, target=0.90) == pytest.approx(2.0)  # 2 x 45 = 90


def test_find_factor_can_narrow():
    y = np.arange(1, 101)
    mean = np.zeros(100)
    q = np.full(100, 180.0)
    assert find_factor(y, mean, q, target=0.90) == pytest.approx(0.5)


def test_apply_factors_keeps_order_and_sign():
    results = pd.DataFrame(
        {
            "unique_id": ["A_CA_1"] * 2,
            "ds": pd.to_datetime(["2016-01-04", "2016-01-05"]),
            "cutoff": pd.to_datetime(["2016-01-03"] * 2),
            "y": [1.0, 2.0],
            "model": "M",
            "mean": [2.0, 2.0],
            "q90": [3.0, 4.0],
            "q95": [3.5, 5.0],
        }
    )
    classes = pd.DataFrame({"item_id": ["A"], "store_id": ["CA_1"], "demand_class": ["smooth"]})
    factors = pd.DataFrame({"model": ["M"], "demand_class": ["smooth"], "q90": [0.5], "q95": [0.1]})
    out = apply_factors(results, classes, factors)
    assert list(out.columns) == list(results.columns)
    assert (out["q90"] >= out["mean"]).all()
    assert (out["q95"] >= out["q90"]).all()  # q95 shrunk below q90 is pushed back up
    assert out["q90"].iloc[1] == pytest.approx(3.0)  # 2 + 0.5 * (4 - 2)
