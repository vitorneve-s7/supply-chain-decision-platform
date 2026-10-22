import pandas as pd
import pytest

from src.forecasting.ml import EXOGENOUS, add_ml_features


@pytest.fixture
def nx():
    n = 60
    return pd.DataFrame(
        {
            "unique_id": "A_CA_1",
            "ds": pd.date_range("2016-01-01", periods=n, freq="D"),
            "y": 1.0,
            "sell_price": [2.0] * 30 + [1.0] * 30,  # price cut on day 31
            "snap_CA": 0,
            "event_name_1": pd.Categorical([None] * (n - 1) + ["SuperBowl"]),
            "event_type_1": pd.Categorical([None] * (n - 1) + ["Sporting"]),
            "event_name_2": pd.Categorical([None] * n),
            "event_type_2": pd.Categorical([None] * n),
            "available": True,
            "closed": False,
        }
    )


def test_future_unknown_columns_are_dropped(nx):
    out = add_ml_features(nx)
    assert "available" not in out.columns and "closed" not in out.columns
    assert list(out.columns) == ["unique_id", "ds", "y", *EXOGENOUS]


def test_price_rel_flags_a_price_cut(nx):
    out = add_ml_features(nx)
    assert out["price_rel"].iloc[0] == pytest.approx(1.0)
    assert out["price_rel"].iloc[30] < 1.0  # cheaper than the 28-day average


def test_events_become_codes(nx):
    out = add_ml_features(nx)
    assert out["event_name_1"].iloc[0] == -1
    assert out["event_name_1"].iloc[-1] >= 0
