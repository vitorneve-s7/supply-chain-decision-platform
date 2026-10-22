import pandas as pd
import pytest

from src.forecasting import forecast as fc


@pytest.fixture(autouse=True)
def fake_forecast_file(monkeypatch):
    """Use a small made-up forecast table instead of the real file."""
    dates = pd.date_range("2016-05-23", periods=28, freq="D")
    data = pd.DataFrame(
        {"unique_id": "FOODS_3_001_CA_1", "ds": dates, "q50": 1.0, "q90": 2.0, "q95": 3.0}
    )
    monkeypatch.setattr(fc, "_load", lambda: data)


def test_returns_horizon_rows_in_order():
    out = fc.forecast("FOODS_3_001", "CA_1", horizon=7)
    assert len(out) == 7
    assert list(out.columns) == ["date", "q50", "q90", "q95"]
    assert out["date"].is_monotonic_increasing
    assert (out["q50"] >= 0).all()
    assert (out["q50"] <= out["q90"]).all() and (out["q90"] <= out["q95"]).all()


def test_unknown_item_raises():
    with pytest.raises(KeyError):
        fc.forecast("FOODS_3_999", "CA_1")


@pytest.mark.parametrize("horizon", [0, 29])
def test_horizon_out_of_range_raises(horizon):
    with pytest.raises(ValueError):
        fc.forecast("FOODS_3_001", "CA_1", horizon=horizon)
