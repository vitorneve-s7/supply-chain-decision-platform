import numpy as np
import pandas as pd
import pytest

from src.inventory.simulate import forecast_ahead, simulate_item

START = pd.Timestamp("2016-01-04")  # a Monday


def make_item(demand, history_demand=10.0, history_days=14):
    """One item: `history_days` days of history, then the replay days with `demand`."""
    n = len(demand)
    days = pd.date_range(START - pd.Timedelta(days=history_days), periods=history_days + n)
    return pd.DataFrame(
        {
            "ds": days,
            "in_replay": days >= START,
            "demand": [history_demand] * history_days + list(demand),
            "sales": [history_demand] * history_days + list(demand),
            "sell_price": 2.0,
            "demand_class": "smooth",
            "cutoff": pd.Timestamp("2016-01-03"),
            "q50": 1.0,
            "q90": 2.0,
            "q95": 3.0,
        }
    )


def never(state):
    return 0


def fixed(units):
    return lambda state: units


def leads(days, item):
    """The same lead time on every Monday of the replay."""
    mondays = item.loc[item["in_replay"] & (item["ds"].dt.day_of_week == 0), "ds"]
    return {d: days for d in mondays}


def test_zero_demand_sells_nothing():
    item = make_item([0] * 14)
    out = simulate_item(item, never, leads(3, item), start_stock=5)
    assert out["sales"].sum() == 0
    assert out["lost"].sum() == 0
    assert out["end_stock"].iloc[-1] == 5


def test_order_arrives_after_lead_time():
    item = make_item([0] * 10)
    out = simulate_item(item, fixed(8), leads(3, item), start_stock=0)
    assert out["received"].iloc[:3].sum() == 0  # days 0, 1, 2: nothing yet
    assert out["received"].iloc[3] == 8  # ordered on Monday (day 0), arrives on day 3


def test_stock_never_negative_and_shortfall_is_lost():
    item = make_item([4, 4, 4])
    out = simulate_item(item, never, leads(3, item), start_stock=6)
    assert (out["end_stock"] >= 0).all()
    assert list(out["sales"]) == [4, 2, 0]
    assert list(out["lost"]) == [0, 2, 4]


def test_stock_balance_holds_every_day():
    rng = np.random.default_rng(0)
    item = make_item(rng.integers(0, 8, size=28))
    out = simulate_item(item, fixed(20), leads(5, item), start_stock=10)
    balance = out["start_stock"] + out["received"] - out["sales"] - out["end_stock"]
    assert balance.abs().max() == pytest.approx(0)
    assert (out["sales"] + out["lost"] == out["demand"]).all()


def test_never_ordering_runs_out_then_loses_everything():
    item = make_item([3] * 10)
    out = simulate_item(item, never, leads(3, item), start_stock=6)
    assert out["end_stock"].iloc[1] == 0
    assert out["lost"].iloc[2:].eq(3).all()


def test_default_start_stock_is_14_days_of_history():
    item = make_item([0] * 7, history_demand=2.0)
    out = simulate_item(item, never, leads(3, item))
    assert out["start_stock"].iloc[0] == 28


def test_forecast_ahead_repeats_last_week_past_window_end():
    item = make_item([1] * 10)
    item.loc[item["in_replay"], "q50"] = np.arange(10.0)  # forecast 0..9 for the 10 replay days
    fc = forecast_ahead(item, START + pd.Timedelta(days=5), n=8)
    assert list(fc["q50"]) == [5, 6, 7, 8, 9, 3, 4, 5]  # then the last 7 days (3..9) again
