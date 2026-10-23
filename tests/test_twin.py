import numpy as np
import pandas as pd
import pytest

from src.simulation.config import NetworkConfig
from src.simulation.dc import dc_mirror
from src.simulation.twin import Twin

START = pd.Timestamp("2016-01-04")  # a Monday


def make_inputs(n_items=2, days=28, demand=5.0, history_days=14):
    """Made-up items with flat demand and a flat forecast, in the build_inputs format."""
    rows = []
    for i in range(n_items):
        for d in pd.date_range(
            START - pd.Timedelta(days=history_days), periods=history_days + days
        ):
            rows.append(
                {
                    "unique_id": f"I{i}_CA_1",
                    "ds": d,
                    "in_replay": d >= START,
                    "demand": demand,
                    "sales": demand,
                    "available": True,
                    "sell_price": 1.0,
                    "demand_class": "smooth",
                    "cutoff": pd.Timestamp("2016-01-03"),
                    "q50": demand,
                    "q90": demand,
                    "q95": demand,
                }
            )
    return pd.DataFrame(rows)


def make_leads(inputs, days):
    mondays = inputs.loc[inputs["in_replay"] & (inputs["ds"].dt.day_of_week == 0)]
    return mondays[["unique_id", "ds"]].assign(lead_time=days)


def fixed(units):
    return lambda state: units


def run(policy, config, n_items=2, days=28, demand=5.0, lead=3, dc_policy=dc_mirror):
    inputs = make_inputs(n_items, days, demand)
    return Twin(inputs, policy, dc_policy, make_leads(inputs, lead), config).run()


def test_stock_balances_at_store_and_dc():
    twin = run(fixed(40), NetworkConfig(supplier_lead=(5, 9)))
    s, d = twin.store_records(), twin.dc_records()
    assert (s["start_stock"] + s["received"] - s["sales"] - s["end_stock"]).abs().max() < 1e-9
    assert (d["dc_start"] + d["dc_received"] - d["dc_shipped"] - d["dc_end"]).abs().max() < 1e-9
    assert (s["end_stock"] >= 0).all() and (d["dc_end"] >= -1e-9).all()


def test_dc_ships_no_more_than_its_stock():
    config = NetworkConfig(dc_start_days=1, supplier_capacity=None)  # DC starts with 5 per item
    d = run(fixed(40), config).dc_records()
    first_monday = d[d["ds"] == START]
    assert first_monday["dc_shipped"].tolist() == pytest.approx([5.0, 5.0])
    assert first_monday["dc_owed"].tolist() == pytest.approx([35.0, 35.0])  # the rest waits


def test_dc_capacity_caps_shipments_over_seven_days():
    # average weekly demand = 2 items x 5 x 7 = 70; capacity 0.5 -> 35 units in any 7 days
    config = NetworkConfig(dc_capacity=0.5, dc_start_days=100, supplier_capacity=None)
    d = run(fixed(40), config).dc_records()
    daily = d.groupby("ds")["dc_shipped"].sum()
    assert daily.rolling(7, min_periods=1).sum().max() == pytest.approx(35.0)


def test_store_delivery_arrives_after_its_lead_time():
    s = run(fixed(10), NetworkConfig(unlimited_dc=True), lead=3).store_records()
    item = s[s["unique_id"] == "I0_CA_1"].reset_index(drop=True)
    assert item["received"].iloc[:3].sum() == 0
    assert item["received"].iloc[3] == 10  # ordered Monday (day 0), arrives on day 3


def test_supplier_outage_stops_deliveries_then_resumes():
    config = NetworkConfig(
        supplier_lead=(5, 5), start_day=0, duration=14, supplier_down=True, dc_start_days=100
    )
    d = run(fixed(40), config, days=35).dc_records()
    daily = d.groupby("ds")["dc_received"].sum()
    assert daily.iloc[:19].sum() == 0  # nothing ships on days 0-13, so nothing lands before day 19
    assert daily.iloc[19:].sum() > 0  # shipped on day 14, lands on day 19


def test_unlimited_dc_matches_phase2_replay():
    from src.inventory.simulate import simulate_item

    inputs = make_inputs(n_items=1, days=21, demand=4.0)
    leads = make_leads(inputs, 5)
    twin = Twin(inputs, fixed(30), dc_mirror, leads, NetworkConfig(unlimited_dc=True)).run()
    item = inputs.drop(columns="unique_id").reset_index(drop=True)
    phase2 = simulate_item(item, fixed(30), dict(zip(leads["ds"], leads["lead_time"])))
    twin_rec = twin.store_records()
    np.testing.assert_allclose(twin_rec["end_stock"], phase2["end_stock"])
    np.testing.assert_allclose(twin_rec["lost"], phase2["lost"])


def test_dc_order_up_to_covers_14_days_minus_its_position():
    from scipy.stats import norm

    from src.simulation.dc import dc_order_up_to

    # flat forecast of 5 a day with q95 = q50 (no daily spread): only lead-time risk remains
    # level = 14 x 5 + 1.645 x sqrt(5**2 x 2) = 70 + 11.6 = 81.6 per item
    inputs = make_inputs(n_items=2, demand=5.0)
    twin = Twin(inputs, fixed(0), dc_order_up_to, make_leads(inputs, 3), NetworkConfig())
    level = 70 + norm.ppf(0.95) * np.sqrt(25 * 2)
    np.testing.assert_allclose(dc_order_up_to(twin, 0), level - twin.dc_position())
    assert twin.dc_position() == pytest.approx([70.0, 70.0])  # starts with 14 days of 5
