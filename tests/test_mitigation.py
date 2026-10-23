import pandas as pd
import pytest

from src.simulation.mitigation import REPLAY_DAYS, a_items, break_even


def test_a_items_are_the_top_sales_value():
    rows = []
    for item, demand, price in [("I0", 1, 1.0), ("I1", 10, 1.0), ("I2", 1, 50.0), ("I3", 2, 1.0)]:
        rows.append({"unique_id": item, "in_replay": False, "demand": demand, "sell_price": price})
    inputs = pd.DataFrame(rows)
    assert a_items(inputs, share=0.5) == ["I2", "I1"]  # values 50 and 10


def test_break_even_divides_yearly_cost_by_damage_avoided():
    runs = pd.DataFrame(
        [
            # normal: the buffer costs 10 more per replay; outage: it saves 40 net
            {
                "seed": 0,
                "scenario": "S0 normal",
                "mitigation": "No buffer",
                "total_cost_with_dc": 100,
            },
            {
                "seed": 0,
                "scenario": "S0 normal",
                "mitigation": "A-item buffer",
                "total_cost_with_dc": 110,
            },
            {"seed": 0, "scenario": "S1", "mitigation": "No buffer", "total_cost_with_dc": 200},
            {"seed": 0, "scenario": "S1", "mitigation": "A-item buffer", "total_cost_with_dc": 170},
        ]
    )
    row = break_even(runs).iloc[0]
    yearly = 10 * 365 / REPLAY_DAYS
    assert row["damage_avoided_per_disruption"] == pytest.approx(40)  # 30 saved + 10 normal cost
    assert row["buffer_cost_per_year"] == pytest.approx(yearly)
    assert row["break_even_disruptions_per_year"] == pytest.approx(yearly / 40)
