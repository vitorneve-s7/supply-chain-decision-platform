import pandas as pd
import pytest

from src.inventory.kpis import scorecard


def test_scorecard_on_a_hand_made_replay():
    days = pd.date_range("2016-01-18", periods=14)  # two full weeks after the warm-up
    rows = []
    for d in days:
        rows.append(["A", "smooth", 2.0, d, 10.0, 10.0, 0.0, 5.0, 0])  # always in stock
        lost = 2.0 if d == days[0] else 0.0  # one stockout day, in week 1
        rows.append(["B", "lumpy", 1.0, d, 4.0, 4.0 - lost, lost, 0.0, 0])
    records = pd.DataFrame(
        rows,
        columns=[
            "unique_id",
            "demand_class",
            "sell_price",
            "ds",
            "demand",
            "sales",
            "lost",
            "end_stock",
            "order",
        ],
    )
    card = scorecard(records)

    assert card.loc["smooth", "fill_rate"] == pytest.approx(1.0)
    assert card.loc["lumpy", "fill_rate"] == pytest.approx(54 / 56)
    assert card.loc["lumpy", "cycle_service"] == pytest.approx(0.5)  # 1 of 2 weeks had a stockout
    assert card.loc["all", "cycle_service"] == pytest.approx(0.75)  # 3 of 4 item-weeks
    assert card.loc["smooth", "days_of_stock"] == pytest.approx(0.5)  # 5 units / 10 per day
    assert card.loc["all", "stock_value"] == pytest.approx(10.0)  # 5 units x price 2, per day
