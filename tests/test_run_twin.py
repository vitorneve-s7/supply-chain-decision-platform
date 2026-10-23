import pandas as pd
import pytest

from src.simulation.run_twin import recovery_days, sample_items, summarise


def test_recovery_counts_to_the_last_bad_day():
    normal = pd.Series([0.95] * 20)
    fill = pd.Series([0.95] * 5 + [0.90, 0.85, 0.94, 0.945] + [0.95] * 11)  # bad on days 5-7
    assert recovery_days(fill, normal, start=4) == 4  # last bad day is day 7: 4 days after day 4


def test_no_damage_means_zero_recovery_days():
    normal = pd.Series([0.95] * 10)
    assert recovery_days(normal.copy(), normal, start=2) == 0


def test_sample_keeps_the_class_mix():
    inputs = pd.DataFrame(
        {"unique_id": [f"I{i}" for i in range(100)], "demand_class": ["a"] * 60 + ["b"] * 40}
    )
    kept = sample_items(inputs, 50).groupby("demand_class")["unique_id"].nunique()
    assert kept["a"] == 30 and kept["b"] == 20


def test_summary_interval_is_196_standard_errors():
    runs = pd.DataFrame(
        {
            "policy": "P",
            "scenario": "S1",
            "extra_lost_cost": [10.0, 20.0, 30.0, 40.0],
            **{
                c: 0.0
                for c in [
                    "fill_rate",
                    "cycle_service",
                    "extra_lost_units",
                    "extra_total_cost",
                    "recovery_days",
                    "total_cost_with_dc",
                    "dc_short_share",
                ]
            },
        }
    )
    row = summarise(runs).iloc[0]
    assert row["extra_lost_cost"] == pytest.approx(25.0)
    assert row["extra_lost_cost_ci"] == pytest.approx(1.96 * runs["extra_lost_cost"].std() / 2)
