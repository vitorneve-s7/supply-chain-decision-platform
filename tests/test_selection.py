import pandas as pd
import pytest

from src.forecasting.selection import build_hybrid


def test_hybrid_takes_point_by_class_and_margin_from_interval_model():
    rows = []
    for uid in ["A_CA_1", "B_CA_1"]:
        rows.append([uid, "ETS", 10.0, 13.0, 15.0])  # margin +3 / +5
        rows.append([uid, "LGB", 12.0, 20.0, 30.0])
    results = pd.DataFrame(rows, columns=["unique_id", "model", "mean", "q90", "q95"])
    results["ds"] = pd.Timestamp("2016-01-04")
    results["cutoff"] = pd.Timestamp("2016-01-03")
    results["y"] = 11.0
    results = results[["unique_id", "ds", "cutoff", "y", "model", "mean", "q90", "q95"]]
    classes = pd.DataFrame(
        {"item_id": ["A", "B"], "store_id": ["CA_1", "CA_1"], "demand_class": ["smooth", "lumpy"]}
    )

    out = build_hybrid(results, classes, {"smooth": "LGB", "lumpy": "ETS"}, "ETS")
    out = out.set_index("unique_id")

    assert list(out.columns) == ["ds", "cutoff", "y", "model", "mean", "q90", "q95"]
    assert out.loc["A_CA_1", "mean"] == pytest.approx(12.0)  # smooth: LightGBM point
    assert out.loc["A_CA_1", "q95"] == pytest.approx(17.0)  # 12 + ETS margin 5
    assert out.loc["B_CA_1", "mean"] == pytest.approx(10.0)  # lumpy: ETS point
    assert out.loc["B_CA_1", "q90"] == pytest.approx(13.0)
