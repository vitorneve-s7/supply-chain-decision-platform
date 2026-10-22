"""Combine models into one forecast: a point model per demand class, intervals from another."""

import pandas as pd

from src.forecasting.calibration import TARGETS

# Chosen on the validation windows (Part G1)
POINT_MODEL = {
    "smooth": "LightGBM",
    "erratic": "LightGBM",
    "intermittent": "AutoETS",
    "lumpy": "AutoETS",
}
INTERVAL_MODEL = "AutoETS"


def build_hybrid(
    results: pd.DataFrame,
    classes: pd.DataFrame,
    point_model: dict[str, str],
    interval_model: str,
    name: str = "Hybrid",
) -> pd.DataFrame:
    """Point forecast from point_model[class]; q90/q95 = that point + the interval model's margin.

    results must hold calibrated forecasts of every model named, in the long format.
    """
    cls = classes.assign(
        unique_id=classes["item_id"].astype(str) + "_" + classes["store_id"].astype(str)
    )[["unique_id", "demand_class"]]
    key = ["unique_id", "ds", "cutoff"]
    df = results.merge(cls, on="unique_id", how="left")

    margin = df[df["model"] == interval_model][key + ["mean", *TARGETS]]
    for col in TARGETS:
        margin[col] = margin[col] - margin["mean"]
    margin = margin.drop(columns="mean")

    df["wanted"] = df["demand_class"].map(point_model)
    point = df[df["model"] == df["wanted"]][key + ["y", "mean"]]

    out = point.merge(margin, on=key, how="inner")
    for col in TARGETS:
        out[col] = out["mean"] + out[col]
    out["model"] = name
    return out[results.columns]
