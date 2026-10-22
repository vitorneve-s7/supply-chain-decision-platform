"""Build the replay inputs: demand, forecasts, prices and lead times for every item and day."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.selection import INTERVAL_MODEL, POINT_MODEL, build_hybrid

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

REPLAY_START = pd.Timestamp("2016-01-04")  # a Monday
REPLAY_END = pd.Timestamp("2016-05-22")
HISTORY_DAYS = 56  # past days kept before the replay, for policies that look back
LEAD_MIN, LEAD_MAX = 3, 7  # lead time in days, uniform
SEED = 42


def hybrid_forecasts(classes: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample Hybrid forecasts for the whole replay: validation windows + test window."""
    calibrated = pd.read_parquet(RESULTS / "calibrated_backtest.parquet")
    validation = build_hybrid(calibrated, classes, POINT_MODEL, INTERVAL_MODEL)
    test = pd.read_parquet(RESULTS / "test_forecasts.parquet")
    test = test[test["model"] == "Hybrid"]
    fc = pd.concat([validation, test], ignore_index=True)
    return fc.rename(columns={"mean": "q50"})[["unique_id", "ds", "cutoff", "q50", "q90", "q95"]]


def impute_demand(df: pd.DataFrame) -> pd.Series:
    """Observed sales; on unavailable days, the average of the last 28 available days, rounded."""
    sales_if_available = df["sales"].where(df["available"])
    recent_avg = (
        sales_if_available.groupby(df["unique_id"])
        .transform(lambda s: s.rolling(28, min_periods=1).mean().ffill())
        .fillna(0)
    )
    return df["sales"].where(df["available"], recent_avg.round())


def build_inputs() -> pd.DataFrame:
    """One row per item and day, from HISTORY_DAYS before the replay to its end."""
    df = pd.read_parquet(PROCESSED / "sales_long.parquet")
    classes = pd.read_parquet(PROCESSED / "item_classes.parquet")
    df["unique_id"] = df["item_id"].astype(str) + "_" + df["store_id"].astype(str)
    df = df.rename(columns={"date": "ds"}).sort_values(["unique_id", "ds"])
    df["sales"] = df["sales"].astype(float)
    df["demand"] = impute_demand(df)

    fc = hybrid_forecasts(classes)
    first = REPLAY_START - pd.Timedelta(days=HISTORY_DAYS)
    df = df[(df["ds"] >= first) & (df["ds"] <= REPLAY_END) & df["unique_id"].isin(fc["unique_id"])]
    df = df.merge(fc, on=["unique_id", "ds"], how="left")

    cls = classes.assign(
        unique_id=classes["item_id"].astype(str) + "_" + classes["store_id"].astype(str)
    )[["unique_id", "demand_class"]]
    df = df.merge(cls, on="unique_id", how="left")
    df["in_replay"] = df["ds"] >= REPLAY_START
    cols = [
        "unique_id",
        "ds",
        "in_replay",
        "demand",
        "sales",
        "available",
        "sell_price",
        "demand_class",
        "cutoff",
        "q50",
        "q90",
        "q95",
    ]
    return df[cols].reset_index(drop=True)


def draw_lead_times(inputs: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """One lead time per item and review day (Monday), drawn once so every policy shares them."""
    reviews = inputs[inputs["in_replay"] & (inputs["ds"].dt.day_of_week == 0)][["unique_id", "ds"]]
    rng = np.random.default_rng(seed)
    return reviews.assign(lead_time=rng.integers(LEAD_MIN, LEAD_MAX + 1, size=len(reviews)))
