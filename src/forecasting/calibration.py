"""Calibrate interval width so q90 and q95 cover 90% and 95% of the scored days.

For each model and demand class, find the factor c so that mean + c * (q - mean) covers
the target share of actuals. The factors are fitted on the validation windows and checked
on the test window in Part G.
"""

import numpy as np
import pandas as pd

TARGETS = {"q90": 0.90, "q95": 0.95}
GRID = np.round(np.arange(0.5, 5.001, 0.05), 2)  # factors tried, from 0.5 to 5


def scored_days(results: pd.DataFrame, train: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """Keep the days that count (available, not closed) and add each item's demand class."""
    flags = train[["unique_id", "ds", "available", "closed"]]
    df = results.merge(flags, on=["unique_id", "ds"], how="left")
    df = df[df["available"] & ~df["closed"]]
    cls = classes.assign(
        unique_id=classes["item_id"].astype(str) + "_" + classes["store_id"].astype(str)
    )[["unique_id", "demand_class"]]
    return df.merge(cls, on="unique_id", how="left")


def find_factor(y, mean, q, target: float) -> float:
    """Smallest factor in GRID whose widened quantile covers `target` of the actuals."""
    error = np.asarray(y) - np.asarray(mean)
    width = np.asarray(q) - np.asarray(mean)
    for c in GRID:
        if np.mean(error <= c * width) >= target:
            return float(c)
    return float(GRID[-1])


def fit_factors(results: pd.DataFrame, train: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """One row per model and demand class, with the factor for q90 and for q95."""
    df = scored_days(results, train, classes)
    rows = []
    for (model, demand_class), g in df.groupby(["model", "demand_class"]):
        row = {"model": model, "demand_class": demand_class}
        for col, target in TARGETS.items():
            row[col] = find_factor(g["y"], g["mean"], g[col], target)
        rows.append(row)
    return pd.DataFrame(rows)


def apply_factors(
    results: pd.DataFrame, classes: pd.DataFrame, factors: pd.DataFrame
) -> pd.DataFrame:
    """Widen or narrow q90 and q95 with the factors; keep 0 <= mean <= q90 <= q95."""
    cls = classes.assign(
        unique_id=classes["item_id"].astype(str) + "_" + classes["store_id"].astype(str)
    )[["unique_id", "demand_class"]]
    out = results.merge(cls, on="unique_id", how="left")
    out = out.merge(factors, on=["model", "demand_class"], how="left", suffixes=("", "_c"))
    for col in TARGETS:
        out[col] = out["mean"] + out[f"{col}_c"] * (out[col] - out["mean"])
    out["q90"] = np.maximum(out["q90"], out["mean"]).clip(lower=0)
    out["q95"] = np.maximum(out["q95"], out["q90"])
    return out[results.columns]
