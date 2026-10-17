"""Rolling-origin backtest: run models on the validation windows and score them the same way."""

import numpy as np
import pandas as pd
from statsforecast import StatsForecast

from src.forecasting import metrics
from src.forecasting.config import HORIZON, N_VAL_WINDOWS, TEST_START

VAL_START = TEST_START - pd.Timedelta(days=HORIZON * N_VAL_WINDOWS)  # 2016-01-04
MIN_HISTORY = 28  # days of history an item needs before VAL_START to be backtested
QUANTILES = {"q90": 0.90, "q95": 0.95}


def train_part(nx: pd.DataFrame) -> pd.DataFrame:
    """Everything before the final test window, for items with enough history to backtest.

    The test window is never touched here. Items launched less than MIN_HISTORY days before
    VAL_START have nothing to learn from in the first window, so they are left out.
    """
    first_day = nx.groupby("unique_id")["ds"].transform("min")
    keep = (nx["ds"] < TEST_START) & (first_day <= VAL_START - pd.Timedelta(days=MIN_HISTORY))
    return nx[keep].reset_index(drop=True)


def run_statsforecast(
    models: list,
    train: pd.DataFrame,
    level: list[int] | None = None,
    history_days: int | None = None,
) -> pd.DataFrame:
    """Cross-validate statsforecast models on the validation windows; return the long format.

    level: interval levels to ask for, e.g. [80, 90] gives the 90th and 95th percentiles.
    history_days: train on the last N days before the test window only (for slow models).
    """
    data = train[["unique_id", "ds", "y"]]
    if history_days is not None:
        data = data[data["ds"] >= TEST_START - pd.Timedelta(days=history_days)]
    sf = StatsForecast(models=models, freq="D", n_jobs=-1)
    cv = sf.cross_validation(
        df=data,
        h=HORIZON,
        step_size=HORIZON,
        n_windows=N_VAL_WINDOWS,
        level=level,
    )
    return to_long(cv, model_names=[repr(m) for m in models])


def to_long(cv: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """One row per (item, date, window, model) with a `mean` column (+ quantiles if present)."""
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    frames = []
    for name in model_names:
        part = cv[["unique_id", "ds", "cutoff", "y"]].copy()
        part["model"] = name
        part["mean"] = cv[name].clip(lower=0).to_numpy()
        for q_col, level in [("q90", 80), ("q95", 90)]:
            hi = f"{name}-hi-{level}"
            if hi in cv.columns:
                part[q_col] = cv[hi].clip(lower=0).to_numpy()
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def evaluate(results: pd.DataFrame, train: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """Metrics per item and model, on scored days only (available and not closed)."""
    flags = train[["unique_id", "ds", "available", "closed"]]
    df = results.merge(flags, on=["unique_id", "ds"], how="left")
    df = df[df["available"] & ~df["closed"]]

    history = train[train["ds"] < VAL_START]
    scale = history.groupby("unique_id")["y"].apply(metrics.naive_scale).rename("scale")

    rows = []
    for (uid, model), g in df.groupby(["unique_id", "model"], sort=False):
        row = {
            "unique_id": uid,
            "model": model,
            "n_days": len(g),
            "y_sum": g["y"].sum(),
            "error_sum": (g["mean"] - g["y"]).sum(),
            "mae": metrics.mae(g["y"], g["mean"]),
            "rmsse": metrics.rmsse(g["y"], g["mean"], scale.get(uid, np.nan)),
        }
        for q_col, q in QUANTILES.items():
            if q_col in g.columns and g[q_col].notna().all():
                row[f"pinball_{q_col}"] = metrics.pinball(g["y"], g[q_col], q)
                row[f"coverage_{q_col}"] = metrics.coverage(g["y"], g[q_col])
        rows.append(row)

    per_item = pd.DataFrame(rows)
    cls = classes.assign(
        unique_id=classes["item_id"].astype(str) + "_" + classes["store_id"].astype(str)
    )[["unique_id", "demand_class"]]
    return per_item.merge(cls, on="unique_id", how="left")


def summarise(per_item: pd.DataFrame) -> pd.DataFrame:
    """Average the per-item metrics by model and demand class, plus an 'all' row per model."""
    both = pd.concat([per_item, per_item.assign(demand_class="all")], ignore_index=True)
    g = both.groupby(["model", "demand_class"])
    out = g[[c for c in per_item.columns if c.startswith(("mae", "rmsse", "pinball", "coverage"))]]
    out = out.mean()
    out["bias"] = g["error_sum"].sum() / g["y_sum"].sum()
    out["n_items"] = g["unique_id"].nunique()
    return out.reset_index().sort_values(["demand_class", "rmsse"])
