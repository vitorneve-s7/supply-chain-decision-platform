"""Turn replay records into a scorecard: service and stock, overall and per demand class."""

import pandas as pd

WARMUP_END = pd.Timestamp("2016-01-18")  # the first 14 replay days are not scored


def scorecard(
    records: pd.DataFrame, start: pd.Timestamp = WARMUP_END, end: pd.Timestamp | None = None
) -> pd.DataFrame:
    """One row per demand class plus 'all': fill rate, cycle service level, stock and orders.

    Only days from `start` to `end` (inclusive) are scored; by default, all days after the warm-up.
    """
    df = records[(records["ds"] >= start) & (records["ds"] <= (end or records["ds"].max()))].copy()
    df["week"] = df["ds"].dt.to_period("W-SUN")
    df["stock_value"] = df["end_stock"] * df["sell_price"]

    weeks = df.groupby(["unique_id", "week"]).agg(
        demand_class=("demand_class", "first"), lost=("lost", "sum")
    )
    weeks["no_stockout"] = weeks["lost"] == 0
    n_days = df["ds"].nunique()

    rows = {}
    groups = [("all", df, weeks)] + [
        (k, g, weeks[weeks["demand_class"] == k]) for k, g in df.groupby("demand_class")
    ]
    for name, g, w in groups:
        rows[name] = {
            "items": g["unique_id"].nunique(),
            "fill_rate": g["sales"].sum() / g["demand"].sum(),
            "cycle_service": w["no_stockout"].mean(),
            "days_of_stock": g["end_stock"].sum() / g["demand"].sum(),
            "stock_value": g["stock_value"].sum() / n_days,
            "orders": int((g["order"] > 0).sum()),
        }
    return pd.DataFrame(rows).T
