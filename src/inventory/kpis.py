"""Turn replay records into a scorecard: service and stock, overall and per demand class."""

import pandas as pd

from src.inventory.policies import HOLDING_RATE, ORDER_COST

LOST_SALES_RATE = 0.30  # penalty per lost unit, as a share of the price (lost margin)
WARMUP_END = pd.Timestamp("2016-01-18")  # the first 14 replay days are not scored


def scorecard(
    records: pd.DataFrame,
    start: pd.Timestamp = WARMUP_END,
    end: pd.Timestamp | None = None,
    holding_rate: float = HOLDING_RATE,
    order_cost: float = ORDER_COST,
    lost_sales_rate: float = LOST_SALES_RATE,
) -> pd.DataFrame:
    """One row per demand class plus 'all': service, stock, orders and costs.

    Only days from `start` to `end` (inclusive) are scored; by default, all days after the warm-up.
    """
    df = records[(records["ds"] >= start) & (records["ds"] <= (end or records["ds"].max()))].copy()
    df["week"] = df["ds"].dt.to_period("W-SUN")
    df["stock_value"] = df["end_stock"] * df["sell_price"]
    df["holding_cost"] = df["stock_value"] * holding_rate / 365
    df["ordering_cost"] = (df["order"] > 0) * order_cost
    df["lost_sales_cost"] = df["lost"] * df["sell_price"] * lost_sales_rate

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
            "holding_cost": g["holding_cost"].sum(),
            "ordering_cost": g["ordering_cost"].sum(),
            "lost_sales_cost": g["lost_sales_cost"].sum(),
        }
    card = pd.DataFrame(rows).T
    card["total_cost"] = card[["holding_cost", "ordering_cost", "lost_sales_cost"]].sum(axis=1)
    return card
