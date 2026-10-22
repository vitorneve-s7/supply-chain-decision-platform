"""Features for the days after the data ends, built from the raw calendar and price files."""

from pathlib import Path

import pandas as pd

from src.forecasting.config import HORIZON

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def future_rows(df: pd.DataFrame, columns: list[str], horizon: int = HORIZON) -> pd.DataFrame:
    """One row per item and future day, with `columns` as in to_nixtla and y unknown.

    df is the processed sales table; the future days are the `horizon` days after its last date.
    """
    last_day = df["date"].max()
    calendar = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    in_horizon = (calendar["date"] > last_day) & (
        calendar["date"] <= last_day + pd.Timedelta(days=horizon)
    )
    calendar = calendar[in_horizon]

    items = df[["item_id", "store_id"]].drop_duplicates().astype(str)
    rows = items.merge(calendar, how="cross")
    prices = pd.read_csv(RAW / "sell_prices.csv")
    rows = rows.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    rows["unique_id"] = rows["item_id"] + "_" + rows["store_id"]
    rows = rows.rename(columns={"date": "ds"})
    rows["y"] = float("nan")
    rows["available"] = True
    rows["closed"] = (rows["ds"].dt.month == 12) & (rows["ds"].dt.day == 25)
    return rows[columns]
