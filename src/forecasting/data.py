"""Convert the processed sales table to the format statsforecast and mlforecast expect."""

import pandas as pd

EXTRA_COLUMNS = [
    "sell_price",
    "snap_CA",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "available",
    "closed",
]


def to_nixtla(df: pd.DataFrame) -> pd.DataFrame:
    """Return unique_id / ds / y plus feature columns, with Christmas filled in for training."""
    out = df.rename(columns={"date": "ds", "sales": "y"})
    out["unique_id"] = out["item_id"].astype(str) + "_" + out["store_id"].astype(str)
    out["y"] = out["y"].astype("float32")
    out = out.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Christmas: average of the day before and the day after
    g = out.groupby("unique_id")["y"]
    prev, nxt = g.shift(1), g.shift(-1)
    fill = ((prev + nxt) / 2).fillna(prev).fillna(nxt).fillna(0).round()
    out.loc[out["closed"], "y"] = fill[out["closed"]]

    return out[["unique_id", "ds", "y", *EXTRA_COLUMNS]]
