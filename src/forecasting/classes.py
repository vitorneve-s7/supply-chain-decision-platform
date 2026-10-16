from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.config import TEST_START

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def demand_profile(sales: pd.Series) -> pd.Series:
    nonzero = sales[sales > 0]
    adi = len(sales) / len(nonzero) if len(nonzero) > 0 else np.nan
    cv2 = (nonzero.std() / nonzero.mean()) ** 2 if len(nonzero) > 1 else np.nan
    return pd.Series({"adi": adi, "cv2": cv2})


def classify(adi: float, cv2: float) -> str:
    if np.isnan(adi) or np.isnan(cv2):
        return "lumpy"
    if adi < 1.32:
        return "smooth" if cv2 < 0.49 else "erratic"
    return "intermittent" if cv2 < 0.49 else "lumpy"


def build_classes(df: pd.DataFrame) -> pd.DataFrame:
    train = df[(df["date"] < TEST_START) & df["available"] & ~df["closed"]]
    out = (
        train.groupby(["item_id", "store_id"], observed=True)["sales"]
        .apply(demand_profile)
        .unstack()
        .reset_index()
    )
    out["demand_class"] = [classify(a, c) for a, c in zip(out["adi"], out["cv2"])]
    return out


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED / "sales_long.parquet")
    classes = build_classes(df)
    classes.to_parquet(PROCESSED / "item_classes.parquet", index=False)
    print(classes["demand_class"].value_counts())
