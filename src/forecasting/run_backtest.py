"""Run a group of models through the backtest and save the results."""

import sys
import time
from pathlib import Path

import pandas as pd

from src.forecasting import models
from src.forecasting.backtest import evaluate, run_statsforecast, summarise, train_part
from src.forecasting.data import to_nixtla

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
GROUPS = {
    "benchmarks": {"models": models.BENCHMARKS},
    "stats": {"models": models.STATS, "level": [80, 90]},
    "arima": {"models": models.ARIMA, "level": [80, 90], "history_days": 730},
}


def update_summary(per_item: pd.DataFrame) -> pd.DataFrame:
    """Merge this group's per-item metrics into the saved file and rebuild the summary."""
    path = RESULTS / "metrics_per_item.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        per_item = pd.concat([old[~old["model"].isin(per_item["model"])], per_item])
    per_item.to_parquet(path, index=False)
    summary = summarise(per_item)
    summary.to_csv(RESULTS / "summary.csv", index=False, float_format="%.4f")
    return summary


def main(group: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_parquet(PROCESSED / "sales_long.parquet")
    classes = pd.read_parquet(PROCESSED / "item_classes.parquet")
    train = train_part(to_nixtla(df))

    start = time.time()
    config = GROUPS[group]
    results = run_statsforecast(
        config["models"], train, config.get("level"), config.get("history_days")
    )
    print(f"{group}: {results['unique_id'].nunique()} items in {time.time() - start:.0f} s")

    results.to_parquet(RESULTS / f"backtest_{group}.parquet", index=False)
    summary = update_summary(evaluate(results, train, classes))
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "benchmarks")
