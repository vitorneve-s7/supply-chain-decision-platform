"""Fit interval calibration factors on the validation windows and report coverage.

Usage: uv run python -m src.forecasting.calibrate
"""

from pathlib import Path

import pandas as pd

from src.forecasting.backtest import evaluate, summarise, train_part
from src.forecasting.calibration import apply_factors, fit_factors, scored_days
from src.forecasting.data import to_nixtla

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def load_interval_results() -> pd.DataFrame:
    """All saved backtest results that have q90 and q95 columns."""
    frames = [pd.read_parquet(p) for p in sorted(RESULTS.glob("backtest_*.parquet"))]
    results = pd.concat([f for f in frames if "q95" in f.columns], ignore_index=True)
    return results.dropna(subset=["q90", "q95"])


def coverage_table(results, train, classes) -> pd.DataFrame:
    df = scored_days(results, train, classes)
    df = df.assign(in90=df["y"] <= df["q90"], in95=df["y"] <= df["q95"])
    return df.groupby(["model", "demand_class"])[["in90", "in95"]].mean().round(3)


def main() -> None:
    df = pd.read_parquet(PROCESSED / "sales_long.parquet")
    classes = pd.read_parquet(PROCESSED / "item_classes.parquet")
    train = train_part(to_nixtla(df))
    results = load_interval_results()

    # 1. Does a factor fitted on early windows hold on later ones?
    cutoffs = sorted(results["cutoff"].unique())
    early = results[results["cutoff"].isin(cutoffs[:2])]
    late = results[results["cutoff"].isin(cutoffs[2:])]
    before = coverage_table(late, train, classes)
    early_factors = fit_factors(early, train, classes)
    after = coverage_table(apply_factors(late, classes, early_factors), train, classes)
    check = before.join(after.add_suffix("_calibrated"))
    print("Coverage on windows 3-4, factors fitted on windows 1-2:")
    print(check.to_string())

    # 2. Final factors from all four validation windows
    factors = fit_factors(results, train, classes)
    factors.to_csv(RESULTS / "calibration_factors.csv", index=False)

    calibrated = apply_factors(results, classes, factors)
    calibrated.to_parquet(RESULTS / "calibrated_backtest.parquet", index=False)
    summary = summarise(evaluate(calibrated, train, classes))
    summary.to_csv(RESULTS / "summary_calibrated.csv", index=False, float_format="%.4f")
    cols = [
        "model",
        "demand_class",
        "rmsse",
        "coverage_q90",
        "coverage_q95",
        "pinball_q90",
        "pinball_q95",
    ]
    print("\nAfter calibration (all validation windows):")
    print(summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
