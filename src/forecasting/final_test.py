"""Train on everything before the test window, forecast it once, and score the result.

Usage: uv run python -m src.forecasting.final_test
"""

import sys
from pathlib import Path

import pandas as pd
from mlforecast.utils import PredictionIntervals
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive, WindowAverage

from src.forecasting.backtest import evaluate, summarise, to_long, train_part
from src.forecasting.calibration import apply_factors
from src.forecasting.config import HORIZON, TEST_START
from src.forecasting.data import to_nixtla
from src.forecasting.ml import add_ml_features, make_mlforecast
from src.forecasting.selection import INTERVAL_MODEL, POINT_MODEL, build_hybrid

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CUTOFF = TEST_START - pd.Timedelta(days=1)  # last training day: 2016-04-24


def statsforecast_test(history: pd.DataFrame) -> pd.DataFrame:
    """AutoETS with intervals, plus two benchmarks (no intervals), fitted once on all history."""
    data = history[["unique_id", "ds", "y"]]
    ets = StatsForecast(models=[AutoETS(season_length=7)], freq="D", n_jobs=-1)
    bench = StatsForecast(
        models=[SeasonalNaive(season_length=7), WindowAverage(window_size=28)], freq="D"
    )
    return ets.forecast(df=data, h=HORIZON, level=[80, 90]).merge(
        bench.forecast(df=data, h=HORIZON), on=["unique_id", "ds"]
    )


def lightgbm_test(nx: pd.DataFrame) -> pd.DataFrame:
    """LightGBM fitted once on all history; future features come from the test rows."""
    feats = add_ml_features(nx)
    fcst = make_mlforecast()
    fcst.fit(
        feats[feats["ds"] < TEST_START],
        static_features=[],
        prediction_intervals=PredictionIntervals(n_windows=2, h=HORIZON),
    )
    future = feats[feats["ds"] >= TEST_START].drop(columns="y")
    return fcst.predict(h=HORIZON, X_df=future, level=[80, 90])


def main(n_items: int | None = None) -> None:
    df = pd.read_parquet(PROCESSED / "sales_long.parquet")
    classes = pd.read_parquet(PROCESSED / "item_classes.parquet")
    nx = to_nixtla(df)

    items = train_part(nx)["unique_id"].unique()  # same 821 items as the backtest
    if n_items:  # quick run on a few items, for checking the code only
        items = items[:n_items]
    nx = nx[nx["unique_id"].isin(items)].reset_index(drop=True)
    history = nx[nx["ds"] < TEST_START]
    actuals = nx[nx["ds"] >= TEST_START][["unique_id", "ds", "y"]]

    wide = statsforecast_test(history).merge(lightgbm_test(nx), on=["unique_id", "ds"], how="inner")
    wide = wide.merge(actuals, on=["unique_id", "ds"], how="left").assign(cutoff=CUTOFF)
    results = to_long(wide, ["AutoETS", "LightGBM", "SeasonalNaive", "WindowAverage"])

    factors = pd.read_csv(RESULTS / "calibration_factors.csv")
    with_q = results[results["model"].isin(factors["model"])]
    calibrated = apply_factors(with_q, classes, factors)
    hybrid = build_hybrid(calibrated, classes, POINT_MODEL, INTERVAL_MODEL)
    benchmarks = results[~results["model"].isin(factors["model"])]
    final = pd.concat([hybrid, calibrated, benchmarks], ignore_index=True)

    summary = summarise(evaluate(final, nx, classes))
    if not n_items:
        final.to_parquet(RESULTS / "test_forecasts.parquet", index=False)
        summary.to_csv(RESULTS / "summary_test.csv", index=False, float_format="%.4f")
    cols = [
        "model",
        "demand_class",
        "mae",
        "rmsse",
        "bias",
        "coverage_q90",
        "coverage_q95",
        "pinball_q90",
        "pinball_q95",
        "n_items",
    ]
    print(summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
