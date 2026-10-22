"""Fit the chosen Hybrid on all data and forecast the next 28 days for every item.

Usage: uv run python -m src.forecasting.train_final
Writes results/future_forecast.parquet, which forecast() reads.
"""

import sys
from pathlib import Path

import pandas as pd
from mlforecast.utils import PredictionIntervals
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from src.forecasting.backtest import to_long
from src.forecasting.calibration import apply_factors
from src.forecasting.config import HORIZON
from src.forecasting.data import to_nixtla
from src.forecasting.future import future_rows
from src.forecasting.ml import add_ml_features, make_mlforecast
from src.forecasting.selection import INTERVAL_MODEL, POINT_MODEL, build_hybrid

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def main(n_items: int | None = None) -> None:
    df = pd.read_parquet(PROCESSED / "sales_long.parquet")
    classes = pd.read_parquet(PROCESSED / "item_classes.parquet")
    nx = to_nixtla(df)
    if n_items:  # quick run on a few items, for checking the code only
        nx = nx[nx["unique_id"].isin(nx["unique_id"].unique()[:n_items])]
    future = future_rows(df, columns=list(nx.columns))
    future = future[future["unique_id"].isin(nx["unique_id"])]
    cutoff = nx["ds"].max()

    ets = StatsForecast(models=[AutoETS(season_length=7)], freq="D", n_jobs=-1)
    ets_fc = ets.forecast(df=nx[["unique_id", "ds", "y"]], h=HORIZON, level=[80, 90])

    feats = add_ml_features(pd.concat([nx, future], ignore_index=True))
    fcst = make_mlforecast()
    fcst.fit(
        feats[feats["ds"] <= cutoff],
        static_features=[],
        prediction_intervals=PredictionIntervals(n_windows=2, h=HORIZON),
    )
    lgb_fc = fcst.predict(
        h=HORIZON, X_df=feats[feats["ds"] > cutoff].drop(columns="y"), level=[80, 90]
    )

    wide = ets_fc.merge(lgb_fc, on=["unique_id", "ds"]).assign(y=float("nan"), cutoff=cutoff)
    results = to_long(wide, ["AutoETS", "LightGBM"])
    factors = pd.read_csv(RESULTS / "calibration_factors.csv")
    calibrated = apply_factors(results, classes, factors)
    hybrid = build_hybrid(calibrated, classes, POINT_MODEL, INTERVAL_MODEL)

    out = hybrid.rename(columns={"mean": "q50"})[["unique_id", "ds", "q50", "q90", "q95"]]
    if n_items:
        print(out.describe().round(2).to_string())
        return
    out.to_parquet(RESULTS / "future_forecast.parquet", index=False)
    print(
        f"Saved {len(out):,} rows for {out['unique_id'].nunique()} items, "
        f"{out['ds'].min().date()} to {out['ds'].max().date()}"
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
