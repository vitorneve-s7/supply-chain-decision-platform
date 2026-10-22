"""Demand forecast for one item: the function the inventory model and the agent call."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.forecasting.config import HORIZON

ROOT = Path(__file__).resolve().parents[2]
FORECAST_FILE = ROOT / "results" / "future_forecast.parquet"


@lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"{FORECAST_FILE} not found. Run: uv run python -m src.forecasting.train_final"
        )
    return pd.read_parquet(FORECAST_FILE)


def forecast(item: str, store: str = "CA_1", horizon: int = HORIZON) -> pd.DataFrame:
    """Daily q50, q90 and q95 demand for the next `horizon` days (1 to 28).

    Example: forecast("FOODS_3_090", "CA_1", horizon=7)
    """
    if not 1 <= horizon <= HORIZON:
        raise ValueError(f"horizon must be between 1 and {HORIZON}, got {horizon}")
    data = _load()
    rows = data[data["unique_id"] == f"{item}_{store}"]
    if rows.empty:
        raise KeyError(f"No forecast for item {item!r} in store {store!r}")
    rows = rows.sort_values("ds").head(horizon)
    return rows.rename(columns={"ds": "date"})[["date", "q50", "q90", "q95"]].reset_index(drop=True)
