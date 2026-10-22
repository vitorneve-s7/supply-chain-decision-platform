"""Global LightGBM model with mlforecast: features and model setup."""

import lightgbm as lgb
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean

EVENT_COLUMNS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
EXOGENOUS = ["sell_price", "price_rel", "snap_CA", *EVENT_COLUMNS]


def add_ml_features(nx: pd.DataFrame) -> pd.DataFrame:
    """Add features known in advance: relative price and events as integer codes.

    `available` and `closed` are dropped: they are not known for future days.
    """
    out = nx.copy()
    avg_price = out.groupby("unique_id")["sell_price"].transform(
        lambda p: p.rolling(28, min_periods=1).mean()
    )
    out["price_rel"] = (out["sell_price"] / avg_price).astype("float32")
    for col in EVENT_COLUMNS:
        out[col] = out[col].astype("category").cat.codes.astype("int16")  # -1 = no event
    return out[["unique_id", "ds", "y", *EXOGENOUS]]


def make_mlforecast(**lgb_params) -> MLForecast:
    params = {
        "objective": "tweedie",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "verbosity": -1,
        "random_state": 42,
    }
    params.update(lgb_params)
    return MLForecast(
        models={"LightGBM": lgb.LGBMRegressor(**params)},
        freq="D",
        lags=[7, 14, 28],
        lag_transforms={7: [RollingMean(window_size=7), RollingMean(window_size=28)]},
        date_features=["dayofweek", "day", "month"],
    )
