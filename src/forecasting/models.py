"""Model lists used in the backtest. Each phase part adds its own group."""

from statsforecast.models import (
    ADIDA,
    IMAPA,
    TSB,
    AutoARIMA,
    AutoETS,
    CrostonOptimized,
    HistoricAverage,
    Naive,
    SeasonalNaive,
    WindowAverage,
)
from statsforecast.utils import ConformalIntervals

from src.forecasting.config import HORIZON

# Intervals from past errors, for models that have no built-in uncertainty
CONFORMAL = ConformalIntervals(h=HORIZON, n_windows=2)

BENCHMARKS = [
    Naive(),
    SeasonalNaive(season_length=7),
    WindowAverage(window_size=28),
    HistoricAverage(),
]

STATS = [
    AutoETS(season_length=7),
    CrostonOptimized(prediction_intervals=CONFORMAL),
    TSB(alpha_d=0.1, alpha_p=0.1, prediction_intervals=CONFORMAL),
    ADIDA(prediction_intervals=CONFORMAL),
    IMAPA(prediction_intervals=CONFORMAL),
]

ARIMA = [AutoARIMA(season_length=7)]
