"""Model lists used in the backtest. Each phase part adds its own group."""

from statsforecast.models import HistoricAverage, Naive, SeasonalNaive, WindowAverage

BENCHMARKS = [
    Naive(),
    SeasonalNaive(season_length=7),
    WindowAverage(window_size=28),
    HistoricAverage(),
]
