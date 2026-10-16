"""Forecast accuracy metrics. Each function takes NumPy arrays or pandas Series of equal length."""

import numpy as np


def mae(y, yhat) -> float:
    """Mean absolute error, in units."""
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat) -> float:
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def naive_scale(y_train) -> float:
    """RMSE of a one-day naive forecast on the training history (the RMSSE denominator)."""
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) < 2:
        return np.nan
    return float(np.sqrt(np.mean(np.diff(y_train) ** 2)))


def rmsse(y, yhat, scale: float) -> float:
    """RMSE divided by the naive scale. Below 1 = better than a one-day naive forecast."""
    if not scale or np.isnan(scale):
        return np.nan
    return rmse(y, yhat) / scale


def bias(y, yhat) -> float:
    """Sum of (forecast - actual) / sum of actuals. Positive = over-forecasting."""
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    total = y.sum()
    return float((yhat - y).sum() / total) if total > 0 else np.nan


def pinball(y, q_pred, q: float) -> float:
    """Average pinball (quantile) loss for quantile level q, e.g. q=0.9."""
    y, q_pred = np.asarray(y, dtype=float), np.asarray(q_pred, dtype=float)
    diff = y - q_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y, q_pred) -> float:
    """Share of actuals at or below the quantile forecast."""
    y, q_pred = np.asarray(y, dtype=float), np.asarray(q_pred, dtype=float)
    return float(np.mean(y <= q_pred))
