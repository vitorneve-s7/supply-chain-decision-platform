import numpy as np
import pytest

from src.forecasting import metrics


def test_mae():
    assert metrics.mae([1, 2, 3], [1, 2, 4]) == pytest.approx(1 / 3)


def test_naive_scale_and_rmsse():
    # naive errors on [1, 3, 5] are 2 and 2 -> scale 2
    assert metrics.naive_scale([1, 3, 5]) == pytest.approx(2.0)
    # RMSE of [2, 2] vs [0, 0] is 2 -> RMSSE 1
    assert metrics.rmsse([2, 2], [0, 0], scale=2.0) == pytest.approx(1.0)


def test_rmsse_flat_history_is_nan():
    assert np.isnan(metrics.rmsse([1, 2], [1, 2], scale=metrics.naive_scale([0, 0, 0])))


def test_bias_sign():
    assert metrics.bias([10, 10], [12, 12]) == pytest.approx(0.2)  # over-forecast
    assert metrics.bias([10, 10], [8, 8]) == pytest.approx(-0.2)  # under-forecast


def test_pinball():
    # q=0.9: under-forecast by 10 costs 9, over-forecast by 10 costs 1
    assert metrics.pinball([10], [0], q=0.9) == pytest.approx(9.0)
    assert metrics.pinball([0], [10], q=0.9) == pytest.approx(1.0)


def test_coverage():
    assert metrics.coverage([1, 2, 3, 4], [2, 2, 2, 2]) == pytest.approx(0.5)
