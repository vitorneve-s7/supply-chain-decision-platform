import pandas as pd
import pytest
from statsforecast.models import Naive

from src.forecasting.backtest import VAL_START, evaluate, run_statsforecast, train_part
from src.forecasting.config import HORIZON, N_VAL_WINDOWS, TEST_START


@pytest.fixture
def toy():
    """Two items with 400 days of made-up sales, ending on the last day of the data."""
    dates = pd.date_range(end="2016-05-22", periods=400, freq="D")
    rows = []
    for uid, level in [("A_CA_1", 5), ("B_CA_1", 1)]:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "y": float(level + i % 7),
                    "available": True,
                    "closed": False,
                }
            )
    nx = pd.DataFrame(rows)
    # one unavailable day inside the validation period, to check it is skipped
    nx.loc[(nx["unique_id"] == "A_CA_1") & (nx["ds"] == VAL_START), "available"] = False
    return nx


def test_no_test_data_in_train(toy):
    assert train_part(toy)["ds"].max() < TEST_START


def test_windows_and_no_leakage(toy):
    res = run_statsforecast([Naive()], train_part(toy))
    assert res["cutoff"].nunique() == N_VAL_WINDOWS
    assert (res["ds"] > res["cutoff"]).all()  # every forecast is after its training end
    assert ((res["ds"] - res["cutoff"]).dt.days <= HORIZON).all()
    assert res["ds"].min() == VAL_START
    assert res["ds"].max() < TEST_START


def test_evaluate_shape(toy):
    train = train_part(toy)
    res = run_statsforecast([Naive()], train)
    classes = pd.DataFrame(
        {"item_id": ["A", "B"], "store_id": ["CA_1", "CA_1"], "demand_class": ["smooth", "smooth"]}
    )
    per_item = evaluate(res, train, classes)
    n_days = per_item.set_index("unique_id")["n_days"]
    assert n_days["A_CA_1"] == HORIZON * N_VAL_WINDOWS - 1  # the unavailable day is skipped
    assert n_days["B_CA_1"] == HORIZON * N_VAL_WINDOWS
