# Supply Chain Decision Platform

End-to-end retail supply chain project on the M5 (Walmart) dataset:
demand forecasting → inventory policies → supply chain digital twin → LLM planning assistant.

## Status

- [x] Phase 0 – Setup and data exploration
- [x] Phase 1 – Demand forecasting
- [ ] Phase 2 – Inventory policies
- [ ] Phase 3 – Digital twin
- [ ] Phase 4 – LLM agent

## Scope

Store CA_1, department FOODS_3: 823 items, daily sales from 2011-01-29 to 2016-05-22.
All assumptions are in [docs/assumptions.md](docs/assumptions.md).

## Phase 1 results: demand forecasting

A hybrid forecast cuts error by **23% against seasonal naive** (RMSSE 0.752 vs 0.976) on a 28-day
test window no model saw during development, and its **95th percentile covers 95.2% of days**.

| Demand class | Items | Hybrid uses | RMSSE Hybrid | RMSSE seasonal naive | q95 coverage |
| --- | --- | --- | --- | --- | --- |
| Smooth | 173 | LightGBM | 0.762 | 0.980 | 95.4% |
| Erratic | 93 | LightGBM | 0.727 | 0.916 | 93.5% |
| Intermittent | 359 | AutoETS | 0.758 | 0.996 | 95.6% |
| Lumpy | 184 | AutoETS | 0.745 | 0.964 | 94.8% |
| **All** | **809** | | **0.752** | **0.976** | **95.2%** |

Test window 2016-04-25 to 2016-05-22; RMSSE below 1 means better than a one-day naive forecast.

**How it was built**

- Rolling-origin backtest: 4 validation windows of 28 days for every model choice; the final test
  window was used once, at the end.
- 12 models compared: benchmarks, AutoETS, AutoARIMA, Croston, TSB, ADIDA, IMAPA (statsforecast)
  and a global LightGBM with lag, price and calendar features (mlforecast).
- Days when an item was out of stock and Christmas Day are excluded from scoring, so models are
  not rewarded for predicting zeros caused by empty shelves.
- Prediction intervals are calibrated per model and demand class; raw conformal intervals covered
  only about 82% at the 95th percentile.

**What I learned**

- LightGBM gives the best point forecast for items that sell every day, but AutoETS gives the
  sharpest quantiles in every class, so the Hybrid combines LightGBM's point with AutoETS's interval.
- The gain over a 28-day moving average is modest (3%): daily item-level demand is mostly noise.
- On lumpy items LightGBM beat AutoETS on the test window, the opposite of validation; the
  validation choice was kept rather than re-picking on test data.
- All models under-forecast (Hybrid bias −9.6%), so stock should be set from q90/q95, not the mean.

Full analysis: [notebooks/02_forecasting.ipynb](notebooks/02_forecasting.ipynb)

## Key findings from the data

- Once stockout days are removed, 66% of items have intermittent or lumpy demand and 21% sell
  steadily every day.
- Weekends sell ~55–60% more than midweek; SNAP days add ~12%.
- 84% of fast-moving items show long zero-sales runs, pointing to stockouts or unavailability.

Exploratory analysis: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)

## How to run

1. Install [uv](https://docs.astral.sh/uv/) and run `uv sync`.
2. Download the [M5 data](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) into `data/raw/`.
3. Prepare the data:
```
   uv run python src/data/prepare.py
   uv run python -m src.forecasting.classes
```
4. Run the backtest (ARIMA is the slow one, up to an hour on 2 cores):
```
   uv run python -m src.forecasting.run_backtest benchmarks
   uv run python -m src.forecasting.run_backtest stats
   uv run python -m src.forecasting.run_backtest arima
   uv run python -m src.forecasting.run_backtest lightgbm
   uv run python -m src.forecasting.calibrate
```
5. Evaluate on the test window and build the future forecast:
```
   uv run python -m src.forecasting.final_test
   uv run python -m src.forecasting.train_final
```
6. Get a forecast in Python:
```python
from src.forecasting.forecast import forecast

forecast("FOODS_3_090", "CA_1", horizon=7)  # date, q50, q90, q95
```
7. Run the tests with `uv run pytest`.