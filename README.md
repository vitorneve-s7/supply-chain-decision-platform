# Supply Chain Decision Platform

Forecast-driven inventory decisions for 821 Walmart food items, from daily demand forecasts to
weekly orders, stress-tested in a digital twin, and explained by an LLM agent that answers
planner questions with the project's own numbers.

**Results so far**

- **Forecasting:** 23% lower error than seasonal naive on a held-out test window, with 95th
  percentile forecasts that cover 95.2% of days.
- **Inventory:** the same 90% service with **36% less stock** than a weeks-of-cover rule, and
  **27% lower total cost** with a forecast-based (s, S) policy.
- **Digital twin:** across five disruptions in a supplier → DC → store network, the (s, S) policy
  loses the least sales in three of them and is the cheapest in all five.
- **LLM agent:** answers planner questions in English or Portuguese by calling the project's
  code; **30 of 30 correct** on an evaluation set, with every number checked against a tool result.

Built with Python, pandas, statsforecast, LightGBM (mlforecast), SciPy, SimPy, the Anthropic API (Claude Haiku
5.5), Streamlit, Plotly and pytest, on the
[M5 dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy).

## Status

- [x] Phase 0 – Setup and data exploration
- [x] Phase 1 – Demand forecasting
- [x] Phase 2 – Inventory policies
- [x] Phase 3 – Digital twin
- [x] Phase 4 – LLM agent

## Scope

Store CA_1, department FOODS_3: 823 items, daily sales from 2011-01-29 to 2016-05-22.
All assumptions are in [docs/assumptions.md](docs/assumptions.md).

## Phase 4 results: LLM agent

A planner asks in plain language, in English or Portuguese. A Claude Haiku 5.5 agent answers by
calling six tools built on Phases 1–3, and code checks that every number in the answer came from
a tool.

![The app: the chat on the left, the numbers behind the answer on the right](docs/app_screenshot.png)

| Question type | Example | Correct |
| --- | --- | --- |
| Forecast | "How much will FOODS_3_090 sell next week?" | 6 / 6 |
| Inventory policy | "What order-up-to level does the (R, S) quantile policy set for FOODS_3_586?" | 5 / 5 |
| Policy comparison | "Which policy loses the least sales in a demand spike?" | 6 / 6 |
| Live twin run | "Run the longer lead times scenario for FOODS_3_090 with (s, S)." | 4 / 4 |
| Assumptions | "Why is the lost-sales cost set at 30% of the price?" | 4 / 4 |
| Out of scope | "What will store TX_2 sell next month?" | 3 / 3 |
| Follow-up | "E para os próximos 14 dias?" (after a forecast) | 2 / 2 |
| **All** | | **30 / 30** |

A question counts as correct only if the agent called the right tools, every expected number and
name is in the answer, and no number in it lacks a source. Expected values are computed by the
tools at run time, not typed in. A full run takes about 2 minutes and costs about 3 US cents.

**How it works**

- Six tools: forecast, inventory policy, policy comparison (precomputed 20-run twin results), a
  live twin run, item search and a search over the documented assumptions. They are plain
  functions, tested without the API.
- A hand-written tool-use loop, no framework: the model picks tools and phrases the answer; the
  code does every calculation, including totals, changes against normal and rankings.
- A grounding check reads every number in the answer (English and Portuguese formats) and looks
  for it in the tool results; if one is missing, the model gets one retry.

**What I learned**

- Let the code do the arithmetic. In the first test the model invented "20 to 31 June", and later
  ranked costs wrongly; returning totals, differences and ranks from the tools removed both.
- The first evaluation failure was the checker, not the agent: "1 134,8", with a Portuguese space
  as thousands separator, was read as two numbers.
- Honest negative: the check verifies numbers, not claims. In the app the agent called weeks of
  cover's 95.2% fill rate in a supplier outage "the lowest of the four", when (R, S) normal's
  89.4% is lower. The number was right, the comparison was not, and the evaluation set, written
  by the author of the tools, did not catch it. Checking ranking words against the tools' rank
  fields is the next step.
- A cheap model is enough when the tools do the work: Haiku 5.5 scored 30 / 30, so a dearer model
  had nothing left to gain on this set.

Full analysis: [notebooks/05_agent.ipynb](notebooks/05_agent.ipynb)

## Phase 3 results: digital twin

The four Phase 2 policies run in a SimPy twin of a small network (supplier → distribution centre
→ store), through five disruption scenarios, 20 runs each with the same random draws for every
policy. Damage is measured against a normal run with the same draws.

![Extra lost sales per disruption and policy](docs/twin_lost_sales.png)

| Policy | Total cost, normal | Extra lost sales: supplier outage (14 d) | Demand spike +50% (10 d) | DC capacity −50% (21 d) | Lead times ×2 (28 d) |
| --- | --- | --- | --- | --- | --- |
| Weeks of cover | 30,621 | **970** | 1,368 | 974 | 3,382 |
| (R, S) normal | 44,288 | 1,154 | 4,041 | 2,238 | 8,162 |
| (R, S) quantile | 31,659 | 1,438 | 2,075 | 1,536 | 4,178 |
| (s, S) | **23,889** | 1,258 | **989** | **893** | **1,913** |

Costs over 126 scored days, store + DC holding; extra lost sales = lost-sales cost above the
normal run. In the supplier outage the differences are within the 95% intervals.

**What I learned**

- The batches that make (s, S) cheap also make it robust: more stock on the shelf absorbs a shock
  the forecast did not see. The forecast policies are hit hardest by the unforecast demand spike.
- But (s, S) pushes problems upstream: the DC is out of stock on 17% of item-days and order
  variability rises from 0.17 to 0.27 between store and DC (bullwhip).
- Longer lead times are the worst disruption for every policy, and recovery takes 20–44 days.
- Extra DC stock for the top 20% of items is not worth it: it pays only above 9.3 two-week
  supplier outages a year, and it does nothing when the bottleneck is handling capacity.

Full analysis: [notebooks/04_digital_twin.ipynb](notebooks/04_digital_twin.ipynb)

## Phase 2 results: inventory policies

Four ordering policies were replayed day by day over 140 days (2016-01-04 to 2016-05-22), with
the same demand and the same random lead times (3–7 days), using only forecasts made before each day.

![Stock against service per policy](docs/inventory_tradeoff.png)

| Policy | Stock for 90% service | Stock for 95% service | Lowest total cost (at service) |
| --- | --- | --- | --- |
| Baseline: weeks of cover | 75,677 | not reachable (max 94.0%) | 23,737 (91.0%) |
| (R, S), textbook normal safety stock | 52,816 | not reachable (max 90.7%) | 30,702 (90.7%) |
| (R, S), calibrated forecast quantiles | **48,177** | **64,170** | 24,970 (96.5%) |
| (s, S), quantiles + economic order quantity | (always above 94%) | 91,313 | **17,430 (98.6%)** |

Stock is the average stock value; cost is holding + ordering + lost sales over 126 scored days.

**What I learned**

- Forecast quantiles beat the textbook safety stock: it promised 95% service and delivered 85%,
  because the forecast mean is biased low and past sales hide the demand lost to stockouts.
- A rule that copies past sales can get stuck: an item that sells nothing for a week is never
  reordered, so the baseline cannot reach 95% at any stock level.
- The forecast's gain is capital, not cost: holding is cheap (0.5% of price per week vs 30% for a
  lost sale), so the weekly forecast policy does not beat the best baseline on cost. Batching
  orders with (s, S) is what cuts the cost, by 72% fewer orders.
- The 95% service target was too low: with these costs, every forecast policy gets cheaper up to
  99.5% service.
- Not modelled: pack sizes, shelf space, supplier capacity and lead-time disruptions (Phase 3).

Full analysis: [notebooks/03_inventory.ipynb](notebooks/03_inventory.ipynb)

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
- Days when an item was out of stock and Christmas Day are excluded from scoring.
- Prediction intervals are calibrated per model and demand class; raw conformal intervals covered
  only about 82% at the 95th percentile.

**What I learned**

- LightGBM gives the best point forecast for items that sell every day, but AutoETS gives the
  sharpest quantiles in every class, so the Hybrid combines LightGBM's point with AutoETS's interval.
- The gain over a 28-day moving average is modest (3%): daily item-level demand is mostly noise.
- On lumpy items LightGBM beat AutoETS on the test window, the opposite of validation; the
  validation choice was kept rather than re-picking on test data.

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
4. Forecasting (ARIMA is the slow one, up to an hour on 2 cores):
```
   uv run python -m src.forecasting.run_backtest benchmarks
   uv run python -m src.forecasting.run_backtest stats
   uv run python -m src.forecasting.run_backtest arima
   uv run python -m src.forecasting.run_backtest lightgbm
   uv run python -m src.forecasting.calibrate
   uv run python -m src.forecasting.final_test
   uv run python -m src.forecasting.train_final
```
5. Inventory replay and sensitivity:
```
   uv run python -m src.inventory.run_replay
   uv run python -m src.inventory.sweep
```
6. Get a forecast in Python:
```python
from src.forecasting.forecast import forecast

forecast("FOODS_3_090", "CA_1", horizon=7)  # date, q50, q90, q95
```
7. Digital twin (about 30 minutes on 4 cores):
```
   uv run python -m src.simulation.run_twin
   uv run python -m src.simulation.mitigation
```
8. LLM agent: put an Anthropic API key in `.env` as `ANTHROPIC_API_KEY=...`, then
```
   uv run python -m src.agent.chat
   uv run python -m src.agent.run_eval
   uv run streamlit run app/streamlit_app.py
```
9. Run the tests with `uv run pytest` (no test calls the API).