# Supply Chain Decision Platform

End-to-end retail supply chain project on the M5 (Walmart) dataset:
demand forecasting → inventory policies → supply chain digital twin → LLM planning assistant.

## Status

- [x] Phase 0 – Setup and data exploration
- [ ] Phase 1 – Demand forecasting
- [ ] Phase 2 – Inventory policies
- [ ] Phase 3 – Digital twin
- [ ] Phase 4 – LLM agent

## Scope

Store CA_1, department FOODS_3: 823 items, daily sales from 2011-01-29 to 2016-05-22.
All assumptions are in [docs/assumptions.md](docs/assumptions.md).

## Key findings so far

- 80% of items have intermittent or lumpy demand; only 14% sell steadily every day.
- Weekends sell ~55–60% more than midweek; SNAP days add ~12%.
- 84% of fast-moving items show long zero-sales runs, pointing to stockouts or unavailability.

Full analysis: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)

## How to run

1. Install [uv](https://docs.astral.sh/uv/) and run `uv sync`.
2. Download the [M5 data](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) into `data/raw/`.
3. Run `uv run python src/data/prepare.py`.
4. Open the notebooks in `notebooks/`.