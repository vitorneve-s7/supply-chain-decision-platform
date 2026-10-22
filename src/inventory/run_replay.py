"""Run every policy through the same replay and save the scorecards.

Usage: uv run python -m src.inventory.run_replay
Writes results/inventory_kpis.csv (all scored days) and results/inventory_kpis_test.csv
(the test window only), plus the tuned z per class in results/inventory_z.json.
"""

import json
import time
from functools import partial
from pathlib import Path

import pandas as pd

from src.inventory import policies
from src.inventory.inputs import build_inputs, draw_lead_times
from src.inventory.kpis import scorecard
from src.inventory.simulate import simulate
from src.inventory.tuning import TUNE_END, tune_z

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TEST_START = TUNE_END + pd.Timedelta(days=1)


def all_policies(z_by_class: dict[str, float]) -> dict:
    return {
        "Baseline": policies.replenish_last_week,
        "(R, S) normal": policies.order_up_to_normal,
        "(R, S) sum of q95": policies.order_up_to_quantile_sum,
        "(R, S) quantile": policies.order_up_to_quantile,
        "(R, S) quantile, tuned z": partial(policies.order_up_to_quantile, z_by_class=z_by_class),
        "(s, S) tuned z": partial(policies.reorder_point, z_by_class=z_by_class),
    }


def main() -> None:
    inputs = build_inputs()
    lead_times = draw_lead_times(inputs)

    start = time.time()
    z_by_class = tune_z(inputs, lead_times)
    (RESULTS / "inventory_z.json").write_text(json.dumps(z_by_class, indent=2))
    print(f"Tuned z per class in {time.time() - start:.0f} s: {z_by_class}")

    full, test = {}, {}
    for name, policy in all_policies(z_by_class).items():
        records = simulate(inputs, policy, lead_times)
        full[name] = scorecard(records)
        test[name] = scorecard(records, start=TEST_START)
        print(f"{name}: done")

    full = pd.concat(full, names=["policy", "demand_class"])
    test = pd.concat(test, names=["policy", "demand_class"])
    full.to_csv(RESULTS / "inventory_kpis.csv", float_format="%.4f")
    test.to_csv(RESULTS / "inventory_kpis_test.csv", float_format="%.4f")

    cols = ["fill_rate", "cycle_service", "days_of_stock", "stock_value", "orders", "total_cost"]
    print(full.xs("all", level="demand_class")[cols].round(3).to_string())


if __name__ == "__main__":
    main()
