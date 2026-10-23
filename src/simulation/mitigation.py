"""Mitigation: a bigger DC safety stock for the A items, and the break-even rate of disruptions.

Usage: uv run python -m src.simulation.mitigation
Writes results/twin_mitigation.csv (one row per run) and prints the value of the buffer.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd

from src.inventory.inputs import build_inputs
from src.simulation.dc import dc_order_up_to
from src.simulation.run_twin import SEEDS, dc_kpis, sample_items
from src.simulation.scenarios import run_scenario

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

POLICY = "(s, S)"  # the most robust store policy in Part G
SCENARIOS = ["S0 normal", "S1 supplier outage", "S3 DC capacity cut"]
A_SHARE = 0.20  # the top 20% of items by sales value
BUFFER_SERVICE = 0.99  # DC service for A items with the buffer (z = 2.33 instead of 1.65)
REPLAY_DAYS = 126  # scored days: costs are annualised with 365 / 126


def a_items(inputs: pd.DataFrame, share: float = A_SHARE) -> list[str]:
    """Items with the highest sales value before the replay (demand x price)."""
    history = inputs[~inputs["in_replay"]]
    value = (history["demand"] * history["sell_price"]).groupby(history["unique_id"]).sum()
    n = max(1, round(share * len(value)))
    return value.sort_values(ascending=False).head(n).index.tolist()


def run_seed(seed: int, n_sample: int | None) -> list[dict]:
    """Without and with the buffer, all three scenarios, for one seed."""
    inputs = build_inputs()
    if n_sample:
        inputs = sample_items(inputs, n_sample)
    buffer = dict.fromkeys(a_items(inputs), BUFFER_SERVICE)
    rows = []
    for name, dc_policy in [
        ("No buffer", dc_order_up_to),
        ("A-item buffer", partial(dc_order_up_to, service_by_item=buffer)),
    ]:
        for scenario in SCENARIOS:
            kpis, twin = run_scenario(POLICY, scenario, seed, inputs=inputs, dc_policy=dc_policy)
            extra = dc_kpis(twin.dc_records(), twin.store_records())
            row = {**kpis, **extra, "mitigation": name}
            row["total_cost_with_dc"] = row["total_cost"] + row["dc_holding_cost"]
            rows.append(row)
    return rows


def break_even(runs: pd.DataFrame) -> pd.DataFrame:
    """Per disruption: damage avoided by the buffer, its yearly cost, and the break-even rate."""
    cost = runs.pivot_table(
        index=["seed", "scenario"], columns="mitigation", values="total_cost_with_dc"
    )
    cost["saving"] = cost["No buffer"] - cost["A-item buffer"]  # positive = the buffer helps
    per_seed = cost["saving"].unstack("scenario")
    normal = per_seed.pop("S0 normal")  # in normal times: usually negative (extra holding)
    yearly_cost = -normal.mean() * 365 / REPLAY_DAYS
    rows = []
    for scenario in per_seed.columns:
        avoided = (per_seed[scenario] - normal).mean()  # damage avoided per disruption
        rows.append(
            {
                "scenario": scenario,
                "damage_avoided_per_disruption": avoided,
                "buffer_cost_per_year": yearly_cost,
                "break_even_disruptions_per_year": yearly_cost / avoided if avoided > 0 else None,
            }
        )
    return pd.DataFrame(rows)


def main(n_sample: int | None, workers: int) -> None:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(run_seed, SEEDS, [n_sample] * len(SEEDS)), start=1):
            rows += r
            print(f"{i}/{len(SEEDS)} seeds done")
    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS / "twin_mitigation.csv", index=False, float_format="%.4f")
    cols = [
        "fill_rate",
        "cycle_service",
        "lost_sales_cost",
        "dc_stock_value",
        "dc_holding_cost",
        "total_cost_with_dc",
    ]
    print(runs.groupby(["scenario", "mitigation"])[cols].mean().round(1).to_string())
    print(break_even(runs).round(2).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="number of items to sample")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    main(args.sample, args.workers)
