"""Run every store policy through every scenario, 20 seeds each, in parallel; save the results.

Usage: uv run python -m src.simulation.run_twin            (all 821 items)
       uv run python -m src.simulation.run_twin --sample 200   (a stratified sample, ~4x faster)
Writes results/twin_runs.csv (one row per run), results/twin_daily.csv (daily store fill rate
around each disruption) and results/twin_summary.csv (mean and 95% interval per policy and
scenario).
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.inventory.inputs import build_inputs
from src.inventory.kpis import WARMUP_END
from src.inventory.policies import HOLDING_RATE
from src.simulation.scenarios import SCENARIOS, run_scenario, store_policies

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SEEDS = range(20)
RECOVERY_GAP = 0.01  # recovered once the 7-day fill rate is within 1 point of S0's


def sample_items(inputs: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Keep n items, the same share of each demand class as in the full set."""
    classes = inputs.groupby("unique_id")["demand_class"].first()
    share = n / len(classes)
    keep = classes.groupby(classes).sample(frac=share, random_state=seed).index
    return inputs[inputs["unique_id"].isin(keep)]


def daily_fill(store: pd.DataFrame) -> pd.Series:
    """Store fill rate per day, as a 7-day rolling average."""
    daily = store.groupby("ds")[["sales", "demand"]].sum()
    return (
        daily["sales"].rolling(7, min_periods=1).sum()
        / daily["demand"].rolling(7, min_periods=1).sum()
    )


def recovery_days(fill: pd.Series, normal: pd.Series, start: int) -> int:
    """Days from the disruption's start to the last day the fill rate was > 1 point below normal."""
    gap = (normal - fill).to_numpy()[start:]
    bad = np.flatnonzero(gap > RECOVERY_GAP)
    return 0 if len(bad) == 0 else int(bad[-1] + 1)


def dc_kpis(dc: pd.DataFrame, store: pd.DataFrame) -> dict:
    dc = dc[dc["ds"] >= WARMUP_END]
    store = store[store["ds"] >= WARMUP_END]
    weekly = lambda df, col: df.groupby(df["ds"].dt.to_period("W-SUN"))[col].sum()  # noqa: E731
    cv = lambda x: float(x.std() / x.mean()) if x.mean() > 0 else np.nan  # noqa: E731
    dc_value = dc["dc_end"] * dc["sell_price"]
    return {
        "dc_stock_value": float(dc_value.groupby(dc["ds"]).sum().mean()),
        "dc_holding_cost": float((dc_value * HOLDING_RATE / 365).sum()),
        "dc_short_share": float(((dc["dc_end"] < 1e-6) & (dc["dc_owed"] > 0)).mean()),
        "store_order_cv": cv(weekly(store, "order")),
        "dc_order_cv": cv(weekly(dc, "dc_order")),
    }


def run_pair(policy: str, seed: int, n_sample: int | None) -> tuple[list[dict], list[dict]]:
    """All five scenarios for one policy and seed; damage measured against S0 with that seed."""
    inputs = build_inputs()
    if n_sample:
        inputs = sample_items(inputs, n_sample)
    rows, daily_rows, normal = [], [], None
    for scenario in SCENARIOS:
        kpis, twin = run_scenario(policy, scenario, seed, inputs=inputs)
        store, dc = twin.store_records(), twin.dc_records()
        fill = daily_fill(store)
        row = {**kpis, **dc_kpis(dc, store), "lost_units": float(store["lost"].sum())}
        row["total_cost_with_dc"] = row["total_cost"] + row["dc_holding_cost"]
        if normal is None:  # S0 runs first
            normal = row | {"fill": fill}
        start = kpis["start_day"]
        row["extra_lost_units"] = row["lost_units"] - normal["lost_units"]
        row["extra_lost_cost"] = row["lost_sales_cost"] - normal["lost_sales_cost"]
        row["extra_total_cost"] = row["total_cost_with_dc"] - normal["total_cost_with_dc"]
        row["recovery_days"] = recovery_days(fill, normal["fill"], start) if start else 0
        rows.append(row)
        if start is not None:
            for t in range(len(fill)):
                daily_rows.append(
                    {
                        "policy": policy,
                        "scenario": scenario,
                        "seed": seed,
                        "day": t - start,
                        "fill": fill.iloc[t],
                        "normal_fill": normal["fill"].iloc[t],
                    }
                )
    return rows, daily_rows


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% interval half-width (1.96 x standard error) per policy and scenario."""
    cols = [
        "fill_rate",
        "cycle_service",
        "extra_lost_units",
        "extra_lost_cost",
        "extra_total_cost",
        "recovery_days",
        "total_cost_with_dc",
        "dc_short_share",
    ]
    g = runs.groupby(["policy", "scenario"])[cols]
    mean, sem = g.mean(), g.std() / np.sqrt(g.count())
    return mean.join((1.96 * sem).add_suffix("_ci")).reset_index()


def main(n_sample: int | None, workers: int) -> None:
    jobs = [(p, s, n_sample) for p in store_policies() for s in SEEDS]
    start = time.time()
    rows, daily = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (r, d) in enumerate(pool.map(run_pair, *zip(*jobs)), start=1):
            rows += r
            daily += d
            print(f"{i}/{len(jobs)} done, {time.time() - start:.0f} s")
    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS / "twin_runs.csv", index=False, float_format="%.4f")
    pd.DataFrame(daily).to_csv(RESULTS / "twin_daily.csv", index=False, float_format="%.4f")
    summary = summarise(runs)
    summary.to_csv(RESULTS / "twin_summary.csv", index=False, float_format="%.4f")
    cols = ["policy", "scenario", "extra_lost_cost", "extra_lost_cost_ci", "recovery_days"]
    print(summary[cols].round(1).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="number of items to sample")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    main(args.sample, args.workers)
