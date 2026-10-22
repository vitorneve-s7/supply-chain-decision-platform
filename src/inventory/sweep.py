"""Service sweep and cost sensitivity: one curve of stock against service per policy.

Usage: uv run python -m src.inventory.sweep
Writes results/inventory_sweep.csv: one row per policy, setting and cost scenario.
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from src.inventory import policies
from src.inventory.inputs import build_inputs, draw_lead_times
from src.inventory.kpis import LOST_SALES_RATE, scorecard
from src.inventory.simulate import simulate

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

SERVICE_LEVELS = [0.80, 0.90, 0.95, 0.98, 0.99, 0.995]
COVER_WEEKS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
FAMILIES = {  # name: (policy, the parameter swept, its values)
    "Baseline (weeks of cover)": (policies.weeks_of_cover, "weeks", COVER_WEEKS),
    "(R, S) normal": (policies.order_up_to_normal, "service", SERVICE_LEVELS),
    "(R, S) quantile": (policies.order_up_to_quantile, "service", SERVICE_LEVELS),
    "(s, S)": (policies.reorder_point, "service", SERVICE_LEVELS),
}
BASE = {
    "holding_rate": policies.HOLDING_RATE,
    "order_cost": policies.ORDER_COST,
    "lost_sales_rate": LOST_SALES_RATE,
}
SCENARIOS = {  # one cost assumption changed at a time
    "base": {},
    "holding 15%": {"holding_rate": 0.15},
    "holding 35%": {"holding_rate": 0.35},
    "lost sales 10%": {"lost_sales_rate": 0.10},
    "lost sales 50%": {"lost_sales_rate": 0.50},
    "order cost 20": {"order_cost": 20.0},
}
COLUMNS = ["fill_rate", "cycle_service", "stock_value", "orders", "total_cost"]


def run_sweep(inputs: pd.DataFrame, lead_times: pd.DataFrame) -> pd.DataFrame:
    """Simulate every policy at every setting and score it under every cost scenario.

    Only (s, S) uses costs to decide (holding and order cost set its batch size Q), so it is
    re-simulated when those change; every other run is simulated once and re-scored.
    """
    rows = []
    for family, (policy, param, values) in FAMILIES.items():
        for value in values:
            runs = {}  # the costs that change decisions -> replay records
            for scenario, change in SCENARIOS.items():
                costs = {**BASE, **change}
                decide = {}
                if family == "(s, S)":
                    decide = {k: costs[k] for k in ("holding_rate", "order_cost")}
                key = tuple(decide.items())
                if key not in runs:
                    run_policy = partial(policy, **{param: value}, **decide)
                    runs[key] = simulate(inputs, run_policy, lead_times)
                card = scorecard(runs[key], **costs).loc["all", COLUMNS]
                rows.append({"policy": family, "setting": value, "scenario": scenario, **card})
            print(f"{family} {param}={value}: done")
    return pd.DataFrame(rows)


def stock_at_service(curve: pd.DataFrame, target: float) -> float:
    """Stock value a policy needs for `target` cycle service; NaN if its curve never gets there."""
    curve = curve.sort_values("cycle_service")
    if target > curve["cycle_service"].max() or target < curve["cycle_service"].min():
        return float("nan")
    return float(np.interp(target, curve["cycle_service"], curve["stock_value"]))


def main() -> None:
    inputs = build_inputs()
    lead_times = draw_lead_times(inputs)
    sweep = run_sweep(inputs, lead_times)
    sweep.to_csv(RESULTS / "inventory_sweep.csv", index=False, float_format="%.4f")

    base = sweep[sweep["scenario"] == "base"]
    for target in (0.90, 0.95):
        print(f"\nStock value needed for {target:.0%} cycle service:")
        for family, curve in base.groupby("policy"):
            print(f"  {family:28s} {stock_at_service(curve, target):>10,.0f}")

    best = sweep.loc[sweep.groupby(["scenario", "policy"])["total_cost"].idxmin()]
    print("\nLowest total cost per policy and scenario:")
    print(best.pivot(index="policy", columns="scenario", values="total_cost").round(0).to_string())


if __name__ == "__main__":
    main()
