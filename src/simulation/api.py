"""The twin as one plain function, ready to be wrapped as a tool by the Phase 4 agent."""

import pandas as pd

from src.simulation.run_twin import dc_kpis
from src.simulation.scenarios import SCENARIOS, run_scenario, store_policies


def simulate_scenario(
    policy: str, scenario: str, seed: int = 0, inputs: pd.DataFrame | None = None
) -> dict:
    """Run the supply chain digital twin once and return its headline KPIs.

    policy: the store ordering policy, one of "Weeks of cover", "(R, S) normal",
        "(R, S) quantile", "(s, S)".
    scenario: one of "S0 normal", "S1 supplier outage" (14 days), "S2 demand spike" (+50%,
        10 days), "S3 DC capacity cut" (-50%, 21 days), "S4 longer lead times" (x2, 28 days).
    seed: 0-19; sets the lead times and the disruption start, the same for every policy.
    inputs: only for tests; by default all 821 items.

    Returns plain numbers over the 126 scored days (2016-01-18 to 2016-05-22): fill rate and
    cycle service (0-1), lost sales cost, total cost including DC holding, average store and DC
    stock value, and the disruption's start date (None in S0).
    """
    if policy not in store_policies():
        raise ValueError(f"Unknown policy {policy!r}; choose from {list(store_policies())}")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {list(SCENARIOS)}")
    kpis, twin = run_scenario(policy, scenario, seed, inputs=inputs)
    dc = dc_kpis(twin.dc_records(), twin.store_records())
    start = kpis["start_day"]
    return {
        "policy": policy,
        "scenario": scenario,
        "seed": seed,
        "disruption_start": None if start is None else str(twin.dates[start].date()),
        "fill_rate": round(float(kpis["fill_rate"]), 4),
        "cycle_service": round(float(kpis["cycle_service"]), 4),
        "lost_sales_cost": round(float(kpis["lost_sales_cost"]), 2),
        "total_cost_with_dc": round(float(kpis["total_cost"] + dc["dc_holding_cost"]), 2),
        "store_stock_value": round(float(kpis["stock_value"]), 2),
        "dc_stock_value": round(dc["dc_stock_value"], 2),
    }
