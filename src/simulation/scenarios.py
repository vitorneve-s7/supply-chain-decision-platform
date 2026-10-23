"""Disruption scenarios and run_scenario(): one call = one run of the twin, returning KPIs."""

import json
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import pandas as pd

from src.inventory import policies
from src.inventory.inputs import build_inputs, draw_lead_times
from src.inventory.kpis import scorecard
from src.simulation.config import NetworkConfig
from src.simulation.dc import dc_order_up_to
from src.simulation.twin import Twin

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Disruptions start on a random replay day between 2016-02-08 (day 35) and 2016-04-04 (day 91)
FIRST_START, LAST_START = 35, 91

SCENARIOS = {  # name: (duration in days, the change)
    "S0 normal": (0, {}),
    "S1 supplier outage": (14, {"supplier_down": True}),
    "S2 demand spike": (10, {"demand_multiplier": 1.5}),
    "S3 DC capacity cut": (21, {"dc_capacity_multiplier": 0.5}),
    "S4 longer lead times": (28, {"store_lead_multiplier": 2.0}),
}


def store_policies() -> dict:
    """The four Phase 2 policies at their best Phase 2 settings, by name."""
    z = json.loads((RESULTS / "inventory_z.json").read_text())
    return {
        "Weeks of cover": partial(policies.weeks_of_cover, weeks=3),
        "(R, S) normal": policies.order_up_to_normal,
        "(R, S) quantile": partial(policies.order_up_to_quantile, z_by_class=z),
        "(s, S)": partial(policies.reorder_point, z_by_class=z),
    }


@lru_cache(maxsize=1)
def _inputs() -> pd.DataFrame:
    return build_inputs()


def start_day(seed: int) -> int:
    """The disruption's first replay day for this seed: the same for every policy."""
    return int(np.random.default_rng(seed).integers(FIRST_START, LAST_START + 1))


def scenario_config(scenario: str, seed: int) -> NetworkConfig:
    duration, change = SCENARIOS[scenario]
    if duration == 0:
        return NetworkConfig()
    return NetworkConfig(start_day=start_day(seed), duration=duration, **change)


def run_scenario(
    policy: str,
    scenario: str,
    seed: int = 0,
    inputs: pd.DataFrame | None = None,
    dc_policy=dc_order_up_to,
) -> tuple[dict, Twin]:
    """Run the twin once and return its KPIs and the twin itself (for the daily records).

    policy: a name from store_policies(), e.g. "(s, S)".
    scenario: a name from SCENARIOS, e.g. "S1 supplier outage".
    seed: sets the lead times on both legs and the disruption start; the same seed gives the
    same draws for every policy.
    dc_policy: the DC ordering rule; by default dc_order_up_to at 95% service.
    """
    inputs = _inputs() if inputs is None else inputs
    config = scenario_config(scenario, seed)
    twin = Twin(
        inputs,
        store_policies()[policy],
        dc_policy,
        draw_lead_times(inputs, seed=seed),
        config,
        seed=seed,
    ).run()
    card = scorecard(twin.store_records()).loc["all"]
    kpis = {
        "policy": policy,
        "scenario": scenario,
        "seed": seed,
        "start_day": config.start_day,
        **card.to_dict(),
    }
    return kpis, twin
