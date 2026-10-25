"""The agent's tools: plain functions with simple inputs that return small JSON-ready dicts.

Every number the agent states comes from one of these. Errors are returned as data
({"error": ..., "valid": [...]}) so the model can correct itself.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.forecast import forecast
from src.inventory.policies import (
    PROTECTION,
    order_up_to_normal,
    order_up_to_quantile,
    quantile_level,
    reorder_point,
    weeks_of_cover,
)
from src.inventory.simulate import MAX_PROTECTION, ReviewState
from src.simulation.api import simulate_scenario
from src.simulation.scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
ASSUMPTIONS = ROOT / "docs" / "assumptions.md"

STORE = "CA_1"
POLICIES = ["Weeks of cover", "(R, S) normal", "(R, S) quantile", "(s, S)"]
REVIEW_DAY = pd.Timestamp("2016-05-23")  # the first Monday after the data ends
HISTORY_DAYS = 56  # recent sales the policies and find_items look at


# ---------- data, loaded once ----------


@lru_cache(maxsize=1)
def _sales() -> pd.DataFrame:
    """Daily sales and prices of every item over the last HISTORY_DAYS days of data."""
    df = pd.read_parquet(PROCESSED / "sales_long.parquet")
    df = df[df["date"] > df["date"].max() - pd.Timedelta(days=HISTORY_DAYS)]
    return df[["item_id", "date", "sales", "sell_price"]].sort_values(["item_id", "date"])


@lru_cache(maxsize=1)
def _classes() -> pd.Series:
    classes = pd.read_parquet(PROCESSED / "item_classes.parquet")
    return classes.set_index("item_id")["demand_class"]


@lru_cache(maxsize=1)
def _z_by_class() -> dict:
    return json.loads((RESULTS / "inventory_z.json").read_text())


def _unknown_item(item: str) -> dict | None:
    if item in _classes().index:
        return None
    return {
        "error": f"Unknown item {item!r}.",
        "hint": "Item ids look like FOODS_3_090; use find_items to list them.",
    }


def _r(x: float, digits: int = 2) -> float:
    return round(float(x), digits)


# ---------- the six tools ----------


def get_forecast(item: str, days: int = 7) -> dict:
    """Daily demand forecast for one item over the next 1-28 days after 2016-05-22."""
    if error := _unknown_item(item):
        return error
    if not 1 <= days <= 28:
        return {"error": f"days must be between 1 and 28, got {days}."}
    fc = forecast(item, STORE, horizon=days)
    spread = fc["q95"] - fc["q50"]  # as in Phase 2: daily spreads add up like variances
    return {
        "item": item,
        "store": STORE,
        "demand_class": _classes()[item],
        "from": str(fc["date"].iloc[0].date()),
        "to": str(fc["date"].iloc[-1].date()),
        "days": days,
        "q50_total_units": _r(fc["q50"].sum(), 1),
        "q95_total_units": _r(fc["q50"].sum() + np.sqrt((spread**2).sum()), 1),
        "daily": [
            {"date": str(r.date.date()), "q50": _r(r.q50), "q90": _r(r.q90), "q95": _r(r.q95)}
            for r in fc.itertuples()
        ],
    }


def get_inventory_policy(item: str, policy: str = "(s, S)") -> dict:
    """Stock levels one policy sets for one item at the review on 2016-05-23."""
    if error := _unknown_item(item):
        return error
    if policy not in POLICIES:
        return {"error": f"Unknown policy {policy!r}.", "valid": POLICIES}
    sales = _sales()[lambda d: d["item_id"] == item]
    fc = forecast(item, STORE, horizon=MAX_PROTECTION)[["q50", "q90", "q95"]]
    state = ReviewState(
        day=REVIEW_DAY,
        on_hand=0.0,  # with nothing in stock, the order equals the order-up-to level
        on_order=0.0,
        forecast=fc,
        past_sales=sales["sales"].to_numpy(dtype=float),
        price=float(sales["sell_price"].iloc[-1]),
        demand_class=_classes()[item],
    )
    z = _z_by_class()
    daily = fc["q50"].head(PROTECTION).mean()
    out = {
        "item": item,
        "policy": policy,
        "review_day": str(REVIEW_DAY.date()),
        "demand_class": state.demand_class,
        "price_usd": _r(state.price),
        "forecast_daily_units": _r(daily),
    }
    if policy == "Weeks of cover":
        level = weeks_of_cover(state, weeks=3)
        out["rule"] = "order up to 3 x last week's sales"
        out["last_week_sales_units"] = _r(state.past_sales[-7:].sum(), 0)
    elif policy == "(R, S) normal":
        level = order_up_to_normal(state)
        out["rule"] = "weekly review; 12 days of forecast + safety stock from past sales, z = 1.645"
    elif policy == "(R, S) quantile":
        level = order_up_to_quantile(state, z_by_class=z)
        out["rule"] = "weekly review; 12 days of forecast + safety stock from forecast quantiles"
        out["z"] = z[state.demand_class]
    else:
        s = quantile_level(state, z_by_class=z)
        level = reorder_point(state, z_by_class=z)
        out["rule"] = "weekly review; order only when stock + on order <= s, then up to S"
        out["z"] = z[state.demand_class]
        out["reorder_point_s_units"] = _r(s, 1)
        out["batch_eoq_units"] = _r(level - s, 1)
    out["order_up_to_S_units"] = int(level)
    out["S_in_days_of_demand"] = _r(level / daily, 1) if daily > 0 else None
    out["S_stock_value_usd"] = _r(level * state.price)
    return out


def compare_policies(scenario: str = "all") -> dict:
    """The four store policies under one disruption scenario (or all), from the 20-seed runs."""
    if scenario != "all" and scenario not in SCENARIOS:
        return {"error": f"Unknown scenario {scenario!r}.", "valid": ["all", *SCENARIOS]}
    summary = pd.read_csv(RESULTS / "twin_summary.csv")
    if scenario != "all":
        summary = summary[summary["scenario"] == scenario]
    rows = [
        {
            "policy": r.policy,
            "scenario": r.scenario,
            "fill_rate": _r(r.fill_rate, 4),
            "cycle_service": _r(r.cycle_service, 4),
            "extra_lost_sales_cost_usd": _r(r.extra_lost_cost, 0),
            "extra_lost_sales_cost_ci_usd": _r(r.extra_lost_cost_ci, 0),
            "total_cost_with_dc_usd": _r(r.total_cost_with_dc, 0),
            "total_cost_with_dc_ci_usd": _r(r.total_cost_with_dc_ci, 0),
            "recovery_days": _r(r.recovery_days, 1),
            "recovery_days_ci": _r(r.recovery_days_ci, 1),
        }
        for r in summary.sort_values(["scenario", "total_cost_with_dc"]).itertuples()
    ]
    return {
        "source": "digital twin, 821 items, 20 seeds per policy and scenario, 126 scored days",
        "notes": "extra = vs the normal run with the same seed; ci = 95% interval half-width",
        "rows": rows,
    }


def run_scenario(policy: str, scenario: str, seed: int = 0, item: str | None = None) -> dict:
    """One live run of the twin, next to the normal run with the same seed."""
    if item and (error := _unknown_item(item)):
        return error
    try:
        result = simulate_scenario(policy, scenario, seed, item=item)
        normal = simulate_scenario(policy, "S0 normal", seed, item=item)
    except ValueError as e:
        return {"error": str(e)}
    change = {
        k: _r(result[k] - normal[k], 4)
        for k in ["fill_rate", "cycle_service", "lost_sales_cost", "total_cost_with_dc"]
    }
    return {"scenario_run": result, "normal_run": normal, "change_vs_normal": change}


def find_items(demand_class: str | None = None, sort_by: str = "sales_value", n: int = 10) -> dict:
    """Items ranked by units sold or sales value over the last 56 days of data."""
    classes = sorted(_classes().unique())
    if demand_class and demand_class not in classes:
        return {"error": f"Unknown demand class {demand_class!r}.", "valid": classes}
    if sort_by not in ("units", "sales_value"):
        return {"error": f"Unknown sort_by {sort_by!r}.", "valid": ["units", "sales_value"]}
    sales = _sales().assign(value=lambda d: d["sales"] * d["sell_price"])
    table = sales.groupby("item_id").agg(units=("sales", "sum"), sales_value=("value", "sum"))
    table["demand_class"] = _classes()
    if demand_class:
        table = table[table["demand_class"] == demand_class]
    table = table.sort_values(sort_by, ascending=False).head(max(1, min(n, 25)))
    return {
        "period": f"last {HISTORY_DAYS} days of data, to 2016-05-22",
        "items_in_scope": int(
            (_classes() == demand_class).sum() if demand_class else len(_classes())
        ),
        "rows": [
            {
                "rank": i,
                "item": item,
                "demand_class": r.demand_class,
                "units": int(r.units),
                "sales_value_usd": _r(r.sales_value),
            }
            for i, (item, r) in enumerate(table.iterrows(), start=1)
        ],
    }


STOPWORDS = set(
    "the and for from why what how with are was doe did this that which when into use used about "
    "there its ha have can will not you your our".split()
)


def _words(text: str) -> set[str]:
    """Lower-case words of 3+ letters, with a plural s dropped ("costs" -> "cost")."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if len(w) > 2}


@lru_cache(maxsize=1)
def _assumption_bullets() -> list[tuple[str, str]]:
    """(section, bullet text) for every bullet of docs/assumptions.md."""
    bullets, section = [], ""
    for line in ASSUMPTIONS.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        elif line.startswith("- "):
            bullets.append((section, line[2:].strip()))
        elif line.startswith("  ") and bullets:  # continuation of the last bullet
            bullets[-1] = (bullets[-1][0], bullets[-1][1] + " " + line.strip())
    return bullets


def explain_assumption(query: str) -> dict:
    """The 3 bullets of docs/assumptions.md that share most words with the query.

    A bullet's section name counts as part of it; ties go to the shorter, more focused bullet.
    """
    words = _words(query) - STOPWORDS
    if not words:
        return {"error": "Give a few topic words, e.g. 'lost sales cost'."}
    scored = [
        (len(words & _words(section + " " + text)), len(text), section, text)
        for section, text in _assumption_bullets()
    ]
    best = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))[:3]
    if not best:
        return {"error": "No assumption matches these words.", "query": query}
    return {"matches": [{"section": s, "text": t} for _, _, s, t in best]}


# ---------- dispatch ----------

TOOLS = {
    "get_forecast": get_forecast,
    "get_inventory_policy": get_inventory_policy,
    "compare_policies": compare_policies,
    "run_scenario": run_scenario,
    "find_items": find_items,
    "explain_assumption": explain_assumption,
}


def run_tool(name: str, inputs: dict) -> dict:
    """Run a tool by name; any failure comes back as an error the model can read."""
    if name not in TOOLS:
        return {"error": f"Unknown tool {name!r}.", "valid": list(TOOLS)}
    try:
        return TOOLS[name](**inputs)
    except TypeError as e:  # wrong or missing inputs
        return {"error": f"Bad inputs for {name}: {e}"}
