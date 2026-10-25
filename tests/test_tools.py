import inspect

import pytest

from src.agent import tools
from src.agent.schemas import TOOL_SCHEMAS
from src.forecasting.forecast import forecast

ITEM = "FOODS_3_090"


def test_every_schema_matches_its_function():
    assert [s["name"] for s in TOOL_SCHEMAS] == list(tools.TOOLS)
    for schema in TOOL_SCHEMAS:
        params = inspect.signature(tools.TOOLS[schema["name"]]).parameters
        assert set(schema["input_schema"]["properties"]) == set(params)
        assert len(schema["description"]) > 50  # the model chooses tools by these


def test_forecast_totals_match_the_forecast_function():
    out = tools.get_forecast(ITEM, days=7)
    fc = forecast(ITEM, "CA_1", horizon=7)
    assert out["q50_total_units"] == pytest.approx(fc["q50"].sum(), abs=0.05)
    assert out["q50_total_units"] < out["q95_total_units"] <= fc["q95"].sum() + 0.05
    assert len(out["daily"]) == 7 and out["from"] == "2016-05-23"


def test_policy_levels_are_consistent():
    s_s = tools.get_inventory_policy(ITEM, "(s, S)")
    quantile = tools.get_inventory_policy(ITEM, "(R, S) quantile")
    assert s_s["order_up_to_S_units"] == pytest.approx(
        s_s["reorder_point_s_units"] + s_s["batch_eoq_units"], abs=1
    )
    # (s, S) reorders at the (R, S) quantile level, then adds a batch on top
    assert s_s["reorder_point_s_units"] == pytest.approx(quantile["order_up_to_S_units"], abs=1)


def test_compare_policies_reads_the_twin_summary():
    out = tools.compare_policies("S2 demand spike")
    assert len(out["rows"]) == 4
    cheapest = out["rows"][0]  # sorted by total cost
    assert cheapest["policy"] == "(s, S)"


def test_find_items_ranks_and_filters():
    out = tools.find_items(demand_class="smooth", sort_by="units", n=5)
    units = [r["units"] for r in out["rows"]]
    assert units == sorted(units, reverse=True) and len(units) == 5
    assert all(r["demand_class"] == "smooth" for r in out["rows"])


def test_explain_assumption_finds_the_cost_section():
    out = tools.explain_assumption("why is the lost sales cost 30%")
    assert out["matches"][0]["section"] == "5. Costs"


def test_run_scenario_on_one_item_compares_with_normal():
    out = tools.run_scenario("(s, S)", "S1 supplier outage", seed=0, item=ITEM)
    assert out["scenario_run"]["disruption_start"] is not None
    assert out["normal_run"]["disruption_start"] is None
    change = out["scenario_run"]["fill_rate"] - out["normal_run"]["fill_rate"]
    assert out["change_vs_normal"]["fill_rate"] == pytest.approx(change, abs=1e-4)


@pytest.mark.parametrize(
    "name, inputs",
    [
        ("get_forecast", {"item": "FOODS_9_999"}),
        ("get_forecast", {"item": ITEM, "days": 40}),
        ("get_inventory_policy", {"item": ITEM, "policy": "(s,S)"}),
        ("compare_policies", {"scenario": "S9 meteor"}),
        ("run_scenario", {"policy": "(s, S)", "scenario": "S1", "item": ITEM}),
        ("find_items", {"demand_class": "fast"}),
        ("get_forecast", {"item": ITEM, "dayz": 3}),
        ("no_such_tool", {}),
    ],
)
def test_bad_inputs_come_back_as_errors(name, inputs):
    out = tools.run_tool(name, inputs)
    assert "error" in out


def test_compare_policies_ranks_within_each_scenario():
    rows = tools.compare_policies("S1 supplier outage")["rows"]
    by_cost = sorted(rows, key=lambda r: r["total_cost_with_dc_usd"])
    assert [r["rank_total_cost"] for r in by_cost] == [1, 2, 3, 4]
    least_lost = min(rows, key=lambda r: r["extra_lost_sales_cost_usd"])
    assert least_lost["rank_extra_lost_sales"] == 1
