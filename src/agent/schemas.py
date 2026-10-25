"""Tool definitions for the Anthropic API: what the model reads to choose a tool and its inputs."""

from src.agent.tools import POLICIES
from src.simulation.scenarios import SCENARIOS

ITEM = {"type": "string", "description": "Item id in department FOODS_3, e.g. FOODS_3_090"}
POLICY = {
    "type": "string",
    "enum": POLICIES,
    "description": "Store ordering policy. (s, S) was the cheapest and most robust in testing.",
}

TOOL_SCHEMAS = [
    {
        "name": "get_forecast",
        "description": (
            "Demand forecast for one item in store CA_1 for the next 1-28 days after the data ends "
            "(2016-05-23 onwards). Returns the total median forecast (q50), a 95% upper total "
            "(q95), and daily q50/q90/q95. Use for any 'how much will X sell' question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item": ITEM,
                "days": {"type": "integer", "minimum": 1, "maximum": 28, "default": 7},
            },
            "required": ["item"],
        },
    },
    {
        "name": "get_inventory_policy",
        "description": (
            "Stock levels a policy sets for one item at the weekly review on 2016-05-23: the "
            "order-up-to level S (units, days of demand, stock value), plus the reorder point s "
            "and the batch size for (s, S). Use for 'how much stock / when to reorder' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"item": ITEM, "policy": POLICY},
            "required": ["item"],
        },
    },
    {
        "name": "compare_policies",
        "description": (
            "Precomputed comparison of the four store policies across disruption scenarios, from "
            "the digital twin: all 821 items, 20 runs each. Per policy: fill rate, cycle service, "
            "extra lost-sales cost vs normal, total cost incl. DC holding, days to recover, each "
            "with a 95% interval. Instant. Prefer this for any store-level 'which policy' or "
            "'what does a disruption cost' question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario": {"type": "string", "enum": ["all", *SCENARIOS], "default": "all"}
            },
        },
    },
    {
        "name": "run_scenario",
        "description": (
            "Run the supply chain digital twin live, once, for one policy and scenario, next to "
            "the normal run with the same seed. With an item: that item only, about 2 seconds. "
            "Without an item: all 821 items, about 40 seconds, so use compare_policies instead "
            "unless the user asks for a specific seed. Scenarios: S1 supplier ships nothing for "
            "14 days; S2 demand +50% for 10 days; S3 DC capacity -50% for 21 days; S4 DC-to-store "
            "lead time doubled for 28 days."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "policy": POLICY,
                "scenario": {"type": "string", "enum": list(SCENARIOS)},
                "seed": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 19,
                    "default": 0,
                    "description": "Sets lead times and the disruption start date",
                },
                "item": ITEM,
            },
            "required": ["policy", "scenario"],
        },
    },
    {
        "name": "find_items",
        "description": (
            "Rank items by units sold or sales value over the last 56 days of data, optionally "
            "within one demand class. Use to find item ids (there are no product names in the "
            "data) or the top sellers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "demand_class": {
                    "type": "string",
                    "enum": ["smooth", "erratic", "intermittent", "lumpy"],
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["units", "sales_value"],
                    "default": "sales_value",
                },
                "n": {"type": "integer", "minimum": 1, "maximum": 25, "default": 10},
            },
        },
    },
    {
        "name": "explain_assumption",
        "description": (
            "Search the project's documented assumptions (costs, lead times, data cleaning, "
            "models, policies, the twin's design and results) and return the 3 best-matching "
            "passages. Use for 'why' and 'how was this set up' questions. Query in English."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A few topic words in English"}
            },
            "required": ["query"],
        },
    },
]
