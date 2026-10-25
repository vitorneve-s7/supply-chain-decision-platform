"""Supply Chain Decision Platform: ask the agent, see the numbers behind its answer.

Usage: uv run streamlit run app/streamlit_app.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.agent.agent import ask  # noqa: E402
from src.agent.charts import comparison_figure, forecast_figure  # noqa: E402

REPO = "https://github.com/vitorneve-s7/supply-chain-decision-platform"
EXAMPLES = [
    "How much will FOODS_3_090 sell next week?",
    "Which policy loses the least sales in a 2-week supplier outage?",
    "Que nível de stock devo ter do FOODS_3_120 com a política (s, S)?",
    "Run a demand spike for FOODS_3_702 with weeks of cover.",
    "Why is the lost-sales cost 30% of the price?",
]

st.set_page_config(page_title="Supply Chain Decision Platform", page_icon="📦", layout="wide")

try:  # on Streamlit Cloud the key comes from the app's secrets; locally from .env
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except FileNotFoundError:
    pass

if "turns" not in st.session_state:
    st.session_state.turns = []  # one dict per question: question, answer
    st.session_state.history = None  # the conversation sent back for follow-ups


# ---------- sidebar ----------

with st.sidebar:
    st.title("Supply Chain Decision Platform")
    st.caption(
        "Walmart store CA_1, department FOODS_3 (M5 data to 2016-05-22). "
        "Forecasts cover 2016-05-23 to 2016-06-19."
    )
    st.markdown("**Try a question**")
    for example in EXAMPLES:
        if st.button(example, width="stretch"):
            st.session_state.pending = example
    if st.button("Clear the chat", type="secondary"):
        st.session_state.turns, st.session_state.history = [], None
    st.markdown(
        "Every number in an answer comes from the project's own forecasting, inventory and "
        f"simulation code, and is checked automatically. [Code and results]({REPO})"
    )


# ---------- the right panel: what the last tool returned ----------


def show_result(call) -> None:
    result = call.result
    if "error" in result:
        st.info("The last tool returned an error; the answer explains it.")
    elif call.name == "get_forecast":
        st.plotly_chart(forecast_figure(result), width="stretch")
        a, b = st.columns(2)
        a.metric(f"Median, {result['days']} days", f"{result['q50_total_units']:,.0f} units")
        b.metric("95% upper total", f"{result['q95_total_units']:,.0f} units")
    elif call.name == "compare_policies":
        st.plotly_chart(comparison_figure(result), width="stretch")
    elif call.name == "run_scenario":
        run, normal = result["scenario_run"], result["normal_run"]
        st.subheader(f"{run['scenario']}: {run['item']}, {run['policy']}")
        st.caption(f"Seed {run['seed']}, disruption from {run['disruption_start']}")
        a, b, c = st.columns(3)
        a.metric(
            "Fill rate", f"{run['fill_rate']:.1%}", f"{run['fill_rate'] - normal['fill_rate']:+.1%}"
        )
        b.metric(
            "Lost-sales cost",
            f"${run['lost_sales_cost']:,.0f}",
            f"{run['lost_sales_cost'] - normal['lost_sales_cost']:+,.0f}",
            delta_color="inverse",
        )
        c.metric(
            "Total cost incl. DC",
            f"${run['total_cost_with_dc']:,.0f}",
            f"{run['total_cost_with_dc'] - normal['total_cost_with_dc']:+,.0f}",
            delta_color="inverse",
        )
        st.caption("Change vs the normal run with the same seed.")
    elif call.name == "get_inventory_policy":
        st.subheader(f"{result['item']}: {result['policy']}")
        st.caption(result["rule"])
        a, b, c = st.columns(3)
        a.metric("Order-up-to S", f"{result['order_up_to_S_units']:,} units")
        b.metric("Days of demand", f"{result['S_in_days_of_demand']}")
        c.metric("Stock value", f"${result['S_stock_value_usd']:,.0f}")
        if "reorder_point_s_units" in result:
            a, b, _ = st.columns(3)
            a.metric("Reorder point s", f"{result['reorder_point_s_units']:,.0f} units")
            b.metric("Batch (EOQ)", f"{result['batch_eoq_units']:,.0f} units")
    elif call.name == "find_items":
        st.subheader("Items")
        st.dataframe(pd.DataFrame(result["rows"]), hide_index=True, width="stretch")
    elif call.name == "explain_assumption":
        st.subheader("From docs/assumptions.md")
        for match in result["matches"]:
            st.markdown(f"**{match['section']}**: {match['text']}".replace("$", r"\$"))


def show_answer(answer) -> None:
    st.markdown(answer.text.replace("$", r"\$"))  # a bare $ would start a LaTeX formula
    if answer.ungrounded:
        st.warning(f"Numbers no tool supports: {', '.join(answer.ungrounded)}")
    show_tools(answer)


def show_tools(answer) -> None:
    with st.expander(f"Tools used ({len(answer.tool_calls)})"):
        for call in answer.tool_calls:
            st.markdown(f"`{call.name}` {call.inputs} ({call.seconds:.1f} s)")
            st.json(call.result, expanded=False)
        st.caption(f"{answer.input_tokens:,} input / {answer.output_tokens:,} output tokens")


# ---------- layout ----------

chat, panel = st.columns([3, 2], gap="large")

question = st.chat_input("Ask about forecasts, stock levels or disruptions")
question = question or st.session_state.pop("pending", None)

with chat:
    for turn in st.session_state.turns:
        st.chat_message("user").write(turn["question"])
        with st.chat_message("assistant"):
            show_answer(turn["answer"])

    if question:
        st.chat_message("user").write(question)
        with st.chat_message("assistant"):
            with st.status("Thinking...", expanded=False) as status:
                answer = ask(
                    question,
                    st.session_state.history,
                    on_tool_call=lambda c: status.write(f"`{c.name}` ({c.seconds:.1f} s)"),
                )
                status.update(label=f"Done: {len(answer.tool_calls)} tool calls", state="complete")
            show_answer(answer)
        st.session_state.turns.append({"question": question, "answer": answer})
        st.session_state.history = answer.messages

with panel:
    calls = [c for t in st.session_state.turns for c in t["answer"].tool_calls]
    if calls:
        show_result(calls[-1])
    else:
        st.subheader("What the twin found")
        st.image(str(ROOT / "docs" / "twin_lost_sales.png"))
        st.caption("Extra lost sales per disruption and policy, 20 runs each (Phase 3).")
