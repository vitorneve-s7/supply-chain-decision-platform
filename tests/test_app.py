"""The app renders without calling the API: empty, and with a stored answer per tool type."""

from streamlit.testing.v1 import AppTest

from src.agent.agent import Answer, ToolCall
from src.agent.charts import comparison_figure, forecast_figure
from src.agent.tools import compare_policies, get_forecast, get_inventory_policy

APP = "../app/streamlit_app.py"  # relative to this test file


def app_with(call: ToolCall) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["turns"] = [{"question": "Q", "answer": Answer("A", tool_calls=[call])}]
    at.session_state["history"] = None
    return at.run()


def test_app_starts_with_the_phase_3_chart():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert len(at.sidebar.button) >= 5  # the example questions


def test_app_shows_each_tool_result():
    calls = [
        ToolCall("get_forecast", {}, get_forecast("FOODS_3_090", 7), 0.0),
        ToolCall("compare_policies", {}, compare_policies("S1 supplier outage"), 0.0),
        ToolCall("get_inventory_policy", {}, get_inventory_policy("FOODS_3_090"), 0.0),
    ]
    for call in calls:
        at = app_with(call)
        assert not at.exception, call.name


def test_charts_have_one_trace_per_band_or_policy():
    assert len(forecast_figure(get_forecast("FOODS_3_090", 14)).data) == 4
    assert len(comparison_figure(compare_policies("all")).data) == 4
