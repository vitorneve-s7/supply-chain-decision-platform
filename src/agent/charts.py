"""Plotly charts for the app's right panel, built from a tool's result."""

import plotly.graph_objects as go

POLICY_ORDER = ["Weeks of cover", "(R, S) normal", "(R, S) quantile", "(s, S)"]


def forecast_figure(result: dict) -> go.Figure:
    """Fan chart: the median forecast with a band up to the 95th percentile."""
    days = result["daily"]
    dates = [d["date"] for d in days]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=[d["q50"] for d in days], line_width=0, showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[d["q95"] for d in days],
            fill="tonexty",
            line_width=0,
            name="up to 95th percentile",
            opacity=0.3,
        )
    )
    fig.add_trace(go.Scatter(x=dates, y=[d["q90"] for d in days], name="90th", line_dash="dot"))
    fig.add_trace(go.Scatter(x=dates, y=[d["q50"] for d in days], name="median", line_width=3))
    fig.update_layout(
        title=f"{result['item']}: daily demand forecast",
        yaxis_title="units per day",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
        margin={"t": 50, "b": 10},
    )
    return fig


def comparison_figure(result: dict) -> go.Figure:
    """Extra lost-sales cost per policy and scenario, with 95% intervals (S0 left out: zero)."""
    rows = [r for r in result["rows"] if r["scenario"] != "S0 normal"]
    if not rows:  # only the normal scenario: show total cost instead
        rows = result["rows"]
        value, ci, label = "total_cost_with_dc_usd", "total_cost_with_dc_ci_usd", "total cost, USD"
    else:
        value, ci = "extra_lost_sales_cost_usd", "extra_lost_sales_cost_ci_usd"
        label = "extra lost-sales cost vs normal, USD"
    fig = go.Figure()
    for policy in POLICY_ORDER:
        mine = [r for r in rows if r["policy"] == policy]
        fig.add_trace(
            go.Bar(
                x=[r["scenario"] for r in mine],
                y=[r[value] for r in mine],
                error_y={"type": "data", "array": [r[ci] for r in mine]},
                name=policy,
            )
        )
    fig.update_layout(
        title="Store policies under disruption (20 runs each)",
        yaxis_title=label,
        barmode="group",
        legend={"orientation": "h", "y": -0.2},
        margin={"t": 50, "b": 10},
    )
    return fig
