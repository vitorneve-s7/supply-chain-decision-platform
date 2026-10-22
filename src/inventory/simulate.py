"""Day-by-day inventory replay: any ordering policy, the same demand and lead times."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_PROTECTION = 14  # review period (7) + longest lead time (7): forecast days a policy may need
START_DAYS = 14  # starting stock = this many days of average demand


@dataclass
class ReviewState:
    """What a policy sees on a review day."""

    day: pd.Timestamp
    on_hand: float
    on_order: float
    forecast: pd.DataFrame  # q50, q90, q95 for the next MAX_PROTECTION days, today first
    past_sales: np.ndarray  # observed sales up to yesterday, oldest first
    price: float
    demand_class: str


Policy = Callable[[ReviewState], int]


def forecast_ahead(item: pd.DataFrame, day: pd.Timestamp, n: int = MAX_PROTECTION) -> pd.DataFrame:
    """The latest forecast for `day` onwards; past the window's end, repeat its last 7 days."""
    cutoff = item.loc[item["ds"] == day, "cutoff"].iloc[0]
    window = item[item["cutoff"] == cutoff]
    return _ahead(window["ds"].to_numpy(), window[["q50", "q90", "q95"]].to_numpy(), day, n)


def _ahead(dates: np.ndarray, values: np.ndarray, day: pd.Timestamp, n: int) -> pd.DataFrame:
    rows = values[dates >= np.datetime64(day)]
    tail = values[-7:]
    while len(rows) < n:
        rows = np.vstack([rows, tail])
    return pd.DataFrame(rows[:n], columns=["q50", "q90", "q95"])


def simulate_item(
    item: pd.DataFrame, policy: Policy, lead_times: dict, start_stock: float | None = None
) -> pd.DataFrame:
    """Replay one item. `item` holds its history and replay rows in date order."""
    history = item[~item["in_replay"]]
    replay = item[item["in_replay"]].reset_index(drop=True)
    if start_stock is None:
        start_stock = round(START_DAYS * history["demand"].mean()) if len(history) else 0

    windows = {  # forecast window per cutoff: its dates and q50, q90, q95
        c: (g["ds"].to_numpy(), g[["q50", "q90", "q95"]].to_numpy())
        for c, g in replay.groupby("cutoff")
    }
    past_sales = list(history["sales"])
    on_hand = float(start_stock)
    pipeline: dict[pd.Timestamp, float] = {}  # arrival day -> units
    records = []

    for row in replay.itertuples(index=False):
        day = row.ds
        start = on_hand
        received = pipeline.pop(day, 0.0)
        on_hand += received

        order = 0
        if day.day_of_week == 0:  # Monday: review
            state = ReviewState(
                day=day,
                on_hand=on_hand,
                on_order=sum(pipeline.values()),
                forecast=_ahead(*windows[row.cutoff], day, MAX_PROTECTION),
                past_sales=np.array(past_sales),
                price=row.sell_price,
                demand_class=row.demand_class,
            )
            order = max(0, int(policy(state)))
            if order:
                arrival = day + pd.Timedelta(days=lead_times[day])
                pipeline[arrival] = pipeline.get(arrival, 0.0) + order

        sales = min(on_hand, row.demand)
        on_hand -= sales
        past_sales.append(sales)
        records.append(
            {
                "ds": day,
                "start_stock": start,
                "received": received,
                "demand": row.demand,
                "sales": sales,
                "lost": row.demand - sales,
                "end_stock": on_hand,
                "order": order,
                "on_order": sum(pipeline.values()),
            }
        )
    return pd.DataFrame(records)


def simulate(inputs: pd.DataFrame, policy: Policy, lead_times: pd.DataFrame) -> pd.DataFrame:
    """Replay every item with the same policy; one row per item and replay day."""
    leads = {uid: dict(zip(g["ds"], g["lead_time"])) for uid, g in lead_times.groupby("unique_id")}
    out = []
    for uid, item in inputs.groupby("unique_id", sort=False):
        rec = simulate_item(item.reset_index(drop=True), policy, leads.get(uid, {}))
        rec.insert(0, "unique_id", uid)
        rec.insert(1, "demand_class", item["demand_class"].iloc[0])
        rec.insert(2, "sell_price", item.loc[item["in_replay"], "sell_price"].to_numpy())
        out.append(rec)
    return pd.concat(out, ignore_index=True)
