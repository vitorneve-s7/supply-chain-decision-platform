"""Supplier -> DC -> store digital twin in SimPy. All items move together as NumPy arrays."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import simpy

from src.inventory.simulate import MAX_PROTECTION, START_DAYS, ReviewState, _ahead
from src.simulation.config import NetworkConfig

StorePolicy = Callable[[ReviewState], int]


class Twin:
    """One run of the network: build it, call run(), read store_records() and dc_records()."""

    def __init__(
        self,
        inputs: pd.DataFrame,
        store_policy: StorePolicy,
        dc_policy: Callable[["Twin"], np.ndarray],
        store_leads: pd.DataFrame,
        config: NetworkConfig | None = None,
        seed: int = 0,
    ):
        self.config = config or NetworkConfig()
        self.store_policy = store_policy
        self.dc_policy = dc_policy
        self.rng = np.random.default_rng(seed)
        self._load(inputs, store_leads)

    # ---------- set-up ----------

    def _load(self, inputs: pd.DataFrame, store_leads: pd.DataFrame) -> None:
        inputs = inputs.sort_values(["unique_id", "ds"])
        replay = inputs[inputs["in_replay"]]
        history = inputs[~inputs["in_replay"]]
        self.items = replay["unique_id"].unique()
        self.dates = pd.DatetimeIndex(replay["ds"].unique()).sort_values()
        n, d = len(self.items), len(self.dates)

        def grid(df, col):
            return df.pivot(index="unique_id", columns="ds", values=col).loc[self.items].to_numpy()

        self.demand = grid(replay, "demand").astype(float)
        self.price = grid(replay, "sell_price").astype(float)
        self.cutoff = replay.pivot(index="unique_id", columns="ds", values="cutoff").loc[self.items]
        self.classes = (
            replay.groupby("unique_id")["demand_class"].first().loc[self.items].to_numpy()
        )
        hist_sales = grid(history, "sales").astype(float)
        hist_demand = grid(history, "demand").astype(float)
        self.windows = [  # per item: {cutoff: (dates, q50/q90/q95)}, as in Phase 2
            {
                c: (g["ds"].to_numpy(), g[["q50", "q90", "q95"]].to_numpy())
                for c, g in item.groupby("cutoff")
            }
            for _, item in replay.groupby("unique_id", sort=True)
        ]

        avg_demand = hist_demand.mean(axis=1)  # per item, per day
        self.avg_total = avg_demand.sum()  # whole store, per day
        self.store_stock = np.round(START_DAYS * avg_demand)
        self.dc_stock = self.config.dc_start_days * avg_demand
        self.sales_history = np.concatenate([hist_sales, np.zeros((n, d))], axis=1)
        self.n_history = hist_sales.shape[1]

        leads = store_leads.pivot(index="unique_id", columns="ds", values="lead_time")
        self.store_leads = {day: leads.loc[self.items, day].to_numpy() for day in leads.columns}
        lo, hi = self.config.supplier_lead
        self.supplier_leads = {day: int(self.rng.integers(lo, hi + 1)) for day in leads.columns}

        # Orders not yet shipped, oldest first: (order day, units per item)
        self.dc_backlog: list[tuple[pd.Timestamp, np.ndarray]] = []  # store -> DC
        self.supplier_backlog: list[tuple[pd.Timestamp, np.ndarray]] = []  # DC -> supplier
        self.to_store = np.zeros(n)  # units in transit DC -> store
        self.to_dc = np.zeros(n)  # units in transit supplier -> DC
        self.arrived_store = np.zeros(n)
        self.arrived_dc = np.zeros(n)
        self.store_order = np.zeros(n)
        self.dc_sent: list[float] = []  # total units the DC shipped each day
        self.supplier_sent: list[float] = []  # total units the supplier shipped each day
        self.store_rows: list[dict] = []
        self.dc_rows: list[dict] = []

    # ---------- what policies see ----------

    def store_on_order(self) -> np.ndarray:
        """What the store is still owed: orders the DC has not shipped + units in transit."""
        owed = sum((o for _, o in self.dc_backlog), np.zeros(len(self.items)))
        return owed + self.to_store

    def dc_position(self) -> np.ndarray:
        """DC stock + on order from the supplier - what it still owes the store."""
        on_order = sum((o for _, o in self.supplier_backlog), np.zeros(len(self.items)))
        owed = sum((o for _, o in self.dc_backlog), np.zeros(len(self.items)))
        return self.dc_stock + on_order + self.to_dc - owed

    def forecast(self, i: int, t: int) -> pd.DataFrame:
        """Item i's latest forecast from replay day t, for MAX_PROTECTION days."""
        day = self.dates[t]
        return _ahead(*self.windows[i][self.cutoff.iat[i, t]], day, MAX_PROTECTION)

    def _capacity_left(self, weekly: float | None, sent: list[float], multiplier: float) -> float:
        """Units that can still ship today: the 7-day capacity minus the last 6 days' shipments."""
        if weekly is None:
            return np.inf
        return max(0.0, weekly * 7 * self.avg_total * multiplier - sum(sent[-6:]))

    # ---------- moving goods ----------

    def _truck(self, env, units, lead, to_store: bool):
        """Units leave today and arrive overnight before day + lead."""
        yield env.timeout(lead - 0.5)
        if to_store:
            self.store_stock += units
            self.to_store -= units
            self.arrived_store += units
        else:
            self.dc_stock += units
            self.to_dc -= units
            self.arrived_dc += units

    def _ship(self, env, backlog, stock, capacity, leads_of, to_store: bool) -> np.ndarray:
        """Ship a backlog oldest first, up to stock and capacity; return units shipped per item."""
        shipped = np.zeros(len(self.items))
        left = capacity
        for order_day, owed in backlog:
            if left <= 0:
                break
            ship = np.minimum(owed, stock - shipped)
            if ship.sum() > left:  # not enough capacity: ship the same share of every item
                ship *= left / ship.sum()
            owed -= ship
            shipped += ship
            left -= ship.sum()
            leads = leads_of(order_day)
            for lead in np.unique(leads):  # one truck per lead time
                part = np.where(leads == lead, ship, 0.0)
                if part.sum() > 0:
                    env.process(self._truck(env, part, int(lead), to_store))
        backlog[:] = [(d, o) for d, o in backlog if o.sum() > 1e-9]
        return shipped

    # ---------- the day ----------

    def _day(self, env):
        cfg = self.config
        for t, day in enumerate(self.dates):
            hit = cfg.disrupted(t)
            received_store, received_dc = self.arrived_store.copy(), self.arrived_dc.copy()
            self.arrived_store[:], self.arrived_dc[:] = 0, 0
            start_store = self.store_stock - received_store  # before last night's deliveries
            start_dc = self.dc_stock - received_dc
            monday = day.day_of_week == 0

            # 1. Monday: the store orders, per item, with the Phase 2 policy
            store_order = np.zeros(len(self.items))
            if monday:
                on_order = self.store_on_order()
                col = self.n_history + t
                for i in range(len(self.items)):
                    state = ReviewState(
                        day=day,
                        on_hand=self.store_stock[i],
                        on_order=on_order[i],
                        forecast=self.forecast(i, t),
                        past_sales=self.sales_history[i, :col],
                        price=self.price[i, t],
                        demand_class=self.classes[i],
                    )
                    store_order[i] = max(0, int(self.store_policy(state)))
                self.dc_backlog.append((day, store_order.copy()))
            self.store_order = store_order

            # 2. The store sells
            demand = self.demand[:, t] * (cfg.demand_multiplier if hit else 1.0)
            sales = np.minimum(self.store_stock, demand)
            self.store_stock -= sales
            self.sales_history[:, self.n_history + t] = sales

            # 3. The DC ships to the store
            if cfg.unlimited_dc:
                stock, capacity = np.full(len(self.items), np.inf), np.inf
            else:
                stock = self.dc_stock
                multiplier = cfg.dc_capacity_multiplier if hit else 1.0
                capacity = self._capacity_left(cfg.dc_capacity, self.dc_sent, multiplier)
            lead_x = cfg.store_lead_multiplier if hit else 1.0
            shipped = self._ship(
                env,
                self.dc_backlog,
                stock,
                capacity,
                lambda d: np.round(self.store_leads[d] * lead_x).astype(int),
                to_store=True,
            )
            if not cfg.unlimited_dc:
                self.dc_stock -= shipped
            self.to_store += shipped
            self.dc_sent.append(shipped.sum())

            # 4. Monday: the DC orders from the supplier
            dc_order = np.zeros(len(self.items))
            if monday and not cfg.unlimited_dc:
                dc_order = np.maximum(0, np.ceil(self.dc_policy(self, t)))
                self.supplier_backlog.append((day, dc_order.copy()))

            # 5. The supplier ships to the DC, unless it is down
            down = hit and cfg.supplier_down
            sup_cap = (
                0.0 if down else self._capacity_left(cfg.supplier_capacity, self.supplier_sent, 1.0)
            )
            sent = self._ship(
                env,
                self.supplier_backlog,
                np.full(len(self.items), np.inf),
                sup_cap,
                lambda d: np.full(len(self.items), self.supplier_leads[d]),
                to_store=False,
            )
            self.to_dc += sent
            self.supplier_sent.append(sent.sum())

            # 6. Record the day
            self.store_rows.append(
                {
                    "t": t,
                    "start_stock": start_store,
                    "received": received_store,
                    "demand": demand,
                    "sales": sales,
                    "end_stock": self.store_stock.copy(),
                    "order": store_order,
                    "on_order": self.store_on_order(),
                }
            )
            self.dc_rows.append(
                {
                    "t": t,
                    "dc_start": start_dc,
                    "dc_received": received_dc,
                    "dc_shipped": shipped,
                    "dc_end": self.dc_stock.copy(),
                    "dc_order": dc_order,
                    "dc_owed": sum((o for _, o in self.dc_backlog), np.zeros(len(self.items))),
                }
            )
            yield env.timeout(1)

    def run(self) -> "Twin":
        env = simpy.Environment()
        env.process(self._day(env))
        env.run(until=len(self.dates))
        return self

    # ---------- results ----------

    def _long(self, rows: list[dict]) -> pd.DataFrame:
        """Daily rows of arrays -> one row per item and day."""
        frames = []
        for r in rows:
            f = pd.DataFrame({k: v for k, v in r.items() if k != "t"})
            f.insert(0, "unique_id", self.items)
            f.insert(1, "ds", self.dates[r["t"]])
            f.insert(2, "demand_class", self.classes)
            f.insert(3, "sell_price", self.price[:, r["t"]])
            frames.append(f)
        return pd.concat(frames, ignore_index=True)

    def store_records(self) -> pd.DataFrame:
        """Phase 2 format: one row per item and day, ready for scorecard()."""
        df = self._long(self.store_rows)
        df["lost"] = df["demand"] - df["sales"]
        return df

    def dc_records(self) -> pd.DataFrame:
        return self._long(self.dc_rows)
