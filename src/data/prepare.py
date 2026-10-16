"""Build the long-format sales table for the project scope"""

from pathlib import Path

import pandas as pd

# SETTINGS
STORE = "CA_1"
DEPT = "FOODS_3"
N_DAYS = 1941  # sales_train_evaluation.csv has columns d_1 ... d_1941
MIN_RATE_FOR_STOCKOUT = 2.0  # avg units/day above which an item "always sells"
MIN_ZERO_RUN = 14  # consecutive zero days that count as unavailable

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_COLS = [
    "d",
    "date",
    "wm_yr_wk",
    "weekday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
]


def load_sales() -> pd.DataFrame:
    """Load the wide sales file and keep only the project scope"""
    day_types = {f"d_{i}": "int16" for i in range(1, N_DAYS + 1)}
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv", dtype=day_types)
    scope = (sales["store_id"] == STORE) & (sales["dept_id"] == DEPT)
    return sales[scope].reset_index(drop=True)


def to_long(sales: pd.DataFrame) -> pd.DataFrame:
    """Turn one column per day into one row per item per day."""
    return sales.melt(id_vars=ID_COLS, var_name="d", value_name="sales")


def add_calendar(long: pd.DataFrame) -> pd.DataFrame:
    """Attach real dates, week numbers, events and SNAP days."""
    calendar = pd.read_csv(RAW / "calendar.csv", usecols=CALENDAR_COLS, parse_dates=["date"])
    return long.merge(calendar, on="d", how="left")


def add_prices(long: pd.DataFrame) -> pd.DataFrame:
    """Attach the weekly selling price of each item."""
    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[prices["store_id"] == STORE]
    return long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")


def drop_before_launch(long: pd.DataFrame) -> pd.DataFrame:
    """Remove the days before each item was on sale (no price = not launched)."""
    first_priced_day = (
        long["date"].where(long["sell_price"].notna()).groupby(long["item_id"]).transform("min")
    )
    return long[long["date"] >= first_priced_day]


def add_closed(df: pd.DataFrame) -> pd.DataFrame:
    """Marca o dia de Natal, em que as lojas estão fechadas."""
    df["closed"] = (df["date"].dt.month == 12) & (df["date"].dt.day == 25)
    return df


def add_available(df: pd.DataFrame) -> pd.DataFrame:
    """Marca como indisponíveis as sequências longas de zeros em produtos que vendem bem."""
    key = ["item_id", "store_id"]
    df = df.sort_values([*key, "date"]).reset_index(drop=True)

    # 1. Produtos que vendem em média 2+ unidades por dia (sem contar o Natal)
    rate = df[~df["closed"]].groupby(key, observed=True)["sales"].mean()
    fast = rate[rate >= MIN_RATE_FOR_STOCKOUT].index
    is_fast = pd.MultiIndex.from_frame(df[key]).isin(fast)

    # 2. Comprimento de cada sequência de zeros
    zero = df["sales"].eq(0)
    new_item = df["item_id"].ne(df["item_id"].shift()) | df["store_id"].ne(df["store_id"].shift())
    run_id = (zero.ne(zero.shift()) | new_item).cumsum()
    run_len = zero.groupby(run_id).transform("size")

    # 3. Indisponível = produto rápido + zero + sequência de 14 dias ou mais
    df["available"] = ~(is_fast & zero & (run_len >= MIN_ZERO_RUN))
    return df


def tidy(long: pd.DataFrame) -> pd.DataFrame:
    """Drop unused columns, shrink data types and sort."""
    long = long.drop(columns=["id", "state_id", "d"])
    for col in ["item_id", "dept_id", "cat_id", "store_id", "weekday"] + [
        c for c in long.columns if c.startswith("event_")
    ]:
        long[col] = long[col].astype("category")
    long["sales"] = long["sales"].astype("int16")
    long["sell_price"] = long["sell_price"].astype("float32")
    long["snap_CA"] = long["snap_CA"].astype("int8")
    return long.sort_values(["item_id", "date"]).reset_index(drop=True)


def check(long: pd.DataFrame, wide_total: int) -> None:
    """Stop with a clear message if something is wrong with the data."""
    assert not long.duplicated(["item_id", "date"]).any(), "Duplicate (item, date) rows"

    has_gap = long.groupby("item_id", observed=True)["date"].apply(
        lambda d: (d.diff().dropna() != pd.Timedelta(days=1)).any()
    )
    assert not has_gap.any(), f"Date gaps in items: {list(has_gap[has_gap].index)[:5]}"

    assert (long["sales"] >= 0).all(), "Negative sales found"

    missing = long["sell_price"].isna().sum()
    assert missing == 0, f"{missing} rows have no price after launch"

    long_total = int(long["sales"].astype("int64").sum())
    assert long_total == wide_total, (
        f"Total sales changed: {wide_total} in the wide file, {long_total} after processing"
    )


def main() -> None:
    sales = load_sales()
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    wide_total = int(sales[day_cols].astype("int64").sum().sum())

    long = to_long(sales)
    long = add_calendar(long)
    long = add_prices(long)
    rows_before = len(long)
    long = drop_before_launch(long)
    long = tidy(long)
    long = add_closed(long)
    long = add_available(long)

    check(long, wide_total)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "sales_long.parquet"
    long.to_parquet(out, index=False)

    print(f"Saved {out}")
    print(f"Items: {long['item_id'].nunique()}")
    print(f"Rows: {len(long):,} (dropped {rows_before - len(long):,} pre-launch rows)")
    print(f"Dates: {long['date'].min().date()} to {long['date'].max().date()}")
    print(f"Total units sold: {long['sales'].astype('int64').sum():,}")
    off = ~long["available"]
    print(f"Christmas item-days (closed): {long['closed'].sum():,}")
    print(
        f"Unavailable item-days: {off.sum():,} ({off.mean():.1%}), "
        f"items affected: {long.loc[off, 'item_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
