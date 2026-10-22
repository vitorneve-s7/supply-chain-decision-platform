# Assumptions

Decisions that the M5 dataset does not provide. Every phase uses these values.

## 1. Scope

- store: CA_1
- items: department FOODS_3
- why: one store keeps runtime low; food items mix fast movers and intermittent items, which test both forecasting and inventory methods

## 2. Forecast horizon

- 28 days ahead
- why: same horizon as the original dataset competition, so results are comparable

## 3. Review policy

- periodic review, once per week
- why: matches how most retailers place orders

## 4. Service target

- base case: 95% cycle service level
- sensitivity cases: 90% and 98%

## 5. Costs

- holding cost: 25% of item price per year (price x 0.25 / 365 per unit per day)
- lost-sales penalty: per unit not sold because of a stockout
- fixed ordering cost: per order placed
- why: holding cost as a % of price is a standard textbook assumption; the other two are calibrated once the inventory model exists.

## 6. Lead times

- distribution centre to store: 3 to 7 days, random within that range
- why: realistic for a regional DC; the variability tests safety stock

## 7. Data notes

- scope after cleaning: 823 items, 1,280,182 rows (317,261 pre-launch rows dropped), 2011-01-29 to 2016-05-22.
- days before each item's first price were dropped (not launched yet).
- 25 December: store closed every year. Flagged in `closed` (3,377 item-days); sales kept as 0 in the
  data, replaced by the average of the neighbouring days for training, never scored.
- unavailability: for items selling 2+ units/day on average, runs of 14+ consecutive zero-sales days are
  flagged `available = False` (59,104 item-days, 4.6% of all, 247 items). These days are excluded from
  scoring and from the demand classification; the true demand on them is unknown.

## 8. Backtest design

- horizon: 28 days
- final test window: 2016-04-25 to 2016-05-22, used once at the end of Phase 1
- validation: 4 rolling windows of 28 days, 2016-01-04 to 2016-04-24, for comparing and tuning models
- scoring skips days with `closed = True` or `available = False`
- demand classes (Syntetos-Boylan, ADI 1.32 / CV² 0.49) are computed on data before 2016-04-25,
  available days only: 361 intermittent, 184 lumpy, 174 smooth, 104 erratic
- items need 28+ days of history before 2016-01-04 to be backtested: 821 of 823 (FOODS_3_296 and
  FOODS_3_595 launched later); 4 more have no available day in the validation windows, so 817 are scored
- benchmark bar: WindowAverage(28) has the best RMSSE in every demand class (0.760 overall)
- statistical models: AutoARIMA trains on the last 2 years only (from 2014-04-26) for speed; TSB uses
  alpha_d = alpha_p = 0.1; Croston, TSB, ADIDA and IMAPA get conformal intervals (2 windows of 28 days)
- best statistical model: AutoETS, RMSSE 0.743 overall, best in every demand class; AutoARIMA 0.750
- LightGBM (mlforecast, global): lags 7/14/28, rolling means of lag 7 over 7 and 28 days, day of week,
  day, month, price, price relative to its 28-day average, SNAP, events; tweedie objective, 500 trees,
  learning rate 0.05, 63 leaves. RMSSE 0.745 overall: best on smooth (0.806) and erratic (0.680);
  AutoETS stays best on intermittent (0.742) and lumpy (0.683). Light tuning (31 or 127 leaves,
  learning rate 0.03 with 800 trees) changed RMSSE by at most 0.002, so the base setting is kept.
- LightGBM feature importance (gain): rolling means of the last 7 and 28 days carry about 86%;
  price, day of week and lag 7 a few % each; events almost nothing.
- quantiles: q50 = point forecast; q90 and q95 = upper edge of the 80% and 90% intervals (native for
  AutoETS and AutoARIMA, conformal with 2 windows for the others and LightGBM), clipped at 0 and
  kept in order q50 <= q90 <= q95
- interval calibration: per model and demand class, q = mean + c * (q - mean), with c the smallest
  factor (0.5 to 5, step 0.05) reaching 90% / 95% coverage on the validation windows. Fitted on
  windows 1-2, it held on windows 3-4 (q95 coverage 0.93-0.97 except conformal models on erratic).
  Raw conformal intervals were far too narrow (q95 coverage about 0.82); AutoETS needed almost none.
- after calibration AutoETS has the lowest pinball loss in every class (q95: 0.329 overall vs 0.470
  for LightGBM), even where LightGBM has the better point forecast (smooth, erratic)
- model choice (validation windows only): Hybrid = point forecast from LightGBM for smooth and
  erratic items and from AutoETS for intermittent and lumpy items; q90/q95 = that point plus the
  calibrated AutoETS margin (q - mean). Validation: RMSSE 0.734 vs 0.743 (AutoETS) and 0.745
  (LightGBM); pinball q95 0.323 vs 0.329 and 0.470; coverage 0.903 / 0.952.
- final test window result (2016-04-25 to 2016-05-22, used once; models refitted on all data up to
  2016-04-24; calibration factors from validation applied unchanged): 809 items scored. Hybrid RMSSE
  0.752 vs 0.976 SeasonalNaive (-23%), 0.776 WindowAverage (-3%), 0.760 AutoETS, 0.756 LightGBM;
  q90 / q95 coverage 0.901 / 0.952. Bias -9.6% (slight under-forecasting).
- honest negative: on the test window LightGBM beat AutoETS on lumpy items (RMSSE 0.735 vs 0.745),
  the opposite of validation; the choice made on validation is kept.
- future forecast (2016-05-23 to 2016-06-19, the 28 days after the data ends): the Hybrid refitted
  on all data up to 2016-05-22, for all 823 items; future prices, SNAP days and events come from
  the raw calendar and price files; calibration factors from validation applied unchanged.
  Forecasts are precomputed into results/future_forecast.parquet, and forecast() reads that file.

## 9. Inventory replay

- period: 140 days, 2016-01-04 to 2016-05-22 (the 4 validation windows plus the test window); every
  forecast in it is out of sample
- forecasts: the Hybrid (q50, q90, q95) from the calibrated backtest for the validation windows and
  from test_forecasts.parquet for the test window; each order uses the forecast from the latest
  28-day cutoff. When an order's protection period reaches past that forecast, the last 7 forecast
  days are repeated, which keeps the weekly pattern.
- items: the 821 backtested items
- review: every Monday (R = 7 days)
- lead time: random, uniform 3 to 7 days, drawn once per item and order date with a fixed seed, so
  every policy sees the same deliveries; policies plan with the mean (5 days) and its variance
- unmet demand is lost, not backordered
- demand: observed sales on available days; on days flagged available = False, the item's average
  sales over its last 28 available days, rounded to whole units (true demand there is unknown, and
  zero would flatter every policy). This affects 4,133 item-days of 79 items in the replay.
- policies see only observed sales (what the store would see), never the imputed demand
- starting stock: 14 days of average demand per item, nothing in transit; the first 14 days are a
  warm-up and are not scored
- holding cost: 25% of price per year, charged daily on end-of-day stock
- lost-sales penalty: 30% of price per lost unit (lost margin)
- ordering cost: 1 per order line (per item per order)
- orders are whole units; no pack sizes or minimum order quantities (not in M5)
