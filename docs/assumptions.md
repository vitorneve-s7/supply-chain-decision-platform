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
- cycle service level: share of item-weeks (Monday to Sunday) with no lost sale; fill rate: units
  sold / units demanded; days of stock: total end-of-day stock / total demand
- (R, S) textbook policy: protection period 12 days (review 7 + mean lead time 5); mean daily demand
  = average Hybrid q50 over those 12 days; sigma = standard deviation of observed daily sales over
  the last 56 days; S = 12 mu + z sqrt(12 sigma^2 + mu^2 Var(L)), Var(L) = 2 for a uniform 3-7 day
  lead time
- quantile policies, same 12-day protection period: E1 S = sum of daily q95; E2 S = sum of q50 +
  z sqrt(sum of sigma_t^2 + mu^2 Var(L)), with sigma_t = (q95_t - q50_t) / 1.645 from the
  calibrated forecast
- z tuned per demand class (E2 tuned): the smallest z reaching 95% cycle service on the replay up to
  2016-04-24, found by bisection on [1, 4] in 7 steps; the test window (2016-04-25 to 2016-05-22)
  checks it. Tuned z: smooth 3.11, erratic 2.80, lumpy 2.12, intermittent 1.66. Smooth and erratic
  need large z because the Hybrid point forecast is biased low on them (-8% to -17%).
- (s, S) policy: reviewed every Monday; s = the E2 order-up-to level (with the tuned z per class);
  if the inventory position is above s, no order; otherwise order up to S = s + Q, with Q the
  economic order quantity sqrt(2 D K / h), D = mean daily q50 x 365, K = 1 per order line,
  h = 25% of price per year
- costs per policy are totals over the 126 scored days (2016-01-18 to 2016-05-22): holding = daily
  stock value x 25% / 365; ordering = 1 per order line; lost sales = 30% of price per lost unit.
  Result: lost sales and ordering dominate, holding is small (25%/year over 4 months). Total cost:
  Baseline 36,166; (R, S) normal 36,738; (R, S) quantile with tuned z 24,672 (-32%); (s, S) with
  tuned z 17,048 (-53%). On the test window alone: 10,463 / 8,268 / 5,847 / 4,162.
- service sweep: each policy at several settings gives one curve of stock value against cycle service.
  Forecast policies: service 80%, 90%, 95%, 98%, 99%, 99.5% (one z for all classes). Baseline: order up
  to k weeks of last week's sales, k = 1.5 to 5 ("weeks of cover"); ordering k x last week's sales was
  dropped because for k > 1 stock piles up without limit. Policies are compared at equal service by
  interpolating along the curves.
- cost sensitivity: one assumption changed at a time (holding 15% / 35%, lost-sales penalty 10% / 50%,
  order cost 20); (s, S) is re-simulated when holding or order cost change, since they set its batch
  size. Lead-time sensitivity is left to Phase 3 (the digital twin), where lead-time disruptions are
  modelled directly.
- sweep result, stock at equal service (interpolated, base costs): for 90% cycle service the
  (R, S) quantile policy needs 48,177 of stock vs 75,677 for the baseline (-36%) and 52,816 for
  (R, S) normal (-9%). For 95% it needs 64,170; the baseline never reaches 95% (it levels off at 94.0%
  with 175,954, because an item that sells nothing for a week is never reordered) and (R, S) normal
  tops out at 90.7%.
- sweep result, lowest total cost per policy (base costs): (s, S) 17,430 at 98.6% service; baseline
  23,737 (3 weeks of cover, 91.0%); (R, S) quantile 24,970 (96.5%); (R, S) normal 30,702 (90.7%).
  (s, S) is cheapest in every cost scenario; with order cost 20 it costs 43,838 against 148,548
  for the baseline and 266,536 for weekly (R, S). The forecast policies' cost keeps falling up to
  the highest service tested (99.5%): with lost sales ~60x dearer than a week of holding, the
  economic service target is well above the 95% chosen in the scoping decisions.
- honest negative: on total cost the weekly (R, S) quantile policy does not beat the best
  baseline setting (24,970 vs 23,737), because holding stock is cheap; its gain is capital, not
  cost. The cost win comes from (s, S), which cuts orders by 72%.

## 10. Digital twin

- network: one supplier -> one distribution centre (DC) -> store CA_1, all 821 backtested items;
  simulated in SimPy, all items moving together as arrays (one process per stage, not per item)
- store side unchanged from Phase 2: weekly review on Monday, the same four policies at their best
  Phase 2 settings (weeks of cover k = 3; (R, S) normal at 95%; (R, S) quantile and (s, S) with the
  tuned z per class), DC -> store lead time uniform 3 to 7 days. The store's "on order" = DC backlog
  (ordered, not yet shipped) + units in transit.
- DC policy: weekly (R, S) on the store's forecast, reviewed every Monday after the store orders;
  covers 14 days (review 7 + mean supplier lead time 7) with z = 1.65 from the forecast quantiles.
  The DC sees the forecasts, not the store's orders in advance.
- supplier -> DC lead time: uniform 5 to 9 days
- capacities on normal days: the DC and the supplier can each ship up to 150% of average weekly
  store demand in any 7 consecutive days (a rolling weekly limit, so a normal Monday order ships the
  same day). Changed from a daily limit of 150% of average daily demand: with one weekly order, a
  daily limit took 4-5 days to ship each order, which added about 2 days of lead time on normal
  days. The DC ships its backlog oldest first, up to its stock and capacity; when capacity binds,
  every item gets the same share; what cannot ship waits.
- event order within a day: trucks arrive overnight (at day - 0.5 in SimPy); then the store orders
  (Monday), the store sells, the DC ships, the DC orders (Monday), the supplier ships. A fixed order,
  so results never depend on SimPy's tie-breaking.
- validation: with an unlimited DC, the twin reproduces the Phase 2 replay exactly (weeks of cover
  and (s, S) checked on all 821 items: every KPI identical)
- starting stock: store 14 days of demand (as in Phase 2); DC 14 days of store demand
- demand: the same 2016 demand as Phase 2 (observed, imputed on stockout days) in every replication,
  so the twin does not favour the forecast-based policies
- randomness: lead times on both legs and the disruption start date, drawn per replication
- replications: 20 per policy and scenario, seeds 0 to 19, the same seeds for every policy (common
  random numbers); results are reported relative to scenario S0 with the same seed
- disruption start: random between 2016-02-08 and 2016-04-04, leaving at least 7 weeks to recover
- scenarios: S0 normal; S1 supplier ships nothing for 14 days; S2 store demand +50% on every item for
  10 days, not known to the policies; S3 DC shipping capacity cut to 50% for 21 days; S4 DC -> store
  lead time 6 to 14 days for 28 days while policies still plan with 3 to 7
- costs: Phase 2 costs (holding 25% of price per year, ordering 1 per line, lost sales 30% of price)
  plus DC holding at the same 25% per year
- speed: all 821 items; if one run takes over a minute, a sample of 200 items stratified by demand
  class is used for the scenario runs
- DC policy as built: S = quantile_level of the store's Hybrid forecast over 14 days at 95% service
  (the Phase 2 formula with protection = 14 and Var(L) = 2 for a uniform 5-9 day supplier lead
  time); order = S - DC position (DC stock + on order from the supplier - still owed to the store)
- S0 with the finite DC (seed 0, all 821 items), store fill rate / cycle service vs the Phase 2
  replay: weeks of cover 95.7% / 90.6% (95.9% / 91.0%); (R, S) normal 90.1% / 84.6% (90.2% / 84.8%);
  (R, S) quantile 96.9% / 94.8% (97.0% / 95.0%); (s, S) 98.3% / 97.7% (98.7% / 98.4%). The DC costs
  each policy under 1 point of service. Item-days with the DC out of stock while owing the store:
  1.7% / 2.4% / 7.8% / 17.2% for normal / quantile / weeks of cover / (s, S): the (s, S) batches catch
  the DC short. Coefficient of variation of weekly orders, store -> DC: quantile 0.07 -> 0.14,
  normal 0.15 -> 0.24, weeks of cover 0.16 -> 0.15, (s, S) 0.17 -> 0.27.
- run_scenario(policy, scenario, seed): the seed sets the DC -> store lead times (draw_lead_times with
  that seed), the supplier lead times and the disruption start day (uniform on replay days 35-91,
  2016-02-08 to 2016-04-04), so one seed gives the same draws for every policy
- single-run check (R, S) quantile, seed 0, disruption from day 83 (2016-03-27), weekly store fill
  rate before -> worst week: S1 96.5% -> 88.0% (in week 3, once the DC runs dry); S2 96.5% -> 90.2%
  (week 2); S3 96.5% -> 90.0% (week 4, while stock piles up in the DC); S4 96.5% -> 83.2% (week 2)
