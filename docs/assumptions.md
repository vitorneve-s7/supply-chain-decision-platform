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
