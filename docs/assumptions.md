1. scope

- store: CA_1
- stems: department FOODS_3
- why: one store keeps runtime low; food items mix fast movers and intermittent items, which test both forecasting and inventory methods

2. forecast horizon

- 28 days ahead
- why: same horizon as the original dataset competition, so results are comparable

3. review policy

- periodic review, once per week
- why: matches how most retailers place orders

4. service target

- base case: 95% cycle service level
- sensitivity cases: 90% and 98%

5. costs

- holding cost: 25% of item price per year (price x 0.25 / 365 per unit per day)
- lost-sales penalty: per unit not sold because of a stockout
- fixed ordering cost: per order placed
- why: holding cost as a % of price is a standard textbook assumption; the other two are calibrated once the inventory model exists.

6. lead times

- distribution centre to store: 3 to 7 days, random within that range
- why: realistic for a regional DC; the variability tests safety stock