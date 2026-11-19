# Experiments

Tracking model runs: setup, hyperparameters, and results.

## Run 1

**Date:** 2026-09-30

**Data:** `data/raw/train.csv`, hard-filtered to `100 < trip_duration < 12600` (seconds).

**Features**
- Categorical (one-hot, drop first): `month`, `hour`, `day_of_week`, `vendor_id`
- Numerical (min-max scaled): `passenger_count`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`

**Preprocessing:** `ColumnTransformer` — `OneHotEncoder(drop='first')` on categoricals, `MinMaxScaler()` on numericals.

**Target:** `log1p(trip_duration)`

**Split:** `test_size=0.1`, `random_state=314`

**Model:** `Ridge(alpha=1)`

**Results**

- **RMSE:** 0.6769 (on `log1p(trip_duration)` scale)
- **R²:** 0.0881 — model explains ~8.8% of the variance; weak fit.

## Run 2 (`--model 2`)

**Date:** 2026-09-30

**Data:** same as Run 1.

**Features:** same as Run 1, plus `haversine_distance` (great-circle distance between pickup/dropoff, via `add_haversine_distance`) added to the numerical (min-max scaled) columns.

**Preprocessing:** same as Run 1.

**Target:** same as Run 1.

**Split:** same as Run 1.

**Model:** `Ridge(alpha=1)`

**Results**

- **RMSE:** 0.5260 (on `log1p(trip_duration)` scale)
- **R²:** 0.4492 — model explains ~44.9% of the variance; big improvement over Run 1.

## Run 3 (`--model 3`)

**Date:** 2026-09-30

**Data:** same as Run 1.

**Features:** same as Run 2 (includes `haversine_distance`), but numerical columns go through `PolynomialFeatures(degree=2)` before scaling.

**Preprocessing:** `ColumnTransformer` — `OneHotEncoder(drop='first')` on categoricals, `PolynomialFeatures(degree=2)` → `MinMaxScaler()` on numericals.

**Target:** same as Run 1.

**Split:** same as Run 1.

**Model:** `Ridge(alpha=1)`

**Results**

- **RMSE:** 0.5100 (on `log1p(trip_duration)` scale)
- **R²:** 0.4823 — model explains ~48.2% of the variance; small improvement over Run 2.

## Run 4 (`--model 4`)

**Date:** 2026-09-30

**Data:** same as Run 1.

**Features:** same as Run 2 (includes `haversine_distance`), plus `haversine_distance_sqrt` (`sqrt(haversine_distance)`). No polynomial features — plain `MinMaxScaler` on all numericals.

**Preprocessing:** `ColumnTransformer` — `OneHotEncoder(drop='first')` on categoricals, `MinMaxScaler()` on numericals.

**Target:** same as Run 1.

**Split:** same as Run 1.

**Model:** `Ridge(alpha=1)`

**Results**

- **RMSE:** 0.4269 (on `log1p(trip_duration)` scale)
- **R²:** 0.6372 — model explains ~63.7% of the variance; best run so far, clear improvement over Run 3.

![sqrt(distance) vs trip_duration](images/sqrt(distance)_vs_trip_duration.png)

I found that `sqrt(haversine_distance)` has a linear relationship with `trip_duration`, which is why adding it increased the model's accuracy — Ridge is a linear model, so a feature that's linearly related to the target fits it much better than the raw (non-linear) distance.

## Run 5 (`--features 5`)

**Date:** 2026-09-30

**Note:** `--model` is now fixed to `1` (ridge); feature engineering moved to its own `--features`/`-f` flag.

**Data:** same as Run 1.

**Features:** same as Run 4 (`haversine_distance`, `haversine_distance_sqrt`), plus `is_rush_hour` and `is_weekday` (from `add_time_features`: `is_rush_hour` = `hour` between 14 and 16, `is_weekday` = `day_of_week` not Sat/Sun).

**Preprocessing:** same as Run 4.

**Target:** same as Run 1.

**Split:** same as Run 1.

**Model:** `Ridge(alpha=1)`

**Results**

- **RMSE:** 0.4269 (on `log1p(trip_duration)` scale)
- **R²:** 0.6372 — essentially unchanged from Run 4. `is_rush_hour`/`is_weekday` added no measurable value, likely because `hour` and `day_of_week` are already one-hot encoded in the categorical features.

## Run 6 (`--features 6`)

**Date:** 2026-09-30

**Data:** same as Run 1.

**Features:** same as Run 5 (`haversine_distance`, `haversine_distance_sqrt`, `is_rush_hour`, `is_weekday`).

**Preprocessing:** `ColumnTransformer` — `OneHotEncoder(drop='first')` on categoricals, `MinMaxScaler()` on numericals, same as Run 5. Additionally, `PolynomialFeatures(degree=2)` is applied to the **full encoded matrix after** one-hot encoding (as a pipeline step after the `ColumnTransformer`), producing interaction terms between all encoded features, not just the numeric ones.

**Target:** same as Run 1.

**Split:** same as Run 1.

**Model:** `Ridge(alpha=1)`

**Results**

- **RMSE:** 0.4074 (on `log1p(trip_duration)` scale)
- **R²:** 0.6696 — best run so far, clear improvement over Run 5. Post-encoding polynomial interactions (e.g. between `hour`/`day_of_week` one-hot columns and distance features) capture signal that plain haversine/rush-hour/weekday features alone missed.
