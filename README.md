# NYC Trip Duration

Course homework: predict NYC taxi `trip_duration` (seconds) with pandas and scikit-learn.

## Setup

Requires Python >= 3.14 and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

Put the data in `data/raw/` (`train.csv`, `val.csv`, `test.csv`). It is not tracked in git.

## Run

```
uv run python main.py --dataset 1 --features 5
```

| Flag | Values |
|------|--------|
| `--dataset` | 1 full train (only 1 is implemented so far) |
| `--features` / `-f` | 1 base features, 2 + haversine distance, 3 + degree-2 polynomial, 4 + sqrt(distance), 5 + rush hour & weekday, 6 = 5 + polynomial after one-hot |
| `--model` | 1 Ridge |
| `--preprocessing` | 0 none, 1 MinMaxScaler, 2 StandardScaler |

## Pipeline

1. Load train data, derive `month`, `hour`, `day_of_week`.
2. Keep trips between 100 and 12600 seconds.
3. One-hot encode the categorical features and scale the numeric ones.
4. Train Ridge (alpha=1) on `log1p(trip_duration)` with a 90/10 split.
5. Print RMSE and R².

## Project layout

- `main.py` – entry point
- `src/` – data loading, preprocessing helpers, CLI
- `notebooks/eda.ipynb` – exploratory analysis
- `reports/` – profiling reports and `EXPERIMENTS.md` (run results)
