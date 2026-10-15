# pulse
Training and tracking an ML model on real-world data: will an MBTA bus be more than
3 minutes late at a timepoint stop? (Pulse Reliability Signal, see `ml_spec.md`.)

## Rebuild the dataset from a fresh clone

Everything runs from the repository root.

**1. Install the packages** (Python 3.12; the versions in `requirements.txt` are pinned for it):
```bash
python -m pip install -r requirements.txt
```

**2. Create `.env`** from the example and fill in your own PostgreSQL login.
`MBTA_API_KEY` is optional: without it the daily pull is limited to 20 requests a minute.
```bash
cp .env.example .env
```

**3. Download the raw data**: the MBTA Bus Arrival/Departure Times files from the
MBTA Open Data Portal (mbta-massdot.opendata.arcgis.com). Put three consecutive monthly CSVs
in `data/raw/`. This repo uses April, May and June 2026.

**4. Create the database and table** (`transit` / `stop_events`, defined in `schema.sql`):
```bash
python src/create_schema.py
```

**5. Load the historical files** (about 7.4 million rows, a few minutes):
```bash
python src/load_historical.py
```
Per file: schema check, parse timestamps as Boston wall-clock time, compute `delay_min` and
`late` (> 3 min), drop delays outside -30..+120 min (quarantined with a reason in
`data/rejects/`), keep one row per key, insert with `ON CONFLICT DO NOTHING`.
The verdict lines go to `logs/load_historical.log`. Rerunning inserts 0 rows.

**6. Append today's planned service from the MBTA API**:
```bash
python src/pull_daily.py
```
API rows have `source = 'api'` and no `actual` yet. Verdict lines go to `logs/pull_daily.log`.

**7. Run the tests**:
```bash
python -m pytest
```

**8. Split by time and freeze the splits with DVC** (Task 2, Step 3):
```bash
python src/make_splits.py
dvc init        # once
dvc add data/splits/train.parquet data/splits/val.parquet data/splits/test.parquet
```
April-May -> train, June 1-15 -> validation, June 16-30 -> test. Only the `.dvc` pointer
files go into Git. The test split stays untouched until the final evaluation.

**9. Train the two baselines, tracked in MLflow** (Task 2, Step 4):
```bash
python src/train_baselines.py
```
Configs are in `configs/baseline_majority.yaml` and `configs/baseline_logreg.yaml`.
Runs are stored locally in `mlflow.db` and `mlruns/` and exported to `mlflow_runs.csv`.
To browse them: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Layout
```
src/        every script that runs: ingestion and preparation
configs/    one YAML file per training configuration
data/       DVC-tracked data; only .dvc pointer files are in Git
notebooks/  exploration only (01 historical files, 02 MBTA API)
tests/      tests for src/
assets/     images used in REPORT.md
```
