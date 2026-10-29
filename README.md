# pulse
Training and tracking an ML model on real-world data: will an MBTA bus be more than
3 minutes late at a timepoint stop? (Pulse Reliability Signal, see `ml_spec.md`.)

## Reproduce the winning model (fresh clone)

Everything runs from the repository root. The data splits and the trained model come from DVC,
stored publicly: no account or credentials needed.

**1. Clone and install** (Python 3.12; the versions in `requirements.txt` are the ones tested):
```bash
git clone https://github.com/farahmassuh-ai/pulse.git
cd pulse
python -m pip install -r requirements.txt
```
On Linux the default `torch` package also downloads GPU (CUDA) libraries. For a smaller CPU-only
install, first run `python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu`.

**2. Get the data splits and the trained model:**
```bash
dvc pull
```

**3. Retrain the winning model (one command):**
```bash
python src/train_latenet.py configs/final.yaml
```
About 10 minutes on the author's laptop CPU: 38 epochs, early stopping keeps epoch 35 (validation AUC 0.7631).
On the author's machine this reproduces `models/latenet.pt` bit-for-bit (REPORT section 6).
It overwrites `models/latenet.pt` and logs the run to a local MLflow store (`mlflow.db`, `mlruns/`).

**Optional checks:**
```bash
python src/evaluate.py val --threshold 0.45
python src/latency_check.py
python -m pytest
```
`evaluate.py` scores the validation split at the chosen threshold, and `latency_check.py` times one
prediction on CPU. The test split was scored once, for the final evaluation (REPORT section 4), and
is not meant to be scored again.

The local MLflow store (`mlflow.db`, `mlruns/`) is not in Git, so a fresh clone starts empty. All 14
original runs are exported in `mlflow_runs.csv`. To browse your own runs:
`mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Full pipeline: rebuild the dataset from raw data

Not needed to reproduce the model: `dvc pull` above already provides the splits. This section
rebuilds them from the MBTA's raw files through PostgreSQL. Everything runs from the repository root.

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
dvc add data/splits/train.parquet data/splits/val.parquet data/splits/test.parquet
```
The repo is already set up for DVC; in a brand-new repo, run `dvc init` once first.
April-May -> train, June 1-15 -> validation, June 16-30 -> test. Only the `.dvc` pointer
files go into Git. The test split stays untouched until the final evaluation.

**9. Train the two baselines, tracked in MLflow** (Task 2, Step 4):
```bash
python src/train_baselines.py
```
Configs are in `configs/baseline_majority.yaml` and `configs/baseline_logreg.yaml`.
Runs are stored locally in `mlflow.db` and `mlruns/` and exported to `mlflow_runs.csv`.
To browse them: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

**10. Train LateNet, evaluate once on test, and register it** (Task 3):
```bash
python src/train_latenet.py configs/final.yaml
python src/evaluate.py val --threshold 0.45
python src/evaluate.py test --threshold 0.45 --run-id RUN_ID
python src/register_model.py RUN_ID
```
Replace `RUN_ID` with the winning run's id. Every LateNet config in `configs/` was trained the same
way, one `train_latenet.py` run per file (the two `baseline_` configs come from step 9). The test
command is for the single final evaluation: it refuses a run that already has test results.
`register_model.py` registers the run as `pulse-latenet` and promotes it to Production; it refuses
if a version already exists.

## Layout
```
src/        every script that runs: ingestion, preparation, training, evaluation, latency, registration
configs/    one YAML file per training configuration; configs/final.yaml is the winning one
data/       DVC-tracked data; only .dvc pointer files are in Git
models/     DVC-tracked winning model (latenet.pt); only its .dvc pointer is in Git
notebooks/  exploration only (01 historical files, 02 MBTA API)
tests/      tests for src/
assets/     images used in REPORT.md
REPORT.md   the model delivery report; ml_spec.md is the ML specification
```
