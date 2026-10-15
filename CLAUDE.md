# Pulse (Reliability Signal)
Predicts whether an MBTA bus will be more than 3 minutes late at a timepoint stop.
Pipeline: MBTA files + API -> Postgres (stop_events) -> splits (DVC) -> models (MLflow). Spec in ml_spec.md.

## Commands (run from the repo root, Python 3.12 via `python`, not `python3`)
- Tests: python -m pytest
- Load historical files: python src/load_historical.py
- Daily API pull: python src/pull_daily.py
- Make splits: python src/make_splits.py
- Train baselines: python src/train_baselines.py

## Rules
- "Late" = strictly more than 3 minutes after scheduled (delay_min > 3).
- Features must be known before the trip runs. Nothing derived from `actual` or `headway`.
- Splits are time-based only (train Apr-May, val Jun 1-15, test Jun 16-30). Never a random split.
- The test split is opened once, at the final evaluation.
- Timestamps are Boston wall-clock; store route_id and direction in the files' form ("01", "Inbound").
- Data logic lives in src/; notebooks explore and import from src/.
- Never modify data/raw/. Never commit .env, data files, mlruns/ or mlflow.db.

## Working agreement
- Explain the reasoning before writing code; keep each change small and scoped.
- Verify every change: run it, check it against the spec and the task rules, and report what was and wasn't verified.
