"""Step 4: train the two baselines and track them in MLflow.

Run from the repo root:
    python3 src/train_baselines.py                    # both configs in configs/baseline_*.yaml
    python3 src/train_baselines.py configs/X.yaml     # one config

Train on data/splits/train.parquet, score on data/splits/val.parquet (the test split is
never opened here). Each run logs the config values, the git commit and the DVC md5 of
each split as parameters, the config file as an artifact, and accuracy / AUC-ROC /
PR-AUC on validation as metrics. Afterwards the run table is exported to mlflow_runs.csv.
MLflow store is local: mlflow.db (run table) and mlruns/ (artifacts), both gitignored.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "splits"
EXPERIMENT = "pulse"


# ---------- models ----------

class MajorityClass:
    """Always predicts the training split's larger class."""

    def fit(self, X, y):
        self.positive_rate_ = float(np.mean(y))
        self.majority_ = self.positive_rate_ >= 0.5
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_, dtype=bool)

    def predict_proba(self, X):
        # Same score for every row -> AUC-ROC is 0.5 by construction.
        p = np.full(len(X), self.positive_rate_)
        return np.column_stack([1 - p, p])


def build_model(config: dict):
    if config["model"] == "majority":
        return MajorityClass()
    if config["model"] == "logreg":
        encoder = ColumnTransformer(
            [("onehot", OneHotEncoder(handle_unknown="ignore"), config["features"])]
        )
        return Pipeline([
            ("encode", encoder),
            ("model", LogisticRegression(**config.get("params", {}))),
        ])
    raise ValueError(f"unknown model {config['model']!r}")


def to_categories(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """One-hot treats every feature as a category, so give them all one type."""
    return df[features].astype(str)


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    return {
        "val_accuracy": accuracy_score(y, model.predict(X)),
        "val_auc_roc": roc_auc_score(y, proba),
        "val_pr_auc": average_precision_score(y, proba),
    }


# ---------- run metadata ----------

def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dvc_md5(split: str) -> str:
    pointer = SPLIT_DIR / f"{split}.parquet.dvc"
    if not pointer.exists():
        return "not-tracked"
    return yaml.safe_load(pointer.read_text())["outs"][0]["md5"]


def flatten(config: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = ",".join(map(str, value)) if isinstance(value, list) else value
    return flat


def setup_mlflow(experiment: str = EXPERIMENT):
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment)


# ---------- main ----------

def run(config_path: Path, train: pd.DataFrame, val: pd.DataFrame) -> dict:
    config = yaml.safe_load(config_path.read_text())
    features, label = config["features"], config["label"]
    model = build_model(config)
    model.fit(to_categories(train, features), train[label].to_numpy())
    metrics = evaluate(model, to_categories(val, features), val[label].to_numpy())

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params(flatten(config))
        mlflow.log_params({
            "git_commit": git_commit(),
            "dvc_md5_train": dvc_md5("train"),
            "dvc_md5_val": dvc_md5("val"),
            "dvc_md5_test": dvc_md5("test"),
            "train_rows": len(train),
            "val_rows": len(val),
        })
        mlflow.log_artifact(str(config_path))
        mlflow.log_metrics(metrics)
    return metrics


def main(argv: list[str]) -> int:
    configs = [Path(a) for a in argv] or sorted((ROOT / "configs").glob("baseline_*.yaml"))
    if not configs:
        print("No configs found in configs/")
        return 1
    train = pd.read_parquet(SPLIT_DIR / "train.parquet")
    val = pd.read_parquet(SPLIT_DIR / "val.parquet")
    print(f"train {len(train):,} rows (late {train['late'].mean():.1%}), "
          f"val {len(val):,} rows (late {val['late'].mean():.1%})")

    setup_mlflow()
    for path in configs:
        metrics = run(path, train, val)
        print(f"{path.stem:<20} accuracy {metrics['val_accuracy']:.3f}  "
              f"AUC-ROC {metrics['val_auc_roc']:.3f}  PR-AUC {metrics['val_pr_auc']:.3f}")

    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    runs.to_csv(ROOT / "mlflow_runs.csv", index=False)
    print(f"Exported {len(runs)} run(s) to mlflow_runs.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
