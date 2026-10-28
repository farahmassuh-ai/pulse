"""
Final evaluation of the saved LateNet model (Step 3).

Part 1: load models/latenet.pt and rebuild the trained network.

Why: Every later part of the evaluation script (metrics, tables, the one test run)
starts from this and it must rebuild the exact model that was trained:
- same shape
- same weights
- same vocabulary
- same scalar
"""
# lets type hints like "Path" work on any Python version
from __future__ import annotations
# Path: a tool for working with file paths
from pathlib import Path
# arrays of numbers (probabilities, labels)
import numpy as np
# tables (the split's rows)
import pandas as pd
# Pytorch: needed to load the saved file
import torch
# reads the command-line options
import argparse
# for the exit code at the end
import sys
# logs the test results
import mlflow
# our network class (src/latenet.py)
from latenet import LateNet
# repo's top folder
from train_baselines import ROOT, SPLIT_DIR, dvc_md5, export_runs, setup_mlflow
# the standard metric functions
from sklearn.metrics import (accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score)
# the label name ("late") and data helpers
from train_latenet import LABEL, batches, to_tensors, verify_split
# the input guards
from features import check_inputs

# where the winning model lives (default file)
MODEL_PATH = ROOT / "models" / "latenet.pt"

def load_model(path: Path = MODEL_PATH):
    """
    Rebuild the trained LateNet from its checkpoint.

    Returns (model, checkpoint). The model is in eval mode (dropout off), ready to score.
    weights_only=True is the safe loader: it refuses anything but plain data and tensors.
    """
    # open the saved file safely and outputs a dictionary
    checkpoint = torch.load(path, weights_only=True)
    # extract the config and vocabulary
    config, vocab = checkpoint["config"], checkpoint["vocab"]
    # known values per feature and yields an empty network of the right shape 
    model = LateNet({col: len(values) for col, values in vocab.items()},
                    config["embedding_dims"], config["hidden_layers"],
                    config["dropout"])
    # pour the trained weights into (checks shape)
    model.load_state_dict(checkpoint["state_dict"])
    # scoreing mode: dropout off
    model.eval()
    # return the model and the rest of the file
    return model, checkpoint


def predict(model, df: pd.DataFrame, checkpoint: dict, batch_size: int = 4096) -> np.ndarray:
    """Probability of being late (strictly more than 3 minutes) for every row of df."""
    # turn the rows into tensors, using the vocabulary and scalar saved with the model
    tensors = to_tensors(df, checkpoint["vocab"], checkpoint["scaler"])
    # dropout off (already set, but be safe)
    model.eval()
    # scoring only: no gradients needed (with is used to cleanup automatically)
    with torch.no_grad():
        # score batch by batch (rows stay in order), then join into one long list of logits
        logits = torch.cat([model(cat, num) for cat, num, _ in batches(tensors, batch_size)])
        # from raw scores to probabilities between 0 and 1
    return torch.sigmoid(logits).numpy()


def headline_metrics(y_true, prob, threshold: float) -> dict:
    """The main numbers, at 0.5 and at the chosen threshold. Flagged late = prob > threshold."""
    # labels as 0/1 (1 = late)
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob)
    # True where the model warns late
    flagged = prob > threshold
    # share the trips that really were late
    late_rate = y_true.mean()
    return {
        "accuracy_at_0.5": accuracy_score(y_true, prob > 0.5),          # acceptance bar
        "accuracy": accuracy_score(y_true, flagged),                    # at the chosen threshold
        "precision": precision_score(y_true, flagged, zero_division=0), # of the warnings, how many were right
        "recall": recall_score(y_true, flagged, zero_division=0),       # of the late buses, how many were caught
        "auc_roc": roc_auc_score(y_true, prob),                         # ranking quality (threshold-free)
        "pr_auc": average_precision_score(y_true, prob),                # precision-recall summary (threshold-free)
        "majority_baseline": float(max(late_rate, 1 - late_rate))       # accuracy of always quessing the bigger class
    }


def confusion(y_true, prob, threshold: float) -> dict:
    """The 2x2 confusion matrix at a threshold (flagged late = prob > threshold)."""
    y_true = np.asarray(y_true, dtype=int)
    flagged = np.asarray(prob) > threshold
    return {
        "tn": int(((y_true==0) & ~flagged).sum()),      # on time, no warning: correct
        "fp": int(((y_true==0) & flagged).sum()),       # on time, warned: false alarm
        "fn": int(((y_true==1) & ~flagged).sum()),      # late, no warning: missed delay
        "tp": int(((y_true==1) & flagged).sum())        # late, warned: correct
    }


def threshold_table(y_true, prob, thresholds=(0.35, 0.40, 0.45, 0.50, 0.55, 0.60)) -> pd.DataFrame:
    """For each threshold: accuracy, precision, recall, and how many stops get a warning."""
    rows = []
    for t in thresholds:
        c = confusion(y_true, prob, t)
        total, flagged = sum(c.values()), c["tp"] + c["fp"]    # all stops; stops warned as late
        rows.append({
            "threshold": t,
            "accuracy": (c["tp"] + c["tn"]) / total,             # share of all answers that were right
            "precision": c["tp"] / flagged if flagged else 0.0,  # of the warnings, how many were right
            "recall": c["tp"] / (c["tp"] + c["fn"]),             # of the late buses, how many were caught
            "share_flagged": flagged / total,                    # share of all stops that get a warning
            "on_time_flagged": c["fp"] / (c["fp"] + c["tn"]),    # share of on-time buses warned (= FPR)
        })
    return pd.DataFrame(rows)


def calibration_by_hour(df: pd.DataFrame, prob) -> pd.DataFrame:
    """For each scheduled hour: mean predicted probability vs actual late rate."""
    table = pd.DataFrame({"hour": df["hour"].to_numpy(),          # the hour of each trip
                          "predicted": prob,                       # the model's probability
                          "actual": df[LABEL].to_numpy(dtype=float)})   # 1.0 = really late
    out = table.groupby("hour").agg(rows=("actual", "size"),       # trips in that hour
                                    predicted=("predicted", "mean"),    # average prediction
                                    actual=("actual", "mean"))          # real late rate
    out["gap"] = out["predicted"] - out["actual"]    # + = too pessimistic, - = too optimistic
    return out


def route_check(df: pd.DataFrame, prob, min_rows: int = 1000, top: int = 5):
    """Top routes by predicted vs actual late rate, and how well the two agree (Pearson r).

    Only routes with at least min_rows trips, so each rate is based on plenty of data.
    """
    table = pd.DataFrame({"route": df["route_id"].to_numpy(),
                          "predicted": prob,
                          "actual": df[LABEL].to_numpy(dtype=float)})
    routes = table.groupby("route").agg(rows=("actual", "size"),
                                        predicted=("predicted", "mean"),
                                        actual=("actual", "mean"))
    routes = routes[routes["rows"] >= min_rows]                 # drop routes with too few trips
    top_predicted = routes.nlargest(top, "predicted")           # the model's 5 latest routes
    top_actual = routes.nlargest(top, "actual")                 # the really 5 latest routes
    pearson = float(routes["predicted"].corr(routes["actual"])) # 1 = perfect agreement, 0 = none
    return top_predicted, top_actual, pearson


def check_test_run(run_name: str, run_metrics: dict, checkpoint: dict) -> None:
    """Refuse to score the test split unless it is the first time, for the right model."""
    model_run = checkpoint["config"]["run_name"]
    if run_name != model_run:                       # the run id must belong to the saved model
        raise ValueError(f"run '{run_name}' is not the run that produced the saved model "
                         f"('{model_run}'); check --run-id and models/latenet.pt")
    already = sorted(key for key in run_metrics if key.startswith("test_"))
    if already:                                     # test results already logged: test was used
        raise ValueError(f"run '{run_name}' already has test metrics {already}; "
                         "the test split is evaluated only once")


def log_test_results(run_id: str, threshold: float, m: dict, c: dict, table, hours,
                     top_predicted, top_actual, pearson: float) -> None:
    """Log the one test evaluation onto the winning run, every name prefixed with test_."""
    with mlflow.start_run(run_id=run_id):              # reopen the finished winning run
        mlflow.log_metrics({f"test_{key}": value for key, value in m.items()})
        mlflow.log_metrics({f"test_{key}": value for key, value in c.items()})
        mlflow.log_metrics({"test_threshold": threshold, "test_route_pearson": pearson})
        mlflow.log_text(table.to_csv(index=False), "test/threshold_table.csv")
        mlflow.log_text(hours.to_csv(), "test/calibration_by_hour.csv")
        mlflow.log_text(top_predicted.to_csv(), "test/top_routes_predicted.csv")
        mlflow.log_text(top_actual.to_csv(), "test/top_routes_actual.csv")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the saved LateNet on one split.")
    parser.add_argument("split", choices=["val", "test"], help="which split to score")
    parser.add_argument("--threshold", type=float, default=0.5, help="operating threshold")
    parser.add_argument("--run-id", help="the winning MLflow run (test only; results are logged to it)")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.run_id:
        parser.error("the test split needs --run-id: its results are logged to the winning run")
    if args.split == "val" and args.run_id:
        parser.error("--run-id is only for the test split")

    model, checkpoint = load_model()
    if args.split == "test":                        # check the rules BEFORE opening the test data
        setup_mlflow()
        run = mlflow.get_run(args.run_id)
        check_test_run(run.info.run_name, run.data.metrics, checkpoint)

    path = SPLIT_DIR / f"{args.split}.parquet"
    verify_split(path, dvc_md5(args.split))         # the file is the version DVC recorded
    df = pd.read_parquet(path)
    check_inputs(df)                                # the same input guards as training
    prob = predict(model, df, checkpoint)
    y = df[LABEL].to_numpy(dtype=int)
    t = args.threshold

    m = headline_metrics(y, prob, t)
    c = confusion(y, prob, t)
    table = threshold_table(y, prob)
    hours = calibration_by_hour(df, prob)
    top_predicted, top_actual, pearson = route_check(df, prob)

    print(f"\n{args.split}: {len(df):,} rows, late rate {y.mean():.1%}, threshold {t}")
    print(f"majority baseline accuracy {m['majority_baseline']:.4f}")
    print(f"accuracy at 0.5 {m['accuracy_at_0.5']:.4f} | at {t}: {m['accuracy']:.4f}")
    print(f"AUC-ROC {m['auc_roc']:.4f} | PR-AUC {m['pr_auc']:.4f}")
    print(f"precision {m['precision']:.4f} | recall {m['recall']:.4f} (at {t})")
    print("\nconfusion matrix at", t, "(rows = actual, columns = predicted)")
    print(f"{'':>10} {'on time':>12} {'late':>12}")
    print(f"{'on time':>10} {c['tn']:>12,} {c['fp']:>12,}")
    print(f"{'late':>10} {c['fn']:>12,} {c['tp']:>12,}")
    print("\nthreshold table\n" + table.round(4).to_string(index=False))
    print("\ncalibration by hour (gap: + too pessimistic, - too optimistic)\n"
          + hours.round(4).to_string())
    print(f"\nroutes with 1,000+ rows: Pearson r, predicted vs actual late rate = {pearson:.3f}")
    print("top 5 by predicted late rate\n" + top_predicted.round(4).to_string())
    print("top 5 by actual late rate\n" + top_actual.round(4).to_string())

    if args.split == "test":
        log_test_results(args.run_id, t, m, c, table, hours, top_predicted, top_actual, pearson)
        export_runs()                               # mlflow_runs.csv now includes the test metrics
        print(f"\ntest results logged to run {args.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
