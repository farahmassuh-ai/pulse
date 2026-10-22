"""Step 1.3/1.4: train LateNet from the frozen splits, tracked in MLflow.

Part 6a: load the splits, check them, and turn them into tensors.
Part 6b: one epoch of training, and scoring a split without learning.
Part 6c: early stopping on validation AUC, keeping the best epoch's weights.
Part 6d-1: one full training run from a YAML config, plus a --pilot check.
Part 6d-2: MLflow logging (params, per-epoch curves, final metrics, artifacts, model)
           and the checkpoint in models/latenet.pt (weights + config + vocabulary + scaler).

Run from the repo root:
    python src/train_latenet.py configs/latenet_base.yaml           # a real run
    python src/train_latenet.py configs/latenet_base.yaml --pilot   # quick check
The test split is never opened here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from features import (CATEGORICAL, NUMERIC, build_vocab, check_inputs, encode, fit_scaler,
                      numeric_inputs)
from latenet import LateNet
from train_baselines import ROOT, SPLIT_DIR, dvc_md5, flatten, git_commit, setup_mlflow

LABEL = "late"


def file_md5(path: Path) -> str:
    """md5 of a file's contents, read in 1 MB chunks (the same hash DVC records)."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_split(path: Path, expected_md5: str) -> None:
    """The file on disk must be the version its .dvc pointer (and so MLflow) records."""
    actual = file_md5(path)
    if actual != expected_md5:
        raise ValueError(f"{path.name} has md5 {actual}, but its .dvc pointer says "
                         f"{expected_md5}. Run `dvc status` before training.")


def check_feature_lists(config: dict) -> None:
    """The config's feature lists must match the ones the code actually uses."""
    for key, in_code in [("categorical", CATEGORICAL), ("numeric", NUMERIC)]:
        if config["features"][key] != in_code:
            raise ValueError(f"config features.{key} {config['features'][key]} "
                             f"does not match the code's {in_code}")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and val only, each checked against its DVC pointer and the input guards."""
    frames = []
    for name in ["train", "val"]:
        path = SPLIT_DIR / f"{name}.parquet"
        verify_split(path, dvc_md5(name))
        df = pd.read_parquet(path)
        check_inputs(df)
        frames.append(df)
    return frames[0], frames[1]


def to_tensors(df: pd.DataFrame, vocab: dict, scaler: dict):
    """(categorical int64, numeric float32, label float32) tensors for one split."""
    categorical = torch.from_numpy(encode(df, vocab).to_numpy())
    numeric = torch.from_numpy(numeric_inputs(df, scaler).to_numpy())
    label = torch.from_numpy(df[LABEL].to_numpy(dtype="float32"))
    return categorical, numeric, label


def batches(tensors, batch_size: int, generator: torch.Generator | None = None):
    """Yield one epoch of batches.

    With a generator (training): a fresh random order on every call, reproducible
    from the seed. Without one (validation): rows stay in their original order.
    """
    n = len(tensors[0])
    order = torch.randperm(n, generator=generator) if generator is not None else torch.arange(n)
    for start in range(0, n, batch_size):
        rows = order[start:start + batch_size]
        yield tuple(t[rows] for t in tensors)


def train_one_epoch(model, tensors, optimizer, batch_size: int, generator: torch.Generator) -> None:
    """One pass over train in a fresh shuffled order; the weights learn from every batch."""
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()                                   # dropout on
    for categorical, numeric, label in batches(tensors, batch_size, generator):
        optimizer.zero_grad()                       # forget the previous batch's gradients
        loss = loss_fn(model(categorical, numeric), label)
        loss.backward()                             # gradients for every weight
        optimizer.step()                            # move the weights


def evaluate(model, tensors, batch_size: int) -> dict[str, float]:
    """Loss, AUC and accuracy on one whole split: dropout off, nothing learned."""
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(c, n) for c, n, _ in batches(tensors, batch_size)])
    label = tensors[2]
    return {
        "loss": nn.functional.binary_cross_entropy_with_logits(logits, label).item(),
        "auc": float(roc_auc_score(label.numpy(), logits.numpy())),
        "pr_auc": float(average_precision_score(label.numpy(), logits.numpy())),
        # late = score above 0 (probability above 0.5), the same rule as sklearn's predict()
        "accuracy": float(((logits > 0).float() == label).float().mean()),
    }


class EarlyStopping:
    """Watch validation AUC (higher is better) and keep the best epoch's weights.

    Stops once AUC has not improved for `patience` epochs in a row.
    """

    def __init__(self, patience: int):
        self.patience = patience
        self.best_score = float("-inf")
        self.best_epoch = None
        self.best_state = None
        self.epochs_without_improvement = 0

    def step(self, score: float, epoch: int, model) -> bool:
        """Record one epoch's validation AUC. Returns True when training should stop."""
        if score > self.best_score:                 # strictly better; a tie is not progress
            self.best_score, self.best_epoch = score, epoch
            self.best_state = copy.deepcopy(model.state_dict())   # a frozen copy, not live weights
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience

    def restore_best(self, model) -> None:
        """Put the best epoch's weights back into the model."""
        if self.best_state is None:
            raise RuntimeError("no epoch improved on -inf; was every validation AUC NaN?")
        model.load_state_dict(self.best_state)


def make_optimizer(model, training: dict):
    """Adam as named in the config; AdamW if weight decay is ever used (chapter 10)."""
    name, lr, decay = training["optimizer"], training["learning_rate"], training["weight_decay"]
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    raise ValueError(f"unknown optimizer {name!r}; expected 'adam' or 'adamw'")


def train(config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, max_epochs: int) -> dict:
    """One training run: fit on train, early-stop on val AUC, return the best model and its record."""
    check_feature_lists(config)
    t = config["training"]
    torch.manual_seed(t["seed"])                    # before the model exists: same start every rerun

    vocab, scaler = build_vocab(train_df), fit_scaler(train_df)
    train_t, val_t = to_tensors(train_df, vocab, scaler), to_tensors(val_df, vocab, scaler)
    model = LateNet({col: len(vocab[col]) for col in CATEGORICAL}, config["embedding_dims"],
                    config["hidden_layers"], config["dropout"])
    optimizer = make_optimizer(model, t)
    generator = torch.Generator().manual_seed(t["seed"])
    stopper = EarlyStopping(t["patience"])
    size = t["batch_size"]

    untrained = evaluate(model, val_t, size)        # the 0.69 check, before any learning
    print(f"epoch  0 (untrained)                  val loss {untrained['loss']:.4f}  "
          f"val AUC {untrained['auc']:.4f}")
    history = []
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        train_one_epoch(model, train_t, optimizer, size, generator)
        on_train, on_val = evaluate(model, train_t, size), evaluate(model, val_t, size)
        history.append({"epoch": epoch, "train_loss": on_train["loss"], "val_loss": on_val["loss"],
                        "val_auc_roc": on_val["auc"], "val_accuracy": on_val["accuracy"],
                        "seconds": time.time() - start})
        print(f"epoch {epoch:2d}  train loss {on_train['loss']:.4f}  val loss {on_val['loss']:.4f}  "
              f"val AUC {on_val['auc']:.4f}  val acc {on_val['accuracy']:.4f}  "
              f"({history[-1]['seconds']:.0f}s)")
        if stopper.step(on_val["auc"], epoch, model):
            print(f"early stop: no AUC improvement for {t['patience']} epochs")
            break

    stopper.restore_best(model)
    final = evaluate(model, val_t, size)            # re-score the weights we keep
    return {"model": model, "vocab": vocab, "scaler": scaler, "untrained": untrained,
            "history": history, "final": final, "best_epoch": stopper.best_epoch,
            "n_params": sum(p.numel() for p in model.parameters()),
            "train_rows": len(train_df), "val_rows": len(val_df)}


def save_checkpoint(path: Path, result: dict, config: dict) -> None:
    """Best weights + config + vocabulary + scaler in one file: everything needed to score
    a single new trip. Only plain types and tensors, so it loads with weights_only=True."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": result["model"].state_dict(), "config": config,
                "vocab": result["vocab"], "scaler": result["scaler"],
                "best_epoch": result["best_epoch"]}, path)


def log_run(config: dict, config_path: Path, result: dict, experiment: str,
            run_name: str, pilot: bool) -> str:
    """Record one run in MLflow; returns its run id. The model is logged, not registered."""
    setup_mlflow(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("pilot", str(pilot).lower())
        mlflow.log_params(flatten(config))
        mlflow.log_params({
            "git_commit": git_commit(),
            "dvc_md5_train": dvc_md5("train"),
            "dvc_md5_val": dvc_md5("val"),
            "dvc_md5_test": dvc_md5("test"),      # read from the pointer file; test is never opened
            "train_rows": result["train_rows"],
            "val_rows": result["val_rows"],
            "n_params": result["n_params"],
        })
        mlflow.log_params({f"vocab_size.{col}": len(v) for col, v in result["vocab"].items()})

        untrained = result["untrained"]             # step 0: before any learning (the 0.69 check)
        mlflow.log_metrics({"epoch_val_loss": untrained["loss"],
                            "epoch_val_auc_roc": untrained["auc"]}, step=0)
        for row in result["history"]:               # the curves
            mlflow.log_metrics({"epoch_train_loss": row["train_loss"],
                                "epoch_val_loss": row["val_loss"],
                                "epoch_val_auc_roc": row["val_auc_roc"],
                                "epoch_val_accuracy": row["val_accuracy"],
                                "epoch_seconds": row["seconds"]}, step=row["epoch"])

        final = result["final"]                     # once, for the weights we keep
        mlflow.log_metrics({"val_auc_roc": final["auc"], "val_pr_auc": final["pr_auc"],
                            "val_accuracy": final["accuracy"], "val_log_loss": final["loss"],
                            "best_epoch": result["best_epoch"]})

        mlflow.log_artifact(str(config_path))
        mlflow.log_dict({"vocab": result["vocab"], "scaler": result["scaler"]}, "vocab.json")
        # "pickle": MLflow 3's newer "pt2" format needs torch >= 2.4, and this project runs 2.2.2.
        info = mlflow.pytorch.log_model(result["model"], name="model",
                                        serialization_format="pickle",
                                        code_paths=[str(ROOT / "src" / "latenet.py"),
                                                    str(ROOT / "src" / "features.py")])
        mlflow.set_tag("model_uri", info.model_uri)
        return run.info.run_id


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train LateNet from a YAML config.")
    parser.add_argument("config", type=Path, help="e.g. configs/latenet_base.yaml")
    parser.add_argument("--pilot", action="store_true",
                        help="quick end-to-end check using the config's 'pilot' section")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())

    train_df, val_df = load_splits()
    max_epochs = config["training"]["max_epochs"]
    if args.pilot:
        pilot = config["pilot"]
        train_df = train_df.sample(n=pilot["train_rows"], random_state=config["training"]["seed"])
        max_epochs = pilot["max_epochs"]
    print(f"{'PILOT ' if args.pilot else ''}{config['run_name']}: train {len(train_df):,} rows, "
          f"val {len(val_df):,} rows, up to {max_epochs} epoch(s)")

    result = train(config, train_df, val_df, max_epochs)
    final = result["final"]
    print(f"best epoch {result['best_epoch']}: val AUC {final['auc']:.4f}  "
          f"PR-AUC {final['pr_auc']:.4f}  accuracy {final['accuracy']:.4f}  "
          f"log loss {final['loss']:.4f}  ({result['n_params']:,} parameters)")
    if result["best_epoch"] == max_epochs and not args.pilot:
        print("note: the best epoch is the last allowed one; the run may still have been improving")

    section = config["pilot"] if args.pilot else config
    checkpoint = ROOT / section["output_path"]
    save_checkpoint(checkpoint, result, config)
    experiment = section["experiment"]
    run_name = config["run_name"] + ("_pilot" if args.pilot else "")
    run_id = log_run(config, args.config, result, experiment, run_name, args.pilot)
    print(f"saved {checkpoint.relative_to(ROOT)}; MLflow run {run_id} in experiment '{experiment}'")

    if not args.pilot:                              # the graded run table, as the baselines export it
        runs = mlflow.search_runs(experiment_names=[experiment])
        runs.to_csv(ROOT / "mlflow_runs.csv", index=False)
        print(f"Exported {len(runs)} run(s) to mlflow_runs.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
