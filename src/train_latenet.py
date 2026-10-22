"""Step 1.3/1.4: train LateNet from the frozen splits, tracked in MLflow.

Part 6a: load the splits, check them, and turn them into tensors.
Part 6b: one epoch of training, and scoring a split without learning.
The test split is never opened here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from features import CATEGORICAL, NUMERIC, check_inputs, encode, numeric_inputs
from train_baselines import SPLIT_DIR, dvc_md5

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
        # late = score above 0 (probability above 0.5), the same rule as sklearn's predict()
        "accuracy": float(((logits > 0).float() == label).float().mean()),
    }
