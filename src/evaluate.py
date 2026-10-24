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
# our network class (src/latenet.py)
from latenet import LateNet
# repo's top folder
from train_baselines import ROOT
# the standard metric functions
from sklearn.metrics import (accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score)
# the label name ("late") and data helpers
from train_latenet import LABEL, batches, to_tensors

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