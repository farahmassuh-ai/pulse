"""Step 1 inputs: turn split rows into the integer indexes LateNet embeds.

The vocabulary is built from the train split only. Index 0 is reserved for
values never seen in training; LateNet keeps that embedding row at zero
(padding_idx=0), so an unseen value adds nothing to the prediction.
"""
from __future__ import annotations

import pandas as pd

CATEGORICAL = ["route_id", "stop_id", "hour", "day_of_week", "point_type"]
UNKNOWN = 0
FLAG_VALUES = {                      # each flag column's only values, and the 0/1 they become
    "direction_id": {"Inbound": 1, "Outbound": 0},
    "standard_type": {"Schedule": 1, "Headway": 0},
}
WEEKEND_DAYS = (5, 6)                # day_of_week: 0 = Monday ... 5 = Saturday, 6 = Sunday
NUMERIC = ["direction", "weekend", "standard_type", "time_point_order"]


def check_inputs(df: pd.DataFrame) -> None:
    """Stop with a clear message instead of quietly training on broken inputs."""
    problems = []
    for col in CATEGORICAL:
        if pd.api.types.is_float_dtype(df[col]):
            problems.append(f"{col} is a float column (index columns must be text or integer)")
    for col in [*CATEGORICAL, "time_point_order", *FLAG_VALUES]:
        missing = int(df[col].isna().sum())
        if missing:
            problems.append(f"{col} has {missing:,} missing values")
    for col, allowed in FLAG_VALUES.items():
        unexpected = set(df[col].dropna().astype(str)) - set(allowed)
        if unexpected:
            problems.append(f"{col} has unexpected values {sorted(unexpected)}, "
                            f"expected only {sorted(allowed)}")
    if problems:
        raise ValueError("Input check failed:\n  " + "\n  ".join(problems))


def build_vocab(train: pd.DataFrame, columns: list[str] = CATEGORICAL) -> dict[str, dict[str, int]]:
    """value -> index (1, 2, 3, ...) for each categorical column, from train only.

    Keys are stored as text so the vocabulary saves cleanly next to the weights
    and an hour of 7 means the same thing whether it arrives as int32 or int.
    """
    vocab = {}
    for col in columns:                     # all five, or fewer for an ablation run
        values = sorted(train[col].astype(str).unique())
        vocab[col] = {value: i for i, value in enumerate(values, start=1)}
    return vocab


def encode(df: pd.DataFrame, vocab: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Replace each categorical value by its index; values not in vocab -> 0."""
    out = pd.DataFrame(index=df.index)
    for col in vocab:                       # the columns the vocabulary was built for, in order
        out[col] = df[col].astype(str).map(vocab[col]).fillna(UNKNOWN).astype("int64")
    return out


def fit_scaler(train: pd.DataFrame) -> dict[str, float]:
    """Min and max of time_point_order in train, kept as plain floats for saving."""
    low, high = float(train["time_point_order"].min()), float(train["time_point_order"].max())
    if high == low:
        raise ValueError(f"time_point_order is constant ({low}) in train; it cannot be scaled")
    return {"min": low, "max": high}


def numeric_inputs(df: pd.DataFrame, scaler: dict[str, float]) -> pd.DataFrame:
    """The four numeric inputs: three 0/1 flags and the min-max scaled timepoint order.

    Train maps to [0, 1]; a value outside the train range lands slightly outside it.
    """
    out = pd.DataFrame(index=df.index)
    out["direction"] = df["direction_id"].map(FLAG_VALUES["direction_id"])
    out["weekend"] = df["day_of_week"].isin(WEEKEND_DAYS).astype(int)
    out["standard_type"] = df["standard_type"].map(FLAG_VALUES["standard_type"])
    order = df["time_point_order"].astype("float64")
    out["time_point_order"] = (order - scaler["min"]) / (scaler["max"] - scaler["min"])
    return out[NUMERIC].astype("float32")
