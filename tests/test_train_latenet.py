import hashlib
import math
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from torch import nn

from features import CATEGORICAL, NUMERIC, build_vocab, fit_scaler
from latenet import LateNet
from train_latenet import (EarlyStopping, batches, check_feature_lists, evaluate, file_md5,
                           to_tensors, train_one_epoch, verify_split)

ROOT = Path(__file__).resolve().parent.parent


# ---------- data safety ----------

def test_file_md5_matches_hashlib(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"pulse" * 1000)
    assert file_md5(path) == hashlib.md5(b"pulse" * 1000).hexdigest()


def test_changed_file_is_caught(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"version 1")
    recorded = file_md5(path)
    path.write_bytes(b"version 2")
    with pytest.raises(ValueError, match="pointer says"):
        verify_split(path, recorded)


def test_config_feature_lists_match_the_code():
    config = yaml.safe_load((ROOT / "configs" / "latenet_base.yaml").read_text())
    check_feature_lists(config)                     # no error raised


def test_feature_list_mismatch_is_caught():
    config = {"features": {"categorical": CATEGORICAL[:-1], "numeric": NUMERIC}}
    with pytest.raises(ValueError, match="features.categorical"):
        check_feature_lists(config)


# ---------- tensors ----------

def sample():
    return pd.DataFrame({
        "route_id": ["01", "01", "28"],
        "stop_id": ["a", "b", "a"],
        "hour": pd.array([7, 8, 9], dtype="int32"),
        "day_of_week": pd.array([0, 5, 6], dtype="int32"),
        "point_type": ["Midpoint"] * 3,
        "time_point_order": pd.array([1, 2, 3], dtype="Int64"),
        "direction_id": ["Inbound", "Outbound", "Inbound"],
        "standard_type": ["Headway", "Schedule", "Headway"],
        "late": [True, False, True],
    })


def test_to_tensors_types_and_shapes():
    df = sample()
    categorical, numeric, label = to_tensors(df, build_vocab(df), fit_scaler(df))
    assert categorical.dtype == torch.int64 and categorical.shape == (3, len(CATEGORICAL))
    assert numeric.dtype == torch.float32 and numeric.shape == (3, len(NUMERIC))
    assert label.dtype == torch.float32 and label.tolist() == [1.0, 0.0, 1.0]


# ---------- batching ----------

def one_epoch(tensors, batch_size, generator=None):
    return torch.cat([batch[0] for batch in batches(tensors, batch_size, generator)])


def test_each_epoch_sees_every_row_once_in_a_new_order():
    rows = (torch.arange(10),)
    g = torch.Generator().manual_seed(42)
    epoch1, epoch2 = one_epoch(rows, 4, g), one_epoch(rows, 4, g)
    assert sorted(epoch1.tolist()) == sorted(epoch2.tolist()) == list(range(10))
    assert not torch.equal(epoch1, epoch2)


def test_same_seed_gives_the_same_order():
    rows = (torch.arange(10),)
    first = one_epoch(rows, 4, torch.Generator().manual_seed(42))
    again = one_epoch(rows, 4, torch.Generator().manual_seed(42))
    assert torch.equal(first, again)


def test_validation_batches_keep_the_original_order():
    rows = (torch.arange(10),)
    assert [b[0].tolist() for b in batches(rows, 4)] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_all_tensors_in_a_batch_stay_aligned():
    rows = (torch.arange(10), torch.arange(10) * 100)
    for ids, hundreds in batches(rows, 3, torch.Generator().manual_seed(0)):
        assert torch.equal(ids * 100, hundreds)


# ---------- one epoch ----------

class FixedScores(nn.Module):
    """Stand-in model: its 'score' for each trip is simply the first numeric input."""

    def forward(self, categorical, numeric):
        return numeric[:, 0]


def scored(logits, labels):
    n = len(logits)
    return (torch.zeros(n, 5, dtype=torch.int64),
            torch.tensor(logits, dtype=torch.float32).reshape(n, 1),
            torch.tensor(labels, dtype=torch.float32))


def test_evaluate_matches_hand_calculation_across_batches():
    logits, labels = [2.0, -1.0, 0.5, -3.0], [1, 0, 0, 0]
    result = evaluate(FixedScores(), scored(logits, labels), batch_size=3)
    p = [1 / (1 + math.exp(-z)) for z in logits]
    loss = -sum(y * math.log(q) + (1 - y) * math.log(1 - q) for y, q in zip(labels, p)) / 4
    assert result["loss"] == pytest.approx(loss, rel=1e-5)
    assert result["auc"] == 1.0                     # the late trip outranks every on-time one
    assert result["accuracy"] == 0.75               # 0.5 is called late but was on time


def test_a_score_of_zero_costs_log_2():
    result = evaluate(FixedScores(), scored([0.0, 0.0], [1, 0]), batch_size=2)
    assert result["loss"] == pytest.approx(math.log(2))   # the 0.69 anchor from chapter 9


def test_a_score_of_exactly_zero_counts_as_on_time():
    # Both classes are needed, or AUC is undefined. With ">=" the first trip would be
    # called late and accuracy would drop to 0.5.
    assert evaluate(FixedScores(), scored([0.0, 1.0], [0, 1]), batch_size=2)["accuracy"] == 1.0


def tiny_problem(n=512):
    """Late exactly when direction == 1, so a working loop must learn it."""
    g = torch.Generator().manual_seed(3)
    categorical = torch.randint(1, 3, (n, 5), generator=g)
    numeric = torch.rand(n, 4, generator=g)
    numeric[:, 0] = torch.randint(0, 2, (n,), generator=g).float()
    return categorical, numeric, numeric[:, 0].clone()


def tiny_model():
    torch.manual_seed(0)
    sizes = {c: 2 for c in CATEGORICAL}
    return LateNet(sizes, {c: 2 for c in CATEGORICAL}, [16], dropout=0.0)


def test_training_loop_learns_a_simple_pattern():
    data, model = tiny_problem(), tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    start = evaluate(model, data, 128)["loss"]
    g = torch.Generator().manual_seed(42)
    for _ in range(5):
        train_one_epoch(model, data, optimizer, batch_size=64, generator=g)
    end = evaluate(model, data, 128)
    assert end["loss"] < start and end["loss"] < 0.3
    assert end["auc"] > 0.99


def test_evaluate_learns_nothing_and_turns_dropout_off():
    data, model = tiny_problem(), tiny_model()
    before = [p.clone() for p in model.parameters()]
    model.train()
    evaluate(model, data, 128)
    assert not model.training
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


# ---------- early stopping ----------

def run(scores, patience=2):
    """Feed validation AUCs epoch by epoch (epochs numbered from 1); return the stopper and its answers."""
    stopper, model = EarlyStopping(patience), nn.Linear(1, 1)
    answers = [stopper.step(s, epoch, model) for epoch, s in enumerate(scores, start=1)]
    return stopper, answers


def test_stops_after_patience_epochs_without_improvement():
    stopper, answers = run([0.70, 0.72, 0.71, 0.715])
    assert answers == [False, False, False, True]
    assert stopper.best_epoch == 2 and stopper.best_score == 0.72


def test_higher_auc_is_better():
    stopper, _ = run([0.75, 0.74, 0.73])
    assert stopper.best_epoch == 1


def test_a_tie_does_not_count_as_improvement():
    stopper, answers = run([0.72, 0.72, 0.72])
    assert stopper.best_epoch == 1 and answers[-1] is True


def test_an_improvement_resets_the_count():
    _, answers = run([0.70, 0.69, 0.71, 0.70, 0.69])
    assert answers == [False, False, False, False, True]


def test_restores_the_best_weights_not_the_last_ones():
    stopper, model = EarlyStopping(patience=2), nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
    stopper.step(0.80, 1, model)                    # best epoch: weight 1.0
    with torch.no_grad():
        model.weight.fill_(5.0)                     # later training changes the live weights
    stopper.step(0.70, 2, model)
    stopper.restore_best(model)
    assert model.weight.item() == 1.0
