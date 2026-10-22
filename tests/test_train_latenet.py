import hashlib
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from features import CATEGORICAL, NUMERIC, build_vocab, fit_scaler
from train_latenet import batches, check_feature_lists, file_md5, to_tensors, verify_split

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
