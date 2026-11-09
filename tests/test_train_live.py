import json

import numpy as np
import pytest
import torch
import yaml

from live_monitor import LiveMonitor, binary_metrics
from test_train_latenet import ROOT, synthetic_split
from train_latenet import train
from train_live import train_live


def small_config():
    config = yaml.safe_load((ROOT / "configs" / "final.yaml").read_text())
    config["training"].update({"batch_size": 64, "learning_rate": 0.05})
    return config


def no_nan(token):
    raise ValueError(f"{token} in the JSON: browsers cannot read it")


# ---------- the monitor must not change training ----------

def test_monitored_run_matches_the_real_one_exactly(tmp_path):
    config, tr, va = small_config(), synthetic_split(600, 1), synthetic_split(200, 2)
    real = train(config, tr, va, max_epochs=6)
    live = train_live(config, tr, va, max_epochs=6, monitor=LiveMonitor(tmp_path, log_every=3))
    assert live["best_epoch"] == real["best_epoch"]
    assert [h["val_auc_roc"] for h in live["history"]] == [h["val_auc_roc"] for h in real["history"]]
    a, b = real["model"].state_dict(), live["model"].state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)                 # every weight identical


# ---------- what the dashboard reads ----------

def test_snapshot_file_has_what_the_dashboard_needs(tmp_path):
    train_live(small_config(), synthetic_split(600, 1), synthetic_split(200, 2), max_epochs=3,
               monitor=LiveMonitor(tmp_path, log_every=3))
    run_id = (tmp_path / "live_runs" / "latest.txt").read_text()
    text = (tmp_path / "live_runs" / run_id / "state.json").read_text()
    state = json.loads(text, parse_constant=no_nan)                  # valid JSON, no NaN

    assert state["run"]["status"] in ("finished", "stopped early")
    assert state["progress"]["phase"] == "done"
    rows = state["epochs"]
    assert rows[0]["epoch"] == 0 and rows[0]["train_loss"] is None  # the untrained check comes first
    assert all(r[k] is not None for r in rows[1:]
               for k in ("train_loss", "val_loss", "val_auc", "val_accuracy"))
    assert state["batch_loss"] and all(len(p) == 2 for p in state["batch_loss"])

    best = state["early_stopping"]["best_epoch"]
    hist, op = state["val"]["hist"], state["val"]["operating"]
    assert len(hist["late"]) == len(hist["ontime"]) == 40
    assert sum(hist["late"]) + sum(hist["ontime"]) == 200            # every validation row, once
    assert op["tp"] + op["fp"] + op["fn"] + op["tn"] == 200
    for split in ("train", "val"):
        assert len(state["fit"][split]["hour"]["pred"]) == 24
        assert state["fit"][split]["hour_gap"] is not None
    assert state["fit"]["epoch"] == state["val"]["epoch"] == best    # at the end it shows the kept epoch
    assert (tmp_path / "live_runs" / run_id / "epochs" / "epoch_01.json").exists()


# ---------- small helpers ----------

def test_binary_metrics_by_hand():
    prob = np.array([0.9, 0.6, 0.4, 0.2])
    y = np.array([1, 0, 1, 0])
    m = binary_metrics(prob, y, 0.45)                                 # warns on the first two
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["on_time_warned"] == pytest.approx(0.5)
