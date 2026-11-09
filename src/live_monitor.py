"""Live training monitor for LateNet: watches a training run and writes the snapshots
that dashboard/index.html draws while the network trains.

It only observes. It reads the loss and the model's predictions, but never changes
the weights, never switches the model between train and eval mode during an epoch,
and never draws random numbers from torch (that would change the dropout masks and
the shuffling, and the run would no longer match the real one).

Files, under live_runs/<run id>/:
    state.json            the latest snapshot (rewritten atomically, read by the dashboard)
    epochs/epoch_XX.json  a copy of the snapshot at the end of every epoch
and live_runs/latest.txt holds the id of the newest run.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

OPERATING_THRESHOLD = 0.45          # the threshold chosen on validation (REPORT section 5)
HIST_BINS = 40                      # bars in the "can it tell late from on-time?" chart


def _r(x, digits: int = 5):
    """Round floats (and lists of floats) so the JSON stays small."""
    if isinstance(x, (list, tuple)):
        return [_r(v, digits) for v in x]
    x = float(x)
    return round(x, digits) if np.isfinite(x) else None      # JSON has no NaN or infinity


def write_json_atomic(path: Path, data: dict) -> None:
    """Write to a temporary file, then swap it in, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    os.replace(tmp, path)


def binary_metrics(prob: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Confusion counts plus precision, recall and the share of on-time arrivals warned.
    Warned = prob > threshold, the same rule as src/evaluate.py."""
    warn = prob > threshold
    tp = int((warn & (y == 1)).sum()); fp = int((warn & (y == 0)).sum())
    fn = int((~warn & (y == 1)).sum()); tn = int((~warn & (y == 0)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "on_time_warned": fp / (fp + tn) if fp + tn else None}


class LiveMonitor:
    """Collects what the dashboard shows. The training loop calls, in order:
    start() once, then per epoch: before_batch / after_backward / after_step for
    every batch, evaluating(), end_epoch(); and finish() at the very end."""

    def __init__(self, root: Path, log_every: int = 20):
        self.root = Path(root)
        self.log_every = log_every
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = self.root / "live_runs" / self.run_id
        self.state: dict = {}
        self._loss_window: list[float] = []
        self._per_epoch: dict[int, tuple] = {}   # each epoch's fit and 0.45 results, to show the kept epoch at the end

    # ---------- setup ----------
    def start(self, model, config: dict, train_df, val_df, max_epochs: int, batches_per_epoch: int,
              train_hours: np.ndarray, train_labels: np.ndarray,
              val_hours: np.ndarray, val_labels: np.ndarray) -> None:
        self.batches_per_epoch = batches_per_epoch
        self.split = {"train": (train_hours, train_labels.astype(int)),
                      "val": (val_hours, val_labels.astype(int))}
        t = config["training"]
        self.state = {
            "run": {"id": self.run_id, "name": config.get("run_name", "run"),
                    "started": datetime.now().isoformat(timespec="seconds"),
                    "status": "starting", "max_epochs": max_epochs,
                    "batches_per_epoch": batches_per_epoch, "batch_size": t["batch_size"],
                    "learning_rate": t["learning_rate"], "seed": t["seed"],
                    "hidden_layers": config["hidden_layers"], "dropout": config["dropout"],
                    "train_rows": int(len(train_df)), "val_rows": int(len(val_df)),
                    "n_params": int(sum(p.numel() for p in model.parameters())),
                    "threshold": OPERATING_THRESHOLD},
            "progress": {"epoch": 0, "batch": 0, "phase": "starting"},
            "batch_loss": [], "epochs": [],
            "early_stopping": {"best_epoch": None, "best_auc": None, "since": 0, "patience": t["patience"]},
            "val": None, "fit": None, "updated": time.time(),
        }
        self._write()
        (self.root / "live_runs" / "latest.txt").write_text(self.run_id)
        print(f"live monitor: writing to {self.dir.relative_to(self.root)}")

    # ---------- per batch ----------
    def before_batch(self, epoch: int, i: int) -> None:
        if i == 0:
            self.state["progress"].update(epoch=epoch, batch=0, phase="training")
            self.state["run"]["status"] = "training"

    def after_backward(self, epoch: int, i: int, loss: torch.Tensor) -> None:
        self._loss_window.append(float(loss.detach()))       # reads the loss, changes nothing

    def after_step(self, epoch: int, i: int) -> None:
        """Every log_every batches (and the last one): one point on the live loss line."""
        if (i + 1) % self.log_every and i + 1 != self.batches_per_epoch:
            return
        x = (epoch - 1) + (i + 1) / self.batches_per_epoch
        self.state["batch_loss"].append([_r(x, 4), _r(np.mean(self._loss_window))])
        self._loss_window = []
        self.state["progress"].update(epoch=epoch, batch=i + 1)
        self._write()

    # ---------- per epoch ----------
    def evaluating(self, epoch: int) -> None:
        self.state["progress"].update(epoch=epoch, phase="evaluating")
        self._write()

    def end_epoch(self, epoch: int, on_train: dict | None, on_val: dict, val_logits: torch.Tensor,
                  stopper, train_logits: torch.Tensor) -> None:
        prob = torch.sigmoid(val_logits).numpy()
        fit = {"epoch": epoch, "train": self._fit("train", torch.sigmoid(train_logits).numpy()),
               "val": self._fit("val", prob)}
        val = self._val_view(epoch, prob)
        self._per_epoch[epoch] = (fit, val)
        self.state["fit"], self.state["val"] = fit, val
        self.state["epochs"].append({"epoch": epoch,
                                     "train_loss": _r(on_train["loss"], 6) if on_train else None,
                                     "val_loss": _r(on_val["loss"], 6), "val_auc": _r(on_val["auc"], 6),
                                     "val_accuracy": _r(on_val["accuracy"], 6)})
        if stopper is not None:
            self.state["early_stopping"].update(
                best_epoch=stopper.best_epoch, since=stopper.epochs_without_improvement,
                best_auc=_r(stopper.best_score, 6) if stopper.best_epoch else None)
        self.state["progress"].update(epoch=epoch, batch=self.batches_per_epoch, phase="between epochs")
        self._write()
        if epoch > 0:
            write_json_atomic(self.dir / "epochs" / f"epoch_{epoch:02d}.json", self.state)

    def _val_view(self, epoch: int, prob: np.ndarray) -> dict:
        """The two hills (predicted chance for late and on-time buses) and the 0.45 results."""
        y = self.split["val"][1]
        edges = np.linspace(0, 1, HIST_BINS + 1)
        return {"epoch": epoch,
                "hist": {"late": np.histogram(prob[y == 1], edges)[0].tolist(),
                         "ontime": np.histogram(prob[y == 0], edges)[0].tolist()},
                "operating": binary_metrics(prob, y, OPERATING_THRESHOLD)}

    def _fit(self, split: str, prob: np.ndarray) -> dict:
        """Real late rate vs the average prediction, for each hour of the day."""
        hours, y = self.split[split]
        n = np.bincount(hours, minlength=24)
        pred = np.bincount(hours, weights=prob, minlength=24) / np.maximum(n, 1)
        act = np.bincount(hours, weights=y, minlength=24) / np.maximum(n, 1)
        seen = n > 0
        return {"hour": {"n": n.tolist(), "pred": _r(pred.tolist(), 4), "actual": _r(act.tolist(), 4)},
                "hour_gap": _r(float(np.abs(pred - act)[seen].mean()), 4)}

    def finish(self, status: str, best_epoch: int | None, final: dict | None = None) -> None:
        self.state["run"]["status"] = status
        self.state["progress"]["phase"] = "done"
        self.state["early_stopping"]["best_epoch"] = best_epoch
        if best_epoch in self._per_epoch:                  # show the model that is kept, not the last epoch
            self.state["fit"], self.state["val"] = self._per_epoch[best_epoch]
        if final is not None:
            self.state["run"]["final"] = {k: _r(v, 6) for k, v in final.items()}
        self._write()

    def _write(self) -> None:
        self.state["updated"] = time.time()
        write_json_atomic(self.dir / "state.json", self.state)
