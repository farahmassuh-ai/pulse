"""Train LateNet exactly like src/train_latenet.py, while the live dashboard watches.

The math is the same as train_latenet.train(): same seed, same vocabulary and
scaler, same model, same optimizer, same shuffling, same early stopping. The only
additions are calls to LiveMonitor, which observes without changing anything.
So with configs/final.yaml the run should end at the same best epoch with the
same weights as models/latenet.pt; --compare checks that at the end.

What it never touches: models/latenet.pt, MLflow, mlflow_runs.csv, the test split.

Run from the repo root, in two terminals:
    python src/serve_dashboard.py                    # then open http://localhost:8765/dashboard/
    python src/train_live.py configs/final.yaml      # the full run, 8 to 10 minutes
    python src/train_live.py configs/final.yaml --pilot   # a quick 1-epoch check first
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from features import build_vocab, fit_scaler
from latenet import LateNet
from live_monitor import LiveMonitor
from train_baselines import ROOT
from train_latenet import (EarlyStopping, batches, check_feature_lists, load_splits,
                           make_optimizer, save_checkpoint, to_tensors)

LIVE_MODEL = ROOT / "models" / "latenet_live.pt"     # never models/latenet.pt


def evaluate_with_logits(model, tensors, batch_size: int):
    """train_latenet.evaluate(), plus the logits it computed (for the dashboard).
    Same formulas, so the numbers are identical."""
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(c, n) for c, n, _ in batches(tensors, batch_size)])
    label = tensors[2]
    metrics = {
        "loss": nn.functional.binary_cross_entropy_with_logits(logits, label).item(),
        "auc": float(roc_auc_score(label.numpy(), logits.numpy())),
        "pr_auc": float(average_precision_score(label.numpy(), logits.numpy())),
        "accuracy": float(((logits > 0).float() == label).float().mean()),
    }
    return metrics, logits


def train_one_epoch_live(model, tensors, optimizer, batch_size, generator, monitor, epoch):
    """train_latenet.train_one_epoch(), with the monitor reading along."""
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()                                   # dropout on
    for i, (categorical, numeric, label) in enumerate(batches(tensors, batch_size, generator)):
        monitor.before_batch(epoch, i)
        optimizer.zero_grad()
        loss = loss_fn(model(categorical, numeric), label)
        loss.backward()
        monitor.after_backward(epoch, i, loss)      # reads the loss, changes nothing
        optimizer.step()
        monitor.after_step(epoch, i)                # every 20 batches: writes a snapshot


def train_live(config: dict, train_df, val_df, max_epochs: int, monitor: LiveMonitor) -> dict:
    """train_latenet.train() step for step, with the monitor attached."""
    check_feature_lists(config)
    t = config["training"]
    torch.manual_seed(t["seed"])                    # before the model exists: same start every rerun

    vocab = build_vocab(train_df, config["features"]["categorical"])
    scaler = fit_scaler(train_df)
    train_t, val_t = to_tensors(train_df, vocab, scaler), to_tensors(val_df, vocab, scaler)
    model = LateNet({col: len(vocab[col]) for col in vocab}, config["embedding_dims"],
                    config["hidden_layers"], config["dropout"])
    optimizer = make_optimizer(model, t)
    generator = torch.Generator().manual_seed(t["seed"])
    stopper = EarlyStopping(t["patience"])
    size = t["batch_size"]

    n_batches = (len(train_df) + size - 1) // size
    monitor.start(model, config, train_df, val_df, max_epochs, n_batches,
                  train_hours=train_df["hour"].astype(int).to_numpy(), train_labels=train_t[2].numpy(),
                  val_hours=val_df["hour"].astype(int).to_numpy(), val_labels=val_t[2].numpy())

    untrained, logits = evaluate_with_logits(model, val_t, size)
    _, untrained_train_logits = evaluate_with_logits(model, train_t, size)   # for the fit picture only
    monitor.end_epoch(0, None, untrained, logits, stopper=None, train_logits=untrained_train_logits)
    print(f"epoch  0 (untrained)                  val loss {untrained['loss']:.4f}  "
          f"val AUC {untrained['auc']:.4f}")
    history, status = [], "finished"
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        train_one_epoch_live(model, train_t, optimizer, size, generator, monitor, epoch)
        monitor.evaluating(epoch)
        on_train, train_logits = evaluate_with_logits(model, train_t, size)
        on_val, logits = evaluate_with_logits(model, val_t, size)
        seconds = time.time() - start
        history.append({"epoch": epoch, "train_loss": on_train["loss"], "val_loss": on_val["loss"],
                        "val_auc_roc": on_val["auc"], "val_accuracy": on_val["accuracy"],
                        "seconds": seconds})
        print(f"epoch {epoch:2d}  train loss {on_train['loss']:.4f}  val loss {on_val['loss']:.4f}  "
              f"val AUC {on_val['auc']:.4f}  val acc {on_val['accuracy']:.4f}  ({seconds:.0f}s)")
        stop = stopper.step(on_val["auc"], epoch, model)
        monitor.end_epoch(epoch, on_train, on_val, logits, stopper=stopper, train_logits=train_logits)
        if stop:
            print(f"early stop: no AUC improvement for {t['patience']} epochs")
            status = "stopped early"
            break

    stopper.restore_best(model)
    final, _ = evaluate_with_logits(model, val_t, size)   # re-score the weights we keep
    monitor.finish(status, stopper.best_epoch, final)
    return {"model": model, "vocab": vocab, "scaler": scaler, "untrained": untrained,
            "history": history, "final": final, "best_epoch": stopper.best_epoch,
            "n_params": sum(p.numel() for p in model.parameters()),
            "train_rows": len(train_df), "val_rows": len(val_df)}


def compare_weights(path_a: Path, path_b: Path) -> float:
    """Largest absolute difference between two checkpoints' weights (0.0 = identical)."""
    a = torch.load(path_a, weights_only=True)["state_dict"]
    b = torch.load(path_b, weights_only=True)["state_dict"]
    if a.keys() != b.keys():
        raise ValueError("the two checkpoints have different layers")
    return max(float((a[k] - b[k]).abs().max()) for k in a)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train LateNet with the live dashboard watching.")
    parser.add_argument("config", type=Path, help="e.g. configs/final.yaml")
    parser.add_argument("--pilot", action="store_true", help="the config's quick 1-epoch sample run")
    parser.add_argument("--log-every", type=int, default=20, help="batches between snapshots")
    parser.add_argument("--compare", type=Path, default=ROOT / "models" / "latenet.pt",
                        help="checkpoint to compare the final weights with")
    args = parser.parse_args(argv)
    if args.log_every < 1:
        parser.error("--log-every must be at least 1")
    config = yaml.safe_load(args.config.read_text())
    compare = args.compare if args.compare.is_absolute() else ROOT / args.compare   # relative = from the repo root

    train_df, val_df = load_splits()
    max_epochs = config["training"]["max_epochs"]
    if args.pilot:
        pilot = config["pilot"]
        train_df = train_df.sample(n=pilot["train_rows"], random_state=config["training"]["seed"])
        max_epochs = pilot["max_epochs"]
    print(f"{'PILOT ' if args.pilot else ''}{config['run_name']} (live): train {len(train_df):,} rows, "
          f"val {len(val_df):,} rows, up to {max_epochs} epoch(s)")

    monitor = LiveMonitor(ROOT, log_every=args.log_every)
    result = train_live(config, train_df, val_df, max_epochs, monitor)
    final = result["final"]
    print(f"best epoch {result['best_epoch']}: val AUC {final['auc']:.4f}  "
          f"PR-AUC {final['pr_auc']:.4f}  accuracy {final['accuracy']:.4f}  "
          f"({result['n_params']:,} parameters)")

    save_checkpoint(LIVE_MODEL, result, config)
    print(f"saved {LIVE_MODEL.relative_to(ROOT)} (models/latenet.pt untouched)")
    if not args.pilot and compare.exists():
        diff = compare_weights(LIVE_MODEL, compare)
        verdict = "identical weights" if diff == 0 else "the weights differ"
        print(f"compared with {compare}: largest weight difference {diff} ({verdict})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
