"""
Latency check for the saved LateNet.

Times one prediction on CPU the way a server answers one request:
one raw row -> input guards -> encode with the saved vocabulary and scaler -> model -> probability.
Model loading is timed separately: a server does it once at startup, not per request.
"""
# lets type hints like "tuple[float, float]" work on any Python version
from __future__ import annotations
# reads the command-line options (--repeats, --warmup)
import argparse
# exit code at the end (0 = within budget)
import sys
# perf_counter: a monotonic, high-resolution clock meant for measuring durations
import time
# median and 95th percentile of the timings
import numpy as np
# reads the val split to get one real row
import pandas as pd
# loaded before evaluate (which brings in MLflow), the same order that avoids the native-library clash
import torch
# the same loader and scoring path as the evaluation script (no second copy to drift)
from evaluate import load_model, predict
# the same input guards as training and evaluation
from features import check_inputs
# where the frozen splits live
from train_baselines import SPLIT_DIR

# acceptance criterion: under ~200 ms per prediction on a laptop CPU
BUDGET_MS = 200.0


def time_one_request(model, checkpoint: dict, row: pd.DataFrame) -> tuple[float, float]:
    """
    One request, timed end to end: input guards -> encode -> model -> probability.

    Returns (milliseconds, probability of being late).
    """
    # start the clock
    start = time.perf_counter()
    # the same input guards as training and evaluation (a server checks every request)
    check_inputs(row)
    # encode with the saved vocabulary and scaler, run the model, sigmoid: one row -> one probability
    prob = predict(model, row, checkpoint)[0]
    # stop the clock and convert seconds to milliseconds
    elapsed_ms = (time.perf_counter() - start) * 1000
    # plain Python numbers, easy to print and test
    return elapsed_ms, float(prob)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Time one LateNet prediction on CPU.")
    # how many requests to time (more = a steadier median and p95)
    parser.add_argument("--repeats", type=int, default=1000, help="timed single-row requests")
    # untimed requests first: the first calls pay one-time setup costs
    parser.add_argument("--warmup", type=int, default=10, help="untimed requests before timing")
    args = parser.parse_args(argv)
    # guard against nonsense values (0 repeats would leave nothing to measure)
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be at least 1 and --warmup at least 0")

    # load once, as a server does at startup, and time it separately from the requests
    start = time.perf_counter()
    model, checkpoint = load_model()
    load_ms = (time.perf_counter() - start) * 1000

    # one real row from validation (test stays untouched); [[0]] keeps it a 1-row table, not a Series
    row = pd.read_parquet(SPLIT_DIR / "val.parquet").iloc[[0]]

    # warm-up: untimed calls absorb one-time setup costs (lazy initialisation in PyTorch and pandas)
    for _ in range(args.warmup):
        time_one_request(model, checkpoint, row)

    # the measurement: each request timed on its own
    timings = []
    for _ in range(args.repeats):
        elapsed_ms, prob = time_one_request(model, checkpoint, row)
        timings.append(elapsed_ms)

    # typical request, slow-but-normal request, slowest request
    median, p95, worst = np.median(timings), np.percentile(timings, 95), max(timings)
    # the verdict uses p95, the stricter number: 95% of requests must fit the budget
    within = p95 < BUDGET_MS

    print(f"torch {torch.__version__} | CPU threads {torch.get_num_threads()}")
    print(f"model load (once, at startup): {load_ms:.1f} ms")
    print(f"{args.repeats} single-row predictions after {args.warmup} warm-up calls:")
    print(f"  median {median:.2f} ms | p95 {p95:.2f} ms | max {worst:.2f} ms")
    print(f"example prediction: P(late) = {prob:.3f}")
    print(f"budget {BUDGET_MS:.0f} ms (judged on p95): {'PASS' if within else 'FAIL'}")
    # exit code 0 = within budget, 1 = over (lets a script or CI check the result)
    return 0 if within else 1


if __name__ == "__main__":
    sys.exit(main())
