# Pulse Model Delivery Report

## 1. Data
- **Months used:** April, May, and June 2026 (MBTA Bus Arrival/Departure files).
- **Row counts before/after cleaning:** 7,432,763 raw rows. I dropped 592 rows with a delay outside -30 to +120 minutes and 6,690 rows that repeated the key (service date, route, direction, stop, scheduled), which left 7,425,481 rows in `stop_events`. 451,273 of them have no `actual`, so 6,974,208 rows have a label.

  | Month | Raw rows | Delay out of range | Duplicate keys | Loaded |
  |---|---|---|---|---|
  | April | 2,459,855 | 231 | 1,698 | 2,457,926 |
  | May | 2,484,472 | 176 | 1,866 | 2,482,430 |
  | June | 2,488,436 | 185 | 3,126 | 2,485,125 |

- **Late rate by split:** the three late rates are within 6 points of each other.

  | Split | Dates | Rows | Late rate |
  |---|---|---|---|
  | Train | Apr 1 to May 31 | 4,645,567 | 45.7% |
  | Validation | Jun 1 to Jun 15 | 1,155,852 | 49.2% |
  | Test | Jun 16 to Jun 30 | 1,172,789 | 43.3% |

- **DVC data version (md5):**
  - train: `6dbba79a079af00a8eb33995744dcce1`
  - val: `59c88b7daeb30ba6dd2a2ba50d360bd8`
  - test: `1984178d852bc39db62c55af6a2e74af`
- **Daily pull (inserted rows):** Wed 2026-10-14: 92,168 (a second run that day inserted 0). Thu 2026-10-15: 92,168. Both days give the same count because Wednesday and Thursday use the same weekday timetable, and both are close to the roughly 90,000 rows of a weekday in the historical files.

## 2. Baselines
Both models were trained on the train split and scored on the validation split. Both runs are in the MLflow experiment `pulse` and exported to `mlflow_runs.csv`.

- **Majority-class:** accuracy 0.508. It always predicts "not late", so its accuracy is 1 minus the validation late rate (49.2%). AUC-ROC 0.500, PR-AUC 0.492.
- **Logistic regression** (one-hot route, direction, stop, point type, standard type, hour, day of week): accuracy 0.657, AUC-ROC 0.719, PR-AUC 0.699.
- **What the baselines tell us:** the logistic regression is about 15 points more accurate than the majority class and raises AUC from 0.50 to 0.72, so route, stop, and time of day carry real signal about whether a bus will be late. The signal is still weak, so there is plenty of room to improve in Task 3.

## 3. Model Development
Every run below is in the MLflow experiment `pulse` (exported to `mlflow_runs.csv`). Each LateNet variant changes one setting relative to latenet_epochs40; latenet_final combines the changes that helped. All scores are on the validation split.

| Run name | Model | Key settings | Val accuracy | Val AUC-ROC | Val PR-AUC |
|---|---|---|---|---|---|
| baseline_majority | Majority class | none | 0.508 | 0.500 | 0.492 |
| baseline_logreg | Logistic regression | one-hot, C=1.0 | 0.657 | 0.719 | 0.699 |
| latenet_base | LateNet (embeddings + MLP) | emb 8/16/4/3/2, hidden 64-32, dropout 0.1, Adam lr 0.001, batch 4096, max 15 epochs (best: 15) | 0.680 | 0.749 | 0.736 |
| latenet_epochs40 | LateNet | as base, max 40 epochs (the reference for runs below; best epoch 26) | 0.682 | 0.752 | 0.740 |
| latenet_wide | LateNet | hidden 128-64 (33,261 params; best epoch 26) | 0.687 | 0.759 | 0.748 |
| latenet_deep | LateNet | hidden 64-64-32 (28,781 params; best epoch 35) | 0.686 | 0.756 | 0.743 |
| latenet_dropout0 | LateNet | dropout 0.0 (stopped early, best epoch 12) | 0.682 | 0.749 | 0.735 |
| latenet_stop_emb32 | LateNet | stop embedding 32 (44,365 params; best epoch 26) | 0.684 | 0.754 | 0.742 |
| latenet_lr003 | LateNet | learning rate 0.003 (best epoch 23) | 0.685 | 0.755 | 0.743 |
| latenet_batch1024 | LateNet | batch size 1024 (~18 s/epoch; best epoch 38) | 0.685 | 0.756 | 0.744 |
| latenet_no_point_type | LateNet | without point_type (24,485 params; best epoch 36) | 0.681 | 0.753 | 0.741 |
| **latenet_final** | LateNet | **hidden 128-64 + lr 0.003 + patience 3 (33,261 params; best epoch 35)** | **0.690** | **0.763** | **0.752** |

**What improved results, what didn't, and my hypothesis why:**
The base run reached a validation AUC of 0.749, 0.030 above the logistic regression's 0.719. That is a modest gain, which is believable for these inputs. Leakage is also ruled out by design: every input (route, stop, hour, day of week, point type, direction, standard type and time-point order) is known from the schedule before the bus runs; nothing comes from the actual arrival time.

Train and validation loss stayed almost equal throughout (0.5943 vs 0.5957 at the end), so the model is not overfitting. The best epoch was the last one allowed (15), and validation AUC was still rising, which suggests the model could improve with more epochs.

For the next run I raised the maximum number of epochs to 40, because the base run showed that AUC kept rising; the best epoch moved to 26 and AUC rose to 0.752.

Adding capacity to the hidden layers helped: the wider network gained 0.0071 AUC over the 40-epoch run and the deeper one 0.0036. Relative to the parameters each added, width and depth did about the same, so the evidence points to capacity helping rather than width being better than depth.

Adding capacity to the stop embedding helped least: doubling it to 32 numbers per stop added 19,744 parameters but gained only 0.0021 AUC, likely because 16 numbers per stop already describe the stops well enough, so the extra room went mostly unused.

A larger learning rate (0.003) learned faster: it reached the 40-epoch run’s final AUC by epoch 12 and ended 0.0032 higher. A smaller batch size (1024) gained a similar amount (0.0037), but each epoch took about 1.8 times as long and it needed 38 epochs, so I left it out of the final model.

Two changes made no clear difference. Removing dropout scored lower (0.749), but the run stopped early at epoch 14 (best epoch 12) on a wobble of a ten-thousandth, so the lower score mostly reflects fewer epochs of training, not worse learning. That fits the curves: the model was not overfitting, so dropout had little to prevent.

Removing point type changed AUC by only 0.0010, within the run-to-run wobble, likely because time-point order already tells the model where in the trip a stop is.

For the final model I combined the two changes that helped most: the wider hidden layers and the learning rate of 0.003. I also raised early-stopping patience from 2 to 3, because two runs had stopped on a wobble. The gains added up almost exactly (0.0103 separately, 0.0108 combined), likely because the two changes affect different things: the network’s capacity and how fast it learns. Patience 3 mattered too: with patience 2 the run would have stopped at epoch 31 instead of reaching its best at epoch 35.

The final model reached a validation AUC of 0.763, 0.045 above the logistic regression (0.719), and the best of all runs. The sweep was flat: all LateNet variants landed within about 0.01 AUC of each other.

**Final architecture and parameter count:**
The network uses nine inputs. Five of them (route, stop, hour, day of week and point type) are turned into integer indexes using a vocabulary built only from the train split. A value that never appeared in training gets the number 0.

The other four are numbers. Direction, weekend and standard type become 0/1 flags, and time-point order is scaled to the range 0–1 using the minimum and maximum from the train split.

The embeddings and the four numeric inputs are joined into one list of 37 numbers, which goes through two hidden layers of 128 and 64 neurons (ReLU, dropout 0.1). The output layer gives one number per trip with no sigmoid, because the loss, BCEWithLogitsLoss, applies it during training.

It is trained with Adam at a learning rate of 0.003, in batches of 4,096 trips, for up to 40 epochs. Training stops when validation AUC has not improved for 3 epochs, and the weights from the best epoch are kept.

Parameter count: 33,261 (latenet_final).

**How I handled the high-cardinality features (route, stop), and why:**
Instead of one-hot columns, route and stop each get an embedding: a short list of numbers learned during training. Stops with similar delay patterns can end up with similar embeddings, so the model can share what it learns between them. Each of the 1,169 stops is described by 16 numbers, and each of the 152 routes by 8. Both sizes are inside a common rule of thumb, which is between about the fourth root of the number of values (about 5.8 for stops and 3.5 for routes) and a few dozen.

Index 0, for values never seen in training, stays all zeros, so an unseen stop adds nothing to the prediction, and the model relies on the other inputs instead.

## 4. Final Evaluation (held-out test weeks)
Test split: June 16 to 30, 2026 (1,172,789 rows, 43.3% late). Opened once, for this evaluation only.

| Metric | Validation | Test | Acceptance bar |
|---|---|---|---|
| Late rate (share of stops actually late) | 49.2% | 43.3% | |
| Majority-class baseline accuracy | 50.8% | 56.7% | |
| Accuracy at 0.5 | 69.0% | **68.0%** | ≥ 65% ✓ |
| AUC-ROC | 0.763 | **0.736** | ≥ 0.70 ✓ |
| PR-AUC | 0.752 | 0.674 | |
| Accuracy at 0.45 (operating threshold) | 69.4% | 67.1% | |
| Recall (late) at 0.45 | 0.682 | 0.686 | |
| Precision (late) at 0.45 | 0.692 | 0.606 | |

**Confusion matrix** (threshold 0.45):

| | Predicted not late | Predicted late |
|---|---|---|
| **Actually not late** | 437,837 | 226,844 |
| **Actually late** | 159,351 | 348,757 |

**Calibration check: predicted vs. actual late rate by hour:**

Gap = predicted − actual: positive means the model was too pessimistic, negative too optimistic.

| Hour | Rows | Predicted late rate | Actual late rate | Gap |
|---|---|---|---|---|
| 0 | 35,672 | 0.334 | 0.371 | -0.037 |
| 1 | 14,506 | 0.328 | 0.344 | -0.016 |
| 2 | 1,881 | 0.195 | 0.248 | -0.053 |
| 3 | 341 | 0.481 | 0.572 | -0.091 |
| 4 | 5,459 | 0.344 | 0.400 | -0.055 |
| 5 | 42,773 | 0.294 | 0.290 | +0.004 |
| 6 | 71,727 | 0.360 | 0.343 | +0.017 |
| 7 | 74,736 | 0.414 | 0.336 | +0.078 |
| 8 | 73,695 | 0.462 | 0.384 | +0.078 |
| 9 | 64,336 | 0.458 | 0.431 | +0.027 |
| 10 | 55,873 | 0.445 | 0.430 | +0.015 |
| 11 | 53,699 | 0.450 | 0.457 | -0.007 |
| 12 | 53,423 | 0.460 | 0.485 | -0.026 |
| 13 | 55,509 | 0.484 | 0.502 | -0.018 |
| 14 | 60,665 | 0.503 | 0.501 | +0.002 |
| 15 | 65,316 | 0.524 | 0.460 | +0.064 |
| 16 | 68,820 | 0.515 | 0.438 | +0.076 |
| 17 | 69,829 | 0.536 | 0.463 | +0.073 |
| 18 | 70,966 | 0.509 | 0.455 | +0.054 |
| 19 | 60,191 | 0.487 | 0.460 | +0.028 |
| 20 | 49,505 | 0.495 | 0.476 | +0.020 |
| 21 | 44,019 | 0.483 | 0.499 | -0.016 |
| 22 | 41,097 | 0.471 | 0.498 | -0.027 |
| 23 | 38,751 | 0.434 | 0.454 | -0.019 |

On the test weeks the model was mostly too pessimistic: its predicted late rate was above the actual one in most daytime hours, by up to 0.078 during the morning and afternoon rush. On validation it had been the opposite, too optimistic in almost every hour. The reason is that the overall late rate moved: about 46% in training, 49.2% in the first half of June and 43.3% in the second half. The model’s predictions sit near the training level, so they ran too low when June got later and too high when it got less late. The model’s average predicted probability was almost identical in both halves of June (45.5% and 45.8%), close to the training late rate of 45.7%, while the actual rate moved to 49.2% and then 43.3%.

A likely explanation for the drop in the second half of June is that schools close and summer vacations begin, so there are fewer commuters and lighter traffic. The data in this project cannot confirm it. The few early-morning hours with very few trips (for example hour 3, with 341 rows) are too small to read anything from.

**Route-level sanity check: top-5 predicted-late routes vs. actual:**

Top 5 routes by **predicted** late rate (routes with 1,000+ rows):

| Route | Rows | Predicted | Actual |
|---|---|---|---|
| 35 | 5,204 | 0.755 | 0.703 |
| 44 | 9,333 | 0.701 | 0.682 |
| 37 | 3,890 | 0.695 | 0.632 |
| 52 | 3,073 | 0.682 | 0.561 |
| 29 | 4,110 | 0.661 | 0.645 |

Top 5 routes by **actual** late rate:

| Route | Rows | Predicted | Actual |
|---|---|---|---|
| 35 | 5,204 | 0.755 | 0.703 |
| 31 | 14,084 | 0.639 | 0.697 |
| 44 | 9,333 | 0.701 | 0.682 |
| 24 | 7,596 | 0.646 | 0.651 |
| 34E | 15,180 | 0.627 | 0.650 |

Pearson correlation between predicted and actual route late rates: 0.913 on test (0.953 on validation).

The model still ranks routes well: across routes with 1,000+ trips, its predicted late rates correlate with the actual ones at r = 0.913 (0.953 on validation). Route 35 is the latest route in both lists, and two of the actual top 5 routes appear in the model’s top 5.

The weaker agreement on test likely explains part of the AUC drop. Some routes changed in the second half of June in ways the model could not know: route 52 ran later than predicted in the first half of June (0.719 actual vs 0.676 predicted) but much less late in the second half (0.561 vs 0.682). Other routes are under-predicted throughout June: route 31 was later than predicted on both validation (0.752 vs 0.635) and test (0.697 vs 0.639), so its lateness likely rose after the April–May training period.

Both acceptance bars are met on the test weeks: accuracy at 0.5 was 68.0% (bar: 65%) and AUC-ROC 0.736 (bar: 0.70), 11.3 percentage points above the majority-class baseline of the test weeks (56.7%). As expected, test came in below validation: AUC fell by 0.027 and accuracy at 0.5 by 1.1 percentage points.

At the operating threshold of 0.45, recall held steady (0.682 on validation, 0.686 on test), so the model still caught about two-thirds of late buses. Precision fell from 0.692 to 0.606, because fewer buses were actually late in the test weeks and the model, still near the training level, warned on more on-time buses (34.1% vs 29.4%). On the test weeks, 0.45 also had slightly lower accuracy than 0.5 (67.1% vs 68.0%). Section 5 discusses what this means for the threshold.

## 5. Threshold Recommendation
| Threshold | Precision (late) | Recall (late) | Share of trips flagged |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

**Chosen operating threshold:** _(value)_

**Rider-experience rationale:** _(your reasoning)_

## 6. Reproducibility
- **Git commit:** `cf46d60` (`cf46d60c380f06fdf0ee8581501061213b26e7b8`), the commit MLflow logged
  for the winning run `latenet_final` (run id `de577f1e165a498aa88476bc974d3673`). The training code
  and `configs/final.yaml` are unchanged between that commit and the latest one.
- **DVC versions (md5):** train `6dbba79a079af00a8eb33995744dcce1`, val `59c88b7daeb30ba6dd2a2ba50d360bd8`, test `1984178d852bc39db62c55af6a2e74af`, model `models/latenet.pt` `49494f5451712ffae780b451163d2098`.
  Stored in the public repo `farahmassuh-ai/pulse-dvc-store`; `dvc pull` needs no credentials.
- **MLflow experiment:** `pulse`
- **Registered model:** `pulse-latenet`, version 1, stage Production (alias `production`)
- **Exact command(s) to reproduce the winning run:**
  ```bash
  dvc pull
  python src/train_latenet.py configs/final.yaml
  ```
  Before training, the script checks the train and val files' md5 against their `.dvc` pointers
  (training never opens test). Expect validation AUC ≈ 0.763 (seed 42; seeds 7 and 123 gave 0.7623
  and 0.7624); the last digits can differ slightly across machines. Running it overwrites
  `models/latenet.pt`, logs a new run to `pulse` and re-exports `mlflow_runs.csv`; the numbers in
  this report come from the registered version 1 and the DVC-tracked file.
- **Logging notes:** the model is logged with `serialization_format="pickle"`, because MLflow 3's
  `pt2` format failed with this project's torch 2.2.2. MLflow 3 stores it as a logged model
  (`models:/m-8b9b67470af943a796f9e14bb5b3ed3f`), and version 1 was registered from it. No input
  signature was logged.
- **Serving latency** (`python src/latency_check.py`): median 10.4 ms, p95 10.7 ms, 
  max 11.0 ms per single-row prediction (1,000 timed requests after 10 warm-up calls; each covers the input
  guards, encoding with the saved vocabulary and scaler, the forward pass and the sigmoid), well
  under the ~200 ms budget. Model load: 67 ms, once at startup. Measured CPU-only on an Intel
  Core i9-9880H @ 2.30 GHz, torch 2.2.2, 8 threads. Note: `predict` also reads the label column
  (via `to_tensors`); a production endpoint would skip that, which can only make it faster.

## 7. Limitations & Next Steps
**What the model can't do; what I'd try with more time:**
_(your reasoning)_

**How would the picture change if "late" meant five minutes instead of three?**
_(your argument, qualitative)_
