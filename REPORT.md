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

| Metric | Value |
|---|---|
| Accuracy | |
| AUC-ROC | |
| PR-AUC | |
| Recall (late) at threshold _t_ | |
| Precision (late) at threshold _t_ | |

**Confusion matrix** (threshold _t_):

| | Predicted not late | Predicted late |
|---|---|---|
| **Actually not late** | | |
| **Actually late** | | |

**Calibration check: predicted vs. actual late rate by hour:**

| Hour | Predicted late rate | Actual late rate |
|---|---|---|
| | | |

_(what the calibration shows)_

**Route-level sanity check: top-5 predicted-late routes vs. actual:**

| Route | Predicted late rate | Actual late rate |
|---|---|---|
| | | |

_(what the route check shows)_

## 5. Threshold Recommendation
| Threshold | Precision (late) | Recall (late) | Share of trips flagged |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

**Chosen operating threshold:** _(value)_

**Rider-experience rationale:** _(your reasoning)_

## 6. Reproducibility
- **Git commit:** _(hash)_
- **DVC versions (md5):** train `6dbba79a079af00a8eb33995744dcce1`, val `59c88b7daeb30ba6dd2a2ba50d360bd8`, test `1984178d852bc39db62c55af6a2e74af`
- **MLflow experiment:** `pulse`
- **Registered model:** `pulse-latenet`, version _(n)_, stage Production
- **Exact command(s) to reproduce the winning run:**
  ```bash
  _(command)_
  ```

## 7. Limitations & Next Steps
**What the model can't do; what I'd try with more time:**
_(your reasoning)_

**How would the picture change if "late" meant five minutes instead of three?**
_(your argument, qualitative)_
