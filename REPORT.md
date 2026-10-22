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
Every run below is in the MLflow experiment `pulse` (exported to `mlflow_runs.csv`). All scores are on the validation split.

| Run name | Model | Key settings | Val accuracy | Val AUC-ROC | Val PR-AUC |
|---|---|---|---|---|---|
| baseline_majority | Majority class | none | 0.508 | 0.500 | 0.492 |
| baseline_logreg | Logistic regression | one-hot, C=1.0 | 0.657 | 0.719 | 0.699 |
| latenet_base | LateNet (embeddings + MLP) | emb 8/16/4/3/2, hidden 64-32, dropout 0.1, Adam lr 0.001, batch 4096, max 15 epochs (best: 15) | 0.680 | 0.749 | 0.736 |
| _run 4_ | | | | | |
| _run 5_ | | | | | |
| _run 6_ | | | | | |
| _run 7_ | | | | | |
| _run 8_ | | | | | |

**What improved results, what didn't, and my hypothesis why:**
The base run reached a validation AUC of 0.749, 0.030 above the logistic regression's 0.719. That's in the range the task expected, a few hundredths and not a tenth, so there is no sign of data leakage.

Train and validation loss stayed almost equal throughout (0.5943 vs 0.5957 at the end), so the model is not overfitting. The best epoch was the last one allowed (15), and validation AUC was still rising, which suggests the model could improve with more epochs.

For the next run I will change the maximum number of epochs, because the base run showed that AUC kept rising.

**Final architecture and parameter count:**
The network uses nine inputs. Five of them (route, stop, hour, day of week and point type) are turned into integer indexes using a vocabulary built only from the train split. A value that never appeared in training gets the number 0.

The other four are numbers. Direction, weekend and standard type become 0/1 flags, and time-point order is scaled to the range 0–1 using the minimum and maximum from the train split.

_(still to add: the layers and the training settings)_

Parameter count: 24,621 (latenet_base)

**How I handled the high-cardinality features (route, stop), and why:**
Instead of one-hot columns, route and stop each get an embedding: a short list of numbers learned during training. Stops with similar delay patterns can end up with similar embeddings, so the model can share what it learns between them. Each of the 1,169 stops is described by 16 numbers, and each of the 152 routes by 8. Both sizes are inside the rule of thumb from chapter 8, which is between about the fourth root of the number of values (about 5.8 for stops and 3.5 for routes) and a few dozen.

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
