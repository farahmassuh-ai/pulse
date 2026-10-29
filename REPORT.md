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
Every run below is in the MLflow experiment `pulse` (exported to `mlflow_runs.csv`). latenet_base is the first network, and latenet_epochs40 gives it more epochs. Each run after that changes one setting relative to latenet_epochs40. latenet_final combines the changes that helped, and the two seed runs repeat it with different random seeds. All scores are on the validation split.

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
| latenet_final_seed7 | LateNet | as final, seed 7 (seed check; best epoch 28) | 0.689 | 0.762 | 0.751 |
| latenet_final_seed123 | LateNet | as final, seed 123 (seed check; best epoch 34) | 0.689 | 0.762 | 0.751 |

**What improved results, what didn't, and my hypothesis why:**
The base run reached a validation AUC of 0.749, 0.030 above the logistic regression (0.719). That is a modest, believable gain for these inputs. Leakage is ruled out by design: every input (route, stop, hour, day of week, point type, direction, weekend, standard type and time-point order) comes from the schedule, which is known before the bus runs. Nothing comes from the actual arrival time.

Train and validation loss stayed almost equal (0.5943 vs 0.5957 at the end), so the model was not overfitting. The best epoch was the last one allowed (15), and validation AUC was still rising, so more epochs could help.

**More epochs:** I raised the limit to 40. The best epoch moved to 26 and AUC rose to 0.752. This run is the reference for the single-change runs below.

**More capacity in the hidden layers helped.** The wider network gained 0.0071 AUC over the 40-epoch run, and the deeper one 0.0037. The wider one also added about twice as many parameters (8,640 vs 4,160), so per parameter they did about the same: more capacity helped, and width was not clearly better than depth.

**A bigger stop embedding helped least.** Going from 16 to 32 numbers per stop added 19,744 parameters but gained only 0.0021 AUC, likely because 16 numbers already describe the stops well enough.

**Training settings.** A larger learning rate (0.003) learned faster: it matched the 40-epoch run's final AUC by epoch 12 and ended 0.0032 higher. A smaller batch size (1024) gained about the same (0.0038), but each epoch took about 1.8 times as long and it needed 38 epochs, so I left it out.

**Two changes made no clear difference.** Removing dropout scored lower (0.749), but that run stopped early at epoch 14 (best epoch 12) after two tiny dips of less than 0.0003, so the lower score likely reflects fewer epochs, not worse learning. That fits the curves: the model was not overfitting, so dropout had little to prevent. Removing point type changed AUC by only 0.0010, about the size of the run-to-run wobble (0.0008, see the seed check below), likely because time-point order already tells the model where in the trip a stop is.

**The final model** combines the two changes that helped most: the wider hidden layers and the 0.003 learning rate. I also raised early-stopping patience from 2 to 3, because most runs had stopped after two tiny dips (each under 0.0005), so some may have stopped too early. The gains added up almost exactly (0.0104 separately, 0.0108 combined), likely because the two changes affect different things: how much the network can learn, and how fast. Patience 3 mattered: with patience 2 the run would have stopped at epoch 31, before its best epoch (35).

It reached a validation AUC of 0.763, 0.045 above the logistic regression and the best of all runs. Overall the sweep was flat: the single-change runs all landed between 0.749 and 0.759 AUC.

**Seed check:** to make sure the final result was not luck, I trained the final setting two more times with different random seeds (`configs/final_seed7.yaml` and `configs/final_seed123.yaml`). Validation AUC was 0.7631, 0.7623 and 0.7624: a spread of 0.0008, much smaller than the final model's 0.0036 lead over the next best run (latenet_wide).

**Final architecture and parameter count:**
The network has nine inputs. Five are categories: route, stop, hour, day of week and point type. Each value becomes an integer index from a vocabulary built on the train split only, and a value never seen in training gets index 0. Each index then picks a small learned embedding.

The other four are numbers: direction, weekend and standard type as 0/1 flags, and time-point order scaled to 0–1 using the train split's minimum and maximum.

The embeddings (33 numbers) and the four numeric inputs are joined into 37 numbers, which go through two hidden layers of 128 and 64 neurons (ReLU, dropout 0.1). The output is one number per trip, without a sigmoid, because the loss (BCEWithLogitsLoss) applies it during training.

Training: Adam with a learning rate of 0.003, batches of 4,096 trips, up to 40 epochs. Training stops when validation AUC has not improved for 3 epochs, and the weights from the best epoch are kept.

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

On the test weeks the model was mostly too pessimistic: in most daytime hours it predicted more lateness than happened, by up to 0.078 in the morning rush. On validation it was the opposite: too optimistic in almost every hour. The reason is that the real late rate moved, but the model's predictions did not. Training was about 46% late, the first half of June 49.2%, and the second half 43.3%. The model's average prediction stayed near the training level in both halves of June (45.5% and 45.8%), so it was too low in early June and too high in late June.

One possible reason for the drop in late June is that schools close and summer vacations start, so there are fewer commuters and lighter traffic. The data in this project can't confirm it. Hours with very few trips, like hour 3 (341 rows), are too small to read anything from.

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

The model still ranks routes well: across routes with 1,000+ trips, its predicted and actual late rates correlate at r = 0.913 (0.953 on validation). Route 35 is the latest route in both lists, and two of the actual top 5 (35 and 44) are also in the model's top 5.

The weaker match on test likely explains part of the AUC drop. Some routes changed in ways the model could not know. Route 52 was later than predicted in the first half of June (0.719 actual vs 0.676 predicted) but much less late in the second half (0.561 vs 0.682). Route 31 was later than predicted in both halves (0.752 vs 0.635 on validation, 0.697 vs 0.639 on test), because its lateness rose after training (64.4% late in April–May).

Both acceptance bars are met on the test weeks: accuracy at 0.5 was 68.0% (bar: 65%), 11.3 points above the majority-class baseline (56.7%), and AUC-ROC was 0.736 (bar: 0.70). As expected, test came in below validation: AUC fell by 0.027 and accuracy at 0.5 by 1.1 points.

At the operating threshold of 0.45, recall held steady (0.682 on validation, 0.686 on test), so the model still caught about two-thirds of late buses. Precision fell from 0.692 to 0.606: fewer buses were late in the test weeks, but the model, still predicting near the training level, warned on more on-time buses (34.1% vs 29.4%). On the test weeks, 0.45 also had slightly lower accuracy than 0.5 (67.1% vs 68.0%). Section 5 explains the threshold choice.

## 5. Threshold Recommendation
Chosen on the validation weeks (June 1–15), before the test split was opened:

| Threshold | Precision (late) | Recall (late) | Share of trips flagged | On-time trips warned |
|---|---|---|---|---|
| 0.40 | 0.657 | 0.766 | 57.4% | 38.8% |
| **0.45** | **0.692** | **0.682** | **48.5%** | **29.4%** |
| 0.50 | 0.726 | 0.596 | 40.5% | 21.9% |

**Chosen operating threshold:** 0.45

**Rider-experience rationale:** A warning tells a rider to leave a few minutes early. The expensive
mistake is a missed late bus: the rider gets no warning and ends up late. A false alarm is cheap:
the rider just leaves a little early. But if the app warns on too many trips, riders stop paying
attention, and the warning means nothing. So the goal is to catch as many late buses as possible
(recall), while keeping enough warnings right that riders still trust them (precision).

At 0.50 the app would miss about 4 in 10 late buses (recall 0.596): too many of the expensive
mistakes. At 0.40 it would catch more (recall 0.766), but it would warn on more than half of all
trips (57.4%), so the warning would stop meaning much. At 0.45 it catches about 7 in 10 late buses
(recall 0.682), and about 7 in 10 warnings are right (precision 0.692, well above the 49.2% that
guessing would give). It also had the best validation accuracy of the six thresholds tested (0.694).

**On the test weeks:** the threshold stayed at 0.45, as decided before test was opened. Fewer
buses were late (43.3% vs 49.2%), but the model did not adjust. The main goal held: it still warned
about 2 in 3 late buses (recall 0.686). The cost was more unnecessary warnings: about 4 in 10
warnings were wrong instead of 3 in 10 (precision 0.606 vs 0.692). That is still well above the
43.3% that guessing would give on those weeks, but it is the direction that wears down trust.

**How often to re-pick the threshold:** check every week and re-pick every two weeks, using the
last two weeks of trips. Rule: choose the lowest threshold (the most late buses caught) that keeps
precision at or above about 0.65 and warns on no more than about half of all trips. If that would
push recall below about 0.60, retrain instead. Re-pick early if precision drops below 0.65, more
than half of trips get a warning, or the late rate moves by more than about 3 points. These limits
are starting values to tune; they were set so that the rule picks 0.45 on the validation weeks.
Changing the threshold only patches the problem; the real fix for shifting data is retraining on
recent trips (section 7).

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
  (training never opens test). Verified: a fresh clone + `dvc pull` + this command reproduced the
  winning run bit-for-bit on the same machine (best epoch 35, val AUC 0.7631, identical vocabulary
  and scaler, largest weight difference 0.0; about 10 minutes, 38 epochs at ~15 s each). A second
  fresh clone, in a new Python environment installed only from `requirements.txt`, gave the same
  numbers at every epoch. On other hardware the last digits may differ slightly (seeds 7 and 123
  gave 0.7623 and 0.7624, for scale). Running it overwrites `models/latenet.pt`, logs a new run to `pulse` and re-exports `mlflow_runs.csv`; the numbers in
  this report come from the registered version 1 and the DVC-tracked file.
- **Logging notes:** the model is logged with `serialization_format="pickle"`, because MLflow 3's
  `pt2` format failed with this project's torch 2.2.2. MLflow 3 stores it as a logged model
  (`models:/m-8b9b67470af943a796f9e14bb5b3ed3f`), and version 1 was registered from it. No input
  signature was logged.
- **Serving latency** (`python src/latency_check.py`): median 10.4 ms, p95 10.7 ms, max 11.0 ms
  per single-row prediction (1,000 timed requests after 10 warm-up calls; each covers the input
  guards, encoding with the saved vocabulary and scaler, the forward pass and the sigmoid), well
  under the ~200 ms budget. Model load: 67 ms, once at startup. Measured CPU-only on an Intel
  Core i9-9880H @ 2.30 GHz, torch 2.2.2, 8 threads. Note: `predict` also reads the label column
  (via `to_tensors`); a production endpoint would skip that, which can only make it faster.

## 7. Limitations & Next Steps
**What the model can't do; what I'd try with more time:**
- **It can't keep up when lateness changes.** The model learned from April–May, when about 46% of
  trips were late. In June it kept predicting about the same (45.5% and 45.8%), but the real rate
  moved: 49.2% in the first half, 43.3% in the second. So it predicted too low first, then too high,
  and in late June riders got more false alarms.
  *Fix:* retrain on the latest weeks of trips, for example every two weeks, and check recall
  and precision every week (section 5).
- **It only knows the timetable.** It sees the route, stop, hour, day, direction and a few details
  about the stop and the schedule, so it only knows what usually happens on that trip. It can't see
  what is happening right now: traffic, weather, events, or whether the bus is already late at
  earlier stops.
  *Fix:* add those, starting with the bus's current delay, likely the strongest clue for the next stop.
- **It can't score live trips yet.** One of its inputs, `standard_type` (whether a stop is judged
  by the timetable or by the gap between buses), isn't in the MBTA API, and the daily pull leaves it
  empty (`ml_spec.md`, section 3). The model rejects missing inputs, so it can't score live trips
  until that value has a source. Two other inputs, `time_point_order` and `point_type`, come from a
  lookup that uses each route, direction and stop's most common value. Those values vary for 31% and
  10% of the combinations in the historical files, so some live values will not match the trip.
  *Fix:* find a reliable lookup for it (for example by trip pattern or time of day), or retrain
  without it and compare the results.
- **It can't see changes on one route.** The model learned how late each route usually was. Route 31
  was late 64.4% of the time in April–May. In June the model still predicted about that (0.635 in
  the first half, 0.639 in the second), but the route got later: 0.752 and 0.697. Something changed
  after training, such as a detour, roadwork or more riders. The data can't tell which.
  *Fix:* regular retraining would catch it, and a weekly per-route check would spot it sooner.
- **The summer reason is a guess.** Fewer late trips in late June fits schools closing and
  vacations starting, but the data doesn't show that, so it stays a guess.
- **Most settings were trained only once.** A run's result depends a little on its random seed
  (the starting weights and the shuffling), so one run can be lucky or unlucky. I trained the final setting 3 times, and its validation AUC changed by only 0.0008. It
  beat the runner-up by 0.0036, about 4–5 times more, so the win is very unlikely to be luck. But
  several other settings differ by 0.0005 or less, so their order in the table could change.
  *Fix:* run 3 seeds for the top few settings, not just the winner.
- **Smaller points:** `requirements.txt` fixes the versions of the packages the project uses, but
  not of the packages those depend on, so a new install can differ slightly (likely why one harmless
  SQLAlchemy warning showed up in a fresh install). `predict` also reads the label column, which a
  real service would not have.

**How would the picture change if "late" meant five minutes instead of three?**
- **Fewer trips would count as late.** Every bus more than 5 minutes late is also more than 3
  minutes late, but not the other way round.
- **Accuracy would become misleading.** If, for example, 80% of trips were not late, a model that
  always says "not late" would be 80% accurate and pass a 65% bar while being useless. AUC and
  PR-AUC would matter more, because they check whether the model ranks late trips above on-time
  ones, which always saying "not late" can't do.
- **It would probably be harder to predict.** Small delays repeat every day on the same routes and
  hours, so the timetable can learn them. Big delays more often come from one-off events, like
  accidents or bad weather, that the model can't see. This is my expectation, not a tested result.
- **For riders, each warning would matter more, but there would be fewer.** A 5-minute delay is
  more likely to make someone miss a connection, but the model would warn less often and probably
  be wrong more often. I'd keep 3 minutes for the main warning and maybe add a separate "big delay"
  alert later.
