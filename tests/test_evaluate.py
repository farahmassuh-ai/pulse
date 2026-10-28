# arrays of numbers (probabilities, labels)
import numpy as np
# PyTorch
import torch
# testing tool needed for pytest.appox
import pytest
# reads the YAML config file
import yaml
# the function we're testing
from evaluate import load_model, headline_metrics, predict, confusion, threshold_table
# repo folder and fake data maker (already been written)
from test_train_latenet import ROOT, synthetic_split
# tools from the training script
from train_latenet import evaluate, save_checkpoint, to_tensors, train

# tmp_path: a throwaway folder pytest provides
def test_loaded_model_matches_the_trained_one(tmp_path):
    # a real config to start from
    config = yaml.safe_load((ROOT / "configs" / "latenet_base.yaml").read_text())
    # small batches, fast learning: tiny data
    config["training"].update({"batch_size":64, "learning_rate":0.05})
    # 200 fake validation rows
    val = synthetic_split(200,2)
    # train a tiny model for 2 epochs
    result = train(config, synthetic_split(600,1), val, max_epochs=2)
    # where to save it (inside the throwaway folder)
    path = tmp_path / "latenet.pt"
    # save it exactly as real training does
    save_checkpoint(path, result, config)

    # load it back with our new function
    model, checkpoint = load_model(path)
    # check: it's in eval mode (dropout off)
    assert not model.training
    # check: the saved info came back intact
    assert checkpoint["best_epoch"] == result["best_epoch"]
    # prepare the data with the saved vocab/scalar
    data = to_tensors(val, checkpoint["vocab"], checkpoint["scaler"])
    # check: same AUC as before saving
    assert evaluate(model, data, 64)["auc"] == pytest.approx(result["final"]["auc"])


def test_headline_metrics_match_a_hand_calculation():
    # 4 trips: two late (1), two on time (0), with the model's probabilities
    y_true, prob = [1,1,0,0], [0.9,0.4,0.6,0.1]
    m = headline_metrics(y_true,prob,threshold=0.3)
    assert m["accuracy_at_0.5"] == 0.5              # at 0.5: flags 0.9 and 0.6 -> 1 right of 2 late, 1 false alarm
    assert m["accuracy"] == 0.75                    # at 0.3: flags 0.9, 0.4, 0.6 -> 3 of 4 correct
    assert m["precision"] == 2/3                    # 3 warnings, 2 were really late
    assert m["recall"] == 1.0                       # both late trips caught
    assert m["auc_roc"] == 0.75                     # 3 of the 4 late/on-time pairs ranked correctly
    assert m["majority_baseline"] == 0.5            # half late, half on time


def test_a_probability_of_exactly_the_threshold_is_not_flagged():
    m = headline_metrics([1,0], [0.5,0.5], threshold=0.5)
    assert m["recall"] == 0.0                       # 0.5 is not > 0.5, so nothing is flagged


def test_predict_gives_one_probability_per_row(tmp_path):
    config = yaml.safe_load((ROOT / "configs" / "latenet_base.yaml").read_text())
    config["training"].update({"batch_size": 64, "learning_rate": 0.05})
    val = synthetic_split(200, 2)
    result = train(config, synthetic_split(600, 1), val, max_epochs=2)
    checkpoint = {"vocab": result["vocab"], "scaler": result["scaler"]}

    prob = predict(result["model"], val, checkpoint, batch_size=64)
    assert prob.shape == (200,)                         # one probability per row, in order
    assert ((prob >= 0) & (prob <= 1)).all()            # real probabilities
    data = to_tensors(val, result["vocab"], result["scaler"])
    with torch.no_grad():                               # same as sigmoid of the model's raw scores
        expected = torch.sigmoid(result["model"](data[0], data[1])).numpy()
    assert np.allclose(prob, expected)


def test_confusion_matches_a_hand_count():
    y_true, prob = [1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1]
    assert confusion(y_true, prob, 0.5) == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert confusion(y_true, prob, 0.3) == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}


def test_threshold_table_row_matches_a_hand_calculation():
    row = threshold_table([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], thresholds=(0.3,)).iloc[0]
    assert row["accuracy"] == 0.75          # 3 of 4 correct
    assert row["precision"] == 2 / 3        # 3 warnings, 2 right
    assert row["recall"] == 1.0             # both late buses caught
    assert row["share_flagged"] == 0.75     # 3 of 4 stops warned
    assert row["on_time_flagged"] == 0.5    # 1 of 2 on-time buses warned


def test_threshold_table_agrees_with_headline_metrics():
    y_true, prob = [1, 1, 0, 0, 1, 0], [0.9, 0.4, 0.6, 0.1, 0.7, 0.45]
    row = threshold_table(y_true, prob, thresholds=(0.45,)).iloc[0]
    m = headline_metrics(y_true, prob, threshold=0.45)
    assert row["accuracy"] == pytest.approx(m["accuracy"])     # two ways of computing,
    assert row["precision"] == pytest.approx(m["precision"])   # same answer
    assert row["recall"] == pytest.approx(m["recall"])