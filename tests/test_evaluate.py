# testing tool needed for pytest.appox
import pytest
# reads the YAML config file
import yaml
# the function we're testing
from evaluate import load_model
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
