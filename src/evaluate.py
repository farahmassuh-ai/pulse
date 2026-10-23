"""
Final evaluation of the saved LateNet model (Step 3).

Part 1: load models/latenet.pt and rebuild the trained network.

Why: Every later part of the evaluation script (metrics, tables, the one test run)
starts from this and it must rebuild the exact model that was trained:
- same shape
- same weights
- same vocabulary
- same scalar
"""
# lets type hints like "Path" work on any Python version
from __future__ import annotations
# Path: a tool for working with file paths
from pathlib import Path
# Pytorch: needed to load the saved file
import torch
# our network class (src/latenet.py)
from latenet import LateNet
# repo's top folder
from train_baselines import ROOT

# where the winning model lives (default file)
MODEL_PATH = ROOT / "models" / "latenet.pt"

def load_model(path: Path = MODEL_PATH):
    """
    Rebuild the trained LateNet from its checkpoint.

    Returns (model, checkpoint). The model is in eval mode (dropout off), ready to score.
    weights_only=True is the safe loader: it refuses anything but plain data and tensors.
    """
    # open the saved file safely and outputs a dictionary
    checkpoint = torch.load(path, weights_only=True)
    # extract the config and vocabulary
    config, vocab = checkpoint["config"], checkpoint["vocab"]
    # known values per feature and yields an empty network of the right shape 
    model = LateNet({col: len(values) for col, values in vocab.items()},
                    config["embedding_dims"], config["hidden_layers"],
                    config["dropout"])
    # pour the trained weights into (checks shape)
    model.load_state_dict(checkpoint["state_dict"])
    # scoreing mode: dropout off
    model.eval()
    # return the model and the rest of the file
    return model, checkpoint