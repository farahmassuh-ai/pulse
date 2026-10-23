"""Shared pytest setup, loaded before any test file.

Import PyTorch before anything else in the test session. On macOS with conda, running the
MLflow export test (which loads MLflow and its database libraries) before the PyTorch tests
made the first training step crash with a segmentation fault: a clash between native
libraries, not a bug in the project code. Loading torch first, as the training script
effectively does, prevents it (checked: the crash reproduced without this file, and the
full suite passes with it).
"""
import torch  # noqa: F401  (imported for its side effect: it loads first)
