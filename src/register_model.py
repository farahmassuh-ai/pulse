"""Step 3: register the winning LateNet run as pulse-latenet and promote it to Production.

Run once, after the single test evaluation:
    python src/register_model.py <run id>
"""
from __future__ import annotations

import sys

import mlflow
from mlflow import MlflowClient

from train_baselines import setup_mlflow

MODEL_NAME = "pulse-latenet"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv     # the words typed after the script name
    if len(argv) != 1:
        print("usage: python src/register_model.py <run id>")
        return 1
    run_id = argv[0]

    setup_mlflow()                                    # point MLflow at mlflow.db
    client = MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    if "test_auc_roc" not in metrics:                 # ship only a model that has been tested
        raise SystemExit("this run has no test metrics: run src/evaluate.py test first")
    if client.search_model_versions(f"name='{MODEL_NAME}'"):   # register only once
        raise SystemExit(f"{MODEL_NAME} already has a version; this script registers the first one")

    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME).version
    client.update_model_version(
        MODEL_NAME, version,
        description=(f"LateNet from MLflow run {run.info.run_name} ({run_id}), experiment 'pulse'. "
                     f"Validation AUC {metrics['val_auc_roc']:.3f}; test AUC "
                     f"{metrics['test_auc_roc']:.3f}, test accuracy at 0.5 "
                     f"{metrics['test_accuracy_at_0.5']:.3f}. Operating threshold 0.45. "
                     "Local checkpoint: models/latenet.pt."))
    for stage in ["Staging", "Production"]:           # the task's promotion path
        client.transition_model_version_stage(MODEL_NAME, version, stage)
    client.set_registered_model_alias(MODEL_NAME, "production", version)

    print(f"{MODEL_NAME} version {version}: stage Production, alias 'production', "
          f"from run {run.info.run_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
