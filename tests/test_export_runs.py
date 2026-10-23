import mlflow
import pandas as pd

from train_baselines import export_runs


def test_export_keeps_finished_runs_sorted_by_validation_auc(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'test.db'}")   # a throwaway store, not yours
    mlflow.set_experiment("export-test")
    for name, auc in [("middle", 0.73), ("best", 0.75), ("worst", 0.71)]:
        with mlflow.start_run(run_name=name):
            mlflow.log_metric("val_auc_roc", auc)
    crashed = mlflow.start_run(run_name="crashed")
    mlflow.end_run(status="FAILED")

    export_runs("export-test", tmp_path / "runs.csv")
    table = pd.read_csv(tmp_path / "runs.csv")
    assert table["tags.mlflow.runName"].tolist() == ["best", "middle", "worst"]
    assert crashed.info.run_id not in table["run_id"].tolist()
