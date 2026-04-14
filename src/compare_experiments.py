import yaml
import mlflow
import pandas as pd


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_experiment_runs(experiment_name: str, tracking_uri: str) -> pd.DataFrame:
    """
    Retrieve all runs for the given MLflow experiment.
    """
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    if runs_df.empty:
        raise ValueError("No runs found in the experiment.")

    return runs_df


def get_best_run(runs_df: pd.DataFrame, metric_name: str) -> pd.Series:
    """
    Return the best run based on the specified metric.
    """
    metric_column = f"metrics.{metric_name}"

    if metric_column not in runs_df.columns:
        raise ValueError(f"Metric column '{metric_column}' not found in runs.")

    best_index = runs_df[metric_column].idxmax()
    return runs_df.loc[best_index]


def main():
    config = load_config("configs/config.yaml")

    experiment_name = config["mlflow"]["experiment_name"]
    tracking_uri = config["mlflow"]["tracking_uri"]
    primary_metric = config["metrics"]["primary"]

    runs_df = get_experiment_runs(experiment_name, tracking_uri)
    best_run = get_best_run(runs_df, primary_metric)

    print("\n=== Experiment Comparison Summary ===")
    print(f"Experiment Name: {experiment_name}")
    print(f"Number of Runs: {len(runs_df)}")
    print(f"Primary Metric: {primary_metric}")

    print("\n=== Best Run ===")
    print(f"Run ID: {best_run['run_id']}")
    print(f"F1 Score: {best_run['metrics.f1']:.4f}")
    print(f"Accuracy: {best_run['metrics.accuracy']:.4f}")
    print(f"ROC-AUC: {best_run['metrics.roc_auc']:.4f}")

    if "params.model_type" in best_run:
        print(f"Model Type: {best_run['params.model_type']}")
    if "params.C" in best_run:
        print(f"C: {best_run['params.C']}")
    if "params.max_iter" in best_run:
        print(f"Max Iterations: {best_run['params.max_iter']}")
    if "params.class_weight" in best_run:
        print(f"Class Weight: {best_run['params.class_weight']}")
    if "params.data_version" in best_run:
        print(f"Data Version: {best_run['params.data_version']}")


if __name__ == "__main__":
    main()