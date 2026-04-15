import os
import sys
import copy
import yaml
import subprocess
import pandas as pd
import mlflow


BASE_CONFIG_PATH = "configs/config.yaml"
EXPERIMENT_DIR = "configs/experiments"
SUMMARY_CSV_PATH = "reports/experiment_runs_summary.csv"
SUMMARY_MD_PATH = "reports/experiment_runs_summary.md"


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_config(config: dict, config_path: str) -> None:
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def run_training(config_path: str) -> None:
    subprocess.run(
        [sys.executable, "src/train.py", "--config", config_path],
        check=True
    )


def main():
    base_config = load_config(BASE_CONFIG_PATH)

    experiment_variants = [
        {"run_name": "lr_c_0_1_no_weights", "C": 0.1, "max_iter": 500, "class_weight": None},
        {"run_name": "lr_c_0_5_no_weights", "C": 0.5, "max_iter": 1000, "class_weight": None},
        {"run_name": "lr_c_1_balanced", "C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
        {"run_name": "lr_c_2_balanced", "C": 2.0, "max_iter": 1500, "class_weight": "balanced"},
        {"run_name": "lr_c_5_balanced", "C": 5.0, "max_iter": 2000, "class_weight": "balanced"},
    ]

    for variant in experiment_variants:
        experiment_config = copy.deepcopy(base_config)
        experiment_config["run_name"] = variant["run_name"]
        experiment_config["model"]["C"] = variant["C"]
        experiment_config["model"]["max_iter"] = variant["max_iter"]
        experiment_config["model"]["class_weight"] = variant["class_weight"]

        config_path = os.path.join(EXPERIMENT_DIR, f"{variant['run_name']}.yaml")
        save_config(experiment_config, config_path)
        run_training(config_path)

    mlflow.set_tracking_uri(base_config["mlflow"]["tracking_uri"])
    experiment = mlflow.get_experiment_by_name(base_config["mlflow"]["experiment_name"])
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    selected_run_names = [variant["run_name"] for variant in experiment_variants]
    runs_df = runs_df[runs_df["tags.mlflow.runName"].isin(selected_run_names)].copy()

    runs_df = runs_df[
        [
            "run_id",
            "tags.mlflow.runName",
            "metrics.f1",
            "metrics.accuracy",
            "metrics.roc_auc",
            "params.C",
            "params.max_iter",
            "params.class_weight",
        ]
    ].sort_values(by="metrics.f1", ascending=False)

    os.makedirs("reports", exist_ok=True)
    runs_df.to_csv(SUMMARY_CSV_PATH, index=False)

    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as file:
        file.write("# MLflow Experiment Runs Summary\n\n")
        for _, row in runs_df.iterrows():
            file.write(f"- **Run Name:** {row['tags.mlflow.runName']}\n")
            file.write(f"  - Run ID: `{row['run_id']}`\n")
            file.write(f"  - F1: `{row['metrics.f1']:.4f}`\n")
            file.write(f"  - Accuracy: `{row['metrics.accuracy']:.4f}`\n")
            file.write(f"  - ROC-AUC: `{row['metrics.roc_auc']:.4f}`\n")
            file.write(f"  - C: `{row['params.C']}`\n")
            file.write(f"  - Max Iterations: `{row['params.max_iter']}`\n")
            file.write(f"  - Class Weight: `{row['params.class_weight']}`\n\n")

    print("Experiment runs complete.")
    print(f"Summary CSV saved to: {SUMMARY_CSV_PATH}")
    print(f"Summary Markdown saved to: {SUMMARY_MD_PATH}")


if __name__ == "__main__":
    main()