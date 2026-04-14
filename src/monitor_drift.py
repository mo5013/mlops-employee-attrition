import os
import sys
import json
import yaml
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def generate_production_data(reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate production drift by making stronger changes
    to selected numeric and categorical features.
    """
    production_df = reference_df.copy()

    if "MonthlyIncome" in production_df.columns:
        production_df["MonthlyIncome"] = production_df["MonthlyIncome"] * 2

    if "DistanceFromHome" in production_df.columns:
        production_df["DistanceFromHome"] = production_df["DistanceFromHome"] + 20

    if "OverTime" in production_df.columns:
        sampled_index = production_df.sample(frac=0.35, random_state=42).index
        production_df.loc[sampled_index, "OverTime"] = "Yes"

    if "JobRole" in production_df.columns:
        sampled_index = production_df.sample(frac=0.25, random_state=42).index
        production_df.loc[sampled_index, "JobRole"] = "Sales Executive"

    return production_df


def build_drift_report(reference_df: pd.DataFrame, production_df: pd.DataFrame):
    """
    Build and run an Evidently drift report.
    Returns the evaluated snapshot object from report.run(...).
    """
    report = Report([DataDriftPreset()])
    evaluation = report.run(reference_df, production_df)
    return evaluation


def export_report_to_dict(evaluation) -> dict:
    """
    Convert Evidently output into a dictionary.
    Supports several Evidently API versions.
    """
    if hasattr(evaluation, "dict"):
        return evaluation.dict()

    if hasattr(evaluation, "as_dict"):
        return evaluation.as_dict()

    if hasattr(evaluation, "json"):
        json_output = evaluation.json()
        if isinstance(json_output, str):
            return json.loads(json_output)

    raise AttributeError(
        "Could not export Evidently result to dict. Supported methods not found."
    )


def extract_drift_summary(report_dict: dict) -> tuple[float, list[str]]:
    metrics = report_dict.get("metrics", [])
    if not metrics:
        raise ValueError("No metrics found in drift report output.")

    result = metrics[0].get("result", {})
    drift_share = result.get("share_of_drifted_columns", 0.0)

    drifted_columns = []
    drift_by_columns = result.get("drift_by_columns", {})

    for column_name, column_info in drift_by_columns.items():
        if column_info.get("drift_detected", False):
            drifted_columns.append(column_name)

    return drift_share, drifted_columns


def save_html_report(evaluation, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if hasattr(evaluation, "save_html"):
        evaluation.save_html(output_path)
        return

    raise AttributeError("This Evidently result object does not support save_html().")


def main():
    config = load_config("configs/config.yaml")

    reference_path = config["data"]["processed_train_path"]
    drift_threshold = config["monitoring"]["drift_threshold"]
    report_path = config["monitoring"]["report_path"]

    reference_df = load_data(reference_path)
    production_df = generate_production_data(reference_df)

    evaluation = build_drift_report(reference_df, production_df)
    report_dict = export_report_to_dict(evaluation)

    drift_share, drifted_columns = extract_drift_summary(report_dict)
    save_html_report(evaluation, report_path)

    print("Drift monitoring complete.")
    print(f"Drift share: {drift_share:.2%}")

    if drifted_columns:
        print("Drifted columns:")
        for column in drifted_columns:
            print(f"- {column}")
    else:
        print("No drifted columns detected.")

    print(f"HTML report saved to: {report_path}")

    if drift_share > drift_threshold:
        print(
            f"Drift threshold exceeded: {drift_share:.2%} > {drift_threshold:.2%}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()