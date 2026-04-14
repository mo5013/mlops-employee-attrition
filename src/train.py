import os
import sys
import json
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV file into a dataframe.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str]
) -> ColumnTransformer:
    """
    Build preprocessing transformer for numeric and categorical columns.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def build_model(config: dict) -> LogisticRegression:
    """
    Build model from config settings.
    """
    model_config = config["model"]

    if model_config["type"] != "logistic_regression":
        raise ValueError("Only logistic_regression is supported in this version.")

    return LogisticRegression(
        C=model_config["C"],
        max_iter=model_config["max_iter"],
        class_weight=model_config["class_weight"],
        random_state=config["project"]["random_state"]
    )


def evaluate_model(
    model_pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evaluate trained model and return metrics.
    """
    predictions = model_pipeline.predict(x_test)
    probabilities = model_pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "f1": f1_score(y_test, predictions),
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }

    return metrics


def save_model(model_pipeline: Pipeline, output_path: str) -> None:
    """
    Save trained pipeline to disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model_pipeline, output_path)


def save_metrics(metrics: dict, output_path: str) -> None:
    """
    Save evaluation metrics to JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def log_to_mlflow(config: dict, metrics: dict, model_pipeline: Pipeline) -> None:
    """
    Log parameters, metrics, and model to MLflow.
    """
    mlflow_config = config["mlflow"]
    data_config = config["data"]
    model_config = config["model"]

    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run():
        mlflow.log_param("model_type", model_config["type"])
        mlflow.log_param("C", model_config["C"])
        mlflow.log_param("max_iter", model_config["max_iter"])
        mlflow.log_param("class_weight", model_config["class_weight"])
        mlflow.log_param("target_column", data_config["target_column"])
        mlflow.log_param("data_version", data_config["data_version"])
        mlflow.log_param("test_size", data_config["test_size"])
        mlflow.log_param("missing_fraction", config["preprocessing"]["missing_fraction"])

        mlflow.log_metric("f1", metrics["f1"])
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"])

        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model"
        )


def check_performance_threshold(metrics: dict, min_f1: float) -> None:
    """
    Exit with non-zero status if model performance is below threshold.
    """
    if metrics["f1"] < min_f1:
        print(
            f"Training failed: F1 score {metrics['f1']:.4f} is below minimum threshold {min_f1:.4f}."
        )
        sys.exit(1)


def main():
    config = load_config("configs/config.yaml")

    train_path = config["data"]["processed_train_path"]
    test_path = config["data"]["processed_test_path"]
    target_column = config["data"]["target_column"]
    numeric_features = config["preprocessing"]["numeric_features"]
    categorical_features = config["preprocessing"]["categorical_features"]
    min_f1 = config["metrics"]["minimum_f1_threshold"]

    train_df = load_data(train_path)
    test_df = load_data(test_path)

    x_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    x_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = build_model(config)

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    model_pipeline.fit(x_train, y_train)

    metrics = evaluate_model(model_pipeline, x_test, y_test)

    save_model(model_pipeline, "models/model.joblib")
    save_metrics(metrics, "reports/metrics.json")
    log_to_mlflow(config, metrics, model_pipeline)
    check_performance_threshold(metrics, min_f1)

    print("Training complete.")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Model saved to: models/model.joblib")
    print("Metrics saved to: reports/metrics.json")
    print("MLflow run logged successfully.")


if __name__ == "__main__":
    main()