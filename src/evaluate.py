import os
import json
import joblib
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV file into a dataframe.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def load_model(model_path: str):
    """
    Load saved model pipeline.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the model on the test set and return results.
    """
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics = {
        "f1": f1_score(y_test, predictions),
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True)
    }

    return metrics


def save_results(results: dict, output_path: str) -> None:
    """
    Save evaluation results to JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def main():
    test_path = "data/processed/test.csv"
    model_path = "models/model.joblib"
    output_path = "reports/evaluation.json"
    target_column = "Attrition"

    test_df = load_data(test_path)
    model = load_model(model_path)

    x_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    results = evaluate_model(model, x_test, y_test)
    save_results(results, output_path)

    print("Evaluation complete.")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"Confusion Matrix: {results['confusion_matrix']}")
    print(f"Evaluation report saved to: {output_path}")


if __name__ == "__main__":
    main()