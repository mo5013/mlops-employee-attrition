# MLOps Employee Attrition Project

## Overview

This project builds an end-to-end MLOps pipeline for predicting employee attrition using machine learning. It includes data preprocessing, model training, experiment tracking, testing, CI/CD automation, data versioning, and drift monitoring.

---

## Project Structure

```
.
├── configs/                # Configuration files
├── data/
│   ├── raw/               # Raw dataset (tracked by DVC)
│   └── processed/         # Processed train/test data
├── models/                # Saved trained models
├── reports/               # Metrics, evaluation, and drift reports
├── src/                   # Source code (pipeline logic)
├── tests/                 # Pytest test suite
├── dvc.yaml               # DVC pipeline definition
├── dvc.lock               # DVC pipeline lock file
├── requirements.txt       # Project dependencies
└── README.md
```

---

## Setup

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd mlops-employee-attrition
pip install -r requirements.txt
```

---

## Reproducibility with DVC

This project uses DVC (Data Version Control) to track datasets and ensure reproducibility.

### DVC Files Included

* `.dvc/`
* `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv.dvc`
* `dvc.yaml`
* `dvc.lock`

### Reproducing the Data Pipeline

From a fresh clone, run:

```bash
dvc pull
dvc repro
```

This will:

* download the dataset
* run preprocessing
* generate train/test splits

---

## Model Training

To train the model:

```bash
python src/train.py
```

The training process:

* loads configuration from YAML
* trains a classification model
* evaluates performance
* saves the model and metrics
* logs results to MLflow

---

## Experiment Tracking (MLflow)

This project uses MLflow to track experiments and compare performance across different configurations.

### Running Experiments

To execute multiple experiments:

```bash
python src/run_experiments.py
```

This script runs several configurations and logs:

* F1 Score
* Accuracy
* ROC-AUC
* Model parameters

### Experiment Results

A summary of experiment runs is saved to:

* `reports/experiment_runs_summary.csv`
* `reports/experiment_runs_summary.md`

These files provide evidence of multiple MLflow runs and allow comparison between models.

---

## Testing

Run tests using pytest:

```bash
pytest tests/ -v
```

Tests cover:

* data validation
* preprocessing logic
* model performance thresholds

---

## CI/CD Pipeline

GitHub Actions is used to automate the workflow.

The pipeline:

* installs dependencies
* runs preprocessing
* executes pytest tests
* trains the model

This ensures code quality and reproducibility on every push.

---

## Data Drift Monitoring

Data drift monitoring is implemented using the Evidently library.

### Approach

* Training data is used as the reference dataset
* A simulated production dataset is created by modifying feature distributions

### Simulated Production Changes

* `MonthlyIncome` multiplied by 2
* `DistanceFromHome` increased by 20
* Increased proportion of `OverTime = "Yes"`
* Shift in `JobRole` distribution toward `"Sales Executive"`

### Results

The drift monitoring script compares datasets and saves an HTML report:

```
reports/drift_report.html
```

Observed output:

* Drift Share: 0.00%
* Drifted Columns: None detected

### Interpretation

No significant drift was detected, suggesting that the model is likely to maintain stable performance under current conditions.

However, the simulation demonstrates how drift can occur, and monitoring remains important in production systems.

### Recommended Actions

* investigate the source of drift
* continue monitoring if performance remains stable
* retrain the model if drift persists

The script is configured to fail when drift exceeds a threshold, enabling automated monitoring.

---

## Key Technologies

* Python
* Scikit-learn
* MLflow
* DVC
* Pytest
* GitHub Actions
* Evidently

---

## Conclusion

This project demonstrates a complete MLOps workflow, including:

* reproducible data pipelines
* experiment tracking
* automated testing and CI/CD
* model monitoring

It reflects real-world practices for deploying and maintaining machine learning systems.
