import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


DATA_PATH = "data/processed/train.csv"


def build_test_pipeline():
    numeric_features = [
        "Age",
        "DailyRate",
        "DistanceFromHome",
        "HourlyRate",
        "MonthlyIncome",
        "MonthlyRate",
        "NumCompaniesWorked",
        "PercentSalaryHike",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager"
    ]

    categorical_features = [
        "BusinessTravel",
        "Department",
        "EducationField",
        "Gender",
        "JobRole",
        "MaritalStatus",
        "OverTime"
    ]

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

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def test_model_predictions_have_correct_shape():
    df = pd.read_csv(DATA_PATH)

    x = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    x_small = x.head(100)
    y_small = y.head(100)

    pipeline = build_test_pipeline()
    pipeline.fit(x_small, y_small)

    predictions = pipeline.predict(x_small)

    assert len(predictions) == len(y_small)
    assert set(predictions).issubset({0, 1})


def test_model_meets_minimum_f1_threshold():
    df = pd.read_csv(DATA_PATH)

    x = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    pipeline = build_test_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    score = f1_score(y_test, predictions)

    assert score >= 0.30