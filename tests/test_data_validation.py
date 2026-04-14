import pandas as pd


DATA_PATH = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"


def test_expected_columns_are_present():
    df = pd.read_csv(DATA_PATH)

    expected_columns = {
        "Age",
        "Department",
        "JobRole",
        "MonthlyIncome",
        "OverTime",
        "Attrition"
    }

    assert expected_columns.issubset(df.columns)


def test_target_contains_only_expected_values():
    df = pd.read_csv(DATA_PATH)

    valid_values = {"Yes", "No"}
    actual_values = set(df["Attrition"].unique())

    assert actual_values == valid_values


def test_numeric_features_are_within_expected_ranges():
    df = pd.read_csv(DATA_PATH)

    assert df["Age"].between(18, 60).all()
    assert (df["MonthlyIncome"] > 0).all()
    assert (df["DistanceFromHome"] >= 0).all()