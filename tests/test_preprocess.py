import pandas as pd
import pytest

from src.preprocess import simulate_missing_values, encode_target, split_data


def make_sample_df():
    return pd.DataFrame({
        "Age": [25, 30, 35, 40, 45, 50],
        "Department": ["HR", "Sales", "IT", "HR", "IT", "Sales"],
        "Attrition": ["Yes", "No", "No", "Yes", "No", "Yes"]
    })


def test_simulate_missing_values_adds_missing_values():
    df = make_sample_df()
    result = simulate_missing_values(df, ["Age"], 0.5, random_state=42)
    assert result["Age"].isna().sum() > 0


def test_simulate_missing_values_does_not_modify_original_dataframe():
    df = make_sample_df()
    df_copy = df.copy(deep=True)

    simulate_missing_values(df, ["Age"], 0.5, random_state=42)

    pd.testing.assert_frame_equal(df, df_copy)


def test_simulate_missing_values_raises_error_for_invalid_column():
    df = make_sample_df()
    with pytest.raises(ValueError):
        simulate_missing_values(df, ["BadColumn"], 0.5, random_state=42)


def test_simulate_missing_values_raises_error_for_invalid_fraction():
    df = make_sample_df()
    with pytest.raises(ValueError):
        simulate_missing_values(df, ["Age"], 1.5, random_state=42)


def test_encode_target_converts_yes_no_to_binary():
    df = make_sample_df()
    result = encode_target(df, "Attrition")
    assert set(result["Attrition"].unique()) == {0, 1}


def test_encode_target_does_not_modify_original_dataframe():
    df = make_sample_df()
    df_copy = df.copy(deep=True)

    encode_target(df, "Attrition")

    pd.testing.assert_frame_equal(df, df_copy)


def test_encode_target_raises_error_for_unexpected_values():
    df = make_sample_df()
    df.loc[0, "Attrition"] = "Maybe"

    with pytest.raises(ValueError):
        encode_target(df, "Attrition")


def test_split_data_returns_correct_shapes():
    df = make_sample_df()
    df = encode_target(df, "Attrition")

    x_train, x_test, y_train, y_test = split_data(
        df, "Attrition", test_size=0.33, random_state=42
    )

    assert len(x_train) + len(x_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert "Attrition" not in x_train.columns