import os
import random
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load raw CSV data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_csv(file_path)


def simulate_missing_values(
    df: pd.DataFrame,
    columns: list[str],
    missing_fraction: float,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Return a copy of the dataframe with simulated missing values
    in the specified columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if not 0 <= missing_fraction <= 1:
        raise ValueError("missing_fraction must be between 0 and 1.")

    df_copy = df.copy()
    random.seed(random_state)

    for column in columns:
        if column not in df_copy.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")

        n_missing = int(len(df_copy) * missing_fraction)
        if n_missing > 0:
            missing_indices = random.sample(range(len(df_copy)), n_missing)
            df_copy.loc[missing_indices, column] = pd.NA

    return df_copy


def encode_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Convert target from Yes/No to 1/0.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    df_copy = df.copy()
    mapping = {"Yes": 1, "No": 0}
    df_copy[target_column] = df_copy[target_column].map(mapping)

    if df_copy[target_column].isna().any():
        raise ValueError("Target column contains unexpected values.")

    return df_copy


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int
):
    """
    Split data into train and test sets using stratification.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    features = df.drop(columns=[target_column])
    target = df[target_column]

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target
    )


def save_split_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    train_path: str,
    test_path: str,
    target_column: str
) -> None:
    """
    Save processed train and test data.
    """
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    train_df = x_train.copy()
    train_df[target_column] = y_train.values

    test_df = x_test.copy()
    test_df[target_column] = y_test.values

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


def main():
    config = load_config("configs/config.yaml")

    raw_data_path = config["data"]["raw_data_path"]
    target_column = config["data"]["target_column"]
    test_size = config["data"]["test_size"]
    train_path = config["data"]["processed_train_path"]
    test_path = config["data"]["processed_test_path"]
    random_state = config["project"]["random_state"]

    df = load_data(raw_data_path)

    if config["preprocessing"]["simulate_missing"]:
        selected_columns = (
            config["preprocessing"]["numeric_features"][:2] +
            config["preprocessing"]["categorical_features"][:2]
        )

        df = simulate_missing_values(
            df=df,
            columns=selected_columns,
            missing_fraction=config["preprocessing"]["missing_fraction"],
            random_state=random_state
        )

    df = encode_target(df, target_column)

    x_train, x_test, y_train, y_test = split_data(
        df=df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state
    )

    save_split_data(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column
    )

    print("Preprocessing complete.")
    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")


if __name__ == "__main__":
    main()