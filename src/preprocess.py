import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_data():
    """
    Load the Malaysia/Kaggle dataset and Kaduna South dataset.
    """

    malaysia_path = DATA_DIR / "data_kaggle.csv"
    kaduna_path = DATA_DIR / "Strict_Kaduna_South_Dataset.csv"

    malaysia_df = pd.read_csv(malaysia_path)
    kaduna_df = pd.read_csv(kaduna_path)

    print("Malaysia dataset loaded successfully.")
    print("Kaduna South dataset loaded successfully.")

    print("\nMalaysia columns:")
    print(malaysia_df.columns.tolist())

    print("\nKaduna South columns:")
    print(kaduna_df.columns.tolist())

    print("\nMalaysia shape:", malaysia_df.shape)
    print("Kaduna South shape:", kaduna_df.shape)

    return malaysia_df, kaduna_df


def clean_data(df):
    """
    Basic cleaning:
    - Remove duplicate rows
    - Remove rows with missing values
    """

    df = df.drop_duplicates()
    df = df.dropna()

    return df


def align_features(malaysia_df, kaduna_df, common_features, target_column):
    """
    Select common features from both datasets.
    """

    X_malaysia = malaysia_df[common_features]
    y_malaysia = malaysia_df[target_column]

    X_kaduna = kaduna_df[common_features]
    y_kaduna = kaduna_df[target_column]

    return X_malaysia, y_malaysia, X_kaduna, y_kaduna


if __name__ == "__main__":
    malaysia_df, kaduna_df = load_data()

    malaysia_df = clean_data(malaysia_df)
    kaduna_df = clean_data(kaduna_df)

    print("\nAfter cleaning:")
    print("Malaysia shape:", malaysia_df.shape)
    print("Kaduna South shape:", kaduna_df.shape)