from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"


def _resolve_data_path(data_path: str | Path = DEFAULT_DATA_PATH) -> Path:
    path = Path(data_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if path.exists():
        return path

    fallback_path = path.with_suffix(path.suffix + ".csv")
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(f"Dataset not found at {path}")


def load_and_preprocess_data(
    data_path: str | Path = DEFAULT_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Load the churn dataset and return train/test splits with a preprocessor."""
    resolved_path = _resolve_data_path(data_path)
    df = pd.read_csv(resolved_path)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    if df["Churn"].isna().any():
        raise ValueError("Churn column contains values other than 'Yes' and 'No'.")

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_columns),
            ("numerical", numerical_pipeline, numerical_columns),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(preprocessor)
