# src/models/train_model.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from pandas.errors import ParserError
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    """
    Read CSV with a resilient fallback parser.

    Pandas' default C engine is faster but can fail on malformed rows or
    certain I/O tokenization issues. If that happens, retry with the Python
    engine to keep the training pipeline moving.
    """
    try:
        return pd.read_csv(csv_path, low_memory=False)
    except ParserError:
        print(
            f"Warning: C parser failed for {csv_path}. "
            "Retrying with engine='python'."
        )
        return pd.read_csv(csv_path, engine="python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split processed merged dataset into train and test sets."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/merged_lirr_weather.csv",
        help="Path to processed merged dataset CSV.",
    )
    parser.add_argument(
        "--train-out",
        type=str,
        default="data/processed/train.csv",
        help="Output path for train split CSV.",
    )
    parser.add_argument(
        "--test-out",
        type=str,
        default="data/processed/test.csv",
        help="Output path for test split CSV.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="minutes_late",
        help="Target column name.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for test set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--predictions-out",
        type=str,
        default="data/processed/linear_regression_predictions.csv",
        help="Output path for linear regression predictions CSV.",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="data/processed/linear_regression_metrics.json",
        help="Output path for linear regression metrics JSON.",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="models/linear_regression_pipeline.joblib",
        help="Output path for serialized trained model.",
    )
    return parser.parse_args()


def split_dataset(
    input_path: Path,
    train_out: Path,
    test_out: Path,
    target_col: str,
    test_size: float,
    random_state: int,
) -> None:
    # Stage 1: create reproducible train/test splits from merged processed data.
    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {input_path}\n"
            "Run: python src/data_processing/data_engineering.py"
        )

    df = read_csv_with_fallback(input_path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found.\n"
            f"Columns available: {list(df.columns)}"
        )

    total_rows = len(df)

    # Drop rows without target (expected due to right-join on weather)
    df = df.dropna(subset=[target_col])
    dropped_rows = total_rows - len(df)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"Input dataset: {input_path}")
    print(f"Total rows: {total_rows:,}")
    print(f"Dropped rows with missing '{target_col}': {dropped_rows:,}")
    print(f"Train rows: {len(train_df):,} -> {train_out}")
    print(f"Test rows:  {len(test_df):,} -> {test_out}")


def train_linear_regression(
    train_path: Path,
    test_path: Path,
    target_col: str,
    predictions_out: Path,
    metrics_out: Path,
    model_out: Path,
) -> None:
    # Stage 2: train/evaluate linear regression from pre-split CSVs.
    if not train_path.exists():
        raise FileNotFoundError(f"Train split not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found at {test_path}")

    train_df = read_csv_with_fallback(train_path)
    test_df = read_csv_with_fallback(test_path)

    if target_col not in train_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in train split.\n"
            f"Columns available: {list(train_df.columns)}"
        )
    if target_col not in test_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in test split.\n"
            f"Columns available: {list(test_df.columns)}"
        )

    x_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    x_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Keep all non-target features; split preprocessing by dtype.
    numeric_features = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x_train.select_dtypes(exclude=["number"]).columns.tolist()

    # Impute missing values and one-hot encode text/date-like columns.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Core regression metrics for model quality.
    mae = mean_absolute_error(y_test, y_pred)
    # Compute RMSE in a version-compatible way for older sklearn releases.
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    # Save row-level predictions to support downstream error analysis.
    predictions_df = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": y_pred,
        }
    )
    predictions_df["residual"] = predictions_df["actual"] - predictions_df["predicted"]

    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(predictions_out, index=False)
    dump(model, model_out)

    # Save summary metrics for quick tracking/reporting.
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "target": target_col,
    }
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nLinear Regression Evaluation")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"Predictions saved to: {predictions_out}")
    print(f"Metrics saved to: {metrics_out}")
    print(f"Model saved to: {model_out}")


def main() -> None:
    args = parse_args()

    # File is in src/models/, so repo root is two levels up
    base_dir = Path(__file__).resolve().parents[2]

    input_path = (base_dir / args.input).resolve()
    train_out = (base_dir / args.train_out).resolve()
    test_out = (base_dir / args.test_out).resolve()
    predictions_out = (base_dir / args.predictions_out).resolve()
    metrics_out = (base_dir / args.metrics_out).resolve()
    model_out = (base_dir / args.model_out).resolve()

    split_dataset(
        input_path=input_path,
        train_out=train_out,
        test_out=test_out,
        target_col=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    train_linear_regression(
        train_path=train_out,
        test_path=test_out,
        target_col=args.target,
        predictions_out=predictions_out,
        metrics_out=metrics_out,
        model_out=model_out,
    )


if __name__ == "__main__":
    main()
