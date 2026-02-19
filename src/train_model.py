from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split processed data, train a linear regression model, "
            "generate predictions, and save the trained model."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/merged_lirr_weather.csv",
        help="Path to processed dataset CSV.",
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
        help="Fraction of rows used for test split.",
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
        help="Output path for test-set predictions.",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="models/linear_regression_pipeline.joblib",
        help="Output path for serialized trained model.",
    )
    return parser.parse_args()


def train_model(
    input_path: Path,
    target_col: str,
    test_size: float,
    random_state: int,
    predictions_out: Path,
    model_out: Path,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {input_path}\n"
            "Run: python src/data_processing/data_engineering.py"
        )

    df = pd.read_csv(input_path, low_memory=False)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found.\n"
            f"Columns available: {list(df.columns)}"
        )

    total_rows = len(df)
    df = df.dropna(subset=[target_col])
    dropped_rows = total_rows - len(df)

    x = df.drop(columns=[target_col])
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    numeric_features = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x_train.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
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

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    predictions = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": y_pred,
        }
    )
    predictions["residual"] = predictions["actual"] - predictions["predicted"]

    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_out, index=False)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_out)

    print(f"Input dataset: {input_path}")
    print(f"Rows before dropna('{target_col}'): {total_rows:,}")
    print(f"Dropped rows with missing '{target_col}': {dropped_rows:,}")
    print(f"Train rows: {len(x_train):,}")
    print(f"Test rows: {len(x_test):,}")
    print("\nLinear Regression Evaluation")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"Predictions saved to: {predictions_out}")
    print(f"Model saved to: {model_out}")


def main() -> None:
    args = parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    input_path = (base_dir / args.input).resolve()
    predictions_out = (base_dir / args.predictions_out).resolve()
    model_out = (base_dir / args.model_out).resolve()

    train_model(
        input_path=input_path,
        target_col=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        predictions_out=predictions_out,
        model_out=model_out,
    )


if __name__ == "__main__":
    main()
