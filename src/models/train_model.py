# src/models/train_model.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


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
    return parser.parse_args()


def split_dataset(
    input_path: Path,
    train_out: Path,
    test_out: Path,
    target_col: str,
    test_size: float,
    random_state: int,
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


def main() -> None:
    args = parse_args()

    # File is in src/models/, so repo root is two levels up
    base_dir = Path(__file__).resolve().parents[2]

    input_path = (base_dir / args.input).resolve()
    train_out = (base_dir / args.train_out).resolve()
    test_out = (base_dir / args.test_out).resolve()

    split_dataset(
        input_path=input_path,
        train_out=train_out,
        test_out=test_out,
        target_col=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()