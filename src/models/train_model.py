from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.evaluate import evaluate_model, plot_results

TEST_YEAR = 2025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost on LIRR delay data.")
    parser.add_argument(
        "--merged-data", type=str, default="data/processed/merged_lirr_weather.csv",
        help="Merged + feature-engineered CSV (output of data_engineering.py)",
    )
    parser.add_argument("--model-out",       type=str, default="models/xgb_model.json")
    parser.add_argument("--predictions-out", type=str, default="data/processed/xgb_predictions.csv")
    parser.add_argument("--metrics-out",     type=str, default="data/processed/xgb_metrics.json")
    parser.add_argument("--plots-out",       type=str, default="docs/Results")
    parser.add_argument("--target",          type=str, default="minutes_late")
    parser.add_argument("--test-year",       type=int, default=TEST_YEAR)
    parser.add_argument("--random-state",    type=int, default=42)
    parser.add_argument("--top-n-features",  type=int, default=20,
                        help="Number of top features to show in importance plot")
    return parser.parse_args()


def split_features_target(
    df: pd.DataFrame,
    target: str,
    fill_value: float,
):
    """Separates features, target, and dates. Fills NaN with fill_value."""
    dates = df["date"].reset_index(drop=True)
    y     = df[target].reset_index(drop=True)
    X     = df.drop(columns=[target, "date", "depart_label", "arrive_label"]).reset_index(drop=True)
    X     = X.fillna(fill_value)
    return X, y, dates


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_out: Path,
    random_state: int = 42,
) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_out)
    print(f"Model saved to {model_out}")
    return model


def main():
    args     = parse_args()
    base_dir = Path(__file__).resolve().parents[2]

    # Load dataset
    df = pd.read_csv(base_dir / args.merged_data)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Train / test split by year
    train_df = df[df["date"].dt.year <  args.test_year].reset_index(drop=True)
    test_df  = df[df["date"].dt.year == args.test_year].reset_index(drop=True)

    target      = args.target
    global_mean = train_df[target].mean()

    # Capture station labels before split_features_target drops them
    depart_test = test_df["depart_label"].reset_index(drop=True)
    arrive_test = test_df["arrive_label"].reset_index(drop=True)

    # Split features & target
    X_train, y_train, _          = split_features_target(train_df, target, fill_value=global_mean)
    X_test,  y_test,  dates_test = split_features_target(test_df,  target, fill_value=global_mean)

    # Train
    model = train_xgb(X_train, y_train, base_dir / args.model_out, args.random_state)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    metrics, y_true_orig, y_pred_orig = evaluate_model(y_test, y_pred)
    print("Evaluation Metrics:", metrics)

    # Save metrics
    metrics_out = base_dir / args.metrics_out
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=4)

    # Load feature dict, persist the exact training column order, then re-save
    feature_dict_path = base_dir / "data/processed/feature_dict.json"
    with open(feature_dict_path) as f:
        feature_dict = json.load(f)
    feature_dict["feature_columns"] = X_train.columns.tolist()
    with open(feature_dict_path, "w") as f:
        json.dump(feature_dict, f, indent=2)

    # Save predictions CSV
    preds_out = base_dir / args.predictions_out
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date":      dates_test,
        "depart":    depart_test,
        "arrive":    arrive_test,
        "actual":    y_true_orig,
        "predicted": y_pred_orig,
    }).to_csv(preds_out, index=False)
    print(f"Predictions saved to {preds_out}")

    # Plot
    plot_results(
        dates_test, y_true_orig, y_pred_orig, metrics,
        base_dir / args.plots_out, args.test_year,
        model=model,
        feature_names=X_train.columns.tolist(),
        top_n=args.top_n_features,
    )


if __name__ == "__main__":
    main()
