from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

TEST_YEAR = 2025

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost on LIRR delay data.")
    parser.add_argument(
        "--merged-data", type=str, default="data/processed/merged_lirr_weather.csv",
        help="Merged + feature-engineered CSV (output of build_dataset.py)",
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
    dates = df["DATE"].reset_index(drop=True)
    y     = df[target].reset_index(drop=True)
    X     = df.drop(columns=[target, "DATE"]).reset_index(drop=True)
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


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Inverts log1p transform on both actual and predicted values before
    computing all metrics, so reported numbers are in original minutes.
    """
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)

    metrics = {
        "MAE":       float(mean_absolute_error(y_true_orig, y_pred_orig)),
        "RMSE":      float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))),
        "R2":        float(r2_score(y_true_orig, y_pred_orig)),
        "MAPE":      float(mean_absolute_percentage_error(y_true_orig, y_pred_orig) * 100),
        "AvgActual": float(y_true_orig.mean()),
        "AvgPred":   float(y_pred_orig.mean()),
    }
    return metrics, y_true_orig, y_pred_orig


def plot_results(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    plots_out: Path,
    test_year: int,
    model: xgb.XGBRegressor,
    feature_names: list[str],
    top_n: int = 20,
) -> None:
    plots_out.mkdir(parents=True, exist_ok=True)

    # Actual vs Predicted
    plt.figure(figsize=(14, 6))
    plt.plot(dates, y_true, label="Actual",    color="black")
    plt.plot(dates, y_pred, label="Predicted", color="green")
    plt.title(f"XGBoost: Actual vs Predicted Delays ({test_year})")
    plt.xlabel("Date")
    plt.ylabel("Minutes Late")
    plt.legend()

    metrics_text = (
        f"MAE: {metrics['MAE']:.2f}\n"
        f"RMSE: {metrics['RMSE']:.2f}\n"
        f"R²: {metrics['R2']:.3f}\n"
        f"MAPE: {metrics['MAPE']:.1f}%"
    )
    plt.gcf().text(
        0.78, 0.75, metrics_text,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    pred_file = plots_out / f"xgb_predictions_{test_year}.png"
    plt.savefig(pred_file, bbox_inches="tight")
    plt.close()
    print(f"Predictions plot saved to {pred_file}")

    # Feature Importance
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
    importance.plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"XGBoost Feature Importance (Top {top_n})")
    ax.set_xlabel("Importance (F-score)")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)

    fi_file = plots_out / "xgb_feature_importance.png"
    fig.savefig(fi_file, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved to {fi_file}")


def main():
    args     = parse_args()
    base_dir = Path(__file__).resolve().parents[2]

    # Load fully prepared dataset
    df = pd.read_csv(base_dir / args.merged_data)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    # Train / test split by year
    train_df = df[df["DATE"].dt.year <  args.test_year].reset_index(drop=True)
    test_df  = df[df["DATE"].dt.year == args.test_year].reset_index(drop=True)

    target      = args.target
    global_mean = train_df[target].mean()

    X_train, y_train, _          = split_features_target(train_df, target, fill_value=global_mean)
    X_test,  y_test,  dates_test = split_features_target(test_df,  target, fill_value=global_mean)

    # Train
    model = train_xgb(X_train, y_train, base_dir / args.model_out, args.random_state)

    # Predict + invert log1p for metrics and reporting
    y_pred = model.predict(X_test)
    metrics, y_true_orig, y_pred_orig = evaluate_model(y_test, y_pred)
    print("Evaluation Metrics:", metrics)

    # Save metrics
    metrics_out = base_dir / args.metrics_out
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save predictions (inverted, original minutes scale)
    preds_out = base_dir / args.predictions_out
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "DATE":      dates_test,
        "Actual":    y_true_orig,
        "Predicted": y_pred_orig,
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