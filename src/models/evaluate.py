from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

TEST_YEAR = 2025


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Inverts log1p on both arrays then computes metrics in original minutes.
    Returns (metrics_dict, y_true_orig, y_pred_orig).
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate a trained model from saved predictions."
    )
    parser.add_argument("--predictions-out", type=str, default="data/processed/xgb_predictions.csv")
    parser.add_argument("--model-out",       type=str, default="models/xgb_model.json")
    parser.add_argument("--metrics-out",     type=str, default="data/processed/xgb_metrics.json")
    parser.add_argument("--plots-out",       type=str, default="docs/Results")
    parser.add_argument("--test-year",       type=int, default=TEST_YEAR)
    parser.add_argument("--top-n-features",  type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """
    Standalone entry point: loads the saved predictions CSV and model,
    recomputes metrics, and regenerates all plots without retraining.
    """
    args     = parse_args()
    base_dir = Path(__file__).resolve().parents[2]

    # Predictions CSV is already in original minutes (not log space)
    preds_path = base_dir / args.predictions_out
    preds_df   = pd.read_csv(preds_path)
    preds_df["date"] = pd.to_datetime(preds_df["date"])

    y_true = preds_df["actual"].values
    y_pred = preds_df["predicted"].values
    dates  = preds_df["date"]

    metrics = {
        "MAE":       float(mean_absolute_error(y_true, y_pred)),
        "RMSE":      float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":        float(r2_score(y_true, y_pred)),
        "MAPE":      float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "AvgActual": float(y_true.mean()),
        "AvgPred":   float(y_pred.mean()),
    }
    print("Evaluation Metrics:", metrics)

    metrics_out = base_dir / args.metrics_out
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=4)

    model = xgb.XGBRegressor()
    model.load_model(str(base_dir / args.model_out))
    feature_names = model.get_booster().feature_names or []

    plot_results(
        dates, y_true, y_pred, metrics,
        base_dir / args.plots_out, args.test_year,
        model=model,
        feature_names=feature_names,
        top_n=args.top_n_features,
    )


if __name__ == "__main__":
    main()
