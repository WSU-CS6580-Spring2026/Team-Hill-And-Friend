# src/models/train_model.py
from __future__ import annotations

import argparse
from html import parser
import json
from pathlib import Path
import os
from joblib import dump
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

RESULTS_DIR = "/doc/Results"
MODEL_DIR = "models"

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
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--predictions-out",
        type=str,
        default="data/processed/xgb_predictions.csv",
        help="Output path for XGBoost predictions CSV.",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="data/processed/xgb_metrics.json",
        help="Output path for XGBoost metrics JSON.",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="models/xgb_model.json",
        help="Output path for serialized trained model.",
    )
    parser.add_argument(
        "--plots-out",
        type=str,
        default="docs/Results",
        help="Directory to save plots and other results."
    )

    return parser.parse_args()

def split_train_test(df: pd.DataFrame, target: str, test_year: int = 2025):
    """
    Split dataframe into train/test based on year in 'DATE' column.
    Fills missing values with global mean of training target.
    Drops DATE column for training.
    Returns X_train, X_test, y_train, y_test, dates_test
    """
    y = df[target]
    X = df.drop(columns=[target])
    
    # Train/test split
    X_train = X[X["DATE"].dt.year < test_year].copy()
    X_test  = X[X["DATE"].dt.year == test_year].copy()
    
    y_train = y.loc[X_train.index].copy()
    y_test  = y.loc[X_test.index].copy()
    
    dates_test = X_test["DATE"].reset_index(drop=True)
    
    # Drop DATE before training
    X_train.drop(columns=["DATE"], inplace=True)
    X_test.drop(columns=["DATE"], inplace=True)
    
    # Fill unseen categories / missing values
    global_mean = y_train.mean()
    X_train.fillna(global_mean, inplace=True)
    X_test.fillna(global_mean, inplace=True)
    
    # Reset indices
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test, dates_test

def train_xgb(X_train: pd.DataFrame, y_train: pd.Series, model_out: Path, **kwargs):
    """
    Train an XGBoost regressor and save the model for later use.

    Parameters:
    - X_train: Training features
    - y_train: Training target
    - save_filename: Name of the file to save the trained model
    - kwargs: Additional keyword arguments passed to xgb.XGBRegressor

    Returns:
    - model: Trained XGBoost model
    """
    # Train model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        **kwargs
    )
    model.fit(X_train, y_train)
    model.save_model(model_out)
    print(f"XGBoost model saved to {model_out}")
    return model

def evaluate_model(y_test: pd.Series, y_pred: np.ndarray):
    """
    Evaluate predictions and return a dictionary of metrics.
    Converts all values to Python floats for JSON compatibility.
    """
    # Convert back from log-transformation if needed
    y_test = np.expm1(y_test)
    y_pred = np.expm1(y_pred)
    
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_test, y_pred) * 100),
        "AvgActual": float(y_test.mean()),
        "AvgPred": float(y_pred.mean())
    }
    return metrics

def plot_xgb_results(
    xgb_model,
    X_train: pd.DataFrame,
    dates_test: pd.Series,
    y_test: pd.Series,
    pred_xgb: np.ndarray,
    plots_out: Path,
    top_n: int = 15,
    filename_prefix: str = "xgb_results"
):
    """
    Plots:
        1) Top XGBoost feature importance
        2) Actual vs Predicted for 2025
    Metrics (MAE, RMSE, R2, MAPE) are displayed on the predictions plot.
    Both plots are saved to /doc/Results
    """
    # Feature Importance Plot
    fi_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": xgb_model.feature_importances_
    }).sort_values(by="importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=fi_df, color="mediumseagreen")
    plt.title("Top XGBoost Feature Importances")
    plt.tight_layout()
    
    fi_path = plots_out / f"{filename_prefix}_feature_importance.png"
    plt.savefig(fi_path)
    print(f"Feature importance plot saved to {fi_path}")

    # Actual vs Predicted Plot for 2025
    # Reset indices
    dates_test = dates_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    pred_xgb = pd.Series(pred_xgb).reset_index(drop=True)

    # Focus on 2025
    mask_2025 = dates_test.dt.year == 2025
    dates_2025 = dates_test[mask_2025]
    y_2025 = y_test[mask_2025]
    pred_2025 = pred_xgb[mask_2025]

    # Compute metrics
    mae = mean_absolute_error(y_2025, pred_2025)
    rmse = np.sqrt(mean_squared_error(y_2025, pred_2025))
    r2 = r2_score(y_2025, pred_2025)
    mape = mean_absolute_percentage_error(y_2025, pred_2025) * 100

    metrics_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}\nMAPE: {mape:.1f}%"

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(dates_2025, y_2025, label="Actual", color="black")
    plt.plot(dates_2025, pred_2025, label="Predicted", color="green")
    plt.title(f"XGBoost: Actual vs Predicted Delays (2025)")
    plt.ylabel("Total Delay")
    plt.xlabel("Date")
    plt.legend()

    # Add metrics text on top right
    plt.gcf().text(0.78, 0.75, metrics_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    pred_path = plots_out / f"{filename_prefix}_predictions_2025.png"
    plt.savefig(pred_path)
    print(f"Predictions plot saved to {pred_path}")

def main() -> None:
    args = parse_args()

    # Base repo directory
    base_dir = Path(__file__).resolve().parents[2]

    input_path = (base_dir / args.input).resolve()
    train_out = (base_dir / args.train_out).resolve()
    test_out = (base_dir / args.test_out).resolve()
    predictions_out = (base_dir / args.predictions_out).resolve()
    metrics_out = (base_dir / args.metrics_out).resolve()
    model_out = (base_dir / args.model_out).resolve()
    plots_out = (base_dir / args.plots_out).resolve()

    # Load data
    df = pd.read_csv(input_path, parse_dates=["DATE"])
    
    # Split train/test
    X_train, X_test, y_train, y_test, dates_test = split_train_test(df, target=args.target, test_year=2025)
    
    # Train XGBoost
    xgb_model = train_xgb(X_train, y_train, model_out=model_out)
    
    # Predict
    pred_xgb = xgb_model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_model(y_test, pred_xgb)
    print("Evaluation metrics:", metrics)

    # Save metrics JSON
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_out}")

    # Save predictions
    predictions_df = pd.DataFrame({
        "DATE": dates_test,
        "Actual": y_test,
        "Predicted": pred_xgb
    })
    predictions_df.to_csv(predictions_out, index=False)
    print(f"Predictions saved to {predictions_out}")

    # Plot results (features + predictions)
    plot_xgb_results(
        xgb_model=xgb_model,
        X_train=X_train,
        dates_test=dates_test,
        y_test=y_test,
        pred_xgb=pred_xgb,
        plots_out=plots_out,
        top_n=15,
        filename_prefix="xgb_plots",
    )

if __name__ == "__main__":
    main()
