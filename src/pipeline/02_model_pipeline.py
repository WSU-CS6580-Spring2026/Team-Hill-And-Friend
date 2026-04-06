"""
Stage 2 — Model pipeline: train XGBoost → evaluate → save outputs.
Requires data/processed/merged_lirr_weather.csv (run 01_data_pipeline.py first).
Outputs: models/xgb_model.json, data/processed/xgb_predictions.csv,
         data/processed/xgb_metrics.json, docs/Results/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.train_model import main as train_main

if __name__ == "__main__":
    train_main()
    print("\nModel pipeline complete.")
