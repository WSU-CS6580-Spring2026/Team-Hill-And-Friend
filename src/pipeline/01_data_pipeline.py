"""
Stage 1 — Data pipeline: clean → merge → feature engineering.
Outputs: data/interim/ cleaned CSVs, data/processed/merged_lirr_weather.csv,
         data/processed/feature_dict.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_processing.data_engineering import merged_dataset, MERGED_OUT

if __name__ == "__main__":
    df = merged_dataset()
    print(f"\nData pipeline complete. {len(df):,} rows saved to {MERGED_OUT}")
