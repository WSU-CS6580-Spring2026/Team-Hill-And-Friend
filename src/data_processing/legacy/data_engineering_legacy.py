import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_processing.legacy.clean_train_legacy import clean_lirr_train_legacy
from src.data_processing.legacy.clean_weather_legacy import clean_weather_legacy

# Define default output paths (consistent with your project structure)
BASE_DIR = Path(__file__).resolve().parents[3]
LIRR_RAW = BASE_DIR / "data/raw/MTA_LIRR_Delays__Beginning_2010_20260122.csv"
WEATHER_RAW = BASE_DIR / "data/raw/Long_Island_Weather.csv"
LIRR_OUT = BASE_DIR / "data/interim/lirr_train_clean_legacy.csv"
WEATHER_OUT = BASE_DIR / "data/interim/weather_clean_legacy.csv"
MERGED_OUT = BASE_DIR / "data/processed/merged_lirr_weather_legacy.csv"

def merged_dataset_legacy(rolling_window: int = 7) -> "pd.DataFrame":
    """
    Cleans the LIRR and Weather datasets using the reusable functions,
    merges them on date, and returns the merged DataFrame.

    Returns
    -------
    pd.DataFrame
        Merged dataset
    """
    # 1) Clean LIRR and Weather datasets
    df_train = clean_lirr_train_legacy(LIRR_RAW, LIRR_OUT)
    df_weather = clean_weather_legacy(WEATHER_RAW, WEATHER_OUT, rolling_window=rolling_window)

    # 2) Merge on date
    df_merged = df_train.merge(
        df_weather,
        left_on="service_date",
        right_on="DATE",
        how="right"
    )

    # 3) Drop duplicate service_date column from train
    df_merged = df_merged.drop(columns=["service_date"])

    # 4) Save merged dataset
    MERGED_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(MERGED_OUT, index=False)

    return df_merged


# Standalone execution
if __name__ == "__main__":
    df = merged_dataset_legacy()
    print("Merged dataset saved to:", MERGED_OUT)
    print("Rows:", len(df))
    print(df.head())
