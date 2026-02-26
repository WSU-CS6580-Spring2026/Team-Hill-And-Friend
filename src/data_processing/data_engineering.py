import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_processing.clean_train import clean_lirr_train
from src.data_processing.clean_weather import clean_weather

# Define default output paths (consistent with your project structure)
BASE_DIR = Path(__file__).resolve().parents[2]
LIRR_RAW = BASE_DIR / "data/raw/MTA_LIRR_Delays__Beginning_2010_20260122.csv"
WEATHER_RAW = BASE_DIR / "data/raw/Long_Island_Weather.csv"
LIRR_OUT = BASE_DIR / "data/interim/lirr_train_clean.csv"
WEATHER_OUT = BASE_DIR / "data/interim/weather_clean.csv"
MERGED_OUT = BASE_DIR / "data/processed/merged_lirr_weather.csv"

def merged_dataset(rolling_window: int = 7) -> "pd.DataFrame":
    """
    Cleans the LIRR and Weather datasets using the reusable functions,
    merges them on date, and returns the merged DataFrame.

    Returns
    -------
    pd.DataFrame
        Merged dataset
    """
    # 1) Clean LIRR and Weather datasets
    df_train = clean_lirr_train(LIRR_RAW, LIRR_OUT)
    df_weather = clean_weather(WEATHER_RAW, WEATHER_OUT, rolling_window=rolling_window)

    # 2) Merge on date
    df_merged = df_train.merge(
        df_weather,
        left_on="service_date",
        right_on="DATE",
        how="right"
    )

    # 3) Drop duplicate service_date column from train
    df_merged = df_merged.drop(columns=["service_date"])

    # 4) Sort by date and drop rows with missing target
    df_merged["DATE"] = pd.to_datetime(df_merged["DATE"])
    df_merged = df_merged.sort_values("DATE")
    df_merged = df_merged.dropna(subset=["minutes_late"]).reset_index(drop=True)

    # 5) Filter to 2022 and later to reduce noise
    df_merged = df_merged[df_merged["DATE"].dt.year >= 2022].reset_index(drop=True)

    # 6) Time features
    df_merged["year"]  = df_merged["DATE"].dt.year
    df_merged["month"] = df_merged["DATE"].dt.month
    df_merged["day"]   = df_merged["DATE"].dt.day
    df_merged["dow"]   = df_merged["DATE"].dt.dayofweek

    # 7) Clip outliers in minutes_late
    y = df_merged["minutes_late"]
    mean = y.mean()
    std = y.std()
    upper = mean + 3*std
    y = y.clip(upper=upper)
    df_merged = df_merged.loc[y.index].reset_index(drop=True)
    
    # 8) Cyclical encoding
    df_merged["cos_month"] = np.cos(2 * np.pi * df_merged["month"]/12)
    df_merged["cos_day"]   = np.cos(2 * np.pi * df_merged["day"]/31)
    df_merged["cos_dow"]   = np.cos(2 * np.pi * df_merged["dow"]/7)
    
    # 9) Rolling delays
    df_merged["rolling_delay3"] = df_merged["minutes_late"].rolling(3, min_periods=1).mean()
    df_merged["rolling_delay7"] = df_merged["minutes_late"].rolling(7, min_periods=1).mean()
    df_merged["total_prcp3"] = df_merged["PRCP_TOTAL"].rolling(3, min_periods=1).sum()

    # 10) Save merged dataset
    MERGED_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(MERGED_OUT, index=False)

    return df_merged


# Standalone execution
if __name__ == "__main__":
    df = merged_dataset()
    print("Merged dataset saved to:", MERGED_OUT)
    print("Rows:", len(df))
    print(df.head())
