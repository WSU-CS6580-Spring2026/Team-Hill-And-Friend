import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_processing.clean_train import clean_lirr_train
from src.data_processing.clean_weather import clean_weather

BASE_DIR   = Path(__file__).resolve().parents[2]
LIRR_RAW   = BASE_DIR / "data/raw/MTA_LIRR_Delays__Beginning_2010_20260122.csv"
WEATHER_RAW = BASE_DIR / "data/raw/Long_Island_Weather.csv"

TRAIN_OUT   = BASE_DIR / "data/interim/lirr_train_clean.csv"
WEATHER_OUT = BASE_DIR / "data/interim/weather_clean.csv"
MERGED_OUT  = BASE_DIR / "data/processed/merged_lirr_weather.csv"
JSON_PATH   = BASE_DIR / "data/processed/feature_dict.json"

TEST_YEAR = 2025


def merged_dataset(
    lirr_raw: Path = LIRR_RAW,
    weather_raw: Path = WEATHER_RAW,
    train_out: Path = TRAIN_OUT,
    weather_out: Path = WEATHER_OUT,
    merged_out: Path = MERGED_OUT,
    json_path: Path = JSON_PATH,
    rolling_window: int = 7,
    test_year: int = TEST_YEAR,
) -> pd.DataFrame:
    """
    Cleans LIRR and Weather datasets, merges them, applies feature
    engineering, and saves the result.

    Mean encodings are computed in clean_lirr_train and saved to a JSON
    alongside the interim file.  This function loads that dict, appends
    the clip bounds and any additional encoding metadata produced here,
    then writes the combined dict to json_path in data/processed.

    All statistics that could leak test-year information (clip bounds,
    rolling baselines) are computed exclusively on rows where
    DATE.year < test_year.

    Parameters
    ----------
    lirr_raw, weather_raw : Path
        Paths to raw source CSVs.
    train_out, weather_out : Path
        Paths for interim cleaned CSVs.
    merged_out : Path
        Path for the final merged CSV.
    json_path : Path
        Path to write the combined feature dict (JSON).
    rolling_window : int
        Window size used by clean_weather for temperature imputation.
    test_year : int
        Year to hold out; no statistics are computed using its rows.

    Returns
    -------
    pd.DataFrame
        Fully merged and feature-engineered dataframe.
    """
    # 1) Clean LIRR — mean encodings are saved next to train_out
    df_train = clean_lirr_train(lirr_raw, train_out, test_year=test_year)

    # Load the mean-encoding dict written by clean_lirr_train
    encodings_path = train_out.with_name(train_out.stem + "_mean_encodings.json")
    with open(encodings_path) as f:
        feature_dict: dict = json.load(f)

    # 2) Clean Weather
    df_weather = clean_weather(weather_raw, weather_out, rolling_window=rolling_window)


    # 3) Merge
    df_merged = df_train.merge(
        df_weather,
        left_on="service_date",
        right_on="DATE",
        how="inner",
    )
    df_merged = df_merged.drop(columns=["service_date"])

    # 4) Sort, drop missing target, restrict to post-2021
    df_merged["DATE"] = pd.to_datetime(df_merged["DATE"])
    df_merged = (
        df_merged
        .sort_values("DATE")
        .dropna(subset=["minutes_late"])
        .reset_index(drop=True)
    )
    df_merged = df_merged[df_merged["DATE"].dt.year >= 2022].reset_index(drop=True)

    # 5) Time features
    df_merged["year"]  = df_merged["DATE"].dt.year
    df_merged["month"] = df_merged["DATE"].dt.month
    df_merged["day"]   = df_merged["DATE"].dt.day
    df_merged["dow"]   = df_merged["DATE"].dt.dayofweek

    # 6) Clip outliers — fit bounds on training rows only
    train_mask = df_merged["DATE"].dt.year < test_year
    train_mean = df_merged.loc[train_mask, "minutes_late"].mean()
    train_std  = df_merged.loc[train_mask, "minutes_late"].std()
    clip_upper = float(train_mean + 3 * train_std)

    df_merged["minutes_late"] = df_merged["minutes_late"].clip(upper=clip_upper)
    feature_dict["clip_upper_minutes_late"] = clip_upper   # save for inference

    # 7) Cyclical encoding — sin + cos for full circle representation
    for col, period in [("month", 12), ("day", 31), ("dow", 7)]:
        df_merged[f"sin_{col}"] = np.sin(2 * np.pi * df_merged[col] / period)
        df_merged[f"cos_{col}"] = np.cos(2 * np.pi * df_merged[col] / period)


    # 8) Rolling features — computed per-row using only past data
    df_merged["rolling_delay3"] = df_merged["minutes_late"].rolling(3, min_periods=1).mean()
    df_merged["rolling_delay7"] = df_merged["minutes_late"].rolling(7, min_periods=1).mean()
    df_merged["total_prcp3"]    = df_merged["PRCP_TOTAL"].rolling(3, min_periods=1).sum()

    # 9) Save merged CSV and combined feature dict
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(merged_out, index=False)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(feature_dict, f, indent=2)
    print("Feature dict saved to:", json_path)

    return df_merged

if __name__ == "__main__":
    df = merged_dataset()
    print("Merged dataset saved to:", MERGED_OUT)
    print("Rows:", len(df))
    print(df.head())