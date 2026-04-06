import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_processing.clean_train import clean_lirr_train
from src.data_processing.clean_weather import clean_weather

BASE_DIR    = Path(__file__).resolve().parents[2]
LIRR_RAW    = BASE_DIR / "data/raw/MTA_LIRR_Delays__Beginning_2010_20260122.csv"
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
    Cleans LIRR and weather datasets, merges them, applies feature
    engineering, and saves the result.

    Mean encodings are computed here on pre-test_year rows only, then
    applied to the full dataset.  All other statistics (clip bounds,
    rolling baselines) are likewise fit on pre-test_year rows so no
    information from the held-out year leaks into training.

    Parameters
    ----------
    lirr_raw, weather_raw : Path
        Paths to raw source CSVs.
    train_out, weather_out : Path
        Paths for interim cleaned CSVs.
    merged_out : Path
        Path for the final merged CSV.
    json_path : Path
        Path to write the feature dict (JSON).
    rolling_window : int
        Window size used by clean_weather for temperature imputation.
    test_year : int
        Year to hold out; no statistics are computed using its rows.

    Returns
    -------
    pd.DataFrame
        Fully merged and feature-engineered dataframe.
    """
    # 1) Clean LIRR
    df_train = clean_lirr_train(lirr_raw, train_out)

    # 2) Clean weather
    df_weather = clean_weather(weather_raw, weather_out, rolling_window=rolling_window)

    # 3) Merge on date
    df_merged = df_train.merge(
        df_weather,
        left_on="service_date",
        right_on="date",
        how="inner",
    )
    df_merged = df_merged.drop(columns=["service_date"])

    # 4) Sort, drop missing target, restrict to post-2021
    df_merged["date"] = pd.to_datetime(df_merged["date"])
    df_merged = (
        df_merged
        .sort_values("date")
        .dropna(subset=["minutes_late"])
        .reset_index(drop=True)
    )
    df_merged = df_merged[df_merged["date"].dt.year >= 2022].reset_index(drop=True)

    # 5) Mean-encode categorical columns — fit on pre-test_year rows only
    #    depart_station and arrive_station are kept as label columns for
    #    downstream use (e.g. predictions CSV); train and station_pair are dropped.
    df_merged["station_pair"] = df_merged["depart_station"] + "_" + df_merged["arrive_station"]
    cat_cols = ["train", "depart_station", "arrive_station", "station_pair"]
    enc_mask = df_merged["date"].dt.year < test_year

    feature_dict: dict = {}
    for col in cat_cols:
        means = df_merged.loc[enc_mask].groupby(col, observed=True)["minutes_late"].mean()
        feature_dict[col] = means.to_dict()
        df_merged[col + "_mean"] = df_merged[col].map(means)

    feature_dict["global_mean"] = float(df_merged.loc[enc_mask, "minutes_late"].mean())
    df_merged = df_merged.drop(columns=["train", "station_pair"])
    df_merged = df_merged.rename(columns={
        "depart_station": "depart_label",
        "arrive_station": "arrive_label",
    })

    # 6) Time features
    df_merged["year"]  = df_merged["date"].dt.year
    df_merged["month"] = df_merged["date"].dt.month
    df_merged["day"]   = df_merged["date"].dt.day
    df_merged["dow"]   = df_merged["date"].dt.dayofweek

    # 7) Clip outliers — fit bounds on pre-test_year rows only
    train_mask = df_merged["date"].dt.year < test_year
    train_mean = df_merged.loc[train_mask, "minutes_late"].mean()
    train_std  = df_merged.loc[train_mask, "minutes_late"].std()
    clip_upper = float(train_mean + 3 * train_std)

    df_merged["minutes_late"] = df_merged["minutes_late"].clip(upper=clip_upper)
    feature_dict["clip_upper_minutes_late"] = clip_upper

    # 8) Cyclical encoding — sin + cos for full circle representation
    for col, period in [("month", 12), ("day", 31), ("dow", 7)]:
        df_merged[f"sin_{col}"] = np.sin(2 * np.pi * df_merged[col] / period)
        df_merged[f"cos_{col}"] = np.cos(2 * np.pi * df_merged[col] / period)

    # 9) Rolling features — computed per-row using only past data
    df_merged["rolling_delay3"] = df_merged["minutes_late"].rolling(3, min_periods=1).mean()
    df_merged["rolling_delay7"] = df_merged["minutes_late"].rolling(7, min_periods=1).mean()
    df_merged["total_prcp3"]    = df_merged["prcp_total"].rolling(3, min_periods=1).sum()

    # 10) Save merged CSV and feature dict
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
