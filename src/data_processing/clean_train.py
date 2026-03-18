from pathlib import Path
import json
import pandas as pd
import numpy as np

# Hard cap for extreme outliers
OUTLIER_CAP_MINUTES = 200
TEST_YEAR = 2025

def clean_lirr_train(
    input_path: Path,
    output_path: Path,
    outlier_cap: int = OUTLIER_CAP_MINUTES,
    test_year: int = TEST_YEAR,
) -> pd.DataFrame:
    """
    Cleans LIRR train delay data.

    Parameters
    ----------
    input_path : Path
        Path to raw CSV file
    output_path : Path
        Path to save cleaned CSV
    outlier_cap : int
        Maximum allowed minutes_late value
    test_year : int
        Year to hold out from mean encoding to prevent data leakage.
        Means are computed on all rows BEFORE this year, then mapped
        onto the full dataset (including test_year rows).

    Returns
    -------
    pd.DataFrame
        Cleaned LIRR train dataframe

    Side Effects
    ------------
    Saves a JSON file of mean-encoding dicts alongside output_path,
    named  <output_stem>_mean_encodings.json.  Each key is a column
    name; each value is a {category: mean} mapping that can be used
    at inference time.
    """
    # 1) Load raw data
    df = pd.read_csv(input_path, low_memory=False)

    # 2) Remove cancelled / partial cancel trips
    df = df[~df["Status"].isin(["Partial Cancel", "Cancelled"])]

    # 3) Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # 4) Convert service_date
    df["service_date"] = pd.to_datetime(
        df["service_date"], errors="coerce"
    ).dt.normalize()

    # 5) Convert minutes_late numeric and cap outliers
    df["minutes_late"] = (
        pd.to_numeric(df["minutes_late"], errors="coerce")
        .clip(upper=outlier_cap)
        .astype("float32")
    )

    # 6) Perform Log Transformation on minutes_late
    df["minutes_late"] = df["minutes_late"].apply(
        lambda x: np.log1p(x) if pd.notnull(x) else x
    )

    # 7) Select + enforce dtypes
    df = df[
        [
            "service_date",
            "train",
            "depart_station",
            "arrive_station",
            "minutes_late",
        ]
    ]
    df = df.astype(
        {
            "train": "string",
            "depart_station": "string",
            "arrive_station": "string",
        }
    )

    # 8) Create station_pair column
    df["station_pair"] = df["depart_station"] + "_" + df["arrive_station"]
    all_cat_cols = ["train", "depart_station", "arrive_station", "station_pair"]

    # 9) Mean-encode using only pre-test_year rows to avoid data leakage
    #    Then save the encoding dict so it can be reused at inference time.
    train_mask = df["service_date"].dt.year < test_year
    df_train = df[train_mask]

    mean_encodings: dict[str, dict] = {}
    for col in all_cat_cols:
        means = df_train.groupby(col, observed=True)["minutes_late"].mean()
        mean_encodings[col] = means.to_dict()           # save for later use
        df[col + "_mean"] = df[col].map(means)          # apply to full df

    df.drop(columns=all_cat_cols, inplace=True)

    # 10) Save cleaned output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # 11) Persist mean-encoding dict next to the cleaned CSV
    encodings_path = output_path.with_name(
        output_path.stem + "_mean_encodings.json"
    )
    with open(encodings_path, "w") as f:
        json.dump(mean_encodings, f)
    print("Saved mean encodings to:", encodings_path)

    return df

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    RAW_PATH = BASE_DIR / "data/raw/MTA_LIRR_Delays__Beginning_2010_20260122.csv"
    OUT_PATH = BASE_DIR / "data/interim/lirr_train_clean.csv"

    df_clean = clean_lirr_train(RAW_PATH, OUT_PATH)
    print("Saved cleaned train data to:", OUT_PATH)
    print("Rows:", len(df_clean))
    print("Hard outlier cap (minutes):", OUTLIER_CAP_MINUTES)
    print(df_clean.head())
    print("Missing minutes_late:", df_clean["minutes_late"].isna().sum())