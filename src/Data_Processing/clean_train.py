from pathlib import Path
import pandas as pd

# Project root (works no matter where you run from)
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = BASE_DIR / "data/raw/MTA_LIRR_Delays__Beginning_2010_20260122.csv"
OUT_PATH = BASE_DIR / "data/processed/lirr_train_clean_late.csv"

# Hard cap for extreme outliers
OUTLIER_CAP_MINUTES = 200


def main() -> None:
    # 1) Load raw data
    df = pd.read_csv(RAW_PATH, low_memory=False)

    # 2) Remove cancelled / partial cancel trips
    df = df[~df['Status'].isin(['Partial Cancel', 'Cancelled'])]

    # 3) Standardize column names (spaces → underscores, lowercase)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 4) Convert service date to date-only field
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce").dt.date

    # 5) Convert minutes_late numeric and cap outliers
    df["minutes_late"] = pd.to_numeric(df["minutes_late"], errors="coerce").clip(upper=OUTLIER_CAP_MINUTES)

    # 6) Rename key columns
    df = df.rename(
        columns={
            "train": "train",
            "branch": "branch",
            "depart_station": "depart_station",
            "arrive_station": "arrive_station",
        }
    )

    # 7) Keep only the columns needed for training
    df = df[["service_date", "train", "branch", "depart_station", "arrive_station", "minutes_late"]]

    # 8) Save cleaned output
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # 9) Print summary
    print("Saved cleaned train data to:", OUT_PATH)
    print("Rows:", len(df))
    print("Hard outlier cap (minutes):", OUTLIER_CAP_MINUTES)
    print(df.head(5))
    print("Missing minutes_late:", df["minutes_late"].isna().sum())


if __name__ == "__main__":
    main()