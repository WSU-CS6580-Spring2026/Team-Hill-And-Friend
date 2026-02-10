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

    # 2) Parse datetimes (needed to derive service_date cleanly)
    time_fmt = "%m/%d/%Y %I:%M:%S %p"
    df["depart_dt"] = pd.to_datetime(df["Depart Time"], format=time_fmt, errors="coerce")
    df["arrive_dt"] = pd.to_datetime(df["Arrive Time"], format=time_fmt, errors="coerce")

    # Convert Service Date to a true date-only field (python datetime.date)
    # (This will become `service_date` after we standardize column names.)
    df["Service Date"] = pd.to_datetime(df["Service Date"], errors="coerce").dt.date

    # 3) (REMOVED) cross-midnight fix — per your request

    # 4) Keep only "Late" rows
    df = df[df["Status"].astype(str).str.strip().str.lower() == "late"].copy()

    # 5) Minutes Late numeric (but DO NOT drop NA/negative rows — per your request)
    df["Minutes Late"] = pd.to_numeric(df["Minutes Late"], errors="coerce")

    # 6) Hard cap outliers at 200 minutes (only affects non-null values)
    df["Minutes Late"] = df["Minutes Late"].clip(upper=OUTLIER_CAP_MINUTES)

    # 7) Standardize column names (spaces → underscores, lowercase)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 8) Drop columns (this is where you wanted to do it)
    cols_to_drop = [
        "depart_time",
        "arrive_time",
        "period",
        "status",
        "delay_category",
        "depart_dt",
        "arrive_dt",
        "hour",
        "dow",
        "month",
        "is_weekend",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 9) Save cleaned output (overwrites each run)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # 10) Print summary
    print("Saved cleaned train data to:", OUT_PATH)
    print("Rows:", len(df))
    print("Hard outlier cap (minutes):", OUTLIER_CAP_MINUTES)
    if "minutes_late" in df.columns:
        print(df["minutes_late"].describe(percentiles=[0.95, 0.99]))


if __name__ == "__main__":
    main()