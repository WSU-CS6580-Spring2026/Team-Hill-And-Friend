# src/Data_Processing/clean_train.py

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Project root (works no matter where you run from)
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = BASE_DIR / "data/raw/MTA_LIRR_Delays__Beginning_2010_20260122.csv"
OUT_PATH = BASE_DIR / "data/processed/lirr_train_clean_late.csv"

# Outlier rule
OUTLIER_MODE = "cap"   # "cap" (recommended) or "drop"
OUTLIER_Q = 0.999      # 99.9th percentile cutoff


def main() -> None:
    # 1) Load raw data
    df = pd.read_csv(RAW_PATH, low_memory=False)

    # 2) Parse datetimes for depart/arrive
    time_fmt = "%m/%d/%Y %I:%M:%S %p"
    df["depart_dt"] = pd.to_datetime(df["Depart Time"], format=time_fmt, errors="coerce")
    df["arrive_dt"] = pd.to_datetime(df["Arrive Time"], format=time_fmt, errors="coerce")

    # 3) Fix trips that cross midnight (arrive earlier than depart → add 1 day)
    cross = df["depart_dt"].notna() & df["arrive_dt"].notna() & (df["arrive_dt"] < df["depart_dt"])
    df.loc[cross, "arrive_dt"] = df.loc[cross, "arrive_dt"] + pd.Timedelta(days=1)

    # 4) Keep only "Late" rows (baseline regression dataset)
    df = df[df["Status"].astype(str).str.strip().str.lower() == "late"].copy()

    # 5) Clean target column: Minutes Late
    df["Minutes Late"] = pd.to_numeric(df["Minutes Late"], errors="coerce")
    df = df.dropna(subset=["Minutes Late", "depart_dt"])  # must have target + depart time
    df = df[df["Minutes Late"] >= 0]  # safety

    # 6) Handle extreme outliers (cap or drop above the chosen quantile)
    cap = df["Minutes Late"].quantile(OUTLIER_Q)
    if OUTLIER_MODE == "drop":
        df = df[df["Minutes Late"] <= cap]
    else:
        df["Minutes Late"] = df["Minutes Late"].clip(upper=cap)

    # 7) Standardize column names (spaces → underscores, lowercase)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 8) Add simple time-based features for later modeling/EDA
    df["hour"] = df["depart_dt"].dt.hour
    df["dow"] = df["depart_dt"].dt.dayofweek
    df["month"] = df["depart_dt"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # 9) Save cleaned output (overwrites each run)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # 10) Print summary (handy for debugging + writeup)
    print("Saved cleaned train data to:", OUT_PATH)
    print("Rows:", len(df))
    print("Outlier mode:", OUTLIER_MODE, "| quantile:", OUTLIER_Q, "| cap used:", float(cap))
    print(df["minutes_late"].describe(percentiles=[0.95, 0.99, 0.999]))


if __name__ == "__main__":
    main()