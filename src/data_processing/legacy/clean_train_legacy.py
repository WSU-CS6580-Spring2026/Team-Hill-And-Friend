from pathlib import Path
import pandas as pd

# Hard cap for extreme outliers
OUTLIER_CAP_MINUTES = 200


def clean_lirr_train_legacy(
    input_path: Path,
    output_path: Path,
    outlier_cap: int = OUTLIER_CAP_MINUTES,
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

    Returns
    -------
    pd.DataFrame
        Cleaned LIRR train dataframe
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

    # 6) Select + enforce dtypes
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

    # 7) convert categorical columns
    categorical_cols = ["depart_station", "arrive_station"]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    # 8) Save cleaned output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]

    RAW_PATH = BASE_DIR / "data/raw/MTA_LIRR_Delays__Beginning_2010_20260122.csv"
    OUT_PATH = BASE_DIR / "data/interim/lirr_train_clean_legacy.csv"

    df_clean = clean_lirr_train_legacy(RAW_PATH, OUT_PATH)

    print("Saved cleaned train data to:", OUT_PATH)
    print("Rows:", len(df_clean))
    print("Hard outlier cap (minutes):", OUTLIER_CAP_MINUTES)
    print(df_clean.head())
    print("Missing minutes_late:", df_clean["minutes_late"].isna().sum())