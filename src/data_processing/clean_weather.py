from pathlib import Path
import pandas as pd
import numpy as np

def clean_weather(
    input_path: Path,
    output_path: Path,
    rolling_window: int = 7
) -> pd.DataFrame:
    """
    Cleans Long Island weather data.

    Parameters
    ----------
    input_path : Path
        Path to raw CSV file.
    output_path : Path
        Path to save cleaned CSV.
    rolling_window : int
        Backward-looking rolling window size for TMIN/TMAX imputation.

    Returns
    -------
    pd.DataFrame
        Cleaned weather dataframe.
    """
    df = pd.read_csv(input_path)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.normalize()

    df[["PRCP", "SNOW", "SNWD"]] = df[["PRCP", "SNOW", "SNWD"]].fillna(0)
    df["PRCP_TOTAL"] = (df["PRCP"] + df["SNOW"]).round(2)

    drop_cols = ["WESD", "WT05", "TOBS", "STATION", "NAME"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.sort_values("DATE").reset_index(drop=True)

    # center=False prevents future data leaking into imputed values
    df["tmin_roll"] = df["TMIN"].rolling(rolling_window, center=False, min_periods=1).mean().apply(np.ceil)
    df["tmax_roll"] = df["TMAX"].rolling(rolling_window, center=False, min_periods=1).mean().apply(np.ceil)
    df["TMIN"] = df["TMIN"].fillna(df["tmin_roll"])
    df["TMAX"] = df["TMAX"].fillna(df["tmax_roll"])
    df = df.drop(columns=["tmin_roll", "tmax_roll"])

    df = df.astype({
        "PRCP": "float32",
        "SNOW": "float32",
        "SNWD": "float32",
        "PRCP_TOTAL": "float32",
        "TMIN": "float32",
        "TMAX": "float32",
    })

    df["TAVG"] = df[["TMIN", "TMAX"]].mean(axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    RAW_PATH = BASE_DIR / "data/raw/Long_Island_Weather.csv"
    CLEAN_OUT = BASE_DIR / "data/interim/Long_Island_Weather_Cleaned.csv"

    df_clean = clean_weather(RAW_PATH, CLEAN_OUT)
    print("Weather data cleaned and saved to:", CLEAN_OUT)
    print(df_clean.head())