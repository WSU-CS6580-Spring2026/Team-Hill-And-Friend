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
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    df[["prcp", "snow", "snwd"]] = df[["prcp", "snow", "snwd"]].fillna(0)
    df["prcp_total"] = (df["prcp"] + df["snow"]).round(2)

    drop_cols = ["wesd", "wt05", "tobs", "station", "name"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.sort_values("date").reset_index(drop=True)

    # center=False prevents future data leaking into imputed values
    df["tmin_roll"] = np.ceil(df["tmin"].rolling(rolling_window, center=False, min_periods=1).mean())
    df["tmax_roll"] = np.ceil(df["tmax"].rolling(rolling_window, center=False, min_periods=1).mean())
    df["tmin"] = df["tmin"].fillna(df["tmin_roll"])
    df["tmax"] = df["tmax"].fillna(df["tmax_roll"])
    df = df.drop(columns=["tmin_roll", "tmax_roll"])

    df = df.astype({
        "prcp": "float32",
        "snow": "float32",
        "snwd": "float32",
        "prcp_total": "float32",
        "tmin": "float32",
        "tmax": "float32",
    })

    df["tavg"] = df[["tmin", "tmax"]].mean(axis=1)

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