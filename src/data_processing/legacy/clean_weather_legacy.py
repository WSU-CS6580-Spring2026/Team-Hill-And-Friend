from pathlib import Path
import pandas as pd
import numpy as np

def clean_weather_legacy(
    input_path: Path,
    output_path: Path,
    rolling_window: int = 7
) -> pd.DataFrame:
    """
    Cleans Long Island weather data.

    Parameters
    ----------
    input_path : Path
        Path to raw weather CSV
    output_path : Path
        Path to save cleaned weather CSV
    rolling_window : int
        Window size for rolling average for missing TMIN/TMAX values

    Returns
    -------
    pd.DataFrame
        Cleaned weather DataFrame
    """
    # Load CSV
    df = pd.read_csv(input_path, parse_dates=["DATE"])

    # Convert DATE to normalized datetime64
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.normalize()

    # Fill missing precipitation/snow with 0
    df[['PRCP', 'SNOW', 'SNWD']] = df[['PRCP', 'SNOW', 'SNWD']].fillna(0)

    # Total precipitation
    df["PRCP_TOTAL"] = (df["PRCP"] + df["SNOW"]).round(2)

    # Drop unused columns if they exist
    drop_cols = ["WESD", "WT05", "TOBS", "STATION", "NAME"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Sort by date
    df = df.sort_values("DATE")

    # Rolling averages for missing TMIN/TMAX
    df["tmin_roll"] = df["TMIN"].rolling(window=rolling_window, center=True, min_periods=1).mean().apply(np.ceil)
    df["tmax_roll"] = df["TMAX"].rolling(window=rolling_window, center=True, min_periods=1).mean().apply(np.ceil)

    df["TMIN"] = df["TMIN"].fillna(df["tmin_roll"])
    df["TMAX"] = df["TMAX"].fillna(df["tmax_roll"])

    # Drop temporary rolling columns
    df = df.drop(columns=["tmin_roll", "tmax_roll"])

    # Derived feature TAVG to get the average between TMIN and TMAX
    df['TAVG'] = df[['TMIN','TMAX']].mean(axis=1)
    df.head()

    # Enforce dtypes
    df = df.astype({
        "PRCP": "float32",
        "SNOW": "float32",
        "SNWD": "float32",
        "PRCP_TOTAL": "float32",
        "TMIN": "float32",
        "TMAX": "float32",
        "TAVG": "float32"
    })

    # Save cleaned CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[3]

    raw_path = BASE_DIR / "data/raw/Long_Island_Weather.csv"
    interim_path = BASE_DIR / "data/interim/Long_Island_Weather_Cleaned_legacy.csv"

    df_clean = clean_weather_legacy(raw_path, interim_path)

    print("Weather data cleaned and saved to:", interim_path)
    print("Rows:", len(df_clean))
    print("Missing values after cleaning:\n", df_clean.isna().sum())
    print(df_clean.head())
