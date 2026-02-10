from pathlib import Path
import pandas as pd
import numpy as np

def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]

    raw_path = base_dir / "data" / "raw" / "Long_Island_Weather.csv"
    processed_dir = base_dir / "data" / "processed"
    processed_path = processed_dir / "Long_Island_Weather_Cleaned.csv"

    df = pd.read_csv(raw_path, parse_dates=["DATE"])
    print(df.head())

    df.info()

    print(df.isna().sum())

    #If values are missing for PRCP, SNOW, or SNWD assume there is none, put 0 in
    df['PRCP'] = df['PRCP'].fillna(0)
    df['SNOW'] = df['SNOW'].fillna(0)
    df['SNWD'] = df['SNWD'].fillna(0)

    #Create a new column for total precipitation, which is the sum of rain and snow (Round to 2 decimal places)
    df['PRCP_TOTAL'] = (df['PRCP'] + df['SNOW']).round(2)

    #Drop these columns, wont be used in our prediction
    df = df.drop(columns=['WESD', 'WT05', 'TOBS', 'STATION', 'NAME'])

    #Uncomment for testing averages
    # first_null_index = df['TMIN'].isnull().idxmax()
    # print(df.loc[first_null_index-3])
    # print(df.loc[first_null_index-2])
    # print(df.loc[first_null_index-1])
    # print(df.loc[first_null_index])
    # print(df.loc[first_null_index+1])
    # print(df.loc[first_null_index+2])
    # print(df.loc[first_null_index+3])

    #If a value is missing for TMAX or TMIN, average it
    df = df.sort_values("DATE")

    df["tmin_roll"] = (
        df["TMIN"]
        .rolling(window=7, center=True, min_periods=1)
        .mean().apply(np.ceil)
    )

    df["tmax_roll"] = (
        df["TMAX"]
        .rolling(window=7, center=True, min_periods=1)
        .mean().apply(np.ceil)
    )

    #Fill in missing values with the rolling average
    df["TMIN"] = df["TMIN"].fillna(df["tmin_roll"])
    df["TMAX"] = df["TMAX"].fillna(df["tmax_roll"])

    df = df.drop(columns=["tmin_roll", "tmax_roll"])
    print(df.isna().sum())

    #Uncomment for testing averages
    # print(df.loc[first_null_index-3])
    # print(df.loc[first_null_index-2])
    # print(df.loc[first_null_index-1])
    # print(df.loc[first_null_index])
    # print(df.loc[first_null_index+1])
    # print(df.loc[first_null_index+2])
    # print(df.loc[first_null_index+3])

    df.to_csv(processed_path, index=False)


if __name__ == "__main__":
    main()