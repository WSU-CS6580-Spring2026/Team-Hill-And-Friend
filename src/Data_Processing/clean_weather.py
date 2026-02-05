#In[1]
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
#D:\mypie\Documents\CS6580\Team-Hill-And-Friend\data\raw\Long_Island_Weather.csv
#D:\mypie\Documents\CS6580\Team-Hill-And-Friend\src\Data_Processing\clean_weather.py
df = (
    pd.read_csv(
        '../../data/raw/Long_Island_Weather.csv',
        parse_dates=['DATE'],
    )
    .rename(columns=str.strip)
    .sort_values('DATE')
)

df.info()
#In[2]
df.isna().sum()

#In[3]
df['PRCP'] = df['PRCP'].fillna(0)
df['SNOW'] = df['SNOW'].fillna(0)
df['SNWD'] = df['SNWD'].fillna(0)
df['WESD'] = df['WESD'].fillna(0)
df['WT05'] = df['WT05'].fillna(0)


df.isna().sum()
#In[4]
df = df.sort_values("DATE")

df["tmin_roll"] = (
    df["TMIN"]
    .rolling(window=7, center=True, min_periods=1)
    .mean()
)

df["tmax_roll"] = (
    df["TMAX"]
    .rolling(window=7, center=True, min_periods=1)
    .mean()
)


df["TMIN"] = df["TMIN"].fillna(df["tmin_roll"])
df["TMAX"] = df["TMAX"].fillna(df["tmax_roll"])

df = df.drop(columns=["tmin_roll", "tmax_roll"])
df.isna().sum()
#In[5]
df.to_csv('../../data/processed/Long_Island_Weather_Cleaned.csv', index=False)

# %%