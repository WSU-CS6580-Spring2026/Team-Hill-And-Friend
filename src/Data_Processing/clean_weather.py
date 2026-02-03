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

# %%