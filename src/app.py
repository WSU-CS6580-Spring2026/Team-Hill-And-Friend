# Streamlit MVP app for LIRR train delay prediction
# Uses pre-saved feature_dict.json for all mean encodings (no recomputation from data)
# Valid station pairs are enforced from historical data before prediction

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# Paths
DATA_ROOT    = Path(__file__).resolve().parents[1]
DATASET_PATH = DATA_ROOT / "data/processed/merged_lirr_weather.csv"
FEATURE_DICT_PATH = DATA_ROOT / "data/processed/feature_dict.json"
MODEL_PATH   = DATA_ROOT / "models/xgb_model.json"

VALID_DATE_START = date(2025, 1, 1)
VALID_DATE_END   = date(2025, 12, 31)

# Feature columns must exactly match the order used during training in data_engineering.py / train_model.py
FEATURE_COLUMNS = [
    "train_mean",
    "depart_station_mean",
    "arrive_station_mean",
    "station_pair_mean",
    "PRCP",
    "SNOW",
    "SNWD",
    "TMAX",
    "TMIN",
    "PRCP_TOTAL",
    "TAVG",
    "year",
    "month",
    "day",
    "dow",
    "sin_month", "cos_month",
    "sin_day",   "cos_day",
    "sin_dow",   "cos_dow",
    "rolling_delay3",
    "rolling_delay7",
    "total_prcp3",
]

st.set_page_config(page_title="LIRR Delay Predictor", layout="wide")

# Load feature dict (mean encodings + clip bounds)
@st.cache_resource
def _load_feature_dict() -> dict:
    with open(FEATURE_DICT_PATH) as f:
        return json.load(f)

# Load dataset (only used for weather lookup and valid pair index)
@st.cache_resource
def _load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    return df

# Load XGBRegressor (saved via model.save_model in train_model.py)
@st.cache_resource
def _load_model() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model

# Build valid station pair index from historical data
@st.cache_data
def _build_station_pair_index(_df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Returns a dict mapping each depart_station to the sorted list of
    arrive_stations it has been paired with in the training data.
    Only pairs present in feature_dict station_pair_mean are included
    (i.e. pairs that have a computed historical mean).
    """
    feature_dict = _load_feature_dict()
    known_pairs  = set(feature_dict.get("station_pair", {}).keys())

    index: dict[str, list[str]] = {}
    for pair_key in known_pairs:
        if "_" not in pair_key:
            continue
        # pair_key format: "{depart}_{arrive}" — split on first underscore only
        # to handle station names that may contain underscores
        depart, arrive = pair_key.split("_", 1)
        index.setdefault(depart, [])
        if arrive not in index[depart]:
            index[depart].append(arrive)

    for depart in index:
        index[depart] = sorted(index[depart])

    return index

# Daily weather stats for the 2025 test window
@st.cache_data
def _daily_weather(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates weather columns by DATE from the merged dataset.
    Only 2025 rows are used since that is the prediction window.
    Rolling features are recomputed on this daily series.
    """
    weather_cols = ["DATE", "PRCP", "SNOW", "SNWD", "PRCP_TOTAL", "TMIN", "TMAX", "TAVG", "minutes_late"]
    subset = (
        _df[weather_cols]
        .dropna(subset=["DATE"])
        .pipe(lambda d: d[d["DATE"].dt.year == 2025])
    )
    daily = subset.groupby("DATE").mean().sort_index()
    daily["rolling_delay3"] = daily["minutes_late"].rolling(3, min_periods=1).mean()
    daily["rolling_delay7"] = daily["minutes_late"].rolling(7, min_periods=1).mean()
    daily["total_prcp3"]    = daily["PRCP_TOTAL"].rolling(3, min_periods=1).sum()
    return daily

# Helpers
def _lookup(mapping: dict, key: str, fallback: float) -> float:
    return float(mapping.get(key, fallback))

def _safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default

def _get_weather_row(travel_dt: pd.Timestamp, daily_df: pd.DataFrame) -> pd.Series:
    if travel_dt in daily_df.index:
        return daily_df.loc[travel_dt].fillna(0)
    earlier = daily_df.index[daily_df.index <= travel_dt]
    if len(earlier):
        return daily_df.loc[earlier[-1]].fillna(0)
    return daily_df.iloc[0].fillna(0)

# Feature builder
def build_feature_frame(
    depart: str,
    arrive: str,
    travel_date: date,
    feature_dict: dict,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    travel_dt   = pd.Timestamp(travel_date)
    weather_row = _get_weather_row(travel_dt, daily_df)
    global_mean = feature_dict.get("global_mean", 0.0)

    # Mean encodings from saved dict — no recomputation
    pair_key          = f"{depart}_{arrive}"
    depart_mean       = _lookup(feature_dict["depart_station"], depart,    global_mean)
    arrive_mean       = _lookup(feature_dict["arrive_station"], arrive,    global_mean)
    pair_mean         = _lookup(feature_dict["station_pair"],   pair_key,  global_mean)
    # train_mean: no train number collected from user; use global mean as neutral fallback
    train_mean        = global_mean

    # Weather from daily stats
    prcp       = _safe_float(weather_row.get("PRCP"),       0.0)
    snow       = _safe_float(weather_row.get("SNOW"),       0.0)
    snwd       = _safe_float(weather_row.get("SNWD"),       0.0)
    prcp_total = _safe_float(weather_row.get("PRCP_TOTAL"), prcp + snow)
    tmin       = _safe_float(weather_row.get("TMIN"),       50.0)
    tmax       = _safe_float(weather_row.get("TMAX"),       55.0)
    tavg       = _safe_float(weather_row.get("TAVG"),       (tmin + tmax) / 2)

    # Time features — sin + cos to match data_engineering.py
    year  = travel_dt.year
    month = travel_dt.month
    day   = travel_dt.day
    dow   = travel_dt.weekday()

    features = {
        "train_mean":          train_mean,
        "depart_station_mean": depart_mean,
        "arrive_station_mean": arrive_mean,
        "station_pair_mean":   pair_mean,
        "PRCP":                prcp,
        "SNOW":                snow,
        "SNWD":                snwd,
        "TMAX":                tmax,
        "TMIN":                tmin,
        "PRCP_TOTAL":          prcp_total,
        "TAVG":                tavg,
        "year":                year,
        "sin_month":           math.sin(2 * math.pi * month / 12),
        "cos_month":           math.cos(2 * math.pi * month / 12),
        "sin_day":             math.sin(2 * math.pi * day   / 31),
        "cos_day":             math.cos(2 * math.pi * day   / 31),
        "sin_dow":             math.sin(2 * math.pi * dow   / 7),
        "cos_dow":             math.cos(2 * math.pi * dow   / 7),
        "rolling_delay3":      _safe_float(weather_row.get("rolling_delay3"), 0.0),
        "rolling_delay7":      _safe_float(weather_row.get("rolling_delay7"), 0.0),
        "total_prcp3":         _safe_float(weather_row.get("total_prcp3"),    0.0),
    }

    return pd.DataFrame([features], columns=FEATURE_COLUMNS)

# Prediction
def predict(feature_frame: pd.DataFrame, model: xgb.XGBRegressor) -> float:
    raw = model.predict(feature_frame[FEATURE_COLUMNS])[0]
    return max(float(np.expm1(raw)), 0.0)

# Load everything
try:
    feature_dict = _load_feature_dict()
except FileNotFoundError:
    st.error(f"Feature dict not found at {FEATURE_DICT_PATH}. Run data_engineering.py first.")
    st.stop()

try:
    df = _load_dataset()
except FileNotFoundError:
    st.error(f"Merged dataset not found at {DATASET_PATH}. Run data_engineering.py first.")
    st.stop()

model_error = None
try:
    xgb_model = _load_model()
except Exception as exc:
    xgb_model  = None
    model_error = str(exc)
    st.error(f"Unable to load model: {exc}")

pair_index  = _build_station_pair_index(df)
daily_df    = _daily_weather(df)
all_departs = sorted(pair_index.keys())

# UI
st.title("🚆 Long Island Rail Road (LIRR) Delay Predictor")
st.markdown(
    "Select a departure station, arrival station, and a travel date in 2025. "
    "Use the dropdown to select by either scrolling to or typing in station. "
    "Weather conditions for that date will be pulled from historical records."
)

col1, col2, col3 = st.columns(3)

with col1:
    default_depart_idx = all_departs.index("Amagansett") if "Amagansett" in all_departs else 0
    depart_station = st.selectbox("Departure Station", all_departs, index=default_depart_idx)

with col2:
    valid_arrivals = pair_index.get(depart_station, [])

    if not valid_arrivals:
        st.warning("No known arrival stations for this departure.")
        arrive_station = None
    else:
        arrive_station = st.selectbox("Arrival Station", valid_arrivals)

with col3:
    travel_date = st.date_input(
        "Travel Date",
        value=VALID_DATE_START,
        min_value=VALID_DATE_START,
        max_value=VALID_DATE_END,
        help="Predictions are supported for dates within 2025.",
    )

# Weather display
travel_dt   = pd.Timestamp(travel_date)
weather_row = _get_weather_row(travel_dt, daily_df)

st.subheader(f"**📅 Weather on {travel_date.strftime('%B %d, %Y')}**")
w1, w2, w3, w4, w5, w6 = st.columns(6)
w1.metric("Avg Temp (°F)",   f"{_safe_float(weather_row.get('TAVG'), 0.0):.1f}°")
w2.metric("High (°F)",       f"{_safe_float(weather_row.get('TMAX'), 0.0):.1f}°")
w3.metric("Low (°F)",        f"{_safe_float(weather_row.get('TMIN'), 0.0):.1f}°")
w4.metric("Precipitation",   f"{_safe_float(weather_row.get('PRCP'), 0.0):.2f} in")
w5.metric("Snowfall",        f"{_safe_float(weather_row.get('SNOW'), 0.0):.2f} in")
w6.metric("Snow Depth",      f"{_safe_float(weather_row.get('SNWD'), 0.0):.2f} in")

st.divider()

# Predict
if xgb_model is None:
    st.error("Model is unavailable — cannot generate predictions.")
    st.stop()

if st.button("Predict Delay", type="primary"):
    with st.spinner("Running prediction..."):
        try:
            feature_frame = build_feature_frame(
                depart=depart_station,
                arrive=arrive_station,
                travel_date=travel_date,
                feature_dict=feature_dict,
                daily_df=daily_df,
            )
            delay_minutes = predict(feature_frame, xgb_model)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    global_mean_orig = float(np.expm1(feature_dict.get("global_mean", 0.0)))
    delta = delay_minutes - global_mean_orig

    st.subheader("Prediction Result")
    r1, r2 = st.columns(2)
    with r1:
        st.metric(
            "Predicted Delay",
            f"{delay_minutes:.1f} min",
        )
    with r2:
        if delay_minutes <= 5:
            st.success("🟢 On time — minimal disruption expected.")
        elif delay_minutes <= 15:
            st.warning("🟡 Moderate delay — build in a short buffer.")
        else:
            st.error("🔴 Significant delay — consider alternatives.")
