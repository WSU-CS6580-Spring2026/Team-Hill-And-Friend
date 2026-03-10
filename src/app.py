# Streamlit MVP app for train delay prediction
# Purpose:
# - allow users to enter trip and weather inputs interactively
# - load the trained XGBoost model from models/xgb_model.json
# - return a prediction in real time
# - run locally for the presentation

# Import necessary libraries
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# Import arrive and depart station options from the dataset
DATA_ROOT = Path(__file__).resolve().parents[1]
train_data = pd.read_csv(
    DATA_ROOT / "data/processed/merged_lirr_weather.csv",
    parse_dates=["DATE"],
    low_memory=False,
)

def _unique_station_list(column: str) -> list[str]:
    subset = (
        train_data[column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    subset = subset[subset != ""]
    return sorted(subset.unique())

arrive_stations = _unique_station_list("arrive_station")
depart_stations = _unique_station_list("depart_station")

# Set up Streamlit page configuration and title
st.set_page_config(page_title="Train Delay Predictor", layout="wide")

# Define constants and paths
MODEL_PATH = DATA_ROOT / "models/xgb_model.json"
VALID_DATE_START = date(2025, 1, 1)
VALID_DATE_END = date(2025, 12, 31)


@st.cache_resource
def _load_xgb_model(path: str) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


model_load_error = None
try:
    xgb_model = _load_xgb_model(str(MODEL_PATH))
except (FileNotFoundError, xgb.core.XGBoostError) as exc:
    xgb_model = None
    model_load_error = str(exc)
    st.error(f"Unable to load the trained XGBoost model: {exc}")
except Exception as exc:
    xgb_model = None
    model_load_error = str(exc)
    st.error(f"Unexpected error loading the model: {exc}")

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
    "cos_month",
    "cos_day",
    "cos_dow",
    "rolling_delay3",
    "rolling_delay7",
    "total_prcp3",
]


def _compute_mean_map(df: pd.DataFrame, column: str) -> dict:
    series = df[column]
    valid = series.notna()
    if not valid.any():
        return {}
    keys = series.loc[valid].astype(str).str.strip()
    target = df.loc[valid, "minutes_late"]
    return target.groupby(keys, observed=True).mean().to_dict()


def _compute_station_pair_map(df: pd.DataFrame) -> dict:
    depart = df["depart_station"]
    arrive = df["arrive_station"]
    valid = depart.notna() & arrive.notna()
    if not valid.any():
        return {}
    depart_keys = depart.loc[valid].astype(str).str.strip()
    arrive_keys = arrive.loc[valid].astype(str).str.strip()
    pair_keys = depart_keys + "_" + arrive_keys
    target = df.loc[valid, "minutes_late"]
    return target.groupby(pair_keys, observed=True).mean().to_dict()


@st.cache_data
def _load_station_mean_mappings() -> dict:
    subset = train_data[["train", "depart_station", "arrive_station", "minutes_late"]].dropna(subset=["minutes_late"])
    mapping = {"global_mean": float(subset["minutes_late"].mean())}
    mapping["train_mean"] = _compute_mean_map(subset, "train")
    mapping["depart_station_mean"] = _compute_mean_map(subset, "depart_station")
    mapping["arrive_station_mean"] = _compute_mean_map(subset, "arrive_station")
    mapping["station_pair_mean"] = _compute_station_pair_map(subset)
    return mapping


@st.cache_data
def _daily_weather_stats() -> pd.DataFrame:
    columns = ["DATE", "minutes_late", "PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "PRCP_TOTAL", "TAVG"]
    subset = train_data[columns].dropna(subset=["DATE"])
    daily = subset.groupby("DATE", observed=True).mean().sort_index()
    daily["rolling_delay3"] = daily["minutes_late"].rolling(3, min_periods=1).mean()
    daily["rolling_delay7"] = daily["minutes_late"].rolling(7, min_periods=1).mean()
    daily["total_prcp3"] = daily["PRCP_TOTAL"].rolling(3, min_periods=1).sum()
    return daily


def _select_daily_row(travel_date: datetime, daily_df: pd.DataFrame) -> pd.Series:
    if travel_date in daily_df.index:
        return daily_df.loc[travel_date].fillna(0)
    earlier = daily_df.index[daily_df.index <= travel_date]
    if len(earlier):
        return daily_df.loc[earlier[-1]].fillna(0)
    return daily_df.iloc[0].fillna(0)


def _normalize_station_key(value: str) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _lookup_mean(mapping: dict, key: str, fallback: float) -> float:
    if not key:
        return fallback
    return float(mapping.get(key, fallback))


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if isinstance(number, float) and math.isnan(number):
        return default
    return number


def preprocess_inputs(arrive_station: str, depart_station: str, travel_date: date, rain: bool, snow: bool, temperature: float) -> pd.DataFrame:
    cleaned_date = pd.to_datetime(travel_date)
    cleaned_arrive = arrive_station.strip() if arrive_station else ""
    cleaned_depart = depart_station.strip() if depart_station else ""
    temperature_value = float(temperature) if temperature is not None else None
    input_data = pd.DataFrame({
        "arrive_station": [cleaned_arrive],
        "depart_station": [cleaned_depart],
        "travel_date": [cleaned_date],
        "rain": [bool(rain)],
        "snow": [bool(snow)],
        "temperature": [temperature_value],
    })
    return input_data


def build_feature_frame(input_data: pd.DataFrame) -> pd.DataFrame:
    if input_data.empty:
        raise ValueError("Input data is required to build feature frame.")
    row = input_data.iloc[0]
    station_means = _load_station_mean_mappings()
    daily_stats = _daily_weather_stats()
    travel_date = pd.to_datetime(row["travel_date"])
    weather_row = _select_daily_row(travel_date, daily_stats)

    depart_key = _normalize_station_key(row["depart_station"])
    arrive_key = _normalize_station_key(row["arrive_station"])
    station_pair_key = f"{depart_key}_{arrive_key}"

    baseline = station_means["global_mean"]
    depart_mean = _lookup_mean(station_means["depart_station_mean"], depart_key, baseline)
    arrive_mean = _lookup_mean(station_means["arrive_station_mean"], arrive_key, baseline)
    pair_mean = _lookup_mean(station_means["station_pair_mean"], station_pair_key, baseline)
    train_mean = baseline

    prcp = _safe_float(weather_row.get("PRCP"), default=0.0)
    if row["rain"]:
        prcp = max(prcp, 0.05)
    snow_val = _safe_float(weather_row.get("SNOW"), default=0.0)
    if row["snow"]:
        snow_val = max(snow_val, 0.05)
    snwd = max(_safe_float(weather_row.get("SNWD"), default=0.0), 0.0)

    temperature_value = row["temperature"]
    if temperature_value is None or pd.isna(temperature_value):
        temperature_value = _safe_float(weather_row.get("TAVG"), default=50.0)
    else:
        temperature_value = float(temperature_value)

    tavg = temperature_value
    tmax = tavg + 3.0
    tmin = tavg - 3.0
    prcp_total = _safe_float(weather_row.get("PRCP_TOTAL"), default=prcp + snow_val)

    year = travel_date.year
    month = travel_date.month
    day = travel_date.day
    dow = travel_date.weekday()

    cos_month = math.cos(2 * math.pi * month / 12)
    cos_day = math.cos(2 * math.pi * day / 31)
    cos_dow = math.cos(2 * math.pi * dow / 7)

    feature_values = {
        "train_mean": train_mean,
        "depart_station_mean": depart_mean,
        "arrive_station_mean": arrive_mean,
        "station_pair_mean": pair_mean,
        "PRCP": prcp,
        "SNOW": snow_val,
        "SNWD": snwd,
        "TMAX": tmax,
        "TMIN": tmin,
        "PRCP_TOTAL": prcp_total,
        "TAVG": tavg,
        "year": year,
        "month": month,
        "day": day,
        "dow": dow,
        "cos_month": cos_month,
        "cos_day": cos_day,
        "cos_dow": cos_dow,
        "rolling_delay3": _safe_float(weather_row.get("rolling_delay3"), default=0.0),
        "rolling_delay7": _safe_float(weather_row.get("rolling_delay7"), default=0.0),
        "total_prcp3": _safe_float(weather_row.get("total_prcp3"), default=0.0),
    }

    return pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)


def predict(feature_frame: pd.DataFrame, booster: xgb.Booster) -> float:
    if booster is None:
        raise ValueError("XGBoost model is not loaded.")
    ordered_features = feature_frame[FEATURE_COLUMNS]
    dmatrix = xgb.DMatrix(ordered_features)
    raw_pred = booster.predict(dmatrix)
    if raw_pred.size == 0:
        return float("nan")
    estimate = float(raw_pred[0])
    delay = np.expm1(estimate)
    return max(delay, 0.0)


def validate_date(selected_date: date) -> bool:
    """
    Validates that the selected date is within the allowed range of 2025.

    Parameters
    ----------
    selected_date : date
        The date selected by the user

    Returns
    -------
    bool
        True if the date is valid, False otherwise
    """
    if selected_date < VALID_DATE_START or selected_date > VALID_DATE_END:
        st.error(f"Please select a date between {VALID_DATE_START} and {VALID_DATE_END}.")
        return False
    return True


@st.cache_data
def _station_category_index() -> dict[str, set[str]]:
    station_means = _load_station_mean_mappings()
    return {
        "arrive": set(station_means.get("arrive_station_mean", {}).keys()),
        "depart": set(station_means.get("depart_station_mean", {}).keys()),
        "pair": set(station_means.get("station_pair_mean", {}).keys()),
    }


def _gather_prediction_validation(
    arrive_station: str,
    depart_station: str,
    date_valid: bool,
    station_index: dict[str, set[str]],
    model: xgb.Booster | None,
    model_error: str | None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not date_valid:
        errors.append(
            f"Travel date must fall between {VALID_DATE_START} and {VALID_DATE_END}."
        )

    if not arrive_station:
        errors.append("Please choose a destination station from the dropdown.")
    elif arrive_station not in arrive_stations:
        errors.append(
            "Selected arrival station is not recognized; pick one of the provided options."
        )
    elif arrive_station not in station_index["arrive"]:
        warnings.append(
            "Arrival station is missing from historical training categories; default station averages will be used."
        )

    if not depart_station:
        errors.append("Please choose a departure station from the dropdown.")
    elif depart_station not in depart_stations:
        errors.append(
            "Selected departure station is not recognized; pick one of the provided options."
        )
    elif depart_station not in station_index["depart"]:
        warnings.append(
            "Departure station is missing from historical training categories; default station averages will be used."
        )

    if arrive_station and depart_station and arrive_station == depart_station:
        errors.append("Departure and arrival stations must be different.")

    if arrive_station and depart_station:
        pair_key = f"{depart_station}_{arrive_station}"
        if pair_key not in station_index["pair"]:
            warnings.append(
                "Historical data for that station pair is unavailable; pair-based means revert to the network average."
            )

    if model is None:
        error_text = "Trained model is unavailable, so predictions cannot be generated."
        if model_error:
            error_text += f" ({model_error})"
        errors.append(error_text)

    return errors, warnings

st.title("Train Delay Predictor")
st.markdown(
    "Train Delay Predictor combines your trip details with current weather conditions to estimate potential delays. "
    "Provide the stations, travel date, and rain/snow/temperature inputs below to see a real-time prediction."
)
st.info(
    "The underlying weather data only covers 2025, so please choose a travel date between January 1, 2025 and December 31, 2025."
)

arrive_station = st.selectbox("Arrive Station", arrive_stations)
arrive_station = _normalize_station_key(arrive_station)

default_depart_station = "Amityville"
depart_default_index = depart_stations.index(default_depart_station) if default_depart_station in depart_stations else 0
depart_station = st.selectbox(
    "Depart Station",
    depart_stations,
    index=depart_default_index,
)
depart_station = _normalize_station_key(depart_station)

travel_date = st.date_input(
    "Travel Date",
    value=VALID_DATE_START,
    min_value=VALID_DATE_START,
    max_value=VALID_DATE_END,
    help="Weather data only exists between January 1, 2025 and December 31, 2025.",
)
st.caption(
    "The weather observations powering this model span exactly the 2025 calendar year, "
    "so you must select a travel date within that range."
)
date_is_valid = validate_date(travel_date)
station_index = _station_category_index()
validation_errors, validation_warnings = _gather_prediction_validation(
    arrive_station,
    depart_station,
    date_is_valid,
    station_index,
    xgb_model,
    model_load_error,
)

for warning in validation_warnings:
    st.warning(warning)
for error in validation_errors:
    st.error(error)

# TODO: rain input
# - use a dropdown, radio button, or checkbox for rain
# - decide whether model expects binary values like 0/1 or labels like Yes/No
# - transform to the format used during training

# TODO: snow input
# - use a dropdown, radio button, or checkbox for snow
# - decide whether model expects binary values like 0/1 or labels like Yes/No
# - transform to the format used during training

# TODO: temperature input
# - use a numeric input or slider for temperature
# - choose units consistent with the training data
# - validate range if needed

# TODO: collect all user inputs
# - gather arrive station, depart station, date, rain, snow, and temperature
# - assemble values into a dictionary or DataFrame
# - ensure feature names, order, and types match the trained model input schema

# TODO: preprocess date and categorical inputs
# - encode arrive station and depart station as needed
# - convert rain and snow inputs to model-ready values
# - extract any needed date features such as month, day, or weekday
# - ensure preprocessing matches the training pipeline exactly

predict_disabled = bool(validation_errors)

if st.button("Predict Delay", disabled=predict_disabled):
    if predict_disabled:
        st.warning("Fix the highlighted issues before running a prediction.")
        st.stop()
    with st.spinner("Generating delay prediction..."):
        try:
            input_frame = preprocess_inputs(
                arrive_station,
                depart_station,
                travel_date,
                rain=False,
                snow=False,
                temperature=None,
            )
            feature_frame = build_feature_frame(input_frame)
            delay_estimate = predict(feature_frame, xgb_model)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            delay_minutes = max(delay_estimate, 0.0)
            baseline = float(feature_frame["train_mean"].iloc[0])
            delta = delay_minutes - baseline
            status_reason = (
                "On schedule: Expect minimal disruption."
                if delay_minutes <= 5
                else "Moderate delay likely; add a short buffer."
                if delay_minutes <= 15
                else "Significant delay predicted; consider alternatives."
            )
            with st.container():
                st.subheader("Prediction result")
                st.metric(
                    "Predicted delay",
                    f"{delay_minutes:.1f} minutes",
                    delta=f"{delta:+.1f} min vs train mean",
                )
                st.caption(
                    "Derived from station-pair history and 2025 weather trends; "
                    "value represents expected arrival minutes late (no probability output)."
                )
                if delay_minutes <= 5:
                    st.success(status_reason)
                elif delay_minutes <= 15:
                    st.warning(status_reason)
                else:
                    st.error(status_reason)