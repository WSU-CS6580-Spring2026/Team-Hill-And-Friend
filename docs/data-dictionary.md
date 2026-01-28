
# Data Dictionary  
**Long Island Rail Road Delays & Weather (GHCN-Daily)**

---

#  Dataset 1: TA LIRR Delays — Beginning 2010

**Source:** Metropolitan Transportation Authority (NY Open Data)  
**Coverage:** New York City & Long Island  
**Time Period:** Beginning 2010 – Present  
**Update Frequency:** Monthly (≈ 1 week processing lag)  
**Granularity:** Date, train, branch, time period, status, delay category  
**Rows:** ~254,000  
**Columns:** 11  

**Description:**  
This dataset lists all LIRR trains that are delayed, including whether a train was cancelled, partially cancelled, or late, the rounded delay minutes, the delay cause category, train number, and the branch the train operates on.

---

## Table: `lirr_delays`

| Column Name | API Field Name | Data Type | Description |
|------------|----------------|-----------|-------------|
| Service Date | service_date | Floating Timestamp | The day on which the delay occurred. |
| Train | train | Text | The number of the train. |
| Branch | branch | Text | The branch the train operates on. |
| Depart Station | depart_station | Text | The station the train was scheduled to depart from. |
| Depart Time | depart_time | Floating Timestamp | The scheduled departure datetime. |
| Arrive Station | arrive_station | Text | The station the train was scheduled to arrive at. |
| Arrive Time | arrive_time | Floating Timestamp | The scheduled arrival datetime. |
| Period | period | Text | Indicates whether the train is Peak or Off-Peak. |
| Status | status | Text | Train status (Late, Cancelled, Partially Cancelled). |
| Minutes Late | minutes_late | Number | Blank if < 6 minutes late; populated if ≥ 6 minutes late. |
| Delay Category | delay_category | Text | Category describing the cause of the delay. |

---

## LIRR Notes
- Minutes Late only appears when delay ≥ 6 minutes.  
- Cancelled / Partial Cancel trains may not have meaningful delay minutes.  
- Dataset is published on a one-week lag.  

---

#  Dataset 2: GHCN-Daily Weather Data

**Source:** NOAA / NCEI — Global Historical Climatology Network (Daily)  
**Coverage:** Global land stations (100,000+)  
**Update Frequency:** Daily  
**Granularity:** One row per station per day  

**Description:**  
GHCN-Daily is the world’s largest archive of daily weather observations, containing temperature, precipitation, snowfall, snow depth, wind, evaporation, cloudiness, soil temperature, sunshine, and weather event indicators.

---

##  Core Record Structure

| Column | Data Type | Description |
|-------|------------|-------------|
| station | Text | NOAA station identification code. |
| station_name* | Text | Station name (optional output field). |
| latitude* | Number | Decimal degrees latitude. |
| longitude* | Number | Decimal degrees longitude. |
| elevation* | Number | Elevation above mean sea level (meters). |
| date | Date | Observation date (YYYYMMDD or ISO). |

---

##  Core Climate Variables

| Field | Description |
|-------|-------------|
| PRCP | Total daily precipitation (rain + melted snow). |
| SNOW | Daily snowfall. |
| SNWD | Snow depth. |
| TMAX | Daily maximum temperature. |
| TMIN | Daily minimum temperature. |

---

##  Common Additional Variables

| Field | Description |
|-------|-------------|
| AWND | Average daily wind speed. |
| TOBS | Temperature at time of observation. |
| WSFG | Peak wind gust speed. |
| WDFG | Direction of peak wind gust. |
| WESD | Water equivalent of snow on ground. |
| PSUN | Percent of possible sunshine. |
| TSUN | Daily sunshine total (minutes). |

---

##  Rain & Event Indicators

| Field | Description |
|-------|-------------|
| PRCP | Primary precipitation (rain) field. |
| WT16 | Rain observed. |
| WT17 | Freezing rain observed. |
| WT03 | Thunder observed. |

---

##  Flags & Attributes

Each element may have companion columns:
- Measurement Flag  
- Quality Flag  
- Source Flag  
- Observation Time  

---

## Integration Notes

- Join key: `lirr_delays.service_date` ↔ `ghcn_daily.date`  
- Most useful weather fields: PRCP, TMAX, TMIN, SNOW, SNWD, AWND, WT16, WT17  
- Common derived fields: rain_flag, heavy_rain_flag, freezing_rain_flag  

