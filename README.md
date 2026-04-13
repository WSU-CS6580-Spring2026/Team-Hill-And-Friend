# Weather-Based LIRR Delay Prediction

## Group: Team Hill And Friend

### Members and Roles
- **Mylee Anderson** - Project Lead / Model Helper
- **Skyler Turner** - Model Lead
- **Taylor Shipley** - Data Engineer / UI
- **Chase Powers** - Reviewer

---

## Project Overview

This project predicts **Long Island Rail Road (LIRR) train delays** using historical train delay data and weather data. The goal is to estimate how many minutes late a train is likely to be so commuters can make better decisions about when and how to travel.

The project includes:
- data cleaning and preprocessing
- a full pipeline for preparing modeling data
- model training using XGBoost
- a local Streamlit app for interactive delay prediction

---

## Repository Setup

You can set up the project with either **Conda** or **Python venv + pip**.

### Option 1: Conda

Make sure you have **Miniconda** or **Anaconda** installed.

Clone the repository and create the environment:

```bash
git clone https://github.com/WSU-CS6580-Spring2026/Team-Hill-And-Friend.git
cd Team-Hill-And-Friend
conda env create -f environment.yml
conda activate hillandfriend
```

### Option 2: venv + pip

Make sure you have a supported version of Python installed.

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/WSU-CS6580-Spring2026/Team-Hill-And-Friend.git
cd Team-Hill-And-Friend
python -m venv venv
```


Activate the virtual environment.

On macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Then install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Project Workflow

Run the project in this order:

1. Set up the environment
2. Run the full pipeline
3. Launch the Streamlit app

---

## Run the Full Pipeline First

Before starting the Streamlit app, run the full pipeline from the project root:

```bash
python src/pipeline/03_full_pipeline.py
```

This step prepares the data and generates the files needed by the model and app.

---

## Run the Streamlit App

After the environment is activated and the full pipeline has been run, start the app from the project root:

```bash
streamlit run src/app.py
```

Streamlit will start a local server and print a URL similar to:

```text
http://localhost:8501
```

Open that link in your browser to use the application.

---

## Streamlit App Overview

The Streamlit app is a local MVP that allows users to predict **LIRR train delays in real time** using the trained XGBoost model.

The app combines:
- historical train delay patterns
- station-level delay averages
- daily weather conditions

Users can enter trip information and receive an estimated **delay in minutes**.

---

## App Inputs

The app allows users to provide the following inputs:

| Input | Description |
|-------|-------------|
| Depart Station | The LIRR station where the trip begins |
| Arrive Station | The destination LIRR station |
| Travel Date | Date of travel, currently limited to 2025 based on available weather coverage |

Weather-related values are automatically provided by the application.

These inputs are used to generate the same feature structure used during training.

---

## App Output

After clicking **Predict Delay**, the app displays:

- predicted delay in minutes
- comparison to the historical train delay average
- a status indicator describing delay severity

Possible delay statuses include:
- On Schedule
- Moderate Delay
- Significant Delay

---

## Model Training
The trained model used by the Streamlit app is generated when running the full pipeline.

For development or debugging only, the training script can also be run separately:

```bash
python src/models/train_model.py
```

### Default input/output

- Input dataset: `data/processed/merged_lirr_weather.csv`
- Predictions CSV: `data/processed/xgb_predictions.csv`
- Metrics JSON: `data/processed/xgb_metrics.json`
- Trained model: `models/xgb_model.json`
- Plots directory: `docs/Results`

The script creates output folders automatically, including `models/` if it does not already exist.

### Optional arguments

```bash
python src/models/train_model.py \
  --merged-data data/processed/merged_lirr_weather.csv \
  --target minutes_late \
  --test-year 2025 \
  --random-state 42 \
  --predictions-out data/processed/xgb_predictions.csv \
  --metrics-out data/processed/xgb_metrics.json \
  --model-out models/xgb_model.json \
  --plots-out docs/Results \
  --top-n-features 20
```

---

## Model Integration

The Streamlit application loads the trained model from:

```text
models/xgb_model.json
```

The model is loaded when the app starts and then reused for predictions to keep the interface responsive.

### Prediction Pipeline

When generating a prediction, the application:

1. collects user inputs from the Streamlit interface
2. preprocesses inputs to match the training feature structure
3. builds the model feature frame
4. runs the XGBoost prediction
5. converts the predicted value back to minutes late
6. displays the result to the user

---

## Reproducibility Notes

To run this project successfully, use the setup instructions above and execute the pipeline before launching the Streamlit app.

If you install new packages with Conda, update the environment file:

```bash
conda env export --no-builds > environment.yml
```

If you install new packages with pip, update `requirements.txt` as needed.

---

## Development Notes

### Sprint 2: Data Cleaning
- The team split work between LIRR data cleaning and weather data cleaning.
- Cleaning work was focused in the `src/data_processing` folder.
- Processed datasets support later EDA, feature engineering, and modeling work.

### Sprint 3: Model Training
- The team trained and evaluated several models.
- The best model artifact was saved as an XGBoost `.json` file.
- Additional exploration and visualizations were completed in notebooks such as:
  - `Models.ipynb`
  - `auto_models.ipynb`

### Sprint 4: Interactive App
- The team built a Streamlit app as a local interactive demonstration.
- The app uses the trained model to make delay predictions based on route and weather context.