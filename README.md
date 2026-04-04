

#  Weather-Based LIRR Delay Prediction

## Group: Team Hill And Friend

### Members and Roles
- **Mylee Anderson** — Project Lead / Model Helper
- **Skyler Turner** — Model Lead
- **Taylor Shipley** — Data Engineer / UI
- **Chase Powers** — Reviewer

---

## High-Level Overview

This project focuses on predicting **Long Island Rail Road (LIRR) train delays** based on weather conditions. Combining historical train delay records with Long Island weather station data to explore how weather patterns affect transit performance. The goal is to estimate how many minutes late a train is likely to be, helping commuters make better decisions about when and how to travel.


---
#  Getting Started

### Using the environment.yml
This single, OS-agnostic `environment.yml` keeps our dependencies and versions aligned so we can all work from the same setup. Create the `hillandfriend` conda environment once and reuse it for the notebooks.

### How to import
- Ensure you have miniconda or anaconda installed on your computer

- Pull the github repo

```
conda env create -f environment.yml
conda activate hillandfriend
```
This keeps us all up to date and ensures we can work in the same environment for reproducible results.

### Setting Up Pre-Commit Hooks and nbstripout
After creating the conda environment, you need to set up pre-commit hooks and nbstripout to ensure consistent code quality and clean Jupyter notebooks in version control.

#### Install Pre-Commit Hooks
Pre-commit hooks automatically run checks before each commit to catch issues early. This includes running nbstripout on Jupyter notebooks:
```
pre-commit install
```

#### Install nbstripout Git Filter
While the pre-commit hook handles nbstripout during commits, installing the git filter provides an additional safety layer that automatically strips notebook outputs during git operations:
```
nbstripout --install
```

Together, these tools help maintain code quality and make collaboration smoother by ensuring notebooks are stripped of execution outputs and unnecessary metadata before being committed to the repository.

### Note
- Review `environment.yml` if you need platform-specific packages, but the file is designed to resolve cleanly on Windows, macOS, and Linux.

### How to update
- If you `conda install` something to use, update `environment.yml`

- Run:
```
conda env export --no-builds > environment.yml
```

- Ensure the file was updated properly

---
#  Sprint 2

### Data cleaning
- We split the team into two groups. We are currently prioritizing cleaning for the LIRR dataset and the weather dataset independently within the src/data_processing folder.
- Once we have the datasets cleaned, ideally we'll be transferring both the raw and processed datasets to be online so that it will more accessible.
- We will need to rerun our EDA and potentially expand upon it for further analysis. Saving result images into a results folder.
- After we will update our data dictionary with the columns we plan to use for our machine learning models.

---
#  Sprint 3
### Model Training 

- Use the training script to split processed data, with our best model we found saved into a .json file (as it saves the best for xgboost), generate predictions, and save the trained model artifact.
- Updated data dictionary with new features used for our new models.
- Models.ipynb has our first run of models with visualizations with some base models, we also have auto_models.ipynb where we made a pipeline and have various models and the best outcomes.

## Run
```bash
python src/models/train_model.py
```

## Default input/output
- Input dataset: `data/processed/merged_lirr_weather.csv`
- Predictions CSV: `data/processed/xgb_predictions.csv`
- Metrics JSON: `data/processed/xgb_metrics.json`
- Trained model: `models/xgb_model.json`
- Plots directory: `docs/Results`

The script creates output folders automatically (including `models/` if it does not exist).

## Optional arguments
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

# Sprint 4

## Interactive Prediction App (Streamlit)

For Sprint 4, we built an interactive **Streamlit application** that allows users to predict **LIRR train delays in real time** using the trained XGBoost model from Sprint 3.

The application combines:

- Historical train delay patterns
- Station-level delay averages
- Daily weather conditions

Users can enter trip and weather information, and the application returns an estimated **delay in minutes**.

This app serves as a **local MVP demonstration** of the model and does not require cloud deployment.

---

## Running the Streamlit App

After activating the project environment, run the following command from the project root:

```bash
streamlit run src/app.py
```

Streamlit will start a local server and print a URL similar to:

http://localhost:8501

Open this link in your browser to use the application.

---

## App Inputs

The app allows users to provide the following inputs:

| Input | Description |
|-------|-------------|
| Depart Station | The LIRR station where the trip begins |
| Arrive Station | The destination LIRR station |
| Travel Date | Date of travel (must fall within 2025 due to weather data coverage) |

Values such as the weather will be already provided.

These inputs are used to generate the same feature structure that the model was trained on.

---

## App Output

After clicking Predict Delay, the application will display:

- Predicted Delay (minutes late)
- Comparison to the historical train delay average
- A status indicator describing delay severity

Possible delay statuses include:

- On Schedule
- Moderate Delay
- Significant Delay

This provides a simple way to explore how weather conditions and station combinations influence train delays.

---

## Model Integration

The application loads the trained model generated in Sprint 3:

`models/xgb_model.json`

The model is loaded when the Streamlit app starts and reused for predictions to keep the interface responsive.

### Prediction Pipeline

The application performs the following steps when generating a prediction:

1. Collect user inputs from the Streamlit interface
2. Preprocess inputs to match the training feature structure
3. Build the model feature frame
4. Run the XGBoost prediction
5. Convert the predicted value back to minutes late
6. Display the result to the user
