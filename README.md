

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

- Use the training script to split processed data, train a Linear Regression model, generate predictions, and save the trained model artifact.
- Updated data dictionary with new features used for regression models.
- Models.ipynb has our first run of models with visualizations, we also have auto_models.ipynb where we made a pipeline and hvae various models and the best outcomes.

## Run
```bash
python src/train_model.py
```

## Default input/output
- Input dataset: `data/processed/merged_lirr_weather.csv`
- Predictions CSV: `data/processed/linear_regression_predictions.csv`
- Trained model: `models/linear_regression_pipeline.joblib`

The script creates output folders automatically (including `models/` if it does not exist).

## Optional arguments
```bash
python src/train_model.py \
  --input data/processed/merged_lirr_weather.csv \
  --target minutes_late \
  --test-size 0.2 \
  --random-state 42 \
  --predictions-out data/processed/linear_regression_predictions.csv \
  --model-out models/linear_regression_pipeline.joblib
```
