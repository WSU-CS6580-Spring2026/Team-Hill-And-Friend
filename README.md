

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
This is how we will stay on the same page with dependencies and library versions.
There is an environment.yml file in our repo now that we can all download and update as we work on this project.
You can also setup your conda environment (hillandfriend) for the jupyter notebooks.
On macOS/Linux, use `environment.macos-linux.yml` (Windows-only and CUDA packages are removed).

### How to import
- Ensure you have miniconda or anaconda installed on your computer

- Pull the github repo

- Run (Windows):
```
conda env create -f environment.yml
conda activate hillandfriend
```

- Run (macOS/Linux):
```
conda env create -f environment.macos-linux.yml
conda activate hillandfriend
```
This will keep us all up to date and ensure we can work in the same environments for reproducible results

### Note
- The Windows file includes Windows-only and CUDA/NVIDIA dependencies; macOS/Linux users should use the macOS/Linux env file to avoid solve errors.

### How to update
- If you `conda install` something to use, be sure to update the environment.yml file

- Run:
```
conda env export --no-builds > environment.yml
```

- Ensure file was updated properly
