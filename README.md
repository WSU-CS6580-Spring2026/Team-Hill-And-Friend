

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

### Note
- Review `environment.yml` if you need platform-specific packages, but the file is designed to resolve cleanly on Windows, macOS, and Linux.

### How to update
- If you `conda install` something to use, update `environment.yml`

- Run:
```
conda env export --no-builds > environment.yml
```

- Ensure the file was updated properly
