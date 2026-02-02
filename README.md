# Singapore GP Predictor

Predicts qualifying lap times and race finishing positions for the Singapore Grand Prix using historical F1 data (FastF1) and machine learning. Train years: 2022–2024; test year is configurable (e.g. 2025).

## Features

- **Data:** Fetches Singapore GP qualifying and race data from [FastF1](https://theoehrly.github.io/Fast-F1/) (lap times, positions, drivers, teams, grid) and caches it locally so you don’t re-download every run.
- **Qualifying:** Predicts each driver’s quali lap time (seconds) from driver, team, year, and their past Singapore quali times; uses XGBoost and compares it to Random Forest and Ridge.
- **Race:** Predicts finishing position from driver, team, grid, predicted quali time, and past Singapore results; same three models, with MAE reported in positions (e.g. 3.0 = about 3 places off on average).
- **Output:** Prints MAE for all models (quali in seconds, race in positions) and a table of predicted vs actual for the test year.

## Installation

Install the packages listed in `requirements.txt` (so your environment matches the project):

```bash
pip install -r requirements.txt
```

## Run

```bash
python singapore_gp.py
```
