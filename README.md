# Singapore GP Predictor

This project predicts qualifying times and race positions for F1 Grand Prix races using historical F1 data and machine learning. Currently is dedicated to Singapore GP only.

## Features
1. Fetches and caches F1 session data (2022–2024) using [FastF1](https://theoehrly.github.io/Fast-F1/).  
2. Predicts qualifying times with XGBoost based on driver performance.  
3. Predicts final race standings using qualifying predictions, past performance, and constructor info.  
4. Provides Mean Absolute Error metrics for model evaluation.  

## Installation
```bash
pip install -r requirements.txt
```
## Run Program
```
python singapore_gp.py
