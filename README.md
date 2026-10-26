# Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning

## Overview
This project predicts whether a flight will arrive substantially later than scheduled using historical flight data, airline information, routes, and schedule-based features. It compares traditional machine learning models with a deep learning approach.

## Objectives
- Analyze flight delays and identify temporal, airline, and route patterns.
- Perform exploratory data analysis (EDA) and feature engineering.
- Train a Logistic Regression baseline and an XGBoost classifier.
- Evaluate models using PR AUC, ROC AUC, recall at a selected precision, and calibration.
- Explore learned airline and airport embeddings using a compact neural network.
- Investigate offline flight rescheduling recommendations under explicit assumptions.

## Dataset
The starter dataset contains US flight records from 2013, including scheduled and actual departure and arrival times, delays, airline codes, origin and destination airports, flight duration, and distance.

The dataset will be stored locally in `data/raw/`.

## Methodology
1. Data loading and quality assessment
2. Exploratory data analysis
3. Feature engineering and leakage prevention
4. Chronological training, validation, and test splits
5. Logistic Regression baseline
6. XGBoost classification
7. Model evaluation and probability calibration
8. Deep learning extension and offline rescheduling analysis

## Data Leakage Prevention
The prediction timestamp will be defined before feature engineering. Actual departure and arrival information and other features unavailable at that timestamp will not be used as prediction inputs. Historical aggregate features will use past information only.

## Repository Structure
- `data/` — raw and processed datasets
- `notebooks/` — exploration and experiments
- `src/` — reusable Python modules
- `models/` — model artifacts and documentation
- `reports/` — evaluation results and figures
- `docs/` — literature review and project report

## Technology Stack
Python, pandas, NumPy, scikit-learn, XGBoost, PyTorch, Matplotlib, Seaborn, and Jupyter Notebook.

## Current Status
Project structure and initial dependencies configured. Dataset preparation, EDA, feature engineering, and model training are planned next.

## Disclaimer
Predictions and rescheduling recommendations are for academic experimentation only and are not intended for real-world airline operational decisions.
