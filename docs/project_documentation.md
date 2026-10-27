# Flight Delay Prediction: Project Documentation

## 1. Problem Statement and Motivation

Flight delays cause major disruptions across the aviation network. They lead to passenger frustration, missed connecting flights, crew re-scheduling issues, and substantial operational costs for airlines and airport operators. 

Early and accurate prediction of flight delays before departure is valuable because it enables:
- **Airlines**: to optimize ground handling, turnaround schedules, gate assignments, and aircraft routing.
- **Airport Managers**: to foresee congestion peaks and allocate runway/gate resources proactively.
- **Passengers**: to plan travel and adjust connections ahead of time.

This project formulates delay prediction as a **binary classification problem** based on flight schedule details, seasonal trends, airline carriers, and airport routes. The goal is to compare an interpretable linear baseline (**Logistic Regression**) against a non-linear gradient-boosted decision tree model (**XGBoost**) under a realistic, chronological evaluation protocol.

---

## 2. Dataset Overview

### 2.1 Source and Size
- **Source**: US Bureau of Transportation Statistics 2013 New York City flight records (the classic `nycflights13` dataset), encompassing flights departing from the three primary New York metropolitan airports:
  - **EWR**: Newark Liberty International Airport
  - **JFK**: John F. Kennedy International Airport
  - **LGA**: LaGuardia Airport
- **Raw Dataset Size**: 336,776 rows and 21 columns.
- **Data Cleaning**: 9,430 flights without recorded arrival delays (`arr_delay` is null / canceled flights) were removed.
- **Labeled Dataset Size**: 327,346 flights.

### 2.2 Target Variable
- **Target**: `delayed` (binary classification).
- **Definition**: A flight is defined as **substantially delayed** (`delayed = 1`) if its arrival delay is 15 minutes or greater (`arr_delay >= 15`), following standard Federal Aviation Administration (FAA) criteria. Otherwise, it is labeled on-time (`delayed = 0`).
- **Target Distribution**:
  - `delayed = 0` (On-time / minor delay): 247,246 flights (75.53%)
  - `delayed = 1` (Delayed $\ge$ 15 min): 80,100 flights (24.47%)
  - Overall positive class rate: **24.47%** (demonstrating moderate class imbalance).

### 2.3 Feature Set (16 Input Features)
To prevent **target leakage**, only information strictly known *before* flight departure is included as predictive features. All post-departure and outcome variables (`dep_time`, `dep_delay`, `arr_time`, `arr_delay`, `air_time`) are explicitly forbidden from the feature set.

The 16 approved features comprise:
1. **13 Numeric Features**:
   - Schedule & Calendar: `month` (1–12), `day` (1–31), `hour` (0–23), `minute` (0–59), `day_of_week` (0–6), `day_of_year` (1–365), `distance` (flight distance in miles).
   - Cyclical Encodings (Sine / Cosine transformations to capture circular continuity of time):
     - `hour_sin`, `hour_cos`: 24-hour cycle ($\sin/\cos(2\pi \cdot \text{hour}/24)$)
     - `dow_sin`, `dow_cos`: 7-day weekly cycle ($\sin/\cos(2\pi \cdot \text{day\_of\_week}/7)$)
     - `day_of_year_sin`, `day_of_year_cos`: 365.25-day annual cycle ($\sin/\cos(2\pi \cdot \text{day\_of\_year}/365.25)$)
2. **3 Categorical Features**:
   - `carrier`: Two-letter airline carrier code (e.g., UA, AA, B6, DL, EV).
   - `origin`: Departure airport (EWR, JFK, LGA).
   - `dest`: Destination airport code (105 unique destinations).

### 2.4 Chronological Data Splitting
Because flight records represent time-series events, random shuffling would leak future temporal patterns into past training samples. A strict **chronological split** was implemented on sorted scheduled departure times:
- **Training Set (70%)**: 229,142 flights (from 2013-01-01 05:15:00 to 2013-09-13 19:05:00) | Delay rate: 25.94%
- **Validation Set (15%)**: 49,102 flights (from 2013-09-13 19:05:00 to 2013-11-06 13:00:00) | Delay rate: 15.39%
- **Test Set (15%)**: 49,102 flights (from 2013-11-06 13:00:00 to 2013-12-31 23:59:00) | Delay rate: 26.69%

---

## 3. Project Pipeline

The pipeline is organized into modular notebooks and supporting Python packages in `src/`:

### Stage 1: Data Loading, Cleaning, and EDA (`notebooks/01_eda.ipynb` & `src/data_loader.py`, `src/preprocessing.py`)
- Loads the raw flight dataset and checks column types, missingness, and duplicates.
- Analyzes univariate delay distributions: average delay rates by carrier (e.g., Frontier at 38.77%, AirTran at 34.65% vs. lowest delay carriers), origin airport (EWR: 26.34%, JFK: 23.70%, LGA: 23.14%), month of the year (June/July peaking above 31%), and departure hour (delays build steadily throughout the day, from ~10% at 5 AM to >30% late evening).

### Stage 2: Feature Engineering and Leakage Prevention (`notebooks/02_feature_engineering.ipynb` & `src/feature_engineering.py`)
- Derives `scheduled_datetime` from `time_hour` and `minute`.
- Extracts calendar features (`day_of_week`, `day_of_year`) and computes cyclical trigonometric features (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `day_of_year_sin`, `day_of_year_cos`).
- Verifies that post-departure fields are excluded.
- Performs the chronological split and exports the train, validation, and test sets.

### Stage 3: Logistic Regression Baseline (`notebooks/03_baseline_models.ipynb` & `src/train.py`)
- Sets up a scikit-learn `Pipeline` with a `ColumnTransformer`:
  - `StandardScaler` for numeric columns.
  - `OneHotEncoder(handle_unknown="ignore")` for categorical columns.
- Fits a `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`.
- Evaluates validation precision-recall trade-offs to pick a classification threshold.
- Saves model bundle to `models/logistic_regression_baseline.joblib` (saved threshold: 0.7223).

### Stage 4: XGBoost Model (`notebooks/04_xgboost.ipynb` & `src/train.py`)
- Constructs an end-to-end `Pipeline` with passthrough numeric features and one-hot encoded categorical features.
- Trains an `XGBClassifier` with:
  - `n_estimators = 400`, `max_depth = 6`, `learning_rate = 0.05`
  - `subsample = 0.8`, `colsample_bytree = 0.8`, `reg_lambda = 1`
  - `tree_method = "hist"`, `objective = "binary:logistic"`, `random_state = 42`
- Computes feature importances: top predictors are carrier effects (`carrier_EV`, `carrier_DL`), scheduled hour (`hour`, `hour_sin`), seasonal cycles (`day_of_year_cos`), and high-traffic destinations (`dest_BOS`).
- Saves model bundle to `models/xgboost_model.joblib` (operational threshold: 0.3000).

### Stage 5: Evaluation and Calibration (`notebooks/05_model_evaluation.ipynb` & `src/evaluate.py`, `src/calibration.py`)
- Loads both model bundles and applies them to the unseen test set (49,102 samples).
- Computes PR-AUC, ROC-AUC, Precision, Recall, F1-Score, Accuracy, and Brier Score.
- Plots bar chart comparisons (`reports/figures/model_comparison.png`) and probability calibration curves.
- Exports tabular metrics to `reports/metrics/model_comparison.csv` and `reports/metrics/model_comparison.md`.

---

## 4. How to Run the Project

### 4.1 Environment Setup
The project requires Python 3.10+ (tested on Python 3.14). A virtual environment should be used:

```bash
# 1. Clone the repository and enter the directory
git clone https://github.com/Jayadeep007/flight-delay-prediction.git
cd flight-delay-prediction

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install required libraries
pip install -r requirements.txt
```

### 4.2 Required Libraries (`requirements.txt`)
The `requirements.txt` file lists the core dependencies:
- `pandas`: Data manipulation and temporal feature creation
- `numpy`: Trigonometric cyclical features and numerical operations
- `matplotlib`: Visualization and metric comparison plotting
- `seaborn`: Statistical EDA plots
- `scikit-learn`: Preprocessing pipelines, Logistic Regression, metric functions, calibration curves, and `joblib`
- `xgboost`: Gradient boosted decision tree classifier
- `jupyter`, `ipykernel`: Notebook execution environment
- `torch`: Reserved for future deep learning extensions

### 4.3 Execution Order of Notebooks
Execute the notebooks in sequence from top to bottom:
1. `notebooks/01_eda.ipynb`: Exploratory data analysis, missing value auditing, and delay patterns.
2. `notebooks/02_feature_engineering.ipynb`: Feature derivation, cyclical transforms, leakage checks, and split creation.
3. `notebooks/03_baseline_models.ipynb`: Logistic regression baseline training and bundle creation.
4. `notebooks/04_xgboost.ipynb`: XGBoost model training, hyperparameter configuration, and feature importance analysis.
5. `notebooks/05_model_evaluation.ipynb`: Final test set evaluation, model comparison, calibration curves, and report generation.

Alternatively, the Python modules in `src/` can be imported directly into automated training/evaluation scripts.

---

## 5. Repository Structure

```text
flight-delay-prediction/
├── README.md                              # Main project README
├── requirements.txt                       # Project Python dependencies
├── .gitignore                             # Git ignore rules for virtualenvs, data, and models
├── data/
│   ├── raw/
│   │   └── flights.csv                    # Raw 2013 NYC flights dataset (local / gitignored)
│   └── processed/
│       └── flights_processed.csv          # Feature-engineered dataset (local / gitignored)
├── notebooks/
│   ├── 01_eda.ipynb                       # Stage 1: Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb       # Stage 2: Feature Engineering & Splits
│   ├── 03_baseline_models.ipynb           # Stage 3: Logistic Regression Baseline
│   ├── 04_xgboost.ipynb                   # Stage 4: XGBoost Model
│   └── 05_model_evaluation.ipynb          # Stage 5: Final Evaluation & Calibration
├── src/
│   ├── __init__.py                        # Python package marker
│   ├── data_loader.py                     # Data loading and column validation
│   ├── preprocessing.py                   # Filtering and chronological split utilities
│   ├── feature_engineering.py             # Feature creation and cyclical encoding
│   ├── train.py                           # Training pipelines for Logistic Regression and XGBoost
│   ├── evaluate.py                        # Metric evaluation routines
│   └── calibration.py                     # Brier score and calibration curves
├── models/
│   ├── README.md                          # Documentation on model artifacts and loading
│   ├── logistic_regression_baseline.joblib# Serialized Logistic Regression bundle
│   └── xgboost_model.joblib               # Serialized XGBoost bundle
├── reports/
│   ├── figures/
│   │   └── model_comparison.png          # Bar chart comparing evaluation metrics
│   └── metrics/
│       ├── model_comparison.csv          # Evaluation metrics in CSV format
│       └── model_comparison.md           # Evaluation metrics in Markdown format
└── docs/
    ├── project_documentation.md           # Comprehensive project guide (this document)
    ├── evaluation_report.md               # Detailed evaluation analysis and metric discussion
    ├── viva_notes.md                      # Viva preparation questions and grounded answers
    ├── literature_review.md               # Methodological literature review
    ├── project_report.md                  # Project report summary
    ├── project_report.tex                 # IEEE conference format report source
    ├── project_report.pdf                 # Compiled IEEE project report
    └── references.bib                     # BibTeX reference entries (Cox, Chen, Brier)
```

---

## 6. Implementation Notes & TODOs

- **TODO: Temporal Deep Learning Extension**: While the original team project title includes *"and Temporal Deep Learning"*, only Phase 1 (Logistic Regression and XGBoost) is implemented in this codebase. Neural network architectures (such as LSTM or airline/airport entity embeddings) remain future work for Phase 2.
- **TODO: Dynamic Weather and Aircraft Turnaround Features**: The current feature set relies purely on scheduled timetable variables. Adding live meteorological data (wind speed, precipitation, convective weather) and previous inbound aircraft arrival status (`tailnum` turnaround delay) is planned for future iterations.
