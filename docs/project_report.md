# Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning

## 1. Project Overview
This project investigates machine learning methods for predicting flight delays using historical flight data. Phase 1 establishes a reproducible classification pipeline and compares Logistic Regression with XGBoost.

## 2. Problem Statement
Passengers and airline operations can benefit from early estimates of flight delay risk. The project predicts whether a flight's arrival delay will be at least 15 minutes.

## 3. Dataset and Target
The dataset contains 2013 New York City flight records. Records without an arrival-delay label are removed. The target is `delayed = 1` when `arr_delay >= 15`, otherwise `delayed = 0`.

The processed dataset contains 327,346 rows and 18 columns, including the scheduled datetime, 16 engineered input features, and the target label.

## 4. Methodology
The workflow includes exploratory data analysis, feature engineering, chronological data splitting, baseline model training, XGBoost training, and evaluation. Input features include scheduled calendar/time information, carrier, origin, destination, and distance. Post-flight outcome fields are excluded from model inputs to reduce data leakage.

## 5. Models
### Logistic Regression
Used as the baseline binary classifier.

### XGBoost
Used as a gradient-boosted tree classifier to model nonlinear relationships between the input features and delay outcomes.

## 6. Evaluation Results
The saved-model evaluation script produced the following test metrics. The threshold shown is the threshold loaded from each saved model bundle.

| Model | Threshold | PR AUC | ROC AUC | Precision | Recall | F1 | Accuracy | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.722301 | 0.308410 | 0.568216 | 0.000000 | 0.000000 | 0.000000 | 0.733086 | 0.209217 |
| XGBoost | 0.300000 | 0.333815 | 0.598460 | 0.336432 | 0.505417 | 0.403964 | 0.601910 | 0.209065 |

The XGBoost model has higher PR AUC and ROC AUC in this evaluation. At its 0.30 threshold, it identifies more delayed flights, with a recall of approximately 0.505, while precision is approximately 0.336.

The Logistic Regression result at its saved threshold has zero precision and recall, meaning it predicted no positive cases at that threshold. This threshold-specific result should not be interpreted as proof that Logistic Regression cannot rank flight-delay risk. Threshold selection and validation results should be reviewed before choosing a model for practical use.

## 7. Outputs
- `data/processed/flights_processed.csv`: engineered dataset.
- `models/logistic_regression_baseline.joblib`: saved baseline model bundle.
- `models/xgboost_model.joblib`: saved XGBoost model bundle.
- `reports/metrics/model_comparison.csv`: evaluation metrics in CSV format.
- `reports/metrics/model_comparison.md`: evaluation metrics in Markdown format.
- `reports/figures/model_comparison.png`: model comparison chart.
- `notebooks/`: notebooks for EDA, feature engineering, model training, and evaluation.

## 8. Limitations
The data represents flights from 2013 and may not reflect present-day operations. The results depend on the dataset, chronological split, features, and classification thresholds. This phase does not include temporal neural networks or flight rescheduling recommendations.

## 9. Conclusion
Phase 1 establishes an end-to-end baseline for flight delay classification. Under the saved-model evaluation, XGBoost performs better on PR AUC and ROC AUC than the Logistic Regression baseline. Further threshold tuning and evaluation on newer data would be needed before operational use.
