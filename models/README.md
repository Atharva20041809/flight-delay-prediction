# Model Artifacts

This directory stores trained machine learning models for the Flight Delay Prediction project.

## Saved Models

- **`logistic_regression_baseline.joblib`** — Stores the trained Logistic Regression pipeline and its classification threshold.
- **`xgboost_model.joblib`** — Stores the trained XGBoost pipeline and its classification threshold.

## Model Training

The models are trained in the following notebooks:

- `notebooks/03_baseline_models.ipynb`
- `notebooks/04_xgboost.ipynb`

Run these notebooks to generate the corresponding model artifacts.

## Model Loading

Each saved model is stored as a Joblib bundle containing the trained pipeline and threshold. The pipeline includes preprocessing and classification steps.

Example:

```python
import joblib

bundle = joblib.load("models/xgboost_model.joblib")
model = bundle["model"]
threshold = bundle["threshold"]
```

Use `model.predict_proba()` to obtain delay probabilities and the stored threshold to classify flights as delayed or not delayed.

## Important Notes

- Model files are generated artifacts and should be recreated by running the training notebooks if they are unavailable.
- Classification thresholds should be selected using validation data.
- The test set should be reserved for final model evaluation.
- Only features available at the intended prediction time should be used to avoid data leakage.