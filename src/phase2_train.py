"""Train and evaluate the Logistic Regression baseline with Phase 2 history features."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    brier_score_loss,
)

from src.feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.phase2_history_features import (
    PHASE2_NUMERIC_FEATURES,
    PHASE2_FEATURE_COLUMNS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def build_phase1_preprocessor():
    """Build ColumnTransformer for Phase 1 (16 features)."""
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def build_phase2_preprocessor():
    """Build ColumnTransformer for Phase 2 (23 features including history)."""
    return ColumnTransformer([
        ("numeric", StandardScaler(), PHASE2_NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def find_threshold_for_precision(y_val, val_probs, target_precision=0.40, n_grid=200):
    """Find the threshold on validation predictions that achieves target precision.

    Scans thresholds across the empirical range. Selects the threshold that achieves
    precision closest to target_precision while ensuring positive predictions are made.
    """
    y_val_arr = np.asarray(y_val)
    val_probs_arr = np.asarray(val_probs)

    candidate_thresholds = np.linspace(0.05, 0.95, n_grid)
    best_threshold = 0.5
    min_diff = float("inf")

    for th in candidate_thresholds:
        preds = (val_probs_arr >= th).astype(int)
        if preds.sum() == 0:
            continue
        prec = precision_score(y_val_arr, preds, zero_division=0)
        diff = abs(prec - target_precision)
        if diff < min_diff:
            min_diff = diff
            best_threshold = float(th)

    return best_threshold


def evaluate_with_threshold(y_true, probabilities, threshold):
    """Compute comprehensive test metrics given a pre-chosen threshold."""
    y_true_arr = np.asarray(y_true)
    probs_arr = np.asarray(probabilities)
    preds = (probs_arr >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true_arr, probs_arr)),
        "roc_auc": float(roc_auc_score(y_true_arr, probs_arr)),
        "precision": float(precision_score(y_true_arr, preds, zero_division=0)),
        "recall": float(recall_score(y_true_arr, preds, zero_division=0)),
        "f1": float(f1_score(y_true_arr, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_true_arr, preds)),
        "brier_score": float(brier_score_loss(y_true_arr, probs_arr)),
    }


def train_logistic_regression(X_train, y_train, use_phase2=True):
    """Train regularized logistic regression baseline."""
    preprocessor = build_phase2_preprocessor() if use_phase2 else build_phase1_preprocessor()
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )),
    ])
    model.fit(X_train, y_train)
    return model


def save_phase2_model(model, threshold, filename):
    """Save model bundle with chosen validation threshold."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    joblib.dump({
        "model": model,
        "threshold": threshold,
    }, path)
    return path

