"""Evaluate binary flight-delay predictions."""
import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score, brier_score_loss,
    classification_report,
)


def evaluate_model(model, X_test, y_test, threshold=0.3):
    """Return test metrics using a chosen probability threshold."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_test, probabilities),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "accuracy": accuracy_score(y_test, predictions),
        "brier_score": brier_score_loss(y_test, probabilities),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0
        ),
    }
