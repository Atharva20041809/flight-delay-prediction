"""Probability calibration utilities."""
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibration_metrics(y_true, probabilities, n_bins=10):
    """Return Brier score and reliability-curve values."""
    probabilities = np.asarray(probabilities)
    fraction_positive, mean_predicted = calibration_curve(
        y_true, probabilities, n_bins=n_bins, strategy="uniform"
    )
    return {
        "brier_score": brier_score_loss(y_true, probabilities),
        "mean_predicted_probability": mean_predicted,
        "fraction_of_positives": fraction_positive,
    }
