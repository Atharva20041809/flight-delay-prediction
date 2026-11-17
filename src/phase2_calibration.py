"""Probability calibration and evaluation utilities for Phase 2."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


def compute_ece(y_true, probabilities, n_bins=10):
    """Compute Expected Calibration Error (ECE) across uniform confidence bins."""
    y_true = np.asarray(y_true)
    probabilities = np.clip(np.asarray(probabilities), 1e-7, 1.0 - 1e-7)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)

        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(probabilities[mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def calibration_metrics(y_true, probabilities, n_bins=10):
    """Return Brier score, ECE, and reliability curve coordinates."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    fraction_positive, mean_predicted = calibration_curve(
        y_true, probabilities, n_bins=n_bins, strategy="uniform"
    )
    brier = float(brier_score_loss(y_true, probabilities))
    ece = compute_ece(y_true, probabilities, n_bins=n_bins)
    return {
        "brier_score": brier,
        "ece": ece,
        "mean_predicted_probability": mean_predicted,
        "fraction_of_positives": fraction_positive,
    }


class TemperatureScaler:
    """Post-hoc temperature scaling for neural network logits (Guo et al. 2017)."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, val_logits, val_labels):
        """Find optimal temperature T on validation set by minimizing negative log-likelihood."""
        val_logits = np.asarray(val_logits, dtype=np.float64)
        val_labels = np.asarray(val_labels, dtype=np.float64)

        def objective(t):
            scaled_logits = val_logits / t
            probs = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -50.0, 50.0)))
            probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
            return log_loss(val_labels, probs)

        res = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)
        return self

    def transform(self, test_logits):
        """Apply learned temperature to test logits and return calibrated probabilities."""
        scaled = np.asarray(test_logits, dtype=np.float64) / self.temperature
        probs = 1.0 / (1.0 + np.exp(-np.clip(scaled, -50.0, 50.0)))
        return np.clip(probs, 1e-12, 1.0 - 1e-12)


class PlattScaler:
    """Post-hoc Platt scaling (sigmoid calibration) for tree or linear models."""

    def __init__(self):
        self.lr = LogisticRegression(solver="lbfgs", random_state=42)

    def fit(self, val_probs, val_labels):
        val_probs = np.clip(np.asarray(val_probs).reshape(-1, 1), 1e-7, 1.0 - 1e-7)
        log_odds = np.log(val_probs / (1.0 - val_probs))
        self.lr.fit(log_odds, val_labels)
        return self

    def transform(self, test_probs):
        test_probs = np.clip(np.asarray(test_probs).reshape(-1, 1), 1e-7, 1.0 - 1e-7)
        log_odds = np.log(test_probs / (1.0 - test_probs))
        return self.lr.predict_proba(log_odds)[:, 1]


def plot_reliability_curves(models_probs_dict, y_true, title, save_filename, n_bins=10):
    """Plot multi-model reliability diagram."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / save_filename

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (name, probs) in enumerate(models_probs_dict.items()):
        f_pos, m_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")
        ece = compute_ece(y_true, probs, n_bins=n_bins)
        brier = brier_score_loss(y_true, probs)
        color = colors[i % len(colors)]
        plt.plot(
            m_pred,
            f_pos,
            marker="o",
            lw=2,
            color=color,
            label=f"{name} (ECE={ece:.3f}, Brier={brier:.3f})",
        )

    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Positives (Empirical)", fontsize=11)
    plt.title(title, fontsize=12, pad=10)
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_curves(models_probs_dict, y_true, curve_type="pr", save_filename="phase2_pr_curves.png"):
    """Plot multi-model PR or ROC curves."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / save_filename

    plt.figure(figsize=(7, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    if curve_type == "pr":
        baseline_rate = float(np.mean(y_true))
        plt.axhline(
            baseline_rate,
            color="k",
            linestyle="--",
            label=f"No Skill (rate={baseline_rate:.3f})",
            alpha=0.7,
        )
        for i, (name, probs) in enumerate(models_probs_dict.items()):
            prec, rec, _ = precision_recall_curve(y_true, probs)
            score = average_precision_score(y_true, probs)
            plt.plot(rec, prec, lw=2, color=colors[i % len(colors)], label=f"{name} (PR AUC = {score:.4f})")
        plt.xlabel("Recall", fontsize=11)
        plt.ylabel("Precision", fontsize=11)
        plt.title("Precision-Recall Curves (Test Set)", fontsize=12, pad=10)
    else:
        plt.plot([0, 1], [0, 1], "k--", label="Chance (ROC AUC = 0.5000)", alpha=0.7)
        for i, (name, probs) in enumerate(models_probs_dict.items()):
            fpr, tpr, _ = roc_curve(y_true, probs)
            score = roc_auc_score(y_true, probs)
            plt.plot(fpr, tpr, lw=2, color=colors[i % len(colors)], label=f"{name} (ROC AUC = {score:.4f})")
        plt.xlabel("False Positive Rate", fontsize=11)
        plt.ylabel("True Positive Rate", fontsize=11)
        plt.title("Receiver Operating Characteristic (ROC) Curves (Test Set)", fontsize=12, pad=10)

    plt.legend(loc="best", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

