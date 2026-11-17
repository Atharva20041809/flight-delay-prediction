"""Ablation studies for Flight Delay Deep Learning Model."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.phase2_history_features import PHASE2_NUMERIC_FEATURES, PHASE2_FEATURE_COLUMNS
from src.dl_model import CategoricalMapper, FlightMLP, build_tensors, train_mlp_model
from src.phase2_train import find_threshold_for_precision, evaluate_with_threshold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


class FlightMLPOneHot(nn.Module):
    """MLP that uses One-Hot Encoding instead of Entity Embeddings."""

    def __init__(self, in_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_onehot_mlp(
    train_df,
    val_df,
    numeric_cols,
    cat_cols,
    dropout=0.3,
    weight_decay=1e-4,
    lr=1e-3,
    batch_size=1024,
    max_epochs=40,
    patience=5,
    seed=42,
):
    """Train MLP with One-Hot encoded categoricals."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(train_df[numeric_cols])
    X_num_val = scaler.transform(val_df[numeric_cols])

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_train = ohe.fit_transform(train_df[cat_cols])
    X_cat_val = ohe.transform(val_df[cat_cols])

    X_tr = np.hstack([X_cat_train, X_num_train])
    X_va = np.hstack([X_cat_val, X_num_val])

    y_tr = train_df["delayed"].values
    y_va_true = val_df["delayed"].values

    x_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32)
    x_va_tensor = torch.tensor(X_va, dtype=torch.float32)
    y_va_tensor = torch.tensor(y_va_true, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(x_tr_tensor, y_tr_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model = FlightMLPOneHot(in_dim=X_tr.shape[1], dropout=dropout)
    pos_count = float(train_df["delayed"].sum())
    neg_count = float(len(train_df) - pos_count)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_pr_auc = -1.0
    best_weights = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_va_tensor)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        val_pr_auc = float(average_precision_score(y_va_true, val_probs))
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    return model, scaler, ohe


def evaluate_mlp_on_test(model, artifacts, test_df, val_df, target_precision=0.40):
    """Compute test metrics for an embedding MLP using validation threshold."""
    mapper = artifacts["mapper"]
    scaler = artifacts["scaler"]
    numeric_cols = artifacts["numeric_cols"]

    c_va, o_va, d_va, x_va = build_tensors(val_df, mapper, scaler, numeric_cols, is_train=False)
    val_probs = model.predict_proba(c_va, o_va, d_va, x_va)
    threshold = find_threshold_for_precision(val_df["delayed"], val_probs, target_precision)

    c_te, o_te, d_te, x_te = build_tensors(test_df, mapper, scaler, numeric_cols, is_train=False)
    test_probs = model.predict_proba(c_te, o_te, d_te, x_te)

    return evaluate_with_threshold(test_df["delayed"], test_probs, threshold)


def evaluate_onehot_mlp_on_test(model, scaler, ohe, test_df, val_df, numeric_cols, cat_cols, target_precision=0.40):
    """Compute test metrics for a one-hot MLP using validation threshold."""
    X_num_val = scaler.transform(val_df[numeric_cols])
    X_cat_val = ohe.transform(val_df[cat_cols])
    X_va = torch.tensor(np.hstack([X_cat_val, X_num_val]), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_va)).cpu().numpy()

    threshold = find_threshold_for_precision(val_df["delayed"], val_probs, target_precision)

    X_num_te = scaler.transform(test_df[numeric_cols])
    X_cat_te = ohe.transform(test_df[cat_cols])
    X_te = torch.tensor(np.hstack([X_cat_te, X_num_te]), dtype=torch.float32)

    with torch.no_grad():
        test_probs = torch.sigmoid(model(X_te)).cpu().numpy()

    return evaluate_with_threshold(test_df["delayed"], test_probs, threshold)


def plot_ablation_group(df_group, group_name, save_filename, metric_col="pr_auc"):
    """Save clean bar chart for an ablation experiment group."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / save_filename

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(df_group["variant"], df_group[metric_col], color="#2c7fb8", width=0.55, edgecolor="black")

    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.title(f"Ablation {group_name}: {metric_col.upper()}", fontsize=12, pad=10)
    plt.ylabel(metric_col.upper())
    plt.ylim(0, max(df_group[metric_col]) * 1.25)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

