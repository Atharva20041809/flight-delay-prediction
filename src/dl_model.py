"""PyTorch Deep Learning Model with Entity Embeddings for Flight Delay Prediction."""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"


class CategoricalMapper:
    """Map string categorical features to 0-indexed integers with out-of-vocabulary handling."""

    def __init__(self):
        self.mappings = {}

    def fit(self, df, columns):
        for col in columns:
            uniques = sorted(df[col].dropna().unique().tolist())
            self.mappings[col] = {val: idx for idx, val in enumerate(uniques)}
        return self

    def transform(self, df, col):
        mapping = self.mappings[col]
        oov_idx = len(mapping)
        return np.array([mapping.get(val, oov_idx) for val in df[col]], dtype=np.int64)

    def num_classes(self, col):
        return len(self.mappings[col]) + 1  # include OOV token


class FlightMLP(nn.Module):
    """Multi-layer Perceptron with Entity Embeddings for tabular flight delay prediction."""

    def __init__(
        self,
        num_carriers,
        num_origins,
        num_dests,
        num_numeric,
        carrier_dim=4,
        origin_dim=3,
        dest_dim=8,
        dropout=0.3,
    ):
        super().__init__()
        self.carrier_embed = nn.Embedding(num_carriers, carrier_dim)
        self.origin_embed = nn.Embedding(num_origins, origin_dim)
        self.dest_embed = nn.Embedding(num_dests, dest_dim)

        in_dim = carrier_dim + origin_dim + dest_dim + num_numeric
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

    def forward(self, carrier, origin, dest, numeric):
        c_emb = self.carrier_embed(carrier)
        o_emb = self.origin_embed(origin)
        d_emb = self.dest_embed(dest)
        x = torch.cat([c_emb, o_emb, d_emb, numeric], dim=-1)
        return self.net(x).squeeze(-1)

    def predict_logits(self, carrier, origin, dest, numeric):
        self.eval()
        with torch.no_grad():
            return self.forward(carrier, origin, dest, numeric).cpu().numpy()

    def predict_proba(self, carrier, origin, dest, numeric):
        logits = self.predict_logits(carrier, origin, dest, numeric)
        return 1.0 / (1.0 + np.exp(-logits))


def build_tensors(df, mapper, scaler, numeric_cols, is_train=False):
    """Convert a DataFrame into PyTorch tensors for categorical and numeric inputs."""
    carrier_tensor = torch.tensor(mapper.transform(df, "carrier"), dtype=torch.long)
    origin_tensor = torch.tensor(mapper.transform(df, "origin"), dtype=torch.long)
    dest_tensor = torch.tensor(mapper.transform(df, "dest"), dtype=torch.long)

    if is_train:
        num_scaled = scaler.fit_transform(df[numeric_cols])
    else:
        num_scaled = scaler.transform(df[numeric_cols])

    num_tensor = torch.tensor(num_scaled, dtype=torch.float32)
    return carrier_tensor, origin_tensor, dest_tensor, num_tensor


def train_mlp_model(
    train_df,
    val_df,
    numeric_cols,
    carrier_dim=4,
    origin_dim=3,
    dest_dim=8,
    dropout=0.3,
    weight_decay=1e-4,
    lr=1e-3,
    batch_size=1024,
    max_epochs=40,
    patience=5,
    use_pos_weight=True,
    seed=42,
):
    """Train FlightMLP with early stopping based on validation PR AUC."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    mapper = CategoricalMapper().fit(train_df, ["carrier", "origin", "dest"])
    scaler = StandardScaler()

    c_tr, o_tr, d_tr, x_tr = build_tensors(train_df, mapper, scaler, numeric_cols, is_train=True)
    c_va, o_va, d_va, x_va = build_tensors(val_df, mapper, scaler, numeric_cols, is_train=False)

    y_tr = torch.tensor(train_df["delayed"].values, dtype=torch.float32)
    y_va = train_df["delayed"].values
    y_va_true = val_df["delayed"].values

    train_dataset = TensorDataset(c_tr, o_tr, d_tr, x_tr, y_tr)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model = FlightMLP(
        num_carriers=mapper.num_classes("carrier"),
        num_origins=mapper.num_classes("origin"),
        num_dests=mapper.num_classes("dest"),
        num_numeric=len(numeric_cols),
        carrier_dim=carrier_dim,
        origin_dim=origin_dim,
        dest_dim=dest_dim,
        dropout=dropout,
    )

    if use_pos_weight:
        pos_count = float(train_df["delayed"].sum())
        neg_count = float(len(train_df) - pos_count)
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_pr_auc": [],
        "val_roc_auc": [],
    }

    best_pr_auc = -1.0
    best_weights = None
    best_epoch = 0
    patience_counter = 0

    val_target_tensor = torch.tensor(y_va_true, dtype=torch.float32)

    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []
        for bc, bo, bd, bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bc, bo, bd, bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))

        # Evaluate on validation
        model.eval()
        with torch.no_grad():
            val_logits = model(c_va, o_va, d_va, x_va)
            val_loss = float(criterion(val_logits, val_target_tensor).item())
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        val_pr_auc = float(average_precision_score(y_va_true, val_probs))
        val_roc_auc = float(roc_auc_score(y_va_true, val_probs))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["val_roc_auc"].append(val_roc_auc)

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    artifacts = {
        "mapper": mapper,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "best_epoch": best_epoch,
        "best_pr_auc": best_pr_auc,
        "config": {
            "carrier_dim": carrier_dim,
            "origin_dim": origin_dim,
            "dest_dim": dest_dim,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "use_pos_weight": use_pos_weight,
        },
    }

    return model, artifacts, history


def plot_training_curves(history, save_path=None):
    """Plot and save training and validation loss curves."""
    if save_path is None:
        save_path = FIGURES_DIR / "dl_mlp_training_curves.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Validation Loss", color="#ff7f0e", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(epochs, history["val_pr_auc"], label="Val PR AUC", color="#2ca02c", lw=2)
    ax2.plot(epochs, history["val_roc_auc"], label="Val ROC AUC", color="#d62728", lw=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Score")
    ax2.set_title("Validation Discrimination Metrics")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_carrier_embeddings(model, mapper, save_path=None):
    """Extract learned carrier embeddings, project via PCA, and plot."""
    if save_path is None:
        save_path = FIGURES_DIR / "dl_carrier_embeddings.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    carrier_weights = model.carrier_embed.weight.detach().cpu().numpy()
    carrier_map = mapper.mappings["carrier"]
    carriers = list(carrier_map.keys())
    indices = [carrier_map[c] for c in carriers]
    weights_subset = carrier_weights[indices]

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(weights_subset)

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], color="#2b5c8f", s=100, edgecolors="black", alpha=0.85)

    for i, carrier in enumerate(carriers):
        plt.annotate(
            carrier,
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=11,
            fontweight="bold",
        )

    plt.title(
        f"Learned Airline Carrier Embeddings (2D PCA)\n"
        f"Variance Explained: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
        f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%",
        fontsize=12,
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

