"""Train and save baseline and XGBoost flight-delay models."""
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def build_preprocessor():
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         CATEGORICAL_FEATURES),
    ])


def train_logistic_regression(X_train, y_train, threshold=0.3):
    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )),
    ])
    model.fit(X_train, y_train)
    return model, threshold


def train_xgboost(X_train, y_train, threshold=0.3):
    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    return model, threshold


def save_model(model, threshold, filename):
    """Save a model bundle compatible with the evaluation notebooks."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    joblib.dump({
        "model": model,
        "threshold": threshold,
    }, path)
    return path
