"""Create pre-departure features for flight-delay prediction."""
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "month", "day", "hour", "minute", "day_of_week", "day_of_year",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "day_of_year_sin", "day_of_year_cos", "distance",
]
CATEGORICAL_FEATURES = ["carrier", "origin", "dest"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_features(df):
    """Create the same scheduled-time features used by the notebooks."""
    df = df.copy()
    scheduled = pd.to_datetime(df["time_hour"], errors="coerce")
    scheduled += pd.to_timedelta(df["minute"], unit="m")
    df["scheduled_datetime"] = scheduled

    df["day_of_week"] = scheduled.dt.dayofweek
    df["day_of_year"] = scheduled.dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    return df


def get_feature_columns():
    """Return the approved model input columns."""
    return FEATURE_COLUMNS.copy()


def get_model_data(df):
    """Return model inputs and the delay target, excluding leakage columns."""
    missing = set(FEATURE_COLUMNS + ["delayed"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature/target columns: {sorted(missing)}")
    return df[FEATURE_COLUMNS].copy(), df["delayed"].copy()
