"""Prepare flight data for model training."""
import pandas as pd


def clean_data(df):
    """Remove rows without a known arrival-delay outcome."""
    df = df.copy()
    df = df.dropna(subset=["arr_delay"])
    return df


def create_target(df, delay_minutes=15):
    """Label flights delayed by at least the specified minutes."""
    df = df.copy()
    df["delayed"] = (df["arr_delay"] >= delay_minutes).astype(int)
    return df


def chronological_split(df, train_ratio=0.70, validation_ratio=0.15):
    """Split chronologically into train, validation and test sets."""
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("Split ratios must be positive.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must sum to less than 1.")

    df = df.sort_values("scheduled_datetime").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + validation_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
