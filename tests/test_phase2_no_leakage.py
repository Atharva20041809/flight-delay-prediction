"""Test suite to mathematically and empirically verify no target leakage occurs.

Leakage Rules:
1. Prediction time T = scheduled departure time minus 2 hours.
2. Forbidden features: dep_time, dep_delay, arr_time, arr_delay, air_time.
3. Rolling origin features must use only flights with actual departure time < T.
4. Historical daily features must use only flights from calendar dates < flight date.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import clean_data, create_target
from src.phase2_history_features import (
    add_history_features,
    get_phase2_feature_columns,
    HISTORY_FEATURES,
    PHASE2_FEATURE_COLUMNS,
)

FORBIDDEN_COLUMNS = ["dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"]


def test_forbidden_columns_excluded():
    """Verify none of the forbidden post-departure features are in model inputs."""
    feature_cols = get_phase2_feature_columns()
    for col in FORBIDDEN_COLUMNS:
        assert col not in feature_cols, f"Forbidden column {col} found in model feature columns!"
    print("✓ Passed: All forbidden post-departure columns are excluded.")


def test_rolling_origin_no_leakage():
    """Verify rolling origin stats only use departures strictly prior to T."""
    raw = load_data()
    df = clean_data(raw).head(5000).copy()
    df = create_target(df)
    df["scheduled_datetime"] = pd.to_datetime(df["time_hour"]) + pd.to_timedelta(df["minute"], unit="m")
    df = df.sort_values("scheduled_datetime").reset_index(drop=True)
    df["actual_dep_datetime"] = df["scheduled_datetime"] + pd.to_timedelta(df["dep_delay"], unit="m")

    processed, _ = add_history_features(df, train_ratio=0.7)

    # Pick an index with substantial flight history
    idx = 500
    flight = df.iloc[idx]
    flight_T = flight["scheduled_datetime"] - pd.Timedelta(hours=2)
    origin = flight["origin"]

    # In raw data, find all flights from this origin that departed at or after flight_T
    future_mask = (df["origin"] == origin) & (df["actual_dep_datetime"] >= flight_T)
    assert future_mask.sum() > 0, "Need at least one future departure to test invariance."

    # Perturb future departures by adding 100 minutes to their departure delay
    df_perturbed = df.copy()
    df_perturbed.loc[future_mask, "dep_delay"] = df_perturbed.loc[future_mask, "dep_delay"] + 100.0

    # Recompute history features
    processed_perturbed, _ = add_history_features(df_perturbed, train_ratio=0.7)

    # The features for the flight at idx must remain EXACTLY identical
    for feat in ["origin_delay_rate_last_3h", "origin_avg_dep_delay_last_3h", "origin_delay_rate_last_24h"]:
        val_orig = processed.iloc[idx][feat]
        val_pert = processed_perturbed.iloc[idx][feat]
        assert np.isclose(val_orig, val_pert, equal_nan=True), (
            f"Leakage detected! Feature {feat} changed from {val_orig} to {val_pert} "
            f"when future departures at or after T were modified."
        )
    print("✓ Passed: Modifying future departures at or after T does not affect rolling features.")


def test_daily_history_no_leakage():
    """Verify daily historical aggregates use only prior calendar days."""
    raw = load_data()
    df = clean_data(raw).head(5000).copy()
    df = create_target(df)
    df["scheduled_datetime"] = pd.to_datetime(df["time_hour"]) + pd.to_timedelta(df["minute"], unit="m")
    df = df.sort_values("scheduled_datetime").reset_index(drop=True)

    processed, _ = add_history_features(df, train_ratio=0.7)

    # Pick a flight in the middle
    idx_target = 1000
    flight = df.iloc[idx_target]
    target_date = flight["scheduled_datetime"].floor("D")
    carrier = flight["carrier"]

    # Perturb same-day and future flight arrival delays for this carrier
    same_or_future = (df["carrier"] == carrier) & (df["scheduled_datetime"] >= target_date)
    assert same_or_future.sum() > 0

    df_perturbed = df.copy()
    # Invert the delay outcomes for future flights of this carrier
    df_perturbed.loc[same_or_future, "arr_delay"] = df_perturbed.loc[same_or_future, "arr_delay"] + 200.0
    df_perturbed.loc[same_or_future, "delayed"] = 1

    processed_pert, _ = add_history_features(df_perturbed, train_ratio=0.7)

    orig_carrier_rate = processed.iloc[idx_target]["carrier_hist_delay_rate"]
    pert_carrier_rate = processed_pert.iloc[idx_target]["carrier_hist_delay_rate"]

    assert np.isclose(orig_carrier_rate, pert_carrier_rate, equal_nan=True), (
        f"Leakage detected in daily history! {orig_carrier_rate} != {pert_carrier_rate}"
    )
    print("✓ Passed: Modifying same-day or future flight outcomes does not affect daily history.")


if __name__ == "__main__":
    test_forbidden_columns_excluded()
    test_rolling_origin_no_leakage()
    test_daily_history_no_leakage()
    print("ALL LEAKAGE TESTS PASSED SUCCESSFULLY!")
