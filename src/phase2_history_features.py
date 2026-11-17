"""Historical and pre-departure congestion features for Phase 2.

All features are strictly computed using information available at prediction
time T = scheduled_datetime - 2 hours.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES, create_features

HISTORY_FEATURES = [
    "origin_delay_rate_last_3h",
    "origin_avg_dep_delay_last_3h",
    "origin_delay_rate_last_24h",
    "origin_flights_scheduled_same_hour",
    "carrier_hist_delay_rate",
    "route_hist_delay_rate",
    "carrier_origin_hour_delay_rate",
]

PHASE2_NUMERIC_FEATURES = NUMERIC_FEATURES + HISTORY_FEATURES
PHASE2_FEATURE_COLUMNS = PHASE2_NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_history_features(df, train_ratio=0.70):
    """Compute leakage-free historical and congestion features.

    Parameters
    ----------
    df : pd.DataFrame
        Flight dataframe containing raw columns (including dep_delay, time_hour, minute, carrier, origin, dest).
    train_ratio : float
        Ratio of chronological training set used to fit imputation statistics.

    Returns
    -------
    pd.DataFrame
        Dataframe with Phase 1 and Phase 2 features added, sorted by scheduled_datetime.
    """
    df = df.copy()
    if "scheduled_datetime" not in df.columns:
        df = create_features(df)

    # Sort strictly chronologically
    df = df.sort_values("scheduled_datetime").reset_index(drop=True)
    if "delayed" not in df.columns and "arr_delay" in df.columns:
        df["delayed"] = (df["arr_delay"] >= 15).astype(int)

    n = len(df)
    train_end = int(n * train_ratio)
    global_train_delay_rate = float(df.iloc[:train_end]["delayed"].mean())

    # Prediction time T = scheduled_datetime - 2 hours
    df["T"] = df["scheduled_datetime"] - pd.Timedelta(hours=2)
    df["route"] = df["origin"] + "_" + df["dest"]
    df["date"] = df["scheduled_datetime"].dt.normalize()

    # 1. Rolling origin features based strictly on actual departures before T
    # A flight departed if dep_delay is not null and dep_delay is available
    valid_dep = df[df["dep_delay"].notna()].copy()
    valid_dep["actual_dep_datetime"] = valid_dep["scheduled_datetime"] + pd.to_timedelta(
        valid_dep["dep_delay"], unit="m"
    )
    valid_dep["is_dep_delayed"] = (valid_dep["dep_delay"] >= 15).astype(float)

    rate_3h = np.full(n, np.nan)
    avg_delay_3h = np.full(n, np.nan)
    rate_24h = np.full(n, np.nan)

    for origin in df["origin"].unique():
        orig_deps = valid_dep[valid_dep["origin"] == origin].sort_values("actual_dep_datetime")
        dep_times = orig_deps["actual_dep_datetime"].values
        dep_delays = orig_deps["dep_delay"].values
        dep_flags = orig_deps["is_dep_delayed"].values

        cum_count = np.pad(np.arange(1, len(dep_times) + 1), (1, 0))
        cum_delay_flags = np.pad(np.cumsum(dep_flags), (1, 0))
        cum_dep_delays = np.pad(np.cumsum(dep_delays), (1, 0))

        idx = df[df["origin"] == origin].index.values
        q_T = df.loc[idx, "T"].values
        q_T_3h = (df.loc[idx, "T"] - pd.Timedelta(hours=3)).values
        q_T_24h = (df.loc[idx, "T"] - pd.Timedelta(hours=24)).values

        # Use side='left' to strictly exclude departures at or after query time
        r = np.searchsorted(dep_times, q_T, side="left")
        l_3h = np.searchsorted(dep_times, q_T_3h, side="left")
        l_24h = np.searchsorted(dep_times, q_T_24h, side="left")

        c_3h = r - l_3h
        del_3h = cum_delay_flags[r] - cum_delay_flags[l_3h]
        sum_3h = cum_dep_delays[r] - cum_dep_delays[l_3h]

        c_24h = r - l_24h
        del_24h = cum_delay_flags[r] - cum_delay_flags[l_24h]

        with np.errstate(divide="ignore", invalid="ignore"):
            rate_3h[idx] = np.where(c_3h > 0, del_3h / c_3h, np.nan)
            avg_delay_3h[idx] = np.where(c_3h > 0, sum_3h / c_3h, np.nan)
            rate_24h[idx] = np.where(c_24h > 0, del_24h / c_24h, np.nan)

    df["origin_delay_rate_last_3h"] = rate_3h
    df["origin_avg_dep_delay_last_3h"] = avg_delay_3h
    df["origin_delay_rate_last_24h"] = rate_24h

    # 2. Origin scheduled departure density in the same hour (schedule known in advance)
    df["origin_flights_scheduled_same_hour"] = df.groupby(
        ["origin", "year", "month", "day", "hour"]
    )["scheduled_datetime"].transform("count").astype(float)

    # 3. Expanding daily historical delay rate prior to date D (shift by at least 1 day)
    def compute_daily_history(group_cols, name):
        daily = df.groupby(group_cols + ["date"])["delayed"].agg(["count", "sum"]).reset_index()
        daily = daily.sort_values(group_cols + ["date"])
        daily["cum_count"] = daily.groupby(group_cols)["count"].cumsum() - daily["count"]
        daily["cum_sum"] = daily.groupby(group_cols)["sum"].cumsum() - daily["sum"]
        with np.errstate(divide="ignore", invalid="ignore"):
            daily[name] = np.where(daily["cum_count"] > 0, daily["cum_sum"] / daily["cum_count"], np.nan)
        return daily[group_cols + ["date", name]]

    carrier_hist = compute_daily_history(["carrier"], "carrier_hist_delay_rate")
    df = df.merge(carrier_hist, on=["carrier", "date"], how="left")

    route_hist = compute_daily_history(["route"], "route_hist_delay_rate")
    df = df.merge(route_hist, on=["route", "date"], how="left")

    coh_hist = compute_daily_history(["carrier", "origin", "hour"], "carrier_origin_hour_delay_rate")
    df = df.merge(coh_hist, on=["carrier", "origin", "hour", "date"], how="left")

    # Impute missing values using train split statistics only
    impute_stats = {}
    for col in HISTORY_FEATURES:
        mean_val = float(df.iloc[:train_end][col].mean())
        if np.isnan(mean_val):
            mean_val = global_train_delay_rate
        impute_stats[col] = mean_val
        df[col] = df[col].fillna(mean_val)

    # Cleanup temporary working columns
    cols_to_drop = ["T", "date", "route"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df, impute_stats


def get_phase2_feature_columns():
    """Return all input feature column names for Phase 2 models."""
    return PHASE2_FEATURE_COLUMNS.copy()

