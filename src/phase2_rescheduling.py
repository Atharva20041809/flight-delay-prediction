"""Offline flight rescheduling what-if analysis under operational constraints."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.phase2_history_features import PHASE2_NUMERIC_FEATURES
from src.dl_model import build_tensors

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def simulate_rescheduling(
    test_df,
    calibrated_probs,
    mlp_model,
    artifacts,
    temp_scaler,
    risk_threshold=0.40,
    penalty=0.05,
    max_moved_per_hour=5,
):
    """Simulate moving high-risk flights to safer departure slots within +/- 2 hours.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test set flights with original schedule and features.
    calibrated_probs : np.ndarray
        Calibrated delay probabilities for test set flights.
    mlp_model : FlightMLP
        Trained PyTorch model.
    artifacts : dict
        Model artifacts (mapper, scaler, numeric_cols).
    temp_scaler : TemperatureScaler
        Calibrated temperature scaling model.
    risk_threshold : float
        Probability threshold above which a flight is flagged for rescheduling.
    penalty : float
        Friction penalty (risk reduction required to justify moving a flight).
    max_moved_per_hour : int
        Maximum number of rescheduled flights allowed to be added to any airport-hour slot.

    Returns
    -------
    dict
        Summary statistics and detailed simulation records.
    """
    df = test_df.copy().reset_index(drop=True)
    df["orig_calibrated_risk"] = calibrated_probs

    flagged_mask = df["orig_calibrated_risk"] >= risk_threshold
    flagged_indices = df[flagged_mask].index.tolist()

    mapper = artifacts["mapper"]
    scaler = artifacts["scaler"]
    numeric_cols = artifacts["numeric_cols"]

    # Slot occupancy tracking for capacity constraint: (origin, date, hour) -> count
    slot_added_count = {}

    results = []

    for idx in flagged_indices:
        row = df.iloc[idx]
        orig_h = int(row["hour"])
        orig_risk = float(row["orig_calibrated_risk"])
        origin = row["origin"]
        date = str(row["scheduled_datetime"])[:10]

        best_slot = None
        best_risk = orig_risk
        best_delta = 0.0

        # Candidate hours within +/- 2 hours, operational window [6, 22]
        candidate_hours = [
            h for h in [orig_h - 2, orig_h - 1, orig_h + 1, orig_h + 2]
            if 6 <= h <= 22
        ]

        if not candidate_hours:
            results.append({
                "flight_idx": idx,
                "moved": False,
                "orig_hour": orig_h,
                "new_hour": orig_h,
                "orig_risk": orig_risk,
                "new_risk": orig_risk,
                "risk_delta": 0.0,
            })
            continue

        # Evaluate candidate slots
        candidates_data = []
        valid_candidates = []

        for cand_h in candidate_hours:
            slot_key = (origin, date, cand_h)
            if slot_added_count.get(slot_key, 0) >= max_moved_per_hour:
                continue

            cand_row = row.copy()
            cand_row["hour"] = cand_h
            cand_row["hour_sin"] = np.sin(2 * np.pi * cand_h / 24)
            cand_row["hour_cos"] = np.cos(2 * np.pi * cand_h / 24)
            candidates_data.append(cand_row)
            valid_candidates.append(cand_h)

        if not candidates_data:
            results.append({
                "flight_idx": idx,
                "moved": False,
                "orig_hour": orig_h,
                "new_hour": orig_h,
                "orig_risk": orig_risk,
                "new_risk": orig_risk,
                "risk_delta": 0.0,
            })
            continue

        cand_df = pd.DataFrame(candidates_data)
        c, o, d, x = build_tensors(cand_df, mapper, scaler, numeric_cols, is_train=False)
        with torch.no_grad():
            raw_logits = mlp_model(c, o, d, x).cpu().numpy()
            cand_probs = temp_scaler.transform(raw_logits)

        # Find best candidate with delta > penalty
        for cand_h, cand_risk in zip(valid_candidates, cand_probs):
            delta = orig_risk - cand_risk
            if delta > penalty and delta > best_delta:
                best_delta = delta
                best_risk = cand_risk
                best_slot = cand_h

        if best_slot is not None:
            slot_key = (origin, date, best_slot)
            slot_added_count[slot_key] = slot_added_count.get(slot_key, 0) + 1
            results.append({
                "flight_idx": idx,
                "moved": True,
                "orig_hour": orig_h,
                "new_hour": best_slot,
                "orig_risk": orig_risk,
                "new_risk": float(best_risk),
                "risk_delta": float(best_delta),
            })
        else:
            results.append({
                "flight_idx": idx,
                "moved": False,
                "orig_hour": orig_h,
                "new_hour": orig_h,
                "orig_risk": orig_risk,
                "new_risk": orig_risk,
                "risk_delta": 0.0,
            })

    results_df = pd.DataFrame(results)
    num_flagged = len(flagged_indices)
    num_moved = int(results_df["moved"].sum()) if not results_df.empty else 0

    if num_moved > 0:
        moved_df = results_df[results_df["moved"]]
        mean_orig_risk_moved = float(moved_df["orig_risk"].mean())
        mean_new_risk_moved = float(moved_df["new_risk"].mean())
        delays_avoided = float(moved_df["risk_delta"].sum())
        rel_delay_reduction = (delays_avoided / (moved_df["orig_risk"].sum())) * 100.0
    else:
        mean_orig_risk_moved = 0.0
        mean_new_risk_moved = 0.0
        delays_avoided = 0.0
        rel_delay_reduction = 0.0

    return {
        "risk_threshold": risk_threshold,
        "penalty": penalty,
        "total_test_flights": len(df),
        "flights_flagged": num_flagged,
        "flights_moved": num_moved,
        "pct_flagged_moved": (num_moved / num_flagged * 100.0) if num_flagged > 0 else 0.0,
        "mean_orig_risk_moved": mean_orig_risk_moved,
        "mean_new_risk_moved": mean_new_risk_moved,
        "expected_delays_avoided": delays_avoided,
        "delays_avoided_pct": rel_delay_reduction,
        "details": results_df,
    }


def run_sensitivity_analysis(
    test_df,
    calibrated_probs,
    mlp_model,
    artifacts,
    temp_scaler,
    thresholds=(0.35, 0.40, 0.45),
    penalties=(0.03, 0.08),
):
    """Run sensitivity analysis across combinations of thresholds and penalties."""
    records = []
    for th in thresholds:
        for p in penalties:
            res = simulate_rescheduling(
                test_df,
                calibrated_probs,
                mlp_model,
                artifacts,
                temp_scaler,
                risk_threshold=th,
                penalty=p,
            )
            records.append({
                "threshold": th,
                "penalty": p,
                "flagged": res["flights_flagged"],
                "moved": res["flights_moved"],
                "pct_moved": res["pct_flagged_moved"],
                "orig_risk": res["mean_orig_risk_moved"],
                "new_risk": res["mean_new_risk_moved"],
                "delays_avoided": res["expected_delays_avoided"],
                "reduction_pct": res["delays_avoided_pct"],
            })
    return pd.DataFrame(records)

