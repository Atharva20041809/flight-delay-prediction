"""End-to-end execution pipeline for Phase 2 Deep Learning flight delay prediction."""
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

from src.data_loader import load_data
from src.preprocessing import clean_data, create_target
from src.feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS
from src.phase2_history_features import (
    add_history_features,
    PHASE2_NUMERIC_FEATURES,
    PHASE2_FEATURE_COLUMNS,
)
from src.phase2_train import (
    train_logistic_regression,
    save_phase2_model,
    find_threshold_for_precision,
    evaluate_with_threshold,
)
from src.dl_model import (
    train_mlp_model,
    plot_training_curves,
    plot_carrier_embeddings,
    build_tensors,
    FlightMLP,
)
from src.phase2_ablations import (
    train_onehot_mlp,
    evaluate_mlp_on_test,
    evaluate_onehot_mlp_on_test,
    plot_ablation_group,
)
from src.phase2_calibration import (
    compute_ece,
    calibration_metrics,
    TemperatureScaler,
    PlattScaler,
    plot_reliability_curves,
    plot_curves,
)
from src.phase2_rescheduling import (
    simulate_rescheduling,
    run_sensitivity_analysis,
)


def run_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("STARTING PHASE 2 PIPELINE: DEEP LEARNING & FLIGHT DELAY PREDICTION")
    print("=" * 80)

    # ---------------------------------------------------------
    # STAGE 1: DATA LOADING & HISTORY FEATURE GENERATION
    # ---------------------------------------------------------
    print("\n[Stage 1/6] Loading data and engineering history features...")
    t0 = time.time()
    raw_df = load_data()
    clean_df = clean_data(raw_df)
    clean_df = create_target(clean_df)
    print(f"Loaded {len(clean_df):,} cleaned flight records.")

    df_p2, impute_stats = add_history_features(clean_df, train_ratio=0.70)
    p2_csv_path = PROJECT_ROOT / "data" / "processed" / "flights_phase2.csv"
    p2_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_p2.to_csv(p2_csv_path, index=False)
    print(f"Computed 7 history features and saved to {p2_csv_path} ({time.time() - t0:.2f}s).")

    # Chronological splits (70/15/15)
    n = len(df_p2)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df_p2.iloc[:train_end].copy()
    val_df = df_p2.iloc[train_end:val_end].copy()
    test_df = df_p2.iloc[val_end:].copy()

    y_train = train_df["delayed"].values
    y_val = val_df["delayed"].values
    y_test = test_df["delayed"].values

    print(f"Splits -> Train: {len(train_df):,} ({y_train.mean()*100:.2f}%), "
          f"Val: {len(val_df):,} ({y_val.mean()*100:.2f}%), "
          f"Test: {len(test_df):,} ({y_test.mean()*100:.2f}%)")

    # ---------------------------------------------------------
    # STAGE 2: RE-RUN ML BASELINES (PHASE 1 VS PHASE 2 FEATURES)
    # ---------------------------------------------------------
    print("\n[Stage 2/6] Training baseline models (Phase 1 vs Phase 2 features)...")
    t0 = time.time()

    # Logistic Regression (Phase 1 features)
    print("  - Training Logistic Regression (Phase 1 features)...")
    lr_p1 = train_logistic_regression(train_df[FEATURE_COLUMNS], y_train, use_phase2=False)
    val_probs_lr_p1 = lr_p1.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    th_lr_p1 = find_threshold_for_precision(y_val, val_probs_lr_p1, target_precision=0.40)
    metrics_lr_p1 = evaluate_with_threshold(y_test, lr_p1.predict_proba(test_df[FEATURE_COLUMNS])[:, 1], th_lr_p1)

    # Logistic Regression (Phase 2 features)
    print("  - Training Logistic Regression (Phase 2 features)...")
    lr_p2 = train_logistic_regression(train_df[PHASE2_FEATURE_COLUMNS], y_train, use_phase2=True)
    val_probs_lr_p2 = lr_p2.predict_proba(val_df[PHASE2_FEATURE_COLUMNS])[:, 1]
    th_lr_p2 = find_threshold_for_precision(y_val, val_probs_lr_p2, target_precision=0.40)
    test_probs_lr_p2 = lr_p2.predict_proba(test_df[PHASE2_FEATURE_COLUMNS])[:, 1]
    metrics_lr_p2 = evaluate_with_threshold(y_test, test_probs_lr_p2, th_lr_p2)
    save_phase2_model(lr_p2, th_lr_p2, "phase2_logistic_regression.joblib")

    print(f"Baselines completed in {time.time() - t0:.2f}s.")

    # ---------------------------------------------------------
    # STAGE 3: DEEP LEARNING MODEL (PYTORCH MLP WITH EMBEDDINGS)
    # ---------------------------------------------------------
    print("\n[Stage 3/6] Training Deep Learning Model (PyTorch MLP with embeddings)...")
    t0 = time.time()
    mlp_model, mlp_artifacts, mlp_history = train_mlp_model(
        train_df,
        val_df,
        PHASE2_NUMERIC_FEATURES,
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
    )
    print(f"MLP trained in {time.time() - t0:.2f}s. Best epoch: {mlp_artifacts['best_epoch']} "
          f"(Val PR AUC: {mlp_artifacts['best_pr_auc']:.4f}).")

    # Save training curves & carrier embeddings
    plot_training_curves(mlp_history)
    plot_carrier_embeddings(mlp_model, mlp_artifacts["mapper"])

    # Save model and artifacts
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mlp_model.state_dict(), models_dir / "dl_mlp_embeddings.pt")
    joblib.dump(mlp_artifacts, models_dir / "dl_artifacts.joblib")

    # Evaluate MLP on test set
    c_va, o_va, d_va, x_va = build_tensors(val_df, mlp_artifacts["mapper"], mlp_artifacts["scaler"], PHASE2_NUMERIC_FEATURES, is_train=False)
    val_probs_mlp = mlp_model.predict_proba(c_va, o_va, d_va, x_va)
    th_mlp = find_threshold_for_precision(y_val, val_probs_mlp, target_precision=0.40)

    c_te, o_te, d_te, x_te = build_tensors(test_df, mlp_artifacts["mapper"], mlp_artifacts["scaler"], PHASE2_NUMERIC_FEATURES, is_train=False)
    test_probs_mlp = mlp_model.predict_proba(c_te, o_te, d_te, x_te)
    metrics_mlp = evaluate_with_threshold(y_test, test_probs_mlp, th_mlp)

    # ---------------------------------------------------------
    # STAGE 4: ABLATION STUDIES
    # ---------------------------------------------------------
    print("\n[Stage 4/6] Running systematic ablation studies...")
    t0 = time.time()
    ablation_records = []

    # Helper function to log ablation record
    def add_ablation(group, variant, metrics):
        ablation_records.append({
            "group": group,
            "variant": variant,
            "threshold": metrics["threshold"],
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "brier_score": metrics["brier_score"],
        })

    # A: Embeddings vs One-Hot
    print("  - Ablation A: Embeddings vs One-Hot Encoding...")
    add_ablation("A: Architecture", "MLP with Embeddings (Default)", metrics_mlp)
    onehot_model, oh_scaler, oh_encoder = train_onehot_mlp(
        train_df, val_df, PHASE2_NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        dropout=0.3, weight_decay=1e-4, max_epochs=25, patience=4
    )
    metrics_onehot = evaluate_onehot_mlp_on_test(
        onehot_model, oh_scaler, oh_encoder, test_df, val_df,
        PHASE2_NUMERIC_FEATURES, CATEGORICAL_FEATURES, target_precision=0.40
    )
    add_ablation("A: Architecture", "MLP with One-Hot Encoding", metrics_onehot)

    # B: With vs Without History Features
    print("  - Ablation B: With vs Without History Features...")
    add_ablation("B: Features", "With History Features (Default)", metrics_mlp)
    mlp_no_hist, art_no_hist, _ = train_mlp_model(
        train_df, val_df, NUMERIC_FEATURES,
        max_epochs=25, patience=4, seed=42
    )
    metrics_no_hist = evaluate_mlp_on_test(mlp_no_hist, art_no_hist, test_df, val_df, target_precision=0.40)
    add_ablation("B: Features", "Without History Features (Phase 1 Features)", metrics_no_hist)

    # C: Dropout (0.0, 0.2, 0.3, 0.5)
    print("  - Ablation C: Dropout Rates (0.0, 0.2, 0.3, 0.5)...")
    for drop in [0.0, 0.2, 0.3, 0.5]:
        if drop == 0.3:
            add_ablation("C: Dropout", "Dropout 0.3 (Default)", metrics_mlp)
        else:
            m_drop, art_drop, _ = train_mlp_model(
                train_df, val_df, PHASE2_NUMERIC_FEATURES,
                dropout=drop, max_epochs=25, patience=4, seed=42
            )
            met_drop = evaluate_mlp_on_test(m_drop, art_drop, test_df, val_df, target_precision=0.40)
            add_ablation("C: Dropout", f"Dropout {drop}", met_drop)

    # D: Weight Decay (0.0, 1e-4, 1e-3)
    print("  - Ablation D: Weight Decay (0.0, 1e-4, 1e-3)...")
    for wd in [0.0, 1e-4, 1e-3]:
        if wd == 1e-4:
            add_ablation("D: Weight Decay", "Weight Decay 1e-4 (Default)", metrics_mlp)
        else:
            m_wd, art_wd, _ = train_mlp_model(
                train_df, val_df, PHASE2_NUMERIC_FEATURES,
                weight_decay=wd, max_epochs=25, patience=4, seed=42
            )
            met_wd = evaluate_mlp_on_test(m_wd, art_wd, test_df, val_df, target_precision=0.40)
            label = "No Weight Decay (0.0)" if wd == 0.0 else "Weight Decay 1e-3"
            add_ablation("D: Weight Decay", label, met_wd)

    # E: Embedding Size (small, default, large)
    print("  - Ablation E: Embedding Size (small, default, large)...")
    add_ablation("E: Embedding Size", "Default (4, 3, 8)", metrics_mlp)
    m_emb_s, art_emb_s, _ = train_mlp_model(
        train_df, val_df, PHASE2_NUMERIC_FEATURES,
        carrier_dim=2, origin_dim=2, dest_dim=4, max_epochs=25, patience=4, seed=42
    )
    add_ablation("E: Embedding Size", "Small (2, 2, 4)", evaluate_mlp_on_test(m_emb_s, art_emb_s, test_df, val_df, target_precision=0.40))

    m_emb_l, art_emb_l, _ = train_mlp_model(
        train_df, val_df, PHASE2_NUMERIC_FEATURES,
        carrier_dim=8, origin_dim=6, dest_dim=16, max_epochs=25, patience=4, seed=42
    )
    add_ablation("E: Embedding Size", "Large (8, 6, 16)", evaluate_mlp_on_test(m_emb_l, art_emb_l, test_df, val_df, target_precision=0.40))

    # F: Class Weighting (pos_weight vs unweighted)
    print("  - Ablation F: Class Weighting (pos_weight vs unweighted)...")
    add_ablation("F: Class Weighting", "With pos_weight (Default)", metrics_mlp)
    m_no_pw, art_no_pw, _ = train_mlp_model(
        train_df, val_df, PHASE2_NUMERIC_FEATURES,
        use_pos_weight=False, max_epochs=25, patience=4, seed=42
    )
    add_ablation("F: Class Weighting", "Without pos_weight (Unweighted)", evaluate_mlp_on_test(m_no_pw, art_no_pw, test_df, val_df, target_precision=0.40))

    # G: Final Comparison (Logistic Regression vs MLP, both Phase 2)
    print("  - Ablation G: Final Model Comparison...")
    add_ablation("G: Model Comparison", "Logistic Regression (Phase 2)", metrics_lr_p2)
    add_ablation("G: Model Comparison", "PyTorch MLP (Phase 2)", metrics_mlp)

    # Save ablation table & bar charts
    df_ablations = pd.DataFrame(ablation_records)
    metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df_ablations.to_csv(metrics_dir / "phase2_ablation_results.csv", index=False)

    md_lines = ["| Group | Variant | Threshold | PR AUC | ROC AUC | Precision | Recall | F1 | Brier Score |",
                "|:---|:---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ablation_records:
        md_lines.append(f"| {r['group']} | {r['variant']} | {r['threshold']:.4f} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['brier_score']:.4f} |")
    with open(metrics_dir / "phase2_ablation_results.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # Generate charts for each group
    plot_ablation_group(df_ablations[df_ablations["group"] == "A: Architecture"], "A (Architecture)", "phase2_ablation_A_embeddings_vs_onehot.png")
    plot_ablation_group(df_ablations[df_ablations["group"] == "B: Features"], "B (History Features)", "phase2_ablation_B_history_features.png")
    plot_ablation_group(df_ablations[df_ablations["group"] == "C: Dropout"], "C (Dropout Rate)", "phase2_ablation_C_dropout.png")
    plot_ablation_group(df_ablations[df_ablations["group"] == "D: Weight Decay"], "D (Weight Decay)", "phase2_ablation_D_weight_decay.png")
    plot_ablation_group(df_ablations[df_ablations["group"] == "E: Embedding Size"], "E (Embedding Size)", "phase2_ablation_E_embedding_size.png")
    plot_ablation_group(df_ablations[df_ablations["group"] == "F: Class Weighting"], "F (Class Weighting)", "phase2_ablation_F_class_weighting.png")
    plot_ablation_group(df_ablations[df_ablations["group"] == "G: Model Comparison"], "G (Model Comparison)", "phase2_ablation_G_model_comparison.png")

    print(f"Ablations finished and saved in {time.time() - t0:.2f}s.")

    # ---------------------------------------------------------
    # STAGE 5: VALIDATION AND CALIBRATION
    # ---------------------------------------------------------
    print("\n[Stage 5/6] Performing probability calibration and reliability diagnostics...")
    t0 = time.time()

    # Temperature scaling for MLP
    val_logits_mlp = mlp_model.predict_logits(c_va, o_va, d_va, x_va)
    test_logits_mlp = mlp_model.predict_logits(c_te, o_te, d_te, x_te)

    temp_scaler = TemperatureScaler().fit(val_logits_mlp, y_val)
    test_probs_mlp_cal = temp_scaler.transform(test_logits_mlp)
    print(f"Learned optimal temperature T for MLP: {temp_scaler.temperature:.4f}")

    # Platt scaling for Logistic Regression
    platt_scaler = PlattScaler().fit(val_probs_lr_p2, y_val)
    test_probs_lr_cal = platt_scaler.transform(test_probs_lr_p2)

    # Uncalibrated and calibrated test dictionaries
    uncal_probs_dict = {
        "Logistic Regression": test_probs_lr_p2,
        "PyTorch MLP": test_probs_mlp,
    }

    cal_probs_dict = {
        "Logistic Regression (Platt Scaled)": test_probs_lr_cal,
        "PyTorch MLP (Temp Scaled)": test_probs_mlp_cal,
    }

    # Plot reliability curves before and after
    plot_reliability_curves(uncal_probs_dict, y_test, "Reliability Curves (Before Calibration)", "phase2_calibration_curves_before.png")
    plot_reliability_curves(cal_probs_dict, y_test, "Reliability Curves (After Calibration)", "phase2_calibration_curves_after.png")

    # Plot PR and ROC curves
    plot_curves(uncal_probs_dict, y_test, curve_type="pr", save_filename="phase2_pr_curves.png")
    plot_curves(uncal_probs_dict, y_test, curve_type="roc", save_filename="phase2_roc_curves.png")

    # Calibration results table
    cal_table_records = []
    for model_name, raw_p, cal_p, is_cal in [
        ("Logistic Regression (Phase 2)", test_probs_lr_p2, test_probs_lr_cal, True),
        ("PyTorch MLP (Phase 2)", test_probs_mlp, test_probs_mlp_cal, True),
    ]:
        raw_brier = brier_score_loss(y_test, raw_p)
        raw_ece = compute_ece(y_test, raw_p)
        cal_brier = brier_score_loss(y_test, cal_p)
        cal_ece = compute_ece(y_test, cal_p)
        pr_auc = average_precision_score(y_test, cal_p)
        roc_auc = roc_auc_score(y_test, cal_p)

        cal_table_records.append({
            "model": model_name,
            "raw_ece": raw_ece,
            "calibrated_ece": cal_ece,
            "raw_brier": raw_brier,
            "calibrated_brier": cal_brier,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
        })

    df_cal = pd.DataFrame(cal_table_records)
    df_cal.to_csv(metrics_dir / "phase2_calibration_results.csv", index=False)

    cal_md_lines = [
        "| Model | Raw ECE | Calibrated ECE | Raw Brier | Calibrated Brier | PR AUC | ROC AUC |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in cal_table_records:
        cal_md_lines.append(
            f"| {r['model']} | {r['raw_ece']:.4f} | {r['calibrated_ece']:.4f} | "
            f"{r['raw_brier']:.4f} | {r['calibrated_brier']:.4f} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} |"
        )
    with open(metrics_dir / "phase2_calibration_results.md", "w") as f:
        f.write("\n".join(cal_md_lines) + "\n")

    print(f"Calibration completed in {time.time() - t0:.2f}s.")

    # ---------------------------------------------------------
    # STAGE 6: OFFLINE RESCHEDULING WHAT-IF ANALYSIS
    # ---------------------------------------------------------
    print("\n[Stage 6/6] Running offline rescheduling what-if analysis...")
    t0 = time.time()
    rescheduling_summary = simulate_rescheduling(
        test_df,
        test_probs_mlp_cal,
        mlp_model,
        mlp_artifacts,
        temp_scaler,
        risk_threshold=0.40,
        penalty=0.05,
    )
    print(f"Rescheduling Simulation (Threshold=0.40, Penalty=0.05):")
    print(f"  - Flagged flights: {rescheduling_summary['flights_flagged']:,} / {len(test_df):,}")
    print(f"  - Successfully moved: {rescheduling_summary['flights_moved']:,} ({rescheduling_summary['pct_flagged_moved']:.1f}%)")
    print(f"  - Mean risk before: {rescheduling_summary['mean_orig_risk_moved']:.4f} -> after: {rescheduling_summary['mean_new_risk_moved']:.4f}")
    print(f"  - Expected delays avoided: {rescheduling_summary['expected_delays_avoided']:.1f} ({rescheduling_summary['delays_avoided_pct']:.2f}% reduction)")

    sensitivity_df = run_sensitivity_analysis(
        test_df,
        test_probs_mlp_cal,
        mlp_model,
        mlp_artifacts,
        temp_scaler,
        thresholds=(0.35, 0.40, 0.45),
        penalties=(0.03, 0.08),
    )
    sensitivity_df.to_csv(metrics_dir / "phase2_rescheduling_sensitivity.csv", index=False)
    print(f"Rescheduling analysis completed in {time.time() - t0:.2f}s.")

    # ---------------------------------------------------------
    # FINAL METRICS TABLE
    # ---------------------------------------------------------
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PHASE 2 PIPELINE COMPLETE IN {total_elapsed:.1f}s")
    print("=" * 80)

    summary_comparison = pd.DataFrame([
        {"Model": "Logistic Regression (Phase 1)", "Features": 16, "Threshold": th_lr_p1, "PR AUC": metrics_lr_p1["pr_auc"], "ROC AUC": metrics_lr_p1["roc_auc"], "Precision": metrics_lr_p1["precision"], "Recall": metrics_lr_p1["recall"], "Brier Score": metrics_lr_p1["brier_score"]},
        {"Model": "Logistic Regression (Phase 2)", "Features": 23, "Threshold": th_lr_p2, "PR AUC": metrics_lr_p2["pr_auc"], "ROC AUC": metrics_lr_p2["roc_auc"], "Precision": metrics_lr_p2["precision"], "Recall": metrics_lr_p2["recall"], "Brier Score": metrics_lr_p2["brier_score"]},
        {"Model": "PyTorch MLP (Phase 2)", "Features": 23, "Threshold": th_mlp, "PR AUC": metrics_mlp["pr_auc"], "ROC AUC": metrics_mlp["roc_auc"], "Precision": metrics_mlp["precision"], "Recall": metrics_mlp["recall"], "Brier Score": metrics_mlp["brier_score"]},
    ])
    print("\nFINAL MODEL EVALUATION TABLE (TEST SET):")
    print(summary_comparison.to_string(index=False))

    summary_comparison.to_csv(metrics_dir / "phase2_final_model_comparison.csv", index=False)


if __name__ == "__main__":
    run_pipeline()

