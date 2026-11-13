#!/usr/bin/env python3
"""
Executable script for Comparative Analysis: Curve Fit vs. Linear Regression.

Runs both estimation methods on identical data, performs coupling and loss landscape
comparisons, exports summary CSV/JSON files, and generates publication-grade
side-by-side comparative figures.

Usage:
    python scripts/run_comparison.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pendulum_modeling.config import PLOTS_DIR, OUTPUTS_DIR, L_DATA
from pendulum_modeling.data import load_and_center_data, get_random_subset, prepare_time_series
from pendulum_modeling.estimation import run_curve_fit_analysis, run_linear_regression_analysis
from pendulum_modeling.analysis import compare_models
from pendulum_modeling.visualization import (
    plot_side_by_side_trajectories,
    plot_side_by_side_correlation,
    plot_side_by_side_heatmaps,
)


def main():
    print("=" * 80)
    print("RUNNING COMPARATIVE EVALUATION: CURVE FIT vs. LINEAR REGRESSION")
    print("=" * 80)

    # 1. Load data
    print("\n[1/5] Loading pendulum data and extracting identical subset...")
    df = load_and_center_data()
    subset_df = get_random_subset(df, num_os=10, random_state=42)
    t_data, theta_data, _ = prepare_time_series(subset_df)
    print(f"Loaded {len(t_data)} observation points across {t_data[-1]:.2f} seconds.")

    # 2. Run Curve Fit (Method 1)
    print("\n[2/5] Executing Non-linear Curve Fit (curve_fit + solve_ivp)...")
    res_cf = run_curve_fit_analysis(t_data, theta_data, run_scans=True, n_grid=60)

    # 3. Run Linear Regression (Method 2)
    print("\n[3/5] Executing Linear Regression (Damped Fourier + JAX + LinReg)...")
    res_lr = run_linear_regression_analysis(t_data, theta_data, run_scans=True, n_grid=60)

    # 4. Comparative Evaluation
    print("\n[4/5] Computing statistical comparison metrics...")
    comparison_df, summary = compare_models(res_cf, res_lr)

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)

    # Save comparison data to disk
    csv_path = OUTPUTS_DIR / "comparison_table.csv"
    json_path = OUTPUTS_DIR / "comparison_summary.json"
    comparison_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved comparison metrics to:\n  - {csv_path}\n  - {json_path}")

    # 5. Generate Comparative Figures
    print("\n[5/5] Generating publication-grade comparative figures...")

    # Plot 1: Trajectory overlay
    plot_side_by_side_trajectories(
        t_data=t_data,
        theta_data=theta_data,
        t_cf=t_data,
        y_cf=res_cf["fit_2param"]["y_model"],
        t_lr=t_data,
        y_lr=res_lr["fit"]["y_model"],
        save_path=str(PLOTS_DIR / "comparison_trajectories.png"),
    )

    # Plot 2: Side-by-side coupling curves
    cf_scan = res_cf["coupling_scan"]
    lr_scan = res_lr["coupling_scan"]
    plot_side_by_side_correlation(
        cf_L=cf_scan["L_values"],
        cf_gamma=cf_scan["gamma_values"],
        cf_err=cf_scan["gamma_errors"],
        lr_L=lr_scan["L_values"],
        lr_C=lr_scan["C_values"],
        lr_C_err=lr_scan["C_errors"],
        lr_L_opt=res_lr["fit"]["L_model"],
        lr_C_opt=res_lr["fit"]["C_model"],
        save_path=str(PLOTS_DIR / "comparison_correlation_curves.png"),
    )

    # Plot 3: Side-by-side error landscapes / heatmaps
    cf_sse = res_cf["sse_surface"]
    lr_loss = res_lr["loss_surface"]
    plot_side_by_side_heatmaps(
        cf_gamma=cf_sse["gamma_grid"],
        cf_L=cf_sse["L_grid"],
        cf_sse=cf_sse["sse_matrix"],
        cf_best_gamma=cf_sse["best_gamma"],
        cf_best_L=cf_sse["best_L"],
        lr_C=lr_loss["C_grid"],
        lr_L=lr_loss["L_grid"],
        lr_rss=lr_loss["rss_matrix"],
        lr_best_C=lr_loss["best_C"],
        lr_best_L=lr_loss["best_L"],
        save_path=str(PLOTS_DIR / "comparison_loss_heatmaps.png"),
    )

    print(f"\nAll comparative plots saved to: {PLOTS_DIR}")
    print("=" * 80)
    print("COMPARATIVE PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
