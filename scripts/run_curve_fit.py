#!/usr/bin/env python3
"""
Executable script for Non-Linear Curve Fit Analysis (Project_PoC.ipynb Replication).

Directly replicates the analysis and outputs from Project_PoC.ipynb:
- 1-parameter curve_fit (fixed length L = 1.615 m).
- 2-parameter curve_fit (free gamma and L).
- Correlation curve across fixed L.
- 3D parameter trajectory plot.
- 2D SSE heatmap / error surface.

Usage:
    python scripts/run_curve_fit.py
"""

import sys
from pathlib import Path
import numpy as np

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pendulum_modeling.config import PLOTS_DIR, L_DATA, POC_SEED, POC_NUM_OSCILLATIONS
from pendulum_modeling.data import load_and_center_data, get_random_subset, prepare_time_series
from pendulum_modeling.models import solve_pendulum_ivp
from pendulum_modeling.estimation import run_curve_fit_analysis
from pendulum_modeling.visualization import (
    plot_model,
    plot_correlation_curve_2d,
    plot_correlation_curve_3d,
    plot_heatmap,
)


def main():
    print("=" * 70)
    print("RUNNING NON-LINEAR CURVE FIT ANALYSIS (Project_PoC.ipynb Replication)")
    print("=" * 70)

    # 1. Load data using exact configuration from Project_PoC.ipynb (Cell 1 & 11)
    print("\n[1/4] Loading and centering pendulum data...")
    df = load_and_center_data()
    # POC_SEED = 42 and 15 oscillations reproduces Project_PoC.ipynb exactly
    subset_df = get_random_subset(df, num_os=POC_NUM_OSCILLATIONS, random_state=POC_SEED)

    t_start_orig = subset_df['Time'].iloc[0]
    t_end_orig = subset_df['Time'].iloc[-1]
    print(f"Subset window: {t_start_orig:.1f} s to {t_end_orig:.1f} s (n = {len(subset_df)} points)")

    t_data, theta_data, subset_zeroed = prepare_time_series(subset_df)

    # 2. Run Curve Fit estimation
    print("\n[2/4] Performing 1-param and 2-param curve_fit optimization...")
    results = run_curve_fit_analysis(t_data, theta_data, run_scans=True, n_grid=100)

    res_1p = results["fit_1param"]
    res_2p = results["fit_2param"]
    scan = results["coupling_scan"]
    sse = results["sse_surface"]

    # Print exact replication outputs from Project_PoC.ipynb (Cells 19, 21, 30)
    print("\n--- Project_PoC.ipynb Exact Replication Outputs ---")
    print(f"Optimized Damping Coefficent (gamma): {res_1p['gamma_fit']:.5f} ± {res_1p['gamma_err']:.5f} s^-1")
    print(f"1-Param Trajectory MSE: {res_1p['mse']:.6e} rad^2")

    print(f"\nOptimized Damping Coefficents -> gamma: {res_2p['gamma_fit']:.5f} ± {res_2p['gamma_err']:.5f} s^-1")
    print(f"                              -> L:     {res_2p['L_fit']:.5f} ± {res_2p['L_err']:.4f} m")
    print(f"2-Param Trajectory MSE: {res_2p['mse']:.6e} rad^2")
    print(f"Difference vs. L_data ({L_DATA:.3f} m): {res_2p['L_error_cm']:.3f} cm")
    print(f"Parameter correlation |rho|: {abs(res_2p['param_correlation']):.4f}")

    print(f"\n--- Error Surface Grid Minimum (Cell 30) ---")
    print(f"Best gamma: {sse['best_gamma']:.5f}")
    print(f"Best L:     {sse['best_L']:.5f}")
    print(f"Minimum SSE: {sse['min_sse']:.6g}")

    # 3. Generate and save plots matching Project_PoC.ipynb
    print("\n[3/4] Generating and saving plots...")

    # Plot 1: Model 1-param (Cell 19)
    plot_model(
        time_data=subset_zeroed['Time'],
        angle_data=subset_zeroed['Angle'],
        model_fun=solve_pendulum_ivp,
        fit_params=res_1p["gamma_fit"],
        description="fixed length pendulum with damping",
        save_path=str(PLOTS_DIR / "curve_fit_trajectory_1param.png"),
    )

    # Plot 2: Model 2-param (Cell 21)
    plot_model(
        time_data=subset_zeroed['Time'],
        angle_data=subset_zeroed['Angle'],
        model_fun=solve_pendulum_ivp,
        fit_params=(res_2p["gamma_fit"], res_2p["L_fit"]),
        description="fixed length pendulum with damping and variable length",
        save_path=str(PLOTS_DIR / "curve_fit_trajectory_2param.png"),
    )

    # Plot 3: 3D Trajectory Curve (Cell 27)
    plot_correlation_curve_3d(
        scan["L_values"], scan["gamma_values"], scan["gamma_errors"],
        L_fit=res_2p["L_fit"], gamma_fit=res_2p["gamma_fit"], gamma_err=res_2p["gamma_err"],
        save_path=str(PLOTS_DIR / "curve_fit_correlation_curve_3d.png"),
    )

    # Plot 4: 2D Correlation Curve (Cell 28)
    plot_correlation_curve_2d(
        scan["L_values"], scan["gamma_errors"],
        L_fit=res_2p["L_fit"], gamma_err=res_2p["gamma_err"],
        save_path=str(PLOTS_DIR / "curve_fit_correlation_curve.png"),
    )

    # Plot 5: Seaborn Heatmap (Cell 30)
    best_idx = np.unravel_index(np.nanargmin(sse["sse_matrix"]), sse["sse_matrix"].shape)
    plot_heatmap(
        sse["gamma_grid"], sse["L_grid"], sse["sse_matrix"],
        best_index=best_idx,
        save_path=str(PLOTS_DIR / "curve_fit_sse_heatmap.png"),
    )

    print(f"\n[4/4] Plots successfully saved to: {PLOTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
