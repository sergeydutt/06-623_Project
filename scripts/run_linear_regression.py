#!/usr/bin/env python3
"""
Executable script for Linear Regression Analysis (Project_Final.ipynb Replication).

Directly replicates the analysis and outputs from Project_Final.ipynb:
- Continuous Damped Fourier series representation with optimized frequency (Cells 11-12).
- JAX automatic differentiation for velocity and acceleration (Cell 14).
- Linear regression against the physics model: ddot(theta) = -a1*dot(theta) - a2*sin(theta) (Cell 15).
- Evaluation via angular acceleration residuals, MSE, and R^2 (Cell 17).
- Parameter coupling diagnostics and correlation (Cell 19).
- Air-resistance coefficient C sensitivity scan across fixed L (Cell 20).
- 2D RSS loss surface computation and heatmap over (C, L) (Cell 21).

Usage:
    python scripts/run_linear_regression.py
"""

import sys
from pathlib import Path
import numpy as np

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pendulum_modeling.config import PLOTS_DIR, L_DATA, FINAL_SEED, FINAL_NUM_OSCILLATIONS
from pendulum_modeling.data import load_and_center_data, get_random_subset, prepare_time_series
from pendulum_modeling.models import (
    build_theta_func,
    evaluate_continuous_theta,
    solve_pendulum_ivp_a,
)
from pendulum_modeling.estimation import run_linear_regression_analysis
from pendulum_modeling.visualization import (
    plot_fourier_fit,
    plot_residuals,
    plot_derivatives,
    plot_model,
    plot_linear_regression_coupling,
    plot_heatmap,
)


def main():
    print("=" * 70)
    print("RUNNING LINEAR REGRESSION ANALYSIS (Project_Final.ipynb Replication)")
    print("=" * 70)

    # 1. Load data using exact configuration from Project_Final.ipynb (Cell 1 & 10)
    print("\n[1/4] Loading and centering pendulum data...")
    df = load_and_center_data()
    # FINAL_SEED = 42 reproduces the exact window: 269.5 s to 295.7 s
    subset_df = get_random_subset(df, num_os=FINAL_NUM_OSCILLATIONS, random_state=FINAL_SEED)

    t_start_orig = subset_df['Time'].iloc[0]
    t_end_orig = subset_df['Time'].iloc[-1]
    print(f"Subset original time window: {t_start_orig:.1f} s to {t_end_orig:.1f} s")

    t_data, theta_data, _ = prepare_time_series(subset_df)
    print(f"Points: {len(t_data)} across {t_data[-1]:.2f} seconds.")

    # 2. Run Linear Regression analysis
    print("\n[2/4] Executing Fourier series fitting, JAX autodiff, & Linear Regression...")
    results = run_linear_regression_analysis(t_data, theta_data, run_scans=True, n_grid=100)

    fit = results["fit"]
    fourier = fit["fourier"]
    scan = results["coupling_scan"]
    loss = results["loss_surface"]

    # Replicate exact printouts from Project_Final.ipynb (Cells 12, 15, 17)
    print("\n--- Project_Final.ipynb Exact Replication Outputs ---")
    print(f"Number of cos/sin series in Fourier Series is k = {fourier['k_total']}")
    print(f"L_model: {fit['L_model']:.3f} m")
    print(f"L_model error: {fit['L_error_cm']:.3f} cm")
    print(f"Air-Resistance Coef (C): {fit['C_model']:.3f}")
    print(f"R^2 is found to be: {fit['r_squared']:.3f}.")

    # Diagnostics from Cell 19
    print("\n--- Parameter Coupling Diagnostics (Cell 19) ---")
    print(f"Parameter Correlation |rho|: {abs(fit['param_correlation']):.4f}")
    print(f"Regressor Overlap Index:    {fit['regressor_overlap']:.4f} (Near-zero indicates orthogonal regressors)")
    print(f"Damping parameter a1:       {fit['a1']:.5f} ± {fit['a1_err']:.5f} s^-1")
    print(f"Restoring parameter a2:     {fit['a2']:.5f} ± {fit['a2_err']:.5f} s^-2")

    # Grid minimum from Cell 21
    print("\n--- Error Surface Grid Minimum (Cell 21) ---")
    print(f"Grid Minimum -> C: {loss['best_C']:.5f}, L: {loss['best_L']:.5f} m (Min RSS: {loss['min_rss']:.4f})")

    # 3. Generate and save plots matching Project_Final.ipynb
    print("\n[3/4] Generating and saving plots...")

    # Continuous Fourier fit plot (Project_Final Cell 13)
    t_dense = np.linspace(t_data[0], t_data[-1], 300)
    theta_func = build_theta_func(
        fourier["a_coefficients"], fourier["Lambda"], fourier["freq"], fourier["k_total"]
    )
    theta_dense = evaluate_continuous_theta(theta_func, t_dense)

    plot_fourier_fit(
        t_data, theta_data, t_dense, theta_dense,
        title="Continuous Fourier Fit vs. Raw Data",
        save_path=str(PLOTS_DIR / "linear_regression_fourier_fit.png"),
    )

    # Residuals plot (Project_Final Cell 17)
    plot_residuals(
        t_data, fit["residuals"],
        title="Residuals vs Time",
        save_path=str(PLOTS_DIR / "linear_regression_residuals.png"),
    )

    # Autodiff derivatives plot (Project_Final Cell 14)
    plot_derivatives(
        t_data, fourier["theta_dot"], fourier["theta_ddot"],
        save_path=str(PLOTS_DIR / "linear_regression_derivatives.png"),
    )

    # Simulated ODE trajectory vs data
    theta_init = float(theta_data[0])
    plot_model(
        subset_df['Time'], subset_df['Angle'],
        lambda t, *p, th0=theta_init: solve_pendulum_ivp_a(t, fit["a1"], fit["a2"], theta_0=th0),
        fit_params=(),
        description="Linear Regression ODE simulation",
        save_path=str(PLOTS_DIR / "linear_regression_trajectory_simulated.png"),
    )

    # Coupling scan across fixed L (Project_Final Cell 20)
    plot_linear_regression_coupling(
        L_values=scan["L_values"],
        C_values=scan["C_values"],
        C_errors=scan["C_errors"],
        L_model=fit["L_model"],
        C_model=fit["C_model"],
        save_path=str(PLOTS_DIR / "linear_regression_coupling_scan.png"),
    )

    # Loss surface heatmap (Project_Final Cell 21)
    plot_heatmap(
        y_values=loss["C_grid"],
        x_values=loss["L_grid"],
        loss_matrix=loss["rss_matrix"],
        best_index=loss["best_index"],
        xlabel="Pendulum length, L (m)",
        ylabel=r"Air-resistance coefficient, $C$",
        cbar_label=r"$\log_{10}(\mathrm{RSS})$",
        title="Parameter-Coupling Error Surface (RSS)",
        save_path=str(PLOTS_DIR / "linear_regression_loss_heatmap.png"),
    )

    print(f"\n[4/4] Plots successfully saved to: {PLOTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
