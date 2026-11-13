"""
Linear Regression Parameter Estimation via Damped Fourier Series & JAX Autodiff.

Directly implements the parameter estimation and evaluation methodology from Project_Final.ipynb:
1. Damped Fourier series continuous trajectory representation (Cells 11-12).
2. JAX automatic differentiation for exact velocity and acceleration (Cell 14).
3. Linear regression against the pendulum physics model: ddot(theta) = -a1*dot(theta) - a2*sin(theta) (Cell 15).
4. Physical parameter extraction (L_model, C_model) and deviation from nominal reference (L_data).
5. Residual analysis, physics MSE, and coefficient of determination R^2 (Cell 17).
6. Parameter coupling diagnostics and correlation (Cell 19).
7. Air-resistance coefficient C sensitivity scan across fixed L (Cell 20).
8. 2D RSS loss surface computation over (C, L) grid (Cell 21).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import jax.numpy as jnp

from pendulum_modeling.config import (
    G_ACCEL,
    L_DATA,
    M_DATA,
    K_MAX_DEFAULT,
    TOLERANCE_DEFAULT,
    C_MIN,
    C_MAX,
    L_MIN,
    L_MAX,
    L_SCAN_VALUES,
    GRID_RESOLUTION,
)
from pendulum_modeling.models.fourier import (
    series,
    build_theta_func,
    get_derivatives,
)
from pendulum_modeling.models.ode import solve_pendulum_ivp_a


def fit_linear_regression(
    t_data: np.ndarray,
    theta_data: np.ndarray,
    g: float = G_ACCEL,
    L_data: float = L_DATA,
    m_data: float = M_DATA,
    k_max: int = K_MAX_DEFAULT,
    tol: float = TOLERANCE_DEFAULT,
) -> Dict[str, Any]:
    """
    Executes Linear Regression parameter estimation exactly as structured in Project_Final.ipynb.
    """
    # Step 1: Fourier Series representation (Project_Final.ipynb Cells 11-12)
    a_fourier, Lambda, freq, k_total = series(t_data, theta_data, K_max=k_max, tol=tol)
    theta_func = build_theta_func(a_fourier, Lambda, freq, k_total)

    # Step 2: JAX Autodiff Derivatives (Project_Final.ipynb Cell 14)
    theta_dot, theta_ddot = get_derivatives(theta_func, t_data)

    # Step 3: Linear Regression against Pendulum Model (Project_Final.ipynb Cell 15)
    Y = theta_ddot
    X = np.column_stack([-theta_dot, -np.sin(theta_data)])

    # Solve normal equations: a = (X^T X)^-1 X^T Y
    XtX = X.T @ X
    XtY = X.T @ Y
    a_vec = np.linalg.solve(XtX, XtY)
    a1 = float(a_vec[0])
    a2 = float(a_vec[1])

    # Step 4: Residual & Performance Calculations (Project_Final.ipynb Cells 15-17)
    Y_pred = X @ a_vec
    residuals = Y - Y_pred

    mse_physics = float(np.mean(residuals ** 2))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Step 5: Physical Parameters
    L_model = float(g / a2)
    L_err_cm = float(np.abs(L_model - L_data) * 100.0)
    C_model = float(a1 * m_data * (L_model ** 2))

    # Parameter Covariance & Coupling Diagnostics (Project_Final.ipynb Cell 19)
    dof = len(t_data) - 2
    sigma2 = float(np.sum(residuals ** 2) / dof) if dof > 0 else 0.0
    cov_matrix = sigma2 * np.linalg.inv(XtX)
    a1_std = float(np.sqrt(cov_matrix[0, 0]))
    a2_std = float(np.sqrt(cov_matrix[1, 1]))

    if a1_std > 0 and a2_std > 0:
        param_corr = float(cov_matrix[0, 1] / (a1_std * a2_std))
    else:
        param_corr = 0.0

    norm_x1 = np.linalg.norm(X[:, 0])
    norm_x2 = np.linalg.norm(X[:, 1])
    regressor_overlap = float(np.abs(X[:, 0] @ X[:, 1]) / (norm_x1 * norm_x2))

    # Forward-integrated trajectory check
    y_simulated = solve_pendulum_ivp_a(
        t_eval=t_data,
        a1=a1,
        a2=a2,
        theta_0=float(theta_data[0]),
    )
    traj_residuals = theta_data - y_simulated
    traj_mse = float(np.mean(traj_residuals ** 2))

    return {
        "method": "Linear Regression (Fourier + JAX)",
        "a1": a1,
        "a1_err": a1_std,
        "a2": a2,
        "a2_err": a2_std,
        "L_model": L_model,
        "L_err": a2_std * (g / (a2 ** 2)),
        "L_error_cm": L_err_cm,
        "C_model": C_model,
        "r_squared": r_squared,
        "mse_physics": mse_physics,
        "residuals": residuals,
        "y_pred_accel": Y_pred,
        "y_true_accel": Y,
        "y_model": y_simulated,
        "trajectory_residuals": traj_residuals,
        "trajectory_mse": traj_mse,
        "cov_matrix": cov_matrix,
        "param_correlation": param_corr,
        "regressor_overlap": regressor_overlap,
        "fourier": {
            "a_coefficients": a_fourier,
            "Lambda": Lambda,
            "freq": freq,
            "k_total": k_total,
            "theta_dot": theta_dot,
            "theta_ddot": theta_ddot,
        },
    }


def coupling_scan_linear_regression(
    t_data: np.ndarray,
    theta_data: np.ndarray,
    L_values: Optional[np.ndarray] = None,
    k_max: int = K_MAX_DEFAULT,
    tol: float = TOLERANCE_DEFAULT,
    g: float = G_ACCEL,
    m_data: float = M_DATA,
) -> Dict[str, np.ndarray]:
    """
    Evaluates the air-resistance coefficient C across a range of fixed L values.
    Direct implementation of Project_Final.ipynb (Cell 20).
    """
    if L_values is None:
        L_values = L_SCAN_VALUES

    a_fourier, Lambda, freq, k_total = series(t_data, theta_data, K_max=k_max, tol=tol)
    theta_func = build_theta_func(a_fourier, Lambda, freq, k_total)
    theta_dot, theta_ddot = get_derivatives(theta_func, t_data)

    Y = theta_ddot
    X1 = -theta_dot
    sum_x1_sq = float(np.dot(X1, X1))

    C_values = []
    C_errors = []
    a1_values = []
    a1_errors = []

    for L_val in L_values:
        a2_fixed = g / L_val
        Y_fixed = Y - (-a2_fixed * np.sin(theta_data))

        a1_opt = float(np.dot(X1, Y_fixed) / sum_x1_sq)
        res_fixed = Y_fixed - a1_opt * X1
        se_a1 = float(np.sqrt(np.sum(res_fixed ** 2) / (len(Y_fixed) - 1) / sum_x1_sq))

        a1_values.append(a1_opt)
        a1_errors.append(se_a1)
        C_values.append(a1_opt * m_data * (L_val ** 2))
        C_errors.append(se_a1 * m_data * (L_val ** 2))

    return {
        "L_values": np.array(L_values),
        "C_values": np.array(C_values),
        "C_errors": np.array(C_errors),
        "a1_values": np.array(a1_values),
        "a1_errors": np.array(a1_errors),
    }


def compute_loss_surface_linear_regression(
    t_data: np.ndarray,
    theta_data: np.ndarray,
    C_values: Optional[np.ndarray] = None,
    L_values: Optional[np.ndarray] = None,
    n_grid: int = GRID_RESOLUTION,
    k_max: int = K_MAX_DEFAULT,
    tol: float = TOLERANCE_DEFAULT,
    g: float = G_ACCEL,
    m_data: float = M_DATA,
) -> Dict[str, Any]:
    """
    Computes the 2D linear regression residual sum of squares (RSS) loss surface
    over a grid of (C, L) parameters.
    Direct implementation of Project_Final.ipynb (Cell 21).
    """
    if C_values is None:
        C_values = np.linspace(C_MIN, C_MAX, n_grid)
    if L_values is None:
        L_values = np.linspace(L_MIN, L_MAX, n_grid)

    a_fourier, Lambda, freq, k_total = series(t_data, theta_data, K_max=k_max, tol=tol)
    theta_func = build_theta_func(a_fourier, Lambda, freq, k_total)
    theta_dot, theta_ddot = get_derivatives(theta_func, t_data)

    Y = theta_ddot
    sin_theta = np.sin(theta_data)

    rss_matrix = np.empty((len(C_values), len(L_values)), dtype=float)

    for i, C_val in enumerate(C_values):
        for j, L_val in enumerate(L_values):
            a2_val = g / L_val
            a1_val = C_val / (m_data * (L_val ** 2))
            pred = -a1_val * theta_dot - a2_val * sin_theta
            rss_matrix[i, j] = float(np.sum((Y - pred) ** 2))

    min_idx = np.unravel_index(np.nanargmin(rss_matrix), rss_matrix.shape)
    best_C = float(C_values[min_idx[0]])
    best_L = float(L_values[min_idx[1]])
    min_rss = float(rss_matrix[min_idx])

    return {
        "C_grid": C_values,
        "L_grid": L_values,
        "rss_matrix": rss_matrix,
        "best_index": min_idx,
        "best_C": best_C,
        "best_L": best_L,
        "min_rss": min_rss,
    }


def run_linear_regression_analysis(
    t_data: np.ndarray,
    theta_data: np.ndarray,
    run_scans: bool = True,
    n_grid: int = GRID_RESOLUTION,
    g: float = G_ACCEL,
    m: float = M_DATA,
) -> Dict[str, Any]:
    """
    Executes complete Linear Regression estimation and diagnostic pipeline.
    """
    fit_res = fit_linear_regression(t_data, theta_data, g=g, m_data=m)

    results = {
        "fit": fit_res,
    }

    if run_scans:
        coupling_res = coupling_scan_linear_regression(t_data, theta_data, g=g, m_data=m)
        results["coupling_scan"] = coupling_res

        loss_res = compute_loss_surface_linear_regression(
            t_data, theta_data, n_grid=n_grid, g=g, m_data=m
        )
        results["loss_surface"] = loss_res

    return results
