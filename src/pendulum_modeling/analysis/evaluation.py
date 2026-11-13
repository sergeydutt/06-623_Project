"""
Evaluation and comparative metrics module for Pendulum Modeling.

Evaluates models based on MSE, residuals, R^2, parameter uncertainties,
and parameter sensitivity, using the exact evaluation metrics from the project notebooks.
Does not assert an absolute ground truth for the estimated parameters; measures
error via MSE, residuals, and difference against experimental reference L_data.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from pendulum_modeling.config import L_DATA


def calculate_trajectory_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes trajectory fit error metrics based on residuals and MSE.

    Parameters
    ----------
    y_true : np.ndarray
        Observed angle array.
    y_pred : np.ndarray
        Model predicted angle array.

    Returns
    -------
    dict
        'sse', 'mse', 'rmse', 'mae', 'r2', and 'max_abs_error'.
    """
    residuals = y_true - y_pred
    sse = float(np.sum(residuals ** 2))
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))

    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (sse / ss_tot)) if ss_tot > 0 else 0.0
    max_err = float(np.max(np.abs(residuals)))

    return {
        "sse": sse,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "max_abs_error": max_err,
    }


def parameter_difference(
    estimated: float,
    reference_val: float = L_DATA,
) -> Dict[str, float]:
    """
    Calculates absolute difference and percentage difference between an estimated
    parameter and the experimental reference value (L_data = 1.615 m).

    Parameters
    ----------
    estimated : float
        Estimated parameter value (e.g. L_model).
    reference_val : float
        Reference measured value from setup.

    Returns
    -------
    dict
        'diff', 'diff_cm', and 'diff_pct'.
    """
    diff = float(np.abs(estimated - reference_val))
    diff_cm = float(diff * 100.0)
    diff_pct = float((diff / reference_val) * 100.0) if reference_val != 0 else 0.0

    return {
        "diff": diff,
        "diff_cm": diff_cm,
        "diff_pct": diff_pct,
    }


def parameter_correlation_from_cov(cov_matrix: np.ndarray) -> float:
    """
    Extracts the normalized correlation coefficient between parameter 1 and parameter 2
    from a 2x2 parameter covariance matrix:

        rho = Cov[0, 1] / sqrt( Cov[0, 0] * Cov[1, 1] )

    Parameters
    ----------
    cov_matrix : np.ndarray
        2x2 covariance matrix.

    Returns
    -------
    float
        Correlation coefficient between -1.0 and +1.0.
    """
    if cov_matrix is None or cov_matrix.shape != (2, 2):
        return 0.0

    var_0 = cov_matrix[0, 0]
    var_1 = cov_matrix[1, 1]
    if var_0 > 0 and var_1 > 0:
        return float(cov_matrix[0, 1] / np.sqrt(var_0 * var_1))
    return 0.0


def compare_models(
    res_curve_fit: Dict[str, Any],
    res_linear_regression: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Constructs a comparative evaluation between Non-linear Curve Fit (curve_fit + solve_ivp)
    and Linear Regression (Damped Fourier Series + JAX autodiff + normal equations).

    Parameters
    ----------
    res_curve_fit : dict
        Results dictionary returned by curve_fit estimation.
    res_linear_regression : dict
        Results dictionary returned by linear regression estimation.

    Returns
    -------
    comparison_df : pd.DataFrame
        Formatted pandas DataFrame comparing all evaluation dimensions.
    summary : dict
        Numerical comparison metrics dictionary.
    """
    cf_fit = res_curve_fit.get("fit_2param", res_curve_fit)
    lr_fit = res_linear_regression.get("fit", res_linear_regression)

    # Extract Curve Fit metrics
    cf_L = cf_fit.get("L_fit", np.nan)
    cf_L_diff_cm = cf_fit.get("L_error_cm", abs(cf_L - L_DATA) * 100.0)
    cf_L_diff_pct = (cf_L_diff_cm / (L_DATA * 100.0)) * 100.0
    cf_gamma = cf_fit.get("gamma_fit", np.nan)
    cf_gamma_err = cf_fit.get("gamma_err", np.nan)
    cf_corr = cf_fit.get("param_correlation", np.nan)
    cf_mse = cf_fit.get("mse", np.nan)

    # Extract Linear Regression metrics
    lr_L = lr_fit.get("L_model", np.nan)
    lr_L_diff_cm = lr_fit.get("L_error_cm", abs(lr_L - L_DATA) * 100.0)
    lr_L_diff_pct = (lr_L_diff_cm / (L_DATA * 100.0)) * 100.0
    lr_a1 = lr_fit.get("a1", np.nan)
    lr_a1_err = lr_fit.get("a1_err", np.nan)
    lr_a2 = lr_fit.get("a2", np.nan)
    lr_a2_err = lr_fit.get("a2_err", np.nan)
    lr_C = lr_fit.get("C_model", np.nan)
    lr_corr = lr_fit.get("param_correlation", np.nan)
    lr_overlap = lr_fit.get("regressor_overlap", np.nan)
    lr_mse_physics = lr_fit.get("mse_physics", np.nan)
    lr_r2 = lr_fit.get("r_squared", np.nan)
    lr_mse_traj = lr_fit.get("trajectory_mse", np.nan)

    col_cf = "Non-linear Curve Fit (curve_fit + solve_ivp)"
    col_lr = "Linear Regression (Fourier + JAX)"

    data = [
        {
            "Metric / Property": "Estimation Approach",
            col_cf: "Non-linear least squares (ODE integration)",
            col_lr: "Linear regression (autodiff derivatives)",
        },
        {
            "Metric / Property": "Estimated Pendulum Length L (m)",
            col_cf: f"{cf_L:.5f} ± {cf_fit.get('L_err', 0.0):.4f}",
            col_lr: f"{lr_L:.5f} ± {lr_fit.get('L_err', 0.0):.4f}",
        },
        {
            "Metric / Property": "Difference vs. Setup Reference L_data (cm)",
            col_cf: f"{cf_L_diff_cm:.3f} cm",
            col_lr: f"{lr_L_diff_cm:.3f} cm",
        },
        {
            "Metric / Property": "Difference vs. Setup Reference (%)",
            col_cf: f"{cf_L_diff_pct:.2f}%",
            col_lr: f"{lr_L_diff_pct:.2f}%",
        },
        {
            "Metric / Property": "Damping Parameter Value",
            col_cf: f"gamma = {cf_gamma:.5f} ± {cf_gamma_err:.5f} s^-1",
            col_lr: f"a1 = {lr_a1:.5f} (C = {lr_C:.3f} kg*m^2/s)",
        },
        {
            "Metric / Property": "Parameter Coupling / Correlation |rho|",
            col_cf: f"{abs(cf_corr):.4f}",
            col_lr: f"{abs(lr_corr):.4f}",
        },
        {
            "Metric / Property": "Regressor Overlap Index",
            col_cf: "N/A (Nonlinear gradients)",
            col_lr: f"{lr_overlap:.4f} (Nearly orthogonal)",
        },
        {
            "Metric / Property": "Trajectory Angle Fit MSE (rad^2)",
            col_cf: f"{cf_mse:.6e}",
            col_lr: f"{lr_mse_traj:.6e}",
        },
        {
            "Metric / Property": "Physics Equation Residual MSE",
            col_cf: "N/A (fits trajectory directly)",
            col_lr: f"{lr_mse_physics:.6f} (rad/s^2)^2",
        },
        {
            "Metric / Property": "Physics Model R^2",
            col_cf: "N/A",
            col_lr: f"{lr_r2:.3f}",
        },
    ]

    df = pd.DataFrame(data)

    summary = {
        "curve_fit": {
            "L_fit": cf_L,
            "L_diff_cm": cf_L_diff_cm,
            "gamma_fit": cf_gamma,
            "param_correlation": cf_corr,
            "mse_trajectory": cf_mse,
        },
        "linear_regression": {
            "L_model": lr_L,
            "L_diff_cm": lr_L_diff_cm,
            "a1": lr_a1,
            "a2": lr_a2,
            "C_model": lr_C,
            "param_correlation": lr_corr,
            "regressor_overlap": lr_overlap,
            "mse_physics": lr_mse_physics,
            "r_squared": lr_r2,
            "mse_trajectory": lr_mse_traj,
        },
    }

    return df, summary
