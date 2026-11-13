"""
Non-Linear Least Squares Parameter Estimation via curve_fit and solve_ivp.

Directly implements the parameter estimation and coupling analysis from Project_PoC.ipynb:
1. 1-parameter optimization (gamma only, with fixed physical length L = 1.615 m).
2. 2-parameter optimization (gamma and L simultaneously).
3. Parameter coupling scan across a range of fixed L values.
4. 2D Sum-of-Squared-Errors (SSE) loss landscape / heatmap computation.
5. Trajectory residual and MSE evaluation.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from pendulum_modeling.config import (
    G_ACCEL,
    L_DATA,
    GAMMA_MIN,
    GAMMA_MAX,
    L_MIN,
    L_MAX,
    GRID_RESOLUTION,
    L_SCAN_VALUES,
)
from pendulum_modeling.models.ode import solve_pendulum_ivp


def fit_curve_fit_1param(
    t_data: np.ndarray,
    theta_data: np.ndarray,
    fixed_L: float = L_DATA,
    p0_gamma: float = 0.05,
    bounds: Tuple[float, float] = (0.0, 1.0),
    g: float = G_ACCEL,
) -> Dict[str, Any]:
    """
    Fits only the damping parameter gamma using non-linear least squares (curve_fit)
    while keeping pendulum length L constrained to its nominal reference value.
    Direct implementation of Project_PoC.ipynb (Cell 19).

    Parameters
    ----------
    t_data : np.ndarray
        Observation times (s), starting at t_0 = 0.
    theta_data : np.ndarray
        Observed angular displacement (rad).
    fixed_L : float
        Constrained pendulum length (m). Default is 1.615.
    p0_gamma : float
        Initial guess for gamma (s^-1).
    bounds : tuple
        (lower_bound, upper_bound) for gamma.
    g : float
        Gravitational acceleration.

    Returns
    -------
    dict
        Dictionary containing optimal gamma, standard error, covariance, residuals, and MSE.
    """
    theta_0 = float(theta_data[0])

    def model_wrapper(t, gamma_val):
        return solve_pendulum_ivp(
            t,
            gamma_val,
            fixed_L,
            theta_0=theta_0,
            g=g,
        )

    popt, pcov = curve_fit(
        f=model_wrapper,
        xdata=t_data,
        ydata=theta_data,
        p0=[p0_gamma],
        bounds=([bounds[0]], [bounds[1]]),
    )

    gamma_fit = float(popt[0])
    gamma_err = float(np.sqrt(pcov[0, 0])) if pcov is not None else 0.0
    y_model = model_wrapper(t_data, gamma_fit)
    residuals = theta_data - y_model
    sse = float(np.sum(residuals ** 2))
    mse = float(np.mean(residuals ** 2))

    return {
        "method": "Curve Fit (1-param)",
        "gamma_fit": gamma_fit,
        "gamma_err": gamma_err,
        "fixed_L": fixed_L,
        "L_fit": fixed_L,
        "L_err": 0.0,
        "pcov": pcov,
        "y_model": y_model,
        "residuals": residuals,
        "sse": sse,
        "mse": mse,
    }


def fit_curve_fit_2param(
    t_data: np.ndarray,
    theta_data: np.ndarray,
    p0: Optional[List[float]] = None,
    bounds: Optional[Tuple[List[float], List[float]]] = None,
    g: float = G_ACCEL,
    L_data: float = L_DATA,
) -> Dict[str, Any]:
    """
    Simultaneously estimates both gamma (damping) and L (length) using curve_fit and solve_ivp.
    Direct implementation of Project_PoC.ipynb (Cell 21).

    Parameters
    ----------
    t_data : np.ndarray
        Observation times (s).
    theta_data : np.ndarray
        Observed angular positions (rad).
    p0 : list of float, optional
        Initial guesses [gamma_0, L_0]. Default is [0.05, 1.615].
    bounds : tuple of lists, optional
        ([lower_gamma, lower_L], [upper_gamma, upper_L]). Default is ([0.0, 1.0], [1.0, 2.0]).
    g : float
        Gravitational acceleration.
    L_data : float
        Reference measured length (m).

    Returns
    -------
    dict
        Optimal parameters, standard errors, covariance, correlation, residuals, and MSE.
    """
    if p0 is None:
        p0 = [0.05, L_DATA]
    if bounds is None:
        bounds = ([0.0, 1.0], [1.0, 2.0])

    theta_0 = float(theta_data[0])

    def model_wrapper(t, gamma_val, L_val):
        return solve_pendulum_ivp(
            t,
            gamma_val,
            L_val,
            theta_0=theta_0,
            g=g,
        )

    popt, pcov = curve_fit(
        f=model_wrapper,
        xdata=t_data,
        ydata=theta_data,
        p0=p0,
        bounds=bounds,
    )

    gamma_fit, L_fit = float(popt[0]), float(popt[1])
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else [0.0, 0.0]
    gamma_err, L_err = float(perr[0]), float(perr[1])

    # Parameter correlation
    if pcov is not None and pcov[0, 0] > 0 and pcov[1, 1] > 0:
        param_corr = float(pcov[0, 1] / np.sqrt(pcov[0, 0] * pcov[1, 1]))
    else:
        param_corr = 0.0

    y_model = model_wrapper(t_data, gamma_fit, L_fit)
    residuals = theta_data - y_model
    sse = float(np.sum(residuals ** 2))
    mse = float(np.mean(residuals ** 2))
    L_error_cm = float(abs(L_fit - L_data) * 100.0)

    return {
        "method": "Curve Fit (2-param)",
        "gamma_fit": gamma_fit,
        "gamma_err": gamma_err,
        "L_fit": L_fit,
        "L_err": L_err,
        "L_error_cm": L_error_cm,
        "pcov": pcov,
        "param_correlation": param_corr,
        "y_model": y_model,
        "residuals": residuals,
        "sse": sse,
        "mse": mse,
    }


def coupling_scan_curve_fit(
    t_data: np.ndarray,
    theta_data: np.ndarray,
    L_values: Optional[np.ndarray] = None,
    g: float = G_ACCEL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits gamma across a sequence of fixed L values using 1-param curve_fit.
    Direct implementation of Project_PoC.ipynb (Cell 26).

    Parameters
    ----------
    t_data : np.ndarray
        Observation times.
    theta_data : np.ndarray
        Observation angles.
    L_values : np.ndarray, optional
        Array of length values. Defaults to L_SCAN_VALUES.
    g : float
        Gravitational acceleration.

    Returns
    -------
    L_values, gamma_values, gamma_errors, mse_values
    """
    if L_values is None:
        L_values = L_SCAN_VALUES

    gamma_fits = []
    gamma_errs = []
    mse_fits = []

    theta_0 = float(theta_data[0])

    for L_val in L_values:
        def model_fixed_L(t, g_param):
            return solve_pendulum_ivp(t, g_param, L_val, theta_0=theta_0, g=g)

        try:
            popt, pcov = curve_fit(
                f=model_fixed_L,
                xdata=t_data,
                ydata=theta_data,
                p0=[0.05],
                bounds=([0.0], [1.0]),
            )
            g_opt = float(popt[0])
            g_std = float(np.sqrt(pcov[0, 0])) if pcov is not None else 0.0
            pred = model_fixed_L(t_data, g_opt)
            mse_val = float(np.mean((theta_data - pred) ** 2))
        except Exception:
            g_opt = np.nan
            g_std = np.nan
            mse_val = np.nan

        gamma_fits.append(g_opt)
        gamma_errs.append(g_std)
        mse_fits.append(mse_val)

    return (
        np.array(L_values),
        np.array(gamma_fits),
        np.array(gamma_errs),
        np.array(mse_fits),
    )


def compute_sse_surface_curve_fit(
    t_data: np.ndarray,
    theta_data: np.ndarray,
    gamma_values: Optional[np.ndarray] = None,
    L_values: Optional[np.ndarray] = None,
    n_grid: int = GRID_RESOLUTION,
    g: float = G_ACCEL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Calculates the 2D SSE surface over a grid of (gamma, L) values:
        SSE(gamma, L) = sum_{i=1}^N ( theta_data[i] - theta_model(t[i]; gamma, L) )^2
    Direct implementation of Project_PoC.ipynb (Cell 30).

    Returns
    -------
    gamma_grid, L_grid, sse_matrix, best_gamma, best_L, min_sse
    """
    if gamma_values is None:
        gamma_values = np.linspace(GAMMA_MIN, GAMMA_MAX, n_grid)
    if L_values is None:
        L_values = np.linspace(L_MIN, L_MAX, n_grid)

    theta_0 = float(theta_data[0])
    sse_matrix = np.empty((len(gamma_values), len(L_values)), dtype=float)

    for i, g_val in enumerate(gamma_values):
        for j, l_val in enumerate(L_values):
            try:
                pred = solve_pendulum_ivp(
                    t_data,
                    g_val,
                    l_val,
                    theta_0=theta_0,
                    g=g,
                )
                sse_matrix[i, j] = np.sum((theta_data - pred) ** 2)
            except Exception:
                sse_matrix[i, j] = np.nan

    min_idx = np.unravel_index(np.nanargmin(sse_matrix), sse_matrix.shape)
    best_gamma = float(gamma_values[min_idx[0]])
    best_L = float(L_values[min_idx[1]])
    min_sse = float(sse_matrix[min_idx])

    return gamma_values, L_values, sse_matrix, best_gamma, best_L, min_sse


def run_curve_fit_analysis(
    t_data: np.ndarray,
    theta_data: np.ndarray,
    run_scans: bool = True,
    n_grid: int = 50,
) -> Dict[str, Any]:
    """
    Executes the complete Non-Linear Curve Fit parameter estimation and coupling analysis pipeline.
    """
    fit_1p = fit_curve_fit_1param(t_data, theta_data)
    fit_2p = fit_curve_fit_2param(t_data, theta_data)

    results = {
        "fit_1param": fit_1p,
        "fit_2param": fit_2p,
    }

    if run_scans:
        L_vals, g_vals, g_errs, mse_vals = coupling_scan_curve_fit(t_data, theta_data)
        results["coupling_scan"] = {
            "L_values": L_vals,
            "gamma_values": g_vals,
            "gamma_errors": g_errs,
            "mse_values": mse_vals,
        }

        g_grid, L_grid, sse_mat, b_gamma, b_L, min_sse = compute_sse_surface_curve_fit(
            t_data, theta_data, n_grid=n_grid
        )
        results["sse_surface"] = {
            "gamma_grid": g_grid,
            "L_grid": L_grid,
            "sse_matrix": sse_mat,
            "best_gamma": b_gamma,
            "best_L": b_L,
            "min_sse": min_sse,
        }

    return results
