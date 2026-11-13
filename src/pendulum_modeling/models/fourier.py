"""
Analytical Damped Fourier Series and JAX autodiff differentiation.

Directly implements the continuous harmonic trajectory formulation from
Project_Final.ipynb (Cells 11, 12, 14):
1. Damped Fourier series with bounded scalar frequency optimization at each harmonic K.
2. JAX autodiff for exact velocity and acceleration time derivatives.
3. Continuous trajectory evaluation.
"""

from typing import Callable, Tuple
import numpy as np
import scipy.signal as signal
from scipy.optimize import minimize_scalar
import jax
import jax.numpy as jnp

from pendulum_modeling.config import K_MAX_DEFAULT, TOLERANCE_DEFAULT


def series(
    t_data: np.ndarray,
    theta_data: np.ndarray,
    K_max: int = K_MAX_DEFAULT,
    tol: float = TOLERANCE_DEFAULT,
) -> Tuple[np.ndarray, float, float, int]:
    """
    Fits an exponentially damped Fourier series to discrete trajectory data,
    optimizing fundamental frequency at each harmonic expansion step.
    Direct implementation of the 'series' function from Project_Final.ipynb (Cell 11).

    Parameters
    ----------
    t_data : np.ndarray
        Time points (s), normalized to t_0 = 0.
    theta_data : np.ndarray
        Angle points (rad).
    K_max : int
        Maximum number of harmonic terms.
    tol : float
        Stopping tolerance on relative MSE improvement.

    Returns
    -------
    a : np.ndarray
        Fitted Fourier coefficients [A_1, B_1, A_2, B_2, ..., A_K, B_K].
    Lambda : float
        Decay constant (s^-1).
    freq_opt : float
        Optimized fundamental angular frequency (rad/s).
    K_total : int
        Total number of harmonic pairs selected.
    """
    peaks, _ = signal.find_peaks(theta_data)

    t_diff = t_data[peaks[-1]] - t_data[peaks[0]]
    T_avg = t_diff / (len(peaks) - 1)
    freq_guess = 2 * np.pi / T_avg

    first_peak = theta_data[peaks[0]]
    last_peak = theta_data[peaks[-1]]
    Lambda = np.log(first_peak / last_peak) / t_diff
    decay = np.exp(-Lambda * t_data)

    freq_opt = freq_guess
    prev_MSE = float('inf')

    for K in range(1, K_max + 1):
        def opt(f):
            cols = []
            for k in range(1, K + 1):
                cols.append(decay * np.cos(k * f * t_data))
                cols.append(decay * np.sin(k * f * t_data))
            X_temp = np.column_stack(cols)
            a_temp = np.linalg.solve(X_temp.T @ X_temp, X_temp.T @ theta_data)
            theta_model_temp = X_temp @ a_temp
            return np.mean((theta_data - theta_model_temp) ** 2)

        res = minimize_scalar(
            opt,
            bounds=(freq_opt * 0.95, freq_opt * 1.05),
            method='bounded',
        )
        freq_opt = res.x

        cols = []
        for k in range(1, K + 1):
            cols.append(decay * np.cos(k * freq_opt * t_data))
            cols.append(decay * np.sin(k * freq_opt * t_data))

        X_k = np.column_stack(cols)
        Y_k = theta_data

        a_k = np.linalg.solve(X_k.T @ X_k, X_k.T @ Y_k)
        theta_model = X_k @ a_k
        MSE = np.mean((theta_data - theta_model) ** 2)

        if K > 1:
            rel_improvement = (prev_MSE - MSE) / prev_MSE
            if rel_improvement < tol:
                break

        a = a_k
        K_total = K
        prev_MSE = MSE

    return a, float(Lambda), float(freq_opt), K_total


def build_theta_func(
    a_coefficients: np.ndarray,
    Lambda: float,
    freq: float,
    k_total: int,
) -> Callable[[float], jnp.ndarray]:
    """
    Constructs the analytical, differentiable JAX trajectory function theta_func(t).
    Direct implementation of theta_func from Project_Final.ipynb (Cell 12).
    """
    a = jnp.array(a_coefficients)

    def theta_func(t_m):
        decay_t = jnp.exp(-Lambda * t_m)
        terms = jnp.array([
            a[2 * (k - 1)] * jnp.cos(k * freq * t_m) +
            a[2 * (k - 1) + 1] * jnp.sin(k * freq * t_m)
            for k in range(1, k_total + 1)
        ])
        return decay_t * jnp.sum(terms)

    return theta_func


def get_derivatives(
    theta_func: Callable[[float], jnp.ndarray],
    t_eval: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates analytical time derivatives (velocity theta_dot and acceleration theta_ddot)
    using JAX automatic differentiation.
    Direct implementation of get_derivatives from Project_Final.ipynb (Cell 14).
    """
    dtheta_dt_fun = jax.vmap(jax.grad(theta_func))
    d2theta_dt2_fun = jax.vmap(jax.grad(jax.grad(theta_func)))

    t_jax = jnp.array(t_eval)
    theta_dot = np.array(dtheta_dt_fun(t_jax))
    theta_ddot = np.array(d2theta_dt2_fun(t_jax))

    return theta_dot, theta_ddot


def evaluate_continuous_theta(
    theta_func: Callable[[float], jnp.ndarray],
    t_eval: np.ndarray,
) -> np.ndarray:
    """
    Evaluates continuous theta_func across dense evaluation points.
    """
    eval_vmap = jax.vmap(theta_func)
    return np.array(eval_vmap(jnp.array(t_eval)))
