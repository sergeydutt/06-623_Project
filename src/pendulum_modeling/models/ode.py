"""
Physical Ordinary Differential Equation (ODE) models for the simple pendulum.

Directly implements the ODE formulation and numerical integration via
scipy.integrate.solve_ivp (Project_PoC.ipynb, Cells 15-16).
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from pendulum_modeling.config import G_ACCEL, L_DATA


def pendulum_model(
    t: float,
    y: List[float],
    gamma: float,
    L: float = L_DATA,
    g: float = G_ACCEL,
) -> List[float]:
    """
    Standard model for a simple pendulum subject to linear air resistance:
        d^2(theta)/dt^2 + gamma * d(theta)/dt + (g / L) * sin(theta) = 0

    Turned into a system of two coupled first-order ODEs:
        d(theta)/dt = omega
        d(omega)/dt = -gamma * omega - (g / L) * sin(theta)

    Parameters
    ----------
    t : float
        Current time point (s).
    y : list of float
        State vector [theta, omega].
    gamma : float
        Damping parameter (s^-1).
    L : float
        Pendulum length (m). Default is 1.615.
    g : float
        Acceleration due to gravity (m/s^2). Default is 9.81.

    Returns
    -------
    list of float
        [dtheta_dt, domega_dt]
    """
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -gamma * omega - (g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]


def solve_pendulum_ivp(
    t_eval: np.ndarray,
    *params,
    gamma: Optional[float] = None,
    L: Optional[float] = None,
    theta_0: Optional[float] = None,
    omega_0: float = 0.0,
    g: float = G_ACCEL,
    method: str = "RK45",
) -> np.ndarray:
    """
    Solves the pendulum ODE initial value problem numerically using solve_ivp.
    Directly compatible with scipy.optimize.curve_fit.

    Parameters
    ----------
    t_eval : np.ndarray
        Array of time evaluation points.
    *params : tuple
        Variable arguments: (gamma,) for 1-param or (gamma, L) for 2-param fit.
    gamma : float, optional
        Damping parameter (s^-1).
    L : float, optional
        Pendulum length (m). Default is 1.615.
    theta_0 : float, optional
        Initial angular displacement (rad). If None, defaults to 0.2138.
    omega_0 : float
        Initial angular velocity (rad/s). Defaults to 0.0 (released from rest).
    g : float
        Gravitational acceleration (m/s^2).
    method : str
        ODE solver method (default 'RK45').

    Returns
    -------
    np.ndarray
        Simulated angular displacement values theta(t).
    """
    if gamma is None:
        if len(params) >= 1:
            gamma = params[0]
        else:
            raise ValueError("At least damping parameter gamma must be provided.")

    if L is None:
        if len(params) >= 2:
            L = params[1]
        else:
            L = L_DATA

    if theta_0 is None:
        if len(params) >= 3:
            theta_0 = params[2]
        else:
            theta_0 = 0.2138

    y0 = [float(theta_0), float(omega_0)]

    sol = solve_ivp(
        fun=pendulum_model,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=y0,
        t_eval=t_eval,
        args=(float(gamma), float(L), float(g)),
        method=method,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    return sol.y[0]


def solve_pendulum_ivp_a(
    t_eval: np.ndarray,
    a1: float,
    a2: float,
    theta_0: Optional[float] = None,
    omega_0: float = 0.0,
    method: str = "RK45",
) -> np.ndarray:
    """
    Numerically integrates the pendulum ODE parameterized directly by
    linear regression parameters a1 (damping) and a2 = g/L.

    Equations:
        d(theta)/dt = omega
        d(omega)/dt = -a1 * omega - a2 * sin(theta)

    Parameters
    ----------
    t_eval : np.ndarray
        Evaluation times.
    a1 : float
        Damping parameter a1 = c / (m * L^2) (s^-1).
    a2 : float
        Restoring parameter a2 = g / L (s^-2).
    theta_0 : float, optional
        Initial angle. If None, defaults to 0.0.
    omega_0 : float
        Initial velocity.
    method : str
        Solver method.

    Returns
    -------
    np.ndarray
        Simulated theta(t).
    """
    if theta_0 is None:
        theta_0 = 0.0

    def ode_system(t, y):
        th, om = y
        return [om, -a1 * om - a2 * np.sin(th)]

    sol = solve_ivp(
        fun=ode_system,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=[float(theta_0), float(omega_0)],
        t_eval=t_eval,
        method=method,
    )
    return sol.y[0]
