"""
Temporal Generalization and Overfitting Validation Analysis for Pendulum Dynamics.

Captures the analysis workflow from overfitting_check.py:
1. Multi-cycle temporal splitting across oscillation peaks (find_peaks).
2. Damped harmonic Fourier representation and analytical differentiation via JAX autograd.
3. Ordinary least-squares regression to identify non-linear ODE parameters (a1, a2).
4. Forward numerical integration (scipy.integrate.solve_ivp) across the full time domain.
5. Residual analysis, in-sample Train RMSE, and out-of-sample Forecast RMSE in degrees.
6. Encapsulation of all data, parameters, trajectories, and metrics into higher-order data objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.integrate import solve_ivp

from pendulum_modeling.config import G_ACCEL, K_MAX_DEFAULT, M_DATA, TOLERANCE_DEFAULT

# ==============================================================================
# Higher-Order Data Objects (Dataclasses)
# ==============================================================================


@dataclass
class TemporalSplitData:
    """
    Higher-order data container for partitioned time-series observations.

    Encapsulates training observations (in-sample) and unseen forecast observations
    (out-of-sample) partitioned at a specific cycle/peak boundary.

    Attributes
    ----------
    t_data : np.ndarray
        Time points (s) for the in-sample training segment.
    theta_data_deg : np.ndarray
        Observed angular displacement (degrees) for the training segment.
    t_test : np.ndarray
        Time points (s) for the out-of-sample unseen testing segment.
    theta_test_deg : np.ndarray
        Observed angular displacement (degrees) for the unseen testing segment.
    t_split : float
        Timestamp (s) dividing the training and testing regimes.
    train_peaks : int | None
        Number of oscillation peaks contained in the training window (e.g., 10).
    test_peaks : int | None
        Number of oscillation peaks contained in the testing window (e.g., 10).
    """

    t_data: np.ndarray
    theta_data_deg: np.ndarray
    t_test: np.ndarray
    theta_test_deg: np.ndarray
    t_split: float
    train_peaks: int | None = None
    test_peaks: int | None = None

    @classmethod
    def from_radians(
        cls,
        t_data: np.ndarray,
        theta_data_rad: np.ndarray,
        t_test: np.ndarray,
        theta_test_rad: np.ndarray,
        t_split: float | None = None,
        train_peaks: int | None = None,
        test_peaks: int | None = None,
    ) -> TemporalSplitData:
        """
        Constructs TemporalSplitData, automatically converting angle arrays
        from radians to degrees for standardized evaluation and plotting.
        """
        split_time = float(t_test[0]) if t_split is None else float(t_split)
        return cls(
            t_data=np.asarray(t_data, dtype=float),
            theta_data_deg=np.degrees(np.asarray(theta_data_rad, dtype=float)),
            t_test=np.asarray(t_test, dtype=float),
            theta_test_deg=np.degrees(np.asarray(theta_test_rad, dtype=float)),
            t_split=split_time,
            train_peaks=train_peaks,
            test_peaks=test_peaks,
        )


@dataclass
class ODEParameters:
    """
    Higher-order data container for identified pendulum ODE physical parameters.

    Governing physics equation:
        ddot(theta) = -a1 * dot(theta) - a2 * sin(theta)
    where:
        a1 = c / (m * L^2)  [damping term]
        a2 = g / L          [gravitational restoring torque term]

    Attributes
    ----------
    a1 : float
        Identified damping parameter (s^-1).
    a2 : float
        Identified restoring torque parameter (s^-2).
    L_model : float | None
        Derived pendulum length L = g / a2 (m).
    C_model : float | None
        Derived air-resistance coefficient C = a1 * m * L^2 (kg*m^2/s).
    """

    a1: float
    a2: float
    L_model: float | None = None
    C_model: float | None = None

    def formatted_label(self) -> str:
        """Returns LaTeX-formatted parameter string for plot legends."""
        return rf"ODE ($a_1={self.a1:.4f}, a_2={self.a2:.4f}$)"


@dataclass
class SimulationTrajectory:
    """
    Higher-order data container for numerical ODE rollout trajectories.

    Attributes
    ----------
    t_sim : np.ndarray
        Evaluation timepoints (s) covering the complete forward simulation.
    theta_sim_deg : np.ndarray
        Simulated angular trajectory across the full time domain (degrees).
    theta_sim_train_deg : np.ndarray
        Simulated trajectory evaluated at in-sample training timepoints (degrees).
    theta_sim_test_deg : np.ndarray
        Simulated trajectory evaluated at out-of-sample testing timepoints (degrees).
    """

    t_sim: np.ndarray
    theta_sim_deg: np.ndarray
    theta_sim_train_deg: np.ndarray
    theta_sim_test_deg: np.ndarray


@dataclass
class GeneralizationMetrics:
    """
    Higher-order data container for temporal forecast error metrics and residuals.

    Attributes
    ----------
    train_rmse_deg : float
        In-sample Root Mean Square Error (degrees) over the training horizon.
    test_rmse_deg : float
        Out-of-sample Forecast Root Mean Square Error (degrees) over the unseen horizon.
    error_train_deg : np.ndarray
        Pointwise angular error residuals on training window (actual - simulated) in degrees.
    error_test_deg : np.ndarray
        Pointwise angular error residuals on testing window (actual - simulated) in degrees.
    """

    train_rmse_deg: float
    test_rmse_deg: float
    error_train_deg: np.ndarray
    error_test_deg: np.ndarray


@dataclass
class TemporalGeneralizationResult:
    """
    Comprehensive aggregated container bundling the complete overfitting
    and temporal generalization analysis results.

    Attributes
    ----------
    split_data : TemporalSplitData
        Observed partitioned time and angle series.
    ode_params : ODEParameters
        Identified physical equation parameters.
    simulation : SimulationTrajectory
        Simulated trajectory covering both regimes.
    metrics : GeneralizationMetrics
        In-sample and out-of-sample forecast accuracy metrics.
    """

    split_data: TemporalSplitData
    ode_params: ODEParameters
    simulation: SimulationTrajectory
    metrics: GeneralizationMetrics


@dataclass
class GeneralizationPlotConfig:
    """
    Configuration options for rendering temporal generalization diagnostic figures.

    Attributes
    ----------
    title : str | None
        Figure title. If None, dynamically constructed from peak counts.
    train_label : str | None
        Legend label for training observations.
    test_label : str | None
        Legend label for test observations.
    ode_label : str | None
        Legend label for the simulated ODE trajectory.
    figsize : tuple[float, float]
        Figure size in inches (width, height). Default is (10, 6).
    dpi : int
        Figure DPI resolution. Default is 100.
    save_path : str | None
        Optional file path to persist the generated figure.
    """

    title: str | None = None
    train_label: str | None = None
    test_label: str | None = None
    ode_label: str | None = None
    figsize: tuple[float, float] = (10, 6)
    dpi: int = 100
    save_path: str | None = None


# ==============================================================================
# Analytical Workflow Functions
# ==============================================================================


def split_temporal_peaks(
    t_full: np.ndarray,
    theta_full: np.ndarray,
    train_peaks: int = 10,
    total_peaks: int = 20,
    in_degrees: bool = False,
) -> TemporalSplitData:
    """
    Partitions a continuous oscillatory time-series into in-sample training and
    out-of-sample unseen testing segments based on detected cycle peaks.

    Context (from overfitting_check.py lines 38-47):
    -------------------------------------------------
    Dynamical system validation requires chronological (temporal) splitting rather
    than random shuffle cross-validation, because random sampling leaks future state
    information and destroys phase continuity. By splitting at the N-th peak, the
    model is trained on initial oscillations (e.g., peaks 1–10) and tasked with
    forecasting subsequent oscillations (e.g., peaks 11–20).

    Parameters
    ----------
    t_full : np.ndarray
        Full time series array (s), normalized to t[0] = 0.
    theta_full : np.ndarray
        Full angular displacement array. Assumed to be in radians unless `in_degrees=True`.
    train_peaks : int, default=10
        Number of oscillation peaks used for the training subset.
    total_peaks : int, default=20
        Total number of oscillation peaks in the evaluation window.
    in_degrees : bool, default=False
        Set to True if `theta_full` is already provided in degrees.

    Returns
    -------
    TemporalSplitData
        Data container holding partitioned training and testing arrays in degrees.
    """
    theta_arr = np.asarray(theta_full, dtype=float)
    t_arr = np.asarray(t_full, dtype=float)

    # Detect local maxima across the trajectory
    peaks, _ = signal.find_peaks(theta_arr)
    if len(peaks) <= train_peaks:
        raise ValueError(
            f"Found only {len(peaks)} peaks, but at least {train_peaks + 1} are required "
            f"for a train split of {train_peaks} peaks."
        )

    split_idx = peaks[train_peaks]
    end_idx = peaks[total_peaks] if len(peaks) > total_peaks else len(t_arr)

    t_data = t_arr[:split_idx]
    theta_data_raw = theta_arr[:split_idx]

    t_test = t_arr[split_idx:end_idx]
    theta_test_raw = theta_arr[split_idx:end_idx]

    t_split = float(t_arr[split_idx])
    actual_test_peaks = min(total_peaks - train_peaks, len(peaks) - train_peaks)

    if in_degrees:
        return TemporalSplitData(
            t_data=t_data,
            theta_data_deg=theta_data_raw,
            t_test=t_test,
            theta_test_deg=theta_test_raw,
            t_split=t_split,
            train_peaks=train_peaks,
            test_peaks=actual_test_peaks,
        )

    return TemporalSplitData.from_radians(
        t_data=t_data,
        theta_data_rad=theta_data_raw,
        t_test=t_test,
        theta_test_rad=theta_test_raw,
        t_split=t_split,
        train_peaks=train_peaks,
        test_peaks=actual_test_peaks,
    )


def compute_generalization_metrics(
    split_data: TemporalSplitData,
    t_sim: np.ndarray,
    theta_sim: np.ndarray,
    in_degrees: bool = False,
) -> tuple[SimulationTrajectory, GeneralizationMetrics]:
    """
    Computes pointwise residual error trajectories and Root Mean Square Error (RMSE)
    separately for the training window and the unseen forecast window.

    Context (from overfitting_check.py lines 205-214):
    -------------------------------------------------
    Compares the forward-integrated ODE rollout against measured data points.
    If the model is overfitted to in-sample noise or damping artifacts, Test RMSE
    will diverge significantly from Train RMSE.

    Parameters
    ----------
    split_data : TemporalSplitData
        Partitioned training and testing observation data (angles in degrees).
    t_sim : np.ndarray
        Time points corresponding to the simulated trajectory (s).
    theta_sim : np.ndarray
        Simulated angle trajectory. Assumed radians unless `in_degrees=True`.
    in_degrees : bool, default=False
        Set to True if `theta_sim` is already in degrees.

    Returns
    -------
    tuple[SimulationTrajectory, GeneralizationMetrics]
        Packaged simulation trajectory and quantitative metrics objects.
    """
    theta_sim_deg = (
        np.asarray(theta_sim, dtype=float)
        if in_degrees
        else np.degrees(np.asarray(theta_sim, dtype=float))
    )
    t_sim_arr = np.asarray(t_sim, dtype=float)

    n_train = len(split_data.t_data)
    n_test = len(split_data.t_test)

    # Fast slicing if t_sim exactly matches the concatenation of train and test time
    if (
        len(t_sim_arr) == n_train + n_test
        and np.allclose(t_sim_arr[:n_train], split_data.t_data)
        and np.allclose(t_sim_arr[n_train:], split_data.t_test)
    ):
        theta_sim_train_deg = theta_sim_deg[:n_train]
        theta_sim_test_deg = theta_sim_deg[n_train:]
    else:
        # Interpolate onto measurement timestamps if sampling grids differ
        theta_sim_train_deg = np.interp(split_data.t_data, t_sim_arr, theta_sim_deg)
        theta_sim_test_deg = np.interp(split_data.t_test, t_sim_arr, theta_sim_deg)

    # Pointwise residuals in degrees (actual - predicted)
    error_train_deg = split_data.theta_data_deg - theta_sim_train_deg
    error_test_deg = split_data.theta_test_deg - theta_sim_test_deg

    # Summary RMSE metrics
    train_rmse_deg = float(np.sqrt(np.mean(error_train_deg**2)))
    test_rmse_deg = float(np.sqrt(np.mean(error_test_deg**2)))

    simulation = SimulationTrajectory(
        t_sim=t_sim_arr,
        theta_sim_deg=theta_sim_deg,
        theta_sim_train_deg=theta_sim_train_deg,
        theta_sim_test_deg=theta_sim_test_deg,
    )

    metrics = GeneralizationMetrics(
        train_rmse_deg=train_rmse_deg,
        test_rmse_deg=test_rmse_deg,
        error_train_deg=error_train_deg,
        error_test_deg=error_test_deg,
    )

    return simulation, metrics


def evaluate_temporal_generalization(
    split_data: TemporalSplitData,
    t_sim: np.ndarray,
    theta_sim: np.ndarray,
    ode_params: ODEParameters,
    in_degrees: bool = False,
) -> TemporalGeneralizationResult:
    """
    Evaluates temporal generalization and packages results into a TemporalGeneralizationResult.

    Parameters
    ----------
    split_data : TemporalSplitData
        Partitioned observation data.
    t_sim : np.ndarray
        Simulation timepoints.
    theta_sim : np.ndarray
        Simulated angle trajectory.
    ode_params : ODEParameters
        Identified ODE parameters.
    in_degrees : bool, default=False
        Whether `theta_sim` is in degrees.

    Returns
    -------
    TemporalGeneralizationResult
        Aggregated result container.
    """
    simulation, metrics = compute_generalization_metrics(
        split_data=split_data,
        t_sim=t_sim,
        theta_sim=theta_sim,
        in_degrees=in_degrees,
    )
    return TemporalGeneralizationResult(
        split_data=split_data,
        ode_params=ode_params,
        simulation=simulation,
        metrics=metrics,
    )


def run_overfitting_analysis(
    t_full: np.ndarray,
    theta_full: np.ndarray,
    train_peaks: int = 10,
    total_peaks: int = 20,
    g: float = G_ACCEL,
    m_data: float = M_DATA,
    k_max: int = K_MAX_DEFAULT,
    tol: float = TOLERANCE_DEFAULT,
) -> TemporalGeneralizationResult:
    """
    Executes the end-to-end overfitting and temporal generalization analysis pipeline
    directly reproducing overfitting_check.py:

    1. Splits full dataset chronologically into train (first train_peaks) and test segments.
    2. Fits damped multi-harmonic Fourier series to the training segment with frequency optimization.
    3. Derives continuous analytical velocity and acceleration via JAX autodiff.
    4. Performs linear regression against ddot(theta) = -a1*dot(theta) - a2*sin(theta).
    5. Numerically rolls out the identified ODE forward across the entire evaluation horizon.
    6. Computes in-sample and out-of-sample forecast accuracy metrics.
    7. Encapsulates all entities into higher-order data objects.

    Parameters
    ----------
    t_full : np.ndarray
        Zero-referenced time array (s).
    theta_full : np.ndarray
        Equilibrium-centered angle array (rad).
    train_peaks : int, default=10
        Number of oscillation peaks used for training.
    total_peaks : int, default=20
        Total number of oscillation peaks in the evaluation window.
    g : float, default=9.81
        Gravitational acceleration (m/s^2).
    m_data : float, default=1.0
        Pendulum bob mass (kg).
    k_max : int, default=10
        Maximum harmonics in Fourier fitting.
    tol : float, default=0.01
        Fourier harmonic convergence tolerance.

    Returns
    -------
    TemporalGeneralizationResult
        Complete aggregated container with split data, ODE parameters, simulation, and metrics.
    """
    # Step 1: Temporal Peak Split
    split_data = split_temporal_peaks(
        t_full=t_full,
        theta_full=theta_full,
        train_peaks=train_peaks,
        total_peaks=total_peaks,
        in_degrees=False,
    )

    # Convert training angles back to radians for mathematical modeling
    t_data = split_data.t_data
    theta_data_rad = np.radians(split_data.theta_data_deg)

    # Step 2: Damped Fourier Series Representation
    # Envelope decay constant Lambda
    peaks, _ = signal.find_peaks(theta_data_rad)
    t_diff = t_data[peaks[-1]] - t_data[peaks[0]]
    avg_period = t_diff / (len(peaks) - 1)
    freq_guess = 2 * np.pi / avg_period

    first_peak = theta_data_rad[peaks[0]]
    last_peak = theta_data_rad[peaks[-1]]
    Lambda = float(np.log(first_peak / last_peak) / t_diff)
    decay = np.exp(-Lambda * t_data)

    def lin_reg(f: float, K: int):
        cols = []
        for k in range(1, K + 1):
            cols.append(decay * np.cos(k * f * t_data))
            cols.append(decay * np.sin(k * f * t_data))
        X = np.column_stack(cols)
        a = np.linalg.solve(X.T @ X, X.T @ theta_data_rad)
        theta_model = X @ a
        mse = float(np.mean((theta_data_rad - theta_model) ** 2))
        return a, mse

    from scipy.optimize import minimize_scalar

    res = minimize_scalar(
        lambda f: lin_reg(f, k_max)[1],
        bounds=(freq_guess * 0.95, freq_guess * 1.05),
        method="bounded",
    )
    freq_opt = float(res.x)

    prev_MSE = None
    a_fourier = None
    K_total = 1
    for K in range(1, k_max + 1):
        a_k, MSE = lin_reg(freq_opt, K)
        if K > 1 and prev_MSE is not None:
            rel_improvement = (prev_MSE - MSE) / prev_MSE
            if rel_improvement < tol:
                break
        a_fourier = a_k
        K_total = K
        prev_MSE = MSE

    # Step 3: Analytical JAX Autodiff Derivatives
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    a_fourier_jnp = jnp.array(a_fourier)

    def theta_func(t_m):
        decay_t = jnp.exp(-Lambda * t_m)
        terms = jnp.array([
            a_fourier_jnp[2 * (k - 1)] * jnp.cos(k * freq_opt * t_m)
            + a_fourier_jnp[2 * (k - 1) + 1] * jnp.sin(k * freq_opt * t_m)
            for k in range(1, K_total + 1)
        ])
        return decay_t * jnp.sum(terms)

    dtheta_dt_fn = jax.vmap(jax.grad(theta_func))
    d2theta_dt2_fn = jax.vmap(jax.grad(jax.grad(theta_func)))

    theta_dot = np.asarray(dtheta_dt_fn(jnp.asarray(t_data)))
    theta_ddot = np.asarray(d2theta_dt2_fn(jnp.asarray(t_data)))

    # Step 4: Linear Regression against Pendulum Equation
    # ddot(theta) = -a1 * dot(theta) - a2 * sin(theta)
    Y = theta_ddot
    X = np.column_stack([-theta_dot, -np.sin(theta_data_rad)])
    a_fit = np.linalg.solve(X.T @ X, X.T @ Y)
    a1 = float(a_fit[0])
    a2 = float(a_fit[1])

    L_model = float(g / a2) if a2 > 0 else None
    C_model = float(a1 * m_data * (L_model**2)) if L_model is not None else None

    ode_params = ODEParameters(
        a1=a1,
        a2=a2,
        L_model=L_model,
        C_model=C_model,
    )

    # Step 5: Forward ODE Numerical Integration (solve_ivp)
    theta_0 = float(theta_func(t_data[0]))
    omega_0 = float(theta_dot[0])
    y0 = [theta_0, omega_0]

    def pendulum_ode(t_val, y):
        th, om = y
        return [om, -a1 * om - a2 * np.sin(th)]

    t_eval_full = np.concatenate([split_data.t_data, split_data.t_test])
    sol = solve_ivp(
        fun=pendulum_ode,
        t_span=(float(t_eval_full[0]), float(t_eval_full[-1])),
        y0=y0,
        t_eval=t_eval_full,
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
    )

    theta_sim_rad = sol.y[0]

    # Step 6 & 7: Evaluate Metrics and Package
    return evaluate_temporal_generalization(
        split_data=split_data,
        t_sim=t_eval_full,
        theta_sim=theta_sim_rad,
        ode_params=ode_params,
        in_degrees=False,
    )
