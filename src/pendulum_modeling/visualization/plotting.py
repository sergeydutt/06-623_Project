"""
Reusable plotting functions for Pendulum Modeling.

Directly implements and extends visualizations from the project notebooks:
- Trajectory fit overlay (Project_PoC.ipynb Cell 17 / Project_Final.ipynb Cell 13).
- Residuals scatter with zero baseline (Project_Final.ipynb Cell 17).
- JAX autodiff velocity and acceleration derivatives.
- 2D & 3D parameter sensitivity curves across fixed L (Project_PoC.ipynb Cells 27, 28).
- Air-resistance coefficient C errorbar curve across L (Project_Final.ipynb Cell 20).
- Seaborn and contour error surface heatmaps (Project_PoC.ipynb Cell 30 / Project_Final.ipynb Cell 21).
- Side-by-side comparative diagnostics.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pendulum_modeling.config import PLOT_STYLE, PLOTS_DIR


def plot_model(
    time_data: pd.Series,
    angle_data: pd.Series,
    model_fun,
    fit_params,
    description: str = "",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Direct implementation of plot_model from Project_PoC.ipynb (Cell 17).
    Generates plot of model line and raw data points.
    """
    plt.close("all")
    t_model = np.linspace(time_data.iloc[0], time_data.iloc[-1], 500)

    if isinstance(fit_params, (list, tuple, np.ndarray)):
        params = tuple(fit_params)
    else:
        params = (fit_params,)

    try:
        theta_init = float(angle_data.iloc[0] if hasattr(angle_data, "iloc") else angle_data[0])
        y_model = model_fun(t_model, *params, theta_0=theta_init)
    except TypeError:
        y_model = model_fun(t_model, *params)

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_default"], dpi=PLOT_STYLE["dpi"])
    ax.plot(
        time_data,
        angle_data,
        "ko",
        markersize=3,
        label="Data",
    )
    ax.plot(t_model, y_model, color="crimson", linewidth=2, label="Model")

    ax.set_xlabel("Time (s)", fontsize=PLOT_STYLE["font_size"])
    ax.set_ylabel("Angle (rad)", fontsize=PLOT_STYLE["font_size"])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(f"Pendulum Angle vs Time for {description}", fontsize=PLOT_STYLE["font_size"] + 2)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_STYLE["dpi_high"], bbox_inches="tight")

    return fig, ax


def plot_fourier_fit(
    t_data: np.ndarray,
    theta_data: np.ndarray,
    t_plot: np.ndarray,
    theta_plot: np.ndarray,
    title: str = "Continuous Fourier Fit vs. Raw Data",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Direct reproduction of continuous Fourier fit plot from Project_Final.ipynb (Cell 13).
    """
    plt.close("all")
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_default"], dpi=PLOT_STYLE["dpi"])

    ax.plot(t_plot, theta_plot, label=r"Continuous $\theta(t)$", color="black", linewidth=2)
    ax.plot(t_data, theta_data, "ro", label="Data Points")

    ax.set_xlabel("Time (s)", fontsize=PLOT_STYLE["font_size"])
    ax.set_ylabel(r"Angle $\theta$ (rad)", fontsize=PLOT_STYLE["font_size"])
    ax.set_title(title, fontsize=PLOT_STYLE["font_size"] + 2)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_STYLE["dpi_high"], bbox_inches="tight")

    return fig, ax


def plot_residuals(
    t_data: np.ndarray,
    residuals: np.ndarray,
    title: str = "Residuals vs Time",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Direct reproduction of residual plot from Project_Final.ipynb (Cell 17).
    """
    plt.close("all")
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_default"], dpi=PLOT_STYLE["dpi"])

    ax.scatter(t_data, residuals, color=PLOT_STYLE["residual_color"])
    ax.axhline(y=0, color=PLOT_STYLE["baseline_color"], linestyle="dashed")

    ax.set_xlabel("Time (s)", fontsize=PLOT_STYLE["font_size"])
    ax.set_ylabel("Residuals", fontsize=PLOT_STYLE["font_size"])
    ax.set_title(title, fontsize=PLOT_STYLE["font_size"] + 2)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_STYLE["dpi_high"], bbox_inches="tight")

    return fig, ax


def plot_derivatives(
    t_data: np.ndarray,
    theta_dot: np.ndarray,
    theta_ddot: np.ndarray,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plots velocity and acceleration obtained from JAX automatic differentiation.
    """
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), dpi=PLOT_STYLE["dpi"], sharex=True)

    ax1.plot(t_data, theta_dot, color="#2b5c8f", linewidth=1.8)
    ax1.set_ylabel(r"$\dot{\theta}(t)$ (rad/s)", fontsize=PLOT_STYLE["font_size"])
    ax1.set_title("Analytical Velocity (Autodiff)", fontsize=PLOT_STYLE["font_size"])
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_data, theta_ddot, color="#d95f02", linewidth=1.8)
    ax2.set_xlabel("Time (s)", fontsize=PLOT_STYLE["font_size"])
    ax2.set_ylabel(r"$\ddot{\theta}(t)$ (rad/s$^2$)", fontsize=PLOT_STYLE["font_size"])
    ax2.set_title("Analytical Acceleration (Autodiff)", fontsize=PLOT_STYLE["font_size"])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_STYLE["dpi_high"], bbox_inches="tight")

    return fig, (ax1, ax2)


def plot_correlation_curve_2d(
    L_values: np.ndarray,
    gamma_errors: np.ndarray,
    L_fit: Optional[float] = None,
    gamma_err: Optional[float] = None,
    xlabel: str = "L (m)",
    ylabel: str = "Gamma errors",
    title: str = "Correlation curve between L and gamma",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Direct reproduction of 2D correlation plot from Project_PoC.ipynb (Cell 28).
    """
    plt.close("all")
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_default"], dpi=PLOT_STYLE["dpi"])

    ax.plot(L_values, gamma_errors, "k-", linewidth=2.0)
    if L_fit is not None and gamma_err is not None:
        ax.plot(L_fit, gamma_err, "ro", markersize=8, label="Optimal Fit")
        ax.legend(loc="upper right")

    ax.set_xlabel(xlabel, fontsize=PLOT_STYLE["font_size"])
    ax.set_ylabel(ylabel, fontsize=PLOT_STYLE["font_size"])
    ax.set_title(title, fontsize=PLOT_STYLE["font_size"] + 2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_STYLE["dpi_high"], bbox_inches="tight")

    return fig, ax


def plot_correlation_curve_3d(
    L_values: np.ndarray,
    gamma_values: np.ndarray,
    gamma_errors: np.ndarray,
    L_fit: float,
    gamma_fit: float,
    gamma_err: float,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Direct reproduction of 3D trajectory plot from Project_PoC.ipynb (Cell 27).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    plt.close("all")
    fig = plt.figure(figsize=(9, 7), dpi=PLOT_STYLE["dpi"])
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(L_values, gamma_values, gamma_errors, color="black", linewidth=2)
    ax.scatter([L_fit], [gamma_fit], [gamma_err], color="red", s=70, depthshade=False)

    ax.set_xlabel("L (m)", labelpad=10)
    ax.set_ylabel("Optimized Gamma (s^-1)", labelpad=10)
    ax.set_zlabel("Gamma Error (s^-1)", labelpad=10)
    ax.view_init(elev=25, azim=-50)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_STYLE["dpi_high"], bbox_inches="tight")

    return fig, ax


def plot_linear_regression_coupling(
    L_values: np.ndarray,
    C_values: np.ndarray,
    C_errors: np.ndarray,
    L_model: float,
    C_model: float,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Direct reproduction of C(L) errorbar curve from Project_Final.ipynb (Cell 20).
    """
    plt.close("all")
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_default"], dpi=PLOT_STYLE["dpi"])

    ax.errorbar(
        L_values,
        C_values,
        yerr=C_errors,
        fmt="-o",
        color="darkgreen",
        ecolor="lightgreen",
        elinewidth=2,
        capsize=3,
        markersize=4,
        label=r"Estimated $C(L)$",
    )
    ax.scatter(L_model, C_model, color="red", s=80, zorder=5, label=f"Optimal ($L = {L_model:.3f}$ m)")

    ax.set_xlabel("Pendulum Length, L (m)")
    ax.set_ylabel(r"Air-resistance coefficient, $C$")
    ax.set_title(r"Air-resistance coefficient $C$ Across $L$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_STYLE["dpi_high"], bbox_inches="tight")

    return fig, ax


def plot_heatmap(
    y_values: np.ndarray,
    x_values: np.ndarray,
    loss_matrix: np.ndarray,
    best_index: Tuple[int, int],
    xlabel: str = "Pendulum length, L (m)",
    ylabel: str = r"Damping coefficient, $\gamma$ ($s^{-1}$)",
    cbar_label: str = r"$\log_{10}(\mathrm{SSE})$",
    title: str = "Parameter-coupling error surface",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Direct reproduction of seaborn error surface heatmap from Project_PoC.ipynb (Cell 30)
    and Project_Final.ipynb (Cell 21).
    """
    plt.close("all")

    error_surface = pd.DataFrame(
        loss_matrix,
        index=np.round(y_values, 4),
        columns=np.round(x_values, 3),
    )
    log_error_surface = np.log10(error_surface)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=PLOT_STYLE["dpi"])

    sns.heatmap(
        log_error_surface,
        cmap="viridis",
        cbar_kws={"label": cbar_label},
        xticklabels=10,
        yticklabels=10,
        ax=ax,
    )

    best_row = best_index[0] + 0.5
    best_col = best_index[1] + 0.5
    ax.scatter(best_col, best_row, color="red", marker="x", s=120, linewidths=3, label="Optimal")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_STYLE["dpi_high"], bbox_inches="tight")

    return fig, ax


def plot_side_by_side_correlation(
    cf_L: np.ndarray,
    cf_gamma: np.ndarray,
    cf_err: np.ndarray,
    lr_L: np.ndarray,
    lr_C: np.ndarray,
    lr_C_err: np.ndarray,
    lr_L_opt: Optional[float] = None,
    lr_C_opt: Optional[float] = None,
    title: str = "Parameter Sensitivity to Fixed Length L: Curve Fit vs. Linear Regression",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Side-by-side comparison of parameter sensitivity across assumed L.
    Left: Curve Fit gamma(L) (PoC Cell 26/28).
    Right: Linear Regression C(L) with errorbars (Final Cell 20).
    """
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5), dpi=PLOT_STYLE["dpi"], sharex=True)

    ax1.plot(cf_L, cf_gamma, color="black", linewidth=2.0)
    ax1.fill_between(cf_L, cf_gamma - cf_err, cf_gamma + cf_err, color="gray", alpha=0.3)
    ax1.set_xlabel("Fixed Pendulum Length L (m)")
    ax1.set_ylabel(r"Optimized $\gamma$ (s$^{-1}$)")
    ax1.set_title("Curve Fit: Parameter Sensitivity across L")
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(
        lr_L, lr_C, yerr=lr_C_err,
        fmt="-o", color="darkgreen", ecolor="lightgreen", elinewidth=2, capsize=3, markersize=4,
        label=r"Estimated $C(L)$"
    )
    if lr_L_opt is not None and lr_C_opt is not None:
        ax2.scatter(lr_L_opt, lr_C_opt, color="red", s=80, zorder=5, label=f"Optimal ($L = {lr_L_opt:.3f}$ m)")
        ax2.legend()
    ax2.set_xlabel("Fixed Pendulum Length L (m)")
    ax2.set_ylabel(r"Air-resistance coefficient, $C$")
    ax2.set_title("Linear Regression: Air-Resistance C across L")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, y=1.02, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_STYLE["dpi_high"], bbox_inches="tight")

    return fig, (ax1, ax2)


def plot_side_by_side_heatmaps(
    cf_gamma: np.ndarray,
    cf_L: np.ndarray,
    cf_sse: np.ndarray,
    cf_best_gamma: float,
    cf_best_L: float,
    lr_C: np.ndarray,
    lr_L: np.ndarray,
    lr_rss: np.ndarray,
    lr_best_C: float,
    lr_best_L: float,
    title: str = "Parameter Loss Surfaces: Curve Fit vs. Linear Regression",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Side-by-side heatmaps showing Curve Fit diagonal SSE valley vs Linear Regression RSS contours.
    """
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 7), dpi=PLOT_STYLE["dpi"])

    log_sse_cf = np.log10(cf_sse)
    c1 = ax1.contourf(cf_L, cf_gamma, log_sse_cf, levels=35, cmap="viridis")
    ax1.contour(cf_L, cf_gamma, log_sse_cf, levels=12, colors="white", alpha=0.3, linewidths=0.6)
    ax1.scatter([cf_best_L], [cf_best_gamma], color="red", marker="x", s=130, linewidth=2.5, label="Optimal")
    fig.colorbar(c1, ax=ax1, label=r"log$_{10}$(SSE)")
    ax1.set_xlabel("Pendulum length, L (m)")
    ax1.set_ylabel(r"Damping coefficient, $\gamma$ ($s^{-1}$)")
    ax1.set_title("Curve Fit: Diagonal Valley (Coupled)")
    ax1.legend(loc="upper right", facecolor="white", framealpha=0.8)

    log_rss_lr = np.log10(lr_rss)
    c2 = ax2.contourf(lr_L, lr_C, log_rss_lr, levels=35, cmap="viridis")
    ax2.contour(lr_L, lr_C, log_rss_lr, levels=12, colors="white", alpha=0.3, linewidths=0.6)
    ax2.scatter([lr_best_L], [lr_best_C], color="red", marker="x", s=130, linewidth=2.5, label="Optimal")
    fig.colorbar(c2, ax=ax2, label=r"log$_{10}$(RSS)")
    ax2.set_xlabel("Pendulum length, L (m)")
    ax2.set_ylabel(r"Air-resistance coefficient, $C$")
    ax2.set_title("Linear Regression: Decoupled Surface (RSS)")
    ax2.legend(loc="upper right", facecolor="white", framealpha=0.8)

    fig.suptitle(title, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_STYLE["dpi_high"], bbox_inches="tight")

    return fig, (ax1, ax2)


def plot_side_by_side_trajectories(
    t_data: np.ndarray,
    theta_data: np.ndarray,
    t_cf: np.ndarray,
    y_cf: np.ndarray,
    t_lr: np.ndarray,
    y_lr: np.ndarray,
    title: str = "Trajectory Overlay: Curve Fit vs. Linear Regression vs. Raw Data",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Overlays predictions from Curve Fit and Linear Regression on top of raw measurements.
    """
    plt.close("all")
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_wide"], dpi=PLOT_STYLE["dpi"])

    ax.plot(t_data, theta_data, "ko", markersize=3, label="Data")
    ax.plot(t_cf, y_cf, color="#1f77b4", linewidth=2.0, linestyle="--", label="Curve Fit (curve_fit + solve_ivp)")
    ax.plot(t_lr, y_lr, color="#2ca02c", linewidth=2.0, label="Linear Regression (Fourier + LinReg)")

    ax.set_xlabel("Time (s)", fontsize=PLOT_STYLE["font_size"])
    ax.set_ylabel("Angle (rad)", fontsize=PLOT_STYLE["font_size"])
    ax.set_title(title, fontsize=PLOT_STYLE["font_size"] + 2)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_STYLE["dpi_high"], bbox_inches="tight")

    return fig, ax
