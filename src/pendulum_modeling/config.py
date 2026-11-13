"""
Configuration module for Pendulum Modeling Project.

Defines project directory paths, physical constants, notebook replication parameters,
hyperparameters, parameter scan ranges, and plot styles.
"""

from pathlib import Path
import numpy as np

# Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# Data File
DATA_FILE_PATH = PROJECT_ROOT / "Pendulum.xlsx"
DEFAULT_SHEET = "Clear"

# Physical Constants & Reference Values
G_ACCEL = 9.81           # Acceleration due to gravity (m/s^2)
L_DATA = 1.615           # Measured nominal pendulum length from experimental setup (m)
M_DATA = 1.0             # Measured pendulum bob mass (kg)

# Notebook Replication Seeds & Parameters
# Project_PoC.ipynb uses seed 42 and 15 oscillations
POC_SEED = 42
POC_NUM_OSCILLATIONS = 15

# Project_Final.ipynb uses seed 42 and 10 oscillations
FINAL_SEED = 42
FINAL_NUM_OSCILLATIONS = 10

RANDOM_SEED = FINAL_SEED
DEFAULT_NUM_OSCILLATIONS = FINAL_NUM_OSCILLATIONS

# Fourier Series Hyperparameters (Project_Final.ipynb)
K_MAX_DEFAULT = 10
TOLERANCE_DEFAULT = 0.01

# Parameter Scan Ranges (Matching Project_PoC.ipynb & Project_Final.ipynb)
GAMMA_MIN, GAMMA_MAX = 0.0, 0.05
C_MIN, C_MAX = 0.00, 0.15
L_MIN, L_MAX = 1.64, 1.75
GRID_RESOLUTION = 100
L_SCAN_VALUES = np.linspace(L_MIN, L_MAX, 30)

# Reusable Plot Styling
PLOT_STYLE = {
    "figsize_default": (9, 5),
    "figsize_wide": (12, 6),
    "dpi": 100,
    "dpi_high": 300,
    "residual_color": "tomato",
    "baseline_color": "dodgerblue",
    "font_size": 11,
}
