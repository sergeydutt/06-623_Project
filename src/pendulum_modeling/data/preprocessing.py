"""
Data loading, preprocessing, and windowing module for Pendulum Modeling.

Provides functions to:
1. Load raw motion data from Excel and center the equilibrium around 0.0 radians
   using midpoint envelope interpolation to correct mechanical/sensor drift.
2. Extract reproducible oscillation subsets starting at local maxima.
3. Prepare zero-referenced time and displacement arrays for model calibration.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from pendulum_modeling.config import (
    DATA_FILE_PATH,
    DEFAULT_SHEET,
    DEFAULT_NUM_OSCILLATIONS,
    RANDOM_SEED,
)


def load_and_center_data(
    filepath: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads pendulum experimental data and centers the angular trajectory around 0.0 rad.

    The physical equilibrium point in Pendulum.xlsx is off-centered and slowly drifts
    over time due to rig relaxation and sensor bias. This function estimates the baseline
    drift by interpolating the midpoints between successive local peaks and troughs,
    then subtracts the dynamic baseline from the recorded angle.

    Parameters
    ----------
    filepath : str, optional
        Path to Pendulum.xlsx. If None, uses default from config.
    sheet_name : str, optional
        Sheet name within workbook (e.g. 'Clear'). If None, uses default from config.

    Returns
    -------
    pd.DataFrame
        DataFrame containing centered ['Time', 'Angle'] columns.
    """
    path = filepath if filepath is not None else str(DATA_FILE_PATH)
    sheet = sheet_name if sheet_name is not None else DEFAULT_SHEET

    df = pd.read_excel(path, sheet_name=sheet, usecols=['Time', 'Angle'])

    angles = df['Angle'].values
    times = df['Time'].values

    # Detect local maxima (peaks) and minima (troughs)
    peaks, _ = find_peaks(angles)
    troughs, _ = find_peaks(-angles)

    if len(peaks) == 0 or len(troughs) == 0:
        raise ValueError("Insufficient extrema found in data to perform centering.")

    # Combine and sort extrema chronologically
    t_extrema = np.concatenate([times[peaks], times[troughs]])
    a_extrema = np.concatenate([angles[peaks], angles[troughs]])

    sort_idx = np.argsort(t_extrema)
    t_extrema = t_extrema[sort_idx]
    a_extrema = a_extrema[sort_idx]

    # Calculate midpoints of adjacent peak-trough pairs
    t_mid = (t_extrema[:-1] + t_extrema[1:]) / 2.0
    baseline_mid = (a_extrema[:-1] + a_extrema[1:]) / 2.0

    # Linearly interpolate baseline across the full experimental time domain
    baseline_func = interp1d(
        t_mid, baseline_mid, kind='linear', fill_value='extrapolate'
    )
    equilibrium_drift = baseline_func(times)

    # Re-center angles
    df['Angle'] = df['Angle'] - equilibrium_drift

    return df


def get_random_subset(
    data: pd.DataFrame,
    num_os: int = DEFAULT_NUM_OSCILLATIONS,
    random_state: Optional[int] = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Selects a contiguous window of pendulum motion containing a specified number
    of full oscillations, beginning at a local maximum (peak).

    Parameters
    ----------
    data : pd.DataFrame
        Centered pendulum DataFrame with 'Time' and 'Angle' columns.
    num_os : int
        Number of complete oscillations required in the subset window.
    random_state : int, optional
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    pd.DataFrame
        Sliced DataFrame subset containing the requested oscillations.
    """
    angles = data['Angle'].values
    peaks, _ = find_peaks(angles)

    # Identify all peaks that leave enough room for num_os subsequent oscillations
    valid_peaks = [
        p for p in peaks if np.where(peaks == p)[0][0] + num_os < len(peaks)
    ]

    if not valid_peaks:
        raise ValueError(f"Not enough oscillations in data to extract {num_os} cycles.")

    rng = np.random.RandomState(random_state)
    start_peak_idx_in_data = rng.choice(valid_peaks)

    # Locate peak index within peak array
    start_idx = np.where(peaks == start_peak_idx_in_data)[0][0]
    end_peak_idx_in_data = peaks[start_idx + num_os]

    subset = data.iloc[start_peak_idx_in_data : end_peak_idx_in_data + 1].copy()
    return subset


def zero_start_time(subset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shifts the time column of a subset DataFrame so that t_0 = 0.0 s.
    """
    df_copy = subset_df.copy()
    t_0 = df_copy['Time'].iloc[0]
    df_copy['Time'] = df_copy['Time'] - t_0
    return df_copy


def prepare_time_series(
    subset_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Normalizes time coordinates so that t_0 = 0.0s for the selected subset window.

    Parameters
    ----------
    subset_df : pd.DataFrame
        DataFrame subset containing 'Time' and 'Angle'.

    Returns
    -------
    t_data : np.ndarray
        Array of elapsed times starting at 0.0 s.
    theta_data : np.ndarray
        Array of angular displacements in radians.
    subset_df_zeroed : pd.DataFrame
        Copy of DataFrame with re-zeroed 'Time' column.
    """
    df_zeroed = zero_start_time(subset_df)
    t_data = df_zeroed['Time'].to_numpy(dtype=float)
    theta_data = df_zeroed['Angle'].to_numpy(dtype=float)

    return t_data, theta_data, df_zeroed
