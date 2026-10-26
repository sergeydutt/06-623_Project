import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

def load_and_center_data(filepath = 'Pendulum.xlsx', sheet_name = 'Clear'):
    """
    Loads pendulum data and centers it around 0.0 rad
    Equilibrium is originally offset and drifts overtime
    This function counteracts the drift by subtracting an interpolated envolope of peak-trough midpoints
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name, usecols=['Time', 'Angle'])

    angles = df['Angle'].values
    times = df['Time'].values

    # find local maxima (peaks) and local minima (troughs)
    peaks, _ = find_peaks(angles)
    troughs, _ = find_peaks(-angles)

    # combine and sort extrema coordinates by time
    t_extrema = np.concatenate([times[peaks], times[troughs]])
    a_extrema = np.concatenate([angles[peaks], angles[troughs]])

    sort_idx = np.argsort(t_extrema)
    t_extrema = t_extrema[sort_idx]
    a_extrema = a_extrema[sort_idx]

    # calculate midpoints between adjacent peaks and troughs
    t_mid = (t_extrema[:-1] + t_extrema[1:]) / 2.0
    baseline_mid = (a_extrema[:-1] + a_extrema[1:]) / 2.0

    # interpolate baseline across full time array (extrapolate ends if needed)
    baseline_func = interp1d(
        t_mid, baseline_mid, kind='linear', fill_value='extrapolate'
    )
    equilibrium_drift = baseline_func(times)

    # subtract baseline shift
    df['Angle'] = df['Angle'] - equilibrium_drift
    
    return df

def get_random_subset(data, num_os):
    """
    Randomly selects a subset of df starting at
    a local max and contaning a specified number 
    of full oscillations.
    """
    angles = data['Angle'].values

    # finds local max (all peaks)
    peaks, _ = find_peaks(angles)

    # only returns peaks that meet the the oscillation requirement
    valid_peaks = [
        p for p in peaks if np.where(peaks == p)[0][0] + num_os < len(peaks)
    ]

    # should only appear for a very large num_os, still good to have
    if not valid_peaks:
        raise ValueError('Not enough peaks found that meet num_os')

    # randomly choose a valid peak to begin the subset
    start = np.random.choice(valid_peaks)

    # find where the subset ends 
    start_peak = np.where(peaks == start)[0][0]
    end = peaks[start_peak + num_os]

    # create subset and find bounds (start + end)
    subset = data.iloc[start : end + 1].copy()
    t_0 = data.loc[start, 'Time']
    t_n = data.loc[end, 'Time']
    print(f't_0 = {t_0:.1f}s , t_n = {t_n:.1f}s, n = {len(subset)} points')

    return subset
