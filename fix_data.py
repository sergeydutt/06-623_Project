import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def load_and_center_data(filepath = 'Pendulum.xlsx', sheet_name = 'Clear'):
    """
    Loads pendulum data and centers it around 0.0 rad
    """
    df = pd.read_excel(filepath, sheet_name = sheet_name, usecols = ['Time', 'Angle'])

    equilibrium_offset = (df['Angle'].max() + df['Angle'].min()) / 2
    df['Angle'] = df['Angle'] - equilibrium_offset

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
