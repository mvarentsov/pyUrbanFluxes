"""Quality flags for observed fluxes, transferred from the former L2 notebook."""

import numpy as np
import pandas as pd


LIMITS = {'Qh': (-300, 600), 'Qle': (-300, 600), 'Qtau': (0, 5), 'Cd': (0, 1)}
SIGMAS = {'Qh': 5, 'Qle': 5, 'Qtau': 5, 'Cd': 10}
STAT_WINDOW = '60D'
EXTENDED_WINDOW = '1h'
FLAG_COLUMNS = ('qc_tot', 'qc_tot_noCd', 'qc_tot_ext', 'qc_tot_noCd_ext')


def _hour_windows(index):
    hours = index.hour.to_numpy()
    for hour in np.unique(hours):
        neighbours = np.arange(hour - 1, hour + 2)
        # Preserve the L2 convention: the lower bound wraps at hour 0.
        # The hour-23 reference uses 22 and 23, as in the original filter.
        neighbours[neighbours < 0] += 24
        yield hours == hour, np.isin(hours, neighbours)


def _outlier_flags(values, range_ok, sigma, hour_windows):
    outlier_ok = pd.Series(True, index=values.index)
    iterations = 0
    previous_count = None
    while True:
        candidate = values.where(outlier_ok)
        background = values.where(range_ok & outlier_ok)
        rejected = np.zeros(len(values), dtype=bool)
        for in_hour, in_reference_hours in hour_windows:
            window = background.where(in_reference_hours).rolling(STAT_WINDOW, center=True, min_periods=1)
            mean, std = window.mean()[in_hour], window.std()[in_hour]
            current = candidate[in_hour]
            rejected[in_hour] = ((current < mean - sigma * std) | (current > mean + sigma * std)).to_numpy()

        outlier_ok.loc[rejected] = False
        count = int(rejected.sum())
        iterations += 1
        if count == 0 or (iterations > 2 and count <= 2 and previous_count <= 2):
            return outlier_ok
        previous_count = count


def apply_flux_qc(frame):
    """Append Cd and the four legacy flags to the original observations."""
    result = frame.copy()
    if {'Ustar', 'Wind'}.issubset(result.columns):
        result['Cd'] = (result['Ustar'] / result['Wind']) ** 2

    hour_windows = tuple(_hour_windows(result.index))
    variable_ok = {}
    for name, (lower, upper) in LIMITS.items():
        if name in result:
            values = result[name]
            range_ok = values.between(lower, upper) | values.isna()
            variable_ok[name] = range_ok & _outlier_flags(values, range_ok, SIGMAS[name], hour_windows)

    checks = pd.DataFrame(variable_ok, index=result.index)
    total = checks.all(axis=1)
    total_without_cd = checks.drop(columns='Cd', errors='ignore').all(axis=1)
    result['qc_tot'] = total.astype('int64')
    result['qc_tot_noCd'] = total_without_cd.astype('int64')
    result['qc_tot_ext'] = total.rolling(EXTENDED_WINDOW, center=True).min().astype('int64')
    result['qc_tot_noCd_ext'] = total_without_cd.rolling(EXTENDED_WINDOW, center=True).min().astype('int64')
    return result
