"""Shared data paths, naming and monthly I/O for L0, L1 and L2.

Configuration lives in code. L0 timestamps are naive UTC; partitioning never
resamples, interpolates, clips or removes rows (including duplicate labels).
"""

import glob
import os
from pathlib import Path
import re

import pandas as pd


DATA_ROOT = Path(os.environ.get('URBAN_FLUXES_DATA_ROOT', r"D:\Data\IFA\Urban_fluxes_NE"))
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
RELEASE = 'release_001'
REPORTS_ROOT = DATA_ROOT / 'reports'
REFERENCE_ROOT = DATA_ROOT / 'reference' / 'ERA5'
CHINA_SHORT_ROOT = RAW_ROOT / "Beijing" / "short DS"
CHINA_LONG_ROOT = RAW_ROOT / "Beijing" / "long dataset"
MSU_OBS_PATH = Path(r"D:\Data\Meteo\MSK\meteo_data\msu\data_msu_fixed.csv")


def to_l0_units(frame, *, temperature='K', pressure='Pa', radiometer_temperature='K'):
    """Convert declared source units once; magnitude guesses are not used."""
    result = frame.copy()
    for column, unit in [('Tair', temperature), ('T_radiometer', radiometer_temperature)]:
        if column in result:
            if unit == 'degC':
                result[column] = result[column].astype('float64') + 273.15
            elif unit != 'K':
                raise ValueError(f'Unsupported {column} unit: {unit}')
    if 'Pair' in result:
        factors = {'Pa': 1., 'hPa': 100., 'mmHg': 101325. / 760.}
        if pressure not in factors:
            raise ValueError(f'Unsupported pressure unit: {pressure}')
        if pressure != 'Pa':
            result['Pair'] = result['Pair'].astype('float64') * factors[pressure]
    return result


def canonical_series(series):
    """Normalize old IDs and L1 names to the L0/L2 naming convention."""
    if series == 'Oblkom_down':
        return 'TFN-Oblkom_23m'
    if series == 'ZNS_56m':
        return 'IAP-Zvenigorod_56m_SNC'
    for old, new in [('MSU_', 'MSU-MO_'), ('IAP_', 'IAP-'), ('TFN_', 'TFN-'),
                     ('ZNS_', 'IAP-Zvenigorod_'), ('Beijing_', 'IAP-Beijing_'),
                     ('Xianghe_', 'IAP-Xianghe_'), ('BEK_', 'TFN-BEK_'),
                     ('Oblkom_', 'TFN-Oblkom_'), ('IMCES', 'TFN-IMCES'),
                     ('Voronino_', 'TFN-Voronino_')]:
        if series.startswith(old):
            series = new + series[len(old):]
            break
    series = re.sub(r'-(\d+m)(?=_|$)', r'_\1', series, count=1)
    if series == 'IAP-Zvenigorod_56m_new':
        return 'IAP-Zvenigorod_56m'
    return series


def l1_series_name(series):
    """Use a dash before a mast height in L1 and retain product suffixes."""
    return re.sub(r'_(\d+m)(?=_|$)', r'-\1', canonical_series(series), count=1)


def site_for(series):
    series = canonical_series(series)
    if series.startswith('TFN-IMCESmezo_'):
        return 'TFN-IMCES'
    for site in ['MSU-MO', 'LOD', 'IAP-Zvenigorod', 'TFN-BEK', 'TFN-Oblkom',
                 'TFN-IMCES', 'TFN-Voronino', 'IAP-Beijing', 'IAP-Xianghe']:
        if series == site or series.startswith(site + '_'):
            return site
    raise ValueError(f'Unknown series: {series}')


def series_directory(city, kind, series, *, root=None, release=None, level='L0'):
    if kind not in {'fluxes', 'meteo', 'radiation', 'auxiliary'}:
        raise ValueError(kind)
    series = canonical_series(series)
    base = Path(PROCESSED_ROOT if root is None else root) / level
    if level == 'L1':
        if release is not None:
            raise ValueError('L1 has no release subdirectories')
        if kind not in {'fluxes', 'meteo', 'radiation'}:
            raise ValueError('L1 stores mast series and standalone meteo or radiation series')
        if kind == 'fluxes':
            site_for(series)
        return base / city
    base = base / city
    if level == 'L0':
        if release is not None:
            raise ValueError('L0 has no release subdirectories')
    else:
        base = base / (RELEASE if release is None else release)
    return base / site_for(series) / kind / series


def series_file(city, series, *, root=None, kind='fluxes'):
    """The full-period L1 file for one source and observation type."""
    if Path(series).name != series:
        raise ValueError('Expected a series name, not a path')
    series = l1_series_name(series)
    separator = '_' if re.search(r'-\d+m(?=_|$)', series) else '-'
    return series_directory(city, kind, series, root=root, level='L1') / f'{series}{separator}{kind}.csv'


def monthly_files(path):
    """Accept a monthly series directory, a CSV, or an explicit list of CSVs."""
    if isinstance(path, (list, tuple)):
        paths = [Path(p) for p in path]
    else:
        path = Path(path)
        paths = sorted(path.glob('[0-9][0-9][0-9][0-9]-[0-9][0-9].csv')) if path.is_dir() else [path]
    if not paths or any(not p.is_file() for p in paths):
        raise FileNotFoundError(f'No monthly data: {path}')
    return paths


def read_monthly(path):
    parts = [pd.read_csv(p, index_col=0, parse_dates=[0], date_format='ISO8601', float_precision='round_trip',
                        dtype={'cloud_coefficient_raw': str})
             for p in monthly_files(path)]
    columns = parts[0].columns
    if any(not frame.columns.equals(columns) for frame in parts):
        raise ValueError(f'Monthly schemas differ: {path}')
    return pd.concat(parts)


class MonthlyWriter:
    """Partition complete native streams, including chunks crossing UTC months.

    Repeated calls append chunks during ONE complete rebuild. Each encountered
    month is staged in a .part file; existing CSVs are replaced and obsolete
    months removed only after successful reading. No metadata sidecars are
    generated. Concurrent writers for the same series are not supported.
    """
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.seen = set()
        self.columns = None
        self.last = None
        self.rows = 0

    def write(self, frame):
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans or frame.index.tz is not None:
            raise ValueError('Expected complete naive UTC timestamps')
        if not frame.index.is_monotonic_increasing:
            raise ValueError('L0 input must be ordered; partitioning cannot reorder rows')
        if not frame.columns.is_unique:
            raise ValueError('L0 variable names must be unique')
        if frame.empty:
            return
        if self.last is not None and frame.index[0] < self.last:
            raise ValueError('Unordered L0 chunks')
        if self.columns is not None and not self.columns.equals(frame.columns):
            raise ValueError('Columns changed across L0 chunks')
        self.columns, self.last = frame.columns.copy(), frame.index[-1]
        months = frame.index.to_period('M')
        for month in months.unique():
            name = str(month) + '.csv'
            append = name in self.seen
            part = self.directory / ('.' + name + '.part')
            frame.loc[months == month].to_csv(part, mode='a' if append else 'w',
                header=not append, index_label='datetime_utc', na_rep='')
            self.seen.add(name)
        self.rows += len(frame)

    def finish(self):
        if not self.rows:
            raise ValueError(f'Refusing to replace a series with no observations: {self.directory}')
        # Existing months remain untouched while source data are being read.
        for name in sorted(self.seen):
            (self.directory / ('.' + name + '.part')).replace(self.directory / name)
        for old in self.directory.glob('[0-9][0-9][0-9][0-9]-[0-9][0-9].csv'):
            if old.name not in self.seen:
                old.unlink()
        for partial in self.directory.glob('.????-??.csv.part'):
            partial.unlink()
        return sorted(self.directory / name for name in self.seen)


def write_l0(frame, city, kind, series, *, root=None, release=None,
             temperature='K', pressure='Pa', radiometer_temperature='K'):
    writer = MonthlyWriter(series_directory(city, kind, series, root=root, release=release))
    writer.write(to_l0_units(frame, temperature=temperature, pressure=pressure,
                            radiometer_temperature=radiometer_temperature))
    return writer.finish()


def flux_series(city, dataset='preferred'):
    """Explicit selection prevents mixing historical processing/source versions."""
    if city == 'Moscow':
        return ['MSU-MO_2m', 'MSU-MO_11m', 'MSU-MO_19m', 'LOD_31m', 'IAP-Zvenigorod_56m']
    if city == 'Tomsk':
        return [canonical_series(series) for series in TOMSK_SITES]
    if city == 'Beijing':
        if dataset not in {'preferred', 'combined'}:
            raise ValueError('Chinese L0 combines deliveries; short is no longer a separate dataset')
        # Optional levels appear here only when explicitly exported by the
        # preparation notebook. Missing directories are skipped by read_city.
        return [f'IAP-Beijing_{h}m' for h in [8, 16, 47, 80, 140, 200, 280]] + ['IAP-Xianghe_32m', 'IAP-Xianghe_80m']
    raise ValueError(city)


def read_city(city, *, level='L0', root=None, release=None, dataset='preferred', include_reference=False):
    """Read the city's flux series and, optionally, its reanalysis."""
    city = city.split('_')[0]  # old notebook selections such as Tomsk_new_py
    frames = {}
    for series in flux_series(city, dataset):
        directory = series_directory(city, 'fluxes', series, root=root, release=release, level=level)
        path = series_file(city, series, root=root) if level == 'L1' else directory
        if path.exists():
            name = l1_series_name(series) if level == 'L1' else series
            frames[name.removesuffix('_short')] = read_monthly(path)
    if not frames:
        raise FileNotFoundError(f'No {level} data for {city}; run the corresponding preparation notebook')
    if include_reference:
        directory = REFERENCE_ROOT / ('L2' if level == 'L2' else 'processed') / city
        if directory.exists():
            frames['ERA5'] = read_monthly(directory)
    return frames


TOMSK_SITES = [
    "BEK_30m", "Oblkom_30m", "IMCES_40m", "IMCES_30m",
    "IMCESmezo_30m", "Voronino_10m", "Oblkom_down",
]


def require_files(pattern):
    """Return deterministic input order; fail clearly for a missing dataset."""
    files = sorted(path for path in glob.glob(str(pattern)) if Path(path).is_file())
    if not files:
        raise FileNotFoundError(f"No L0 input files match: {pattern}")
    return files
