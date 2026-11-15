"""Readers for the annual Beijing/Xianghe delivery; preserve native resolution.

Flux formulas and output column order match the short-dataset notebooks.
Meteorology and radiation remain separate: their heights, units and sampling
intervals differ from the eddy-covariance measurements. No interpolation, clock
correction (apart from the existing UTC+8 conversion), clipping or gap filling.
"""

import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from l0_data import CHINA_LONG_ROOT, PROCESSED_ROOT


FLUX_MAP = {
    'Beijing': {
        'H': 'Qh', 'LE': 'Qle', 'Tau': 'Qtau', 'u*': 'Ustar',
        'air_temperature': 'Tair', 'RH': 'RHair', 'air_pressure': 'Pair',
        'air_density': 'RHOair', 'v_unrot': 'Wind_V', 'u_unrot': 'Wind_U',
        'w_unrot': 'Wind_W', 'wind_speed': 'Wind', 'u_sigma': 'Wind_U_std',
        'v_sigma': 'Wind_V_std', 'w_sigma': 'Wind_W_std',
        'wind_dir': 'Wind_dir', 'co2_flux': 'Qco2',
    },
    'Xianghe': {
        'H': 'Qh', 'LE': 'Qle', 'Tau': 'Qtau', 'u_star_R': 'Ustar',
        'rho_a_Avg': 'RHOair', 'u_Avg_R': 'Wind',
        'u_Std_R': 'Wind_U_std', 'v_Std_R': 'Wind_V_std', 'w_Std_R': 'Wind_W_std',
    },
}
FLUX_INPUT = {
    'Beijing': ['H', 'LE', 'u*', 'air_temperature', 'RH', 'air_pressure',
                'air_density', 'v_unrot', 'u_unrot', 'w_unrot', 'wind_speed',
                'u_var', 'v_var', 'w_var', 'wind_dir', 'co2_flux'],
    'Xianghe': ['wTs_Cov_R_F_SND', 'Cp', 'rho_a_Avg', 'wH2O_Cov_R_F',
                'rho_d_Avg', 'u_star_R', 'u_Avg_R', 'u_Std_R', 'v_Std_R', 'w_Std_R'],
}
RADIATION_MAP = {
    'Beijing': {'DR_Avg': 'SWdown', 'UR_Avg': 'SWup', 'DLR_Avg': 'LWdown',
                'ULR_Avg': 'LWup', 'Rn_Avg': 'Rn', 'T_CNR1_C_Avg': 'T_radiometer'},
    'Xianghe': {'SWUpper': 'SWdown', 'SWLower': 'SWup', 'LWUpperCo': 'LWdown',
                'LWLowerCo': 'LWup', 'CNR4TC': 'T_radiometer', 'Rn': 'Rn', 'Albedo': 'Albedo'},
}
RADIATION_REFERENCE = 'https://s.campbellsci.com/documents/es/manuals/874%20-%20cnr4.pdf'
UTC_OFFSET = pd.Timedelta(hours=8)


def source_folder(city, kind, root=CHINA_LONG_ROOT):
    """Resolve the supplied folder names without hardcoding punctuation/dates."""
    if city not in FLUX_MAP or kind not in {'flux', 'Radiation', 'W,T,H'}:
        raise ValueError((city, kind))
    matches = sorted(p for p in Path(root).iterdir()
                     if p.is_dir() and p.name.startswith(city + ' ') and p.name.endswith(kind))
    if len(matches) != 1:
        raise ValueError(f'Expected one {city} {kind} directory in {root}, found {matches}')
    return matches[0]


def level_from_path(path):
    for part in reversed(Path(path).parts[:-1]):
        match = re.search(r'(\d+)\s*[mM]$', part)
        if match:
            return f'{int(match[1])}m'
    raise ValueError(f'Cannot identify measurement height: {path}')


def discover(city, kind, root=CHINA_LONG_ROOT):
    folder = source_folder(city, kind, root)
    pattern = '*.txt' if kind == 'W,T,H' or (city == 'Beijing' and kind == 'flux') else '*.dat'
    files = sorted(folder.rglob(pattern))
    if city == 'Beijing' and kind == 'Radiation':
        # The other .dat files are TOB3 binary originals, not additional records.
        files = [p for p in files if p.name.startswith('TOA5_')]
    if not files:
        raise FileNotFoundError(f'No {city} {kind} inputs in {folder}')
    if kind == 'W,T,H':
        return files
    groups = {}
    for path in files:
        groups.setdefault(level_from_path(path), []).append(path)
    return {level: groups[level] for level in sorted(groups, key=lambda v: int(v[:-1]))}


def _numeric(frame, context):
    try:
        return frame.apply(pd.to_numeric, errors='raise')
    except (TypeError, ValueError) as error:
        raise ValueError(f'Non-numeric measurement in {context}: {error}') from error


def read_toa5(path, columns):
    with Path(path).open('r', encoding='utf-8-sig') as stream:
        rows = [next(csv.reader([stream.readline()])) for _ in range(4)]
    if rows[0][0] != 'TOA5':
        raise ValueError(f'Expected converted TOA5 text, not a binary logger file: {path}')
    required = ['TIMESTAMP', *columns]
    missing = set(required) - set(rows[1])
    if missing:
        raise ValueError(f'Missing fields in {path}: {sorted(missing)}')
    frame = pd.read_csv(path, header=1, skiprows=[2, 3], usecols=required,
                        na_values=['NAN', 'NaN', 'nan'], encoding='utf-8-sig')
    index = pd.to_datetime(frame.pop('TIMESTAMP'), errors='raise', format='mixed')
    frame.index = pd.DatetimeIndex(index, name='datetime_local')
    if frame.index.hasnans:
        raise ValueError(f'Missing timestamps in {path}')
    units = dict(zip(rows[1], rows[2]))
    return _numeric(frame.reindex(columns=columns), path), {name: units[name] for name in columns}


def read_beijing_flux(path):
    columns = FLUX_INPUT['Beijing']
    with Path(path).open('r', encoding='utf-8-sig') as stream:
        header = stream.readline().rstrip('\r\n').split('\t')
        units = stream.readline().rstrip('\r\n').split('\t')
    frame = pd.read_csv(path, sep='\t', skiprows=[1], usecols=['date', 'time', *columns])
    index = pd.to_datetime(frame.pop('date') + ' ' + frame.pop('time'), errors='raise', format='mixed')
    frame.index = pd.DatetimeIndex(index, name='datetime_local')
    if frame.index.hasnans:
        raise ValueError(f'Missing timestamps in {path}')
    return _numeric(frame.reindex(columns=columns), path), {name: dict(zip(header, units))[name] for name in columns}


def flux_frame(city, native):
    """Apply exactly the calculations used by the corresponding short notebook."""
    frame = native.copy()
    if city == 'Beijing':
        frame['Tau'] = frame['air_density'] * frame['u*'] ** 2
        for axis in ['u', 'v', 'w']:
            frame[f'{axis}_sigma'] = np.sqrt(frame[f'{axis}_var'])
    else:
        frame['H'] = frame['wTs_Cov_R_F_SND'] * frame['Cp'] * frame['rho_a_Avg']
        frame['LE'] = frame['wH2O_Cov_R_F'] * (2.501 * 10 ** 6) / 1000
        frame['Tau'] = frame['rho_d_Avg'] * frame['u_star_R'] ** 2
    frame = frame[list(FLUX_MAP[city])].rename(columns=FLUX_MAP[city])
    return to_utc(frame)


def to_utc(frame):
    result = frame.copy()
    if not isinstance(result.index, pd.DatetimeIndex) or result.index.tz is not None:
        raise ValueError('Expected naive source timestamps in UTC+8')
    result.index = (result.index - UTC_OFFSET).rename('datetime_utc')
    return result


def unique_observations(frame, context):
    """Resolve copies and all-missing stubs; reject conflicting measurements."""
    result = frame.sort_index(kind='stable')
    # EddyPro can emit both a not_enough_data stub and the actual measurement
    # for one time. Keep standalone empty rows; remove a stub only if a nonempty
    # observation at precisely the same timestamp is present.
    empty = result.isna().all(axis=1)
    redundant_empty = empty & result.index.isin(result.index[~empty])
    placeholders = int(redundant_empty.sum())
    result = result.loc[~redundant_empty]
    duplicates = result.index.duplicated(keep=False)
    if duplicates.any():
        repeated = result.loc[duplicates]
        distinct = repeated.reset_index().drop_duplicates()
        if distinct.iloc[:, 0].duplicated().any():
            bad = distinct.loc[distinct.iloc[:, 0].duplicated(keep=False), distinct.columns[0]]
            raise ValueError(f'Conflicting observations at duplicate timestamps in {context}: {list(bad.head())}')
    identical = int(result.index.duplicated(keep='first').sum())
    result = result.loc[~result.index.duplicated(keep='first')]
    result.attrs.update(identical_duplicate_rows_removed=identical,
                        empty_duplicate_placeholders_removed=placeholders)
    return result, identical + placeholders


def _source_record(path, frame, units):
    return {'source_file': str(path), 'rows': len(frame),
            'first_time_local': str(frame.index.min()), 'last_time_local': str(frame.index.max()),
            'bytes': Path(path).stat().st_size, 'units': units}


def load_flux(city, root=CHINA_LONG_ROOT):
    frames, records = {}, []
    for level, files in discover(city, 'flux', root).items():
        parts, common_units = [], None
        for number, path in enumerate(files, start=1):
            if path.stat().st_size == 0:
                # An empty daily delivery contains no observations to parse.
                records.append({'kind': 'flux', 'level': level, 'source_file': str(path),
                                'rows': 0, 'bytes': 0, 'status': 'empty_file', 'units': {}})
                continue
            native, units = (read_beijing_flux(path) if city == 'Beijing'
                             else read_toa5(path, FLUX_INPUT[city]))
            if common_units is not None and units != common_units:
                raise ValueError(f'Flux units changed within {city} {level}: {path}')
            common_units = units
            parts.append(flux_frame(city, native))
            records.append({'kind': 'flux', 'level': level, **_source_record(path, native, units)})
            if number % 1000 == 0:
                print(f'{city} flux {level}: {number}/{len(files)} files read', flush=True)
        if not parts:
            raise ValueError(f'All flux files are empty for {city} {level}')
        frames[level], removed = unique_observations(pd.concat(parts), f'{city} flux {level}')
        frames[level].attrs.update(source_files=len(files), duplicate_rows_removed=removed,
                                   source_units=common_units)
        print(f'{city} flux {level}: {len(files)} files, {len(frames[level])} rows', flush=True)
        empty_files = sum(record.get('status') == 'empty_file' and record['level'] == level for record in records)
        if empty_files:
            print(f'{city} flux {level}: {empty_files} zero-byte source files recorded in diagnostics', flush=True)
    return frames, records


def read_meteorology(path, city):
    """Parse height blocks and use the printed END of each averaging interval."""
    if city not in FLUX_MAP:
        raise ValueError(city)
    wind_names = ['Wind_nw', 'Wind_se'] if city == 'Beijing' else ['Wind_ne', 'Wind_sw']
    columns = ['height', *wind_names, 'Wind_dir', 'RHair', 'Tair']
    rows, block, date = [], [], None
    expected_date = re.fullmatch(r'(\d{6})-(30|60)\.txt', Path(path).name)
    if not expected_date:
        raise ValueError(f'Unexpected meteorology filename: {path}')
    expected_minutes = int(expected_date[2])
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        parts = line.split()
        if re.match(r'^\s*\d+\s+[-+\d.]', line):
            if len(parts) != 6:
                raise ValueError(f'Expected height and five meteorological variables: {path}: {line}')
            block.append([float(value) for value in parts])
        elif match := re.search(r'date\s*=\s*(\d{6})', line):
            date = match[1]
            if date != expected_date[1]:
                raise ValueError(f'Meteorology date differs from filename: {path}: {date}')
        elif match := re.search(r'hour\s*=\s*(\d+):(\d+)\s*-\s*(\d+):(\d+)', line):
            if date is None or not block:
                raise ValueError(f'Missing date or observation block: {path}: {line}')
            start = int(match[1]) * 60 + int(match[2])
            end = int(match[3]) * 60 + int(match[4])
            if end <= start:
                end += 24 * 60
            if end - start != expected_minutes:
                raise ValueError(f'Unexpected averaging period: {path}: {line}')
            timestamp = pd.to_datetime(date, format='%y%m%d') + pd.Timedelta(minutes=end)
            heights = [row[0] for row in block]
            if len(set(heights)) != len(heights):
                raise ValueError(f'Repeated height within meteorology block: {path}: {line}')
            rows.extend([[timestamp, *row] for row in block])
            block, date = [], None
    if block or date is not None or not rows:
        raise ValueError(f'Incomplete meteorology block: {path}')
    frame = pd.DataFrame(rows, columns=['datetime_local', *columns]).set_index('datetime_local')
    frame['height'] = frame['height'].astype(int)
    # 999 is the documented missing/QC-rejected marker, not a measurement.
    frame.loc[:, columns[1:]] = frame.loc[:, columns[1:]].replace(999, np.nan)
    return frame


def load_meteorology(city, root=CHINA_LONG_ROOT):
    parts, records = [], []
    for path in discover(city, 'W,T,H', root):
        frame = read_meteorology(path, city)
        parts.append(frame)
        records.append({'kind': 'meteorology', 'level': 'all', **_source_record(path, frame, {})})
    all_rows = pd.concat(parts)
    frames = {}
    for height, data in all_rows.groupby('height', sort=True):
        level = f'{height}m'
        frames[level], removed = unique_observations(to_utc(data.drop(columns='height')), f'{city} meteo {level}')
        frames[level].attrs['duplicate_rows_removed'] = removed
    return frames, records


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_frames(city, frames, output_root=PROCESSED_ROOT / 'Beijing_long', kind='flux'):
    directory = Path(output_root) / ({'flux': '', 'meteorology': 'meteo'}[kind])
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for level, frame in frames.items():
        path = directory / f'{city}_{level}.csv'
        frame.to_csv(path, index_label='datetime_utc', date_format='%Y-%m-%d %H:%M:%S', na_rep='')
        records.append({'kind': kind, 'level': level, 'file': str(path.relative_to(output_root)),
                        'rows': len(frame), 'columns': list(frame.columns),
                        'first_time_utc': str(frame.index.min()), 'last_time_utc': str(frame.index.max()),
                        'identical_duplicate_rows_removed': frame.attrs.get('identical_duplicate_rows_removed', 0),
                        'empty_duplicate_placeholders_removed': frame.attrs.get('empty_duplicate_placeholders_removed', 0),
                        'sha256': _digest(path)})
    return records


def export_radiation(city, output_root=PROCESSED_ROOT / 'Beijing_long', root=CHINA_LONG_ROOT):
    """Stream the 3-second Xianghe records without loading a whole year into RAM."""
    output, sources = [], []
    directory = Path(output_root) / 'radiation'
    directory.mkdir(parents=True, exist_ok=True)
    for level, files in discover(city, 'Radiation', root).items():
        path = directory / f'{city}_{level}.csv'
        mapping = RADIATION_MAP[city]
        if city == 'Beijing':
            # Monthly logger downloads can overlap; compare every repeated time.
            parts, common_units = [], None
            for source in files:
                frame, units = read_toa5(source, list(mapping))
                if common_units is not None and units != common_units:
                    raise ValueError(f'Radiation units changed: {source}')
                common_units = units
                parts.append(to_utc(frame.rename(columns=mapping)))
                sources.append({'kind': 'radiation', 'level': level, **_source_record(source, frame, units)})
            combined, removed = unique_observations(pd.concat(parts), f'{city} radiation {level}')
            combined.to_csv(path, index_label='datetime_utc', date_format='%Y-%m-%d %H:%M:%S', na_rep='')
            rows, first, last = len(combined), combined.index.min(), combined.index.max()
        else:
            rows, first, last, removed = 0, None, None, 0
            previous = None
            history_rows = 0
            common_units = None
            with path.open('w', newline='', encoding='utf-8') as stream:
                for number, source in enumerate(files, start=1):
                    native, units = read_toa5(source, list(mapping))
                    if common_units is not None and units != common_units:
                        raise ValueError(f'Radiation units changed: {source}')
                    common_units = units
                    sources.append({'kind': 'radiation', 'level': level, **_source_record(source, native, units)})
                    # Files with _1/_2/_3 suffixes are growing re-exports of the
                    # same logger interval. Retain the whole largest interval
                    # seen, not just the new tail written on the previous step.
                    history_rows = max(history_rows, len(native))
                    frame, count = unique_observations(to_utc(native.rename(columns=mapping)), source)
                    removed += count
                    if previous is not None and len(frame) and frame.index.min() <= previous.index.max():
                        overlap = frame.index <= previous.index.max()
                        repeated = frame.loc[overlap]
                        if not repeated.index.isin(previous.index).all():
                            raise ValueError(f'Overlapping logger intervals need review: {source}')
                        if not repeated.equals(previous.loc[repeated.index]):
                            raise ValueError(f'Conflicting radiation observations: {source}')
                        removed += len(repeated)
                        frame = frame.loc[~overlap]
                    if not frame.empty:
                        frame.to_csv(stream, header=(rows == 0), index_label='datetime_utc',
                                     date_format='%Y-%m-%d %H:%M:%S', na_rep='')
                        first = frame.index.min() if first is None else first
                        last = frame.index.max()
                        rows += len(frame)
                        previous = (frame if previous is None else pd.concat([previous, frame])).tail(history_rows)
                    if number % 1000 == 0:
                        print(f'{city} radiation {level}: {number}/{len(files)} files, {rows} rows', flush=True)
        output.append({'kind': 'radiation', 'level': level, 'file': str(path.relative_to(output_root)),
                       'rows': rows, 'columns': list(mapping.values()), 'first_time_utc': str(first),
                       'last_time_utc': str(last), 'duplicate_rows_removed': removed,
                       'sha256': _digest(path)})
        print(f'{city} radiation {level}: {len(files)} files, {rows} rows', flush=True)
    return output, sources


def write_metadata(city, outputs, sources, output_root=PROCESSED_ROOT / 'Beijing_long', root=CHINA_LONG_ROOT):
    directory = Path(output_root)
    directory.mkdir(parents=True, exist_ok=True)
    flux_units = {'Qh': 'W m-2', 'Qle': 'W m-2', 'Qtau': 'N m-2', 'Ustar': 'm s-1',
                  'RHOair': 'kg m-3', 'Wind': 'm s-1', 'Wind_U_std': 'm s-1',
                  'Wind_V_std': 'm s-1', 'Wind_W_std': 'm s-1'}
    if city == 'Beijing':
        flux_units.update(Tair='K', RHair='%', Pair='Pa', Wind_U='m s-1', Wind_V='m s-1',
                          Wind_W='m s-1', Wind_dir='degree', Qco2='micromol m-2 s-1')
    metadata = {'city': city, 'source_root': str(root), 'time_column': 'datetime_utc',
                'time_shown_in': 'UTC', 'source_utc_offset_hours': 8,
                'time_processing': 'source local time minus 8 hours, as in the short-dataset notebook',
                'resampling': False, 'interpolation': False, 'gap_filling': False, 'clipping': False,
                'period_selection': 'actual timestamps in files; folder date ranges do not crop observations',
                'duplicate_policy': 'remove identical copies and all-missing stubs duplicated by a nonempty observation; conflicting measurements raise an error',
                'flux_units': flux_units,
                'meteorology_units': {'Tair': 'degC', 'RHair': '%', 'Wind_dir': 'degree',
                    **{name: 'm s-1' for name in (['Wind_nw', 'Wind_se'] if city == 'Beijing' else ['Wind_ne', 'Wind_sw'])}},
                'radiation_units': {'SWdown': 'W m-2', 'SWup': 'W m-2', 'LWdown': 'W m-2',
                    'LWup': 'W m-2', 'Rn': 'W m-2', 'T_radiometer': 'degC', 'Albedo': '1'},
                'radiation_mapping': RADIATION_MAP[city], 'radiation_reference': RADIATION_REFERENCE,
                'meteorology_missing_marker': 999, 'meteorology_timestamp': 'printed end of averaging interval',
                'empty_source_files': [record['source_file'] for record in sources if record.get('status') == 'empty_file'],
                'wind_coordinate_frame': 'unchanged from the short-dataset notebook; no geographical rotation',
                'notes': ['Meteorological wind sensors on separate booms are retained separately.',
                          'Xianghe upper meteorological height is labelled 100 m in the data, 102 m in the description.',
                          'Xianghe radiometer longwave fields are already temperature-corrected; no second correction.'],
                'outputs': outputs}
    (directory / f'{city}.metadata.json').write_text(json.dumps(metadata, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    with (directory / f'{city}.sources.json').open('w', encoding='utf-8') as stream:
        json.dump(sources, stream, ensure_ascii=True, indent=2)
        stream.write('\n')
    summary = pd.DataFrame([{k: v for k, v in record.items() if k != 'columns'} for record in outputs])
    diagnostics = directory / 'diagnostics'
    diagnostics.mkdir(exist_ok=True)
    summary.to_csv(diagnostics / f'{city}.availability.csv', index=False)
    return summary
