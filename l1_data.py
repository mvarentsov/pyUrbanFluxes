"""Put independent L0 observation streams on the local flux time grid."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from flux_qc import FLAG_COLUMNS, apply_flux_qc
from l0_data import PROCESSED_ROOT, flux_series, monthly_files, read_monthly, series_directory, series_file, site_for


CITIES = ('Moscow', 'Tomsk', 'Beijing')
CHUNK_SIZE = 250000


@dataclass(frozen=True)
class L1Source:
    city: str
    kind: str
    path: Path

    @property
    def site(self):
        return site_for(self.path.name)

    @property
    def output_series(self):
        return self.path.name if re.search(r'_\d+m$', self.path.name) else self.site

    def output_path(self, root):
        return series_file(self.city, self.output_series, root=root, kind=self.kind)


@dataclass(frozen=True)
class FluxGrid:
    interval: pd.Timedelta
    phase: pd.Timedelta

    @classmethod
    def from_index(cls, index):
        """Take the usual observed step and require every label to share its phase."""
        if len(index) < 2:
            raise ValueError('At least two flux timestamps are needed to determine the grid')
        times = index.as_unit('ns').asi8
        step = int(pd.Series(np.diff(times)).mode().iloc[0])
        phase = int(times[0] % step) if step > 0 else 0
        if step <= 0 or np.any(times % step != phase):
            raise ValueError('Flux timestamps do not lie on the observed fixed grid')
        return cls(pd.Timedelta(step, unit='ns'), pd.Timedelta(phase, unit='ns'))

    def labels(self, first, last, *, cover_last_window=False):
        start = (first - self.phase).ceil(self.interval) + self.phase
        end = last - self.phase
        end = end.ceil(self.interval) if cover_last_window else end.floor(self.interval)
        return pd.date_range(start, end + self.phase, freq=self.interval, name='datetime_utc')


def validate_frame(frame, name):
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.tz is not None:
        raise ValueError(f'{name}: expected complete timezone-naive UTC timestamps')
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(f'{name}: timestamps must be unique and sorted')
    if not frame.columns.is_unique or any(not pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError(f'{name}: expected unique numeric measurement columns')


def read_l0(path):
    frame = read_monthly(path)
    validate_frame(frame, str(path))
    return frame


def source_configs(dataset_root=PROCESSED_ROOT, cities=CITIES):
    """Select active fluxes and find every meteo/radiation L0 height at the selected locations."""
    root = Path(dataset_root)
    sources = []
    for city in cities:
        fluxes = []
        for series in flux_series(city):
            if series == 'LOD_31m':
                continue
            directory = series_directory(city, 'fluxes', series, root=root)
            if directory.is_dir():
                monthly_files(directory)
                fluxes.append(L1Source(city, 'fluxes', directory))
        if not fluxes:
            raise FileNotFoundError(f'No active L0 flux data for {city}')
        sources.extend(fluxes)
        for kind in ('meteo', 'radiation'):
            for directory in sorted((root / 'L0' / city).glob(f'*/{kind}/*')):
                if not directory.is_dir() or directory.parents[1].name == 'LOD':
                    continue
                source = L1Source(city, kind, directory)
                if source.site != directory.parents[1].name:
                    raise ValueError(f'Site code differs from the L0 folder: {directory}')
                monthly_files(directory)
                sources.append(source)
    targets = [s.output_path(root) for s in sources]
    if len(set(targets)) != len(targets):
        raise ValueError('Several L0 sources have the same L1 filename')
    return sources


def source_state(sources):
    state = {}
    for source in sources:
        for path in monthly_files(source.path):
            stat = path.stat()
            state[str(path)] = [stat.st_size, stat.st_mtime_ns]
    return state


def prepare_flux(flux, source):
    """Preserve the original flux series and the agreed Zvenigorod exclusion."""
    validate_frame(flux, str(source.path))
    if source.path.name == 'IAP-Zvenigorod_56m':
        on_grid = flux.index == flux.index.floor('30min')
        selected, excluded = flux.loc[on_grid], flux.loc[~on_grid]
    else:
        selected, excluded = flux, flux.iloc[:0]
    try:
        grid = FluxGrid.from_index(selected.index)
    except ValueError as error:
        raise ValueError(f'{source.path}: {error}') from error
    return selected, excluded, grid


def select_grid(source, grids):
    matching = [grid for flux, grid in grids.items() if flux.city == source.city]
    same_site = [grid for flux, grid in grids.items() if flux.city == source.city and flux.site == source.site]
    exact = [grid for flux, grid in grids.items() if flux.city == source.city and flux.path.name == source.path.name]
    choices = set(exact or same_site or matching)
    if len(choices) != 1:
        raise ValueError(f'Cannot identify one flux grid for {source.path}')
    return choices.pop()


def prepare_meteo(data, grid):
    """Select exact meteorological timestamps over that source's coverage."""
    validate_frame(data, 'meteorology')
    if data.empty:
        raise ValueError('No meteorological observations')
    index = grid.labels(data.index[0], data.index[-1])
    result = data.reindex(index)
    diagnostic = result.notna().astype('int64').add_suffix('_valid')
    diagnostic.insert(0, 'meteo_timestamp_present', index.isin(data.index).astype('int64'))
    return result, diagnostic


def radiation_chunks(path, chunksize=CHUNK_SIZE, expected_columns=None):
    """Stream monthly files without loading the full 3-second record in memory."""
    last = None
    for part in monthly_files(path):
        with pd.read_csv(part, index_col=0, parse_dates=[0], date_format='ISO8601', float_precision='round_trip',
                         chunksize=chunksize) as reader:
            for chunk in reader:
                validate_frame(chunk, str(part))
                if expected_columns is not None and list(chunk.columns) != list(expected_columns):
                    raise ValueError(f'{part}: unexpected radiation columns/order')
                if chunk.empty:
                    continue
                if last is not None and chunk.index[0] <= last:
                    raise ValueError(f'{part}: duplicate or unordered timestamps across monthly chunks')
                last = chunk.index[-1]
                yield chunk


def aggregate_radiation_csv(path, index, interval, columns, chunksize=CHUNK_SIZE, *, complete=False):
    """Compute means over (t - interval, t] using sums and valid counts across chunks."""
    rule = pd.Timedelta(interval)
    if index.empty or rule <= pd.Timedelta(0):
        raise ValueError('Expected a nonempty target grid and positive interval')
    times = index.as_unit('ns').asi8
    if ((times - times[0]) % rule.value != 0).any():
        raise ValueError('Target timestamps do not lie on one fixed grid')
    totals = pd.DataFrame(0., index=index, columns=columns)
    counts = pd.DataFrame(0, index=index, columns=columns, dtype='int64')
    rows = pd.Series(0, index=index, dtype='int64')
    source_rows = 0
    for chunk in radiation_chunks(path, chunksize, columns):
        bins = chunk.resample(rule, origin=index[0], closed='right', label='right')
        totals += bins.sum().reindex(index, fill_value=0.)
        counts += bins.count().reindex(index, fill_value=0).astype('int64')
        rows += bins.size().reindex(index, fill_value=0).astype('int64')
        source_rows += len(chunk)
        if source_rows % 2000000 < len(chunk):
            print(f'{Path(path).name}: aggregated {source_rows} native radiation rows', flush=True)
    if source_rows == 0 or (complete and rows.sum() != source_rows):
        raise ValueError(f'Empty or incomplete radiation coverage: {path}')
    return totals.divide(counts.where(counts.gt(0))), counts, rows


def radiation_diagnostics(counts, rows):
    diagnostics = counts.add_suffix('_samples')
    diagnostics.insert(0, 'radiation_rows', rows)
    return diagnostics


def prepare_radiation(path, grid):
    parts = monthly_files(path)
    first = pd.read_csv(parts[0], usecols=[0], nrows=1).iloc[:, 0]
    last = pd.read_csv(parts[-1], usecols=[0]).iloc[:, 0]
    if first.empty or last.empty:
        raise ValueError(f'Empty first or last radiation month: {path}')
    index = grid.labels(pd.Timestamp(first.iloc[0]), pd.Timestamp(last.iloc[-1]), cover_last_window=True)
    columns = list(pd.read_csv(parts[0], index_col=0, nrows=0).columns)
    result, counts, rows = aggregate_radiation_csv(path, index, grid.interval, columns, complete=True)
    return result, radiation_diagnostics(counts, rows)


def verify_radiation_csv(path, index, interval, result, diagnostics, columns,
                         chunksize=CHUNK_SIZE, rtol=1e-11, atol=1e-9):
    """Independently assign each native observation to a grid interval via searchsorted."""
    times = index.as_unit('ns').asi8
    step = pd.Timedelta(interval).value
    sums = np.zeros((len(times), len(columns)), dtype='float64')
    counts = np.zeros_like(sums, dtype='int64')
    rows = np.zeros(len(times), dtype='int64')
    source_rows = 0
    for chunk in radiation_chunks(path, chunksize, columns):
        native_times = chunk.index.as_unit('ns').asi8
        labels = np.searchsorted(times, native_times, side='left')
        positions = np.flatnonzero(labels < len(times))
        inside = native_times[positions] > times[labels[positions]] - step
        positions = positions[inside]
        labels = labels[positions]
        rows += np.bincount(labels, minlength=len(times))
        values = chunk.to_numpy()[positions]
        for i, name in enumerate(columns):
            valid = ~np.isnan(values[:, i])
            counts[:, i] += np.bincount(labels[valid], minlength=len(times))
            sums[:, i] += np.bincount(labels[valid], weights=values[valid, i], minlength=len(times))
        source_rows += len(chunk)
        if source_rows % 4000000 < len(chunk):
            print(f'{Path(path).name}: independently checked {source_rows} native radiation rows', flush=True)
    if not np.array_equal(rows, diagnostics['radiation_rows'].to_numpy()):
        raise AssertionError('Radiation window counts differ from the independent reference')
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    report = []
    for i, name in enumerate(columns):
        if not np.array_equal(counts[:, i], diagnostics[f'{name}_samples'].to_numpy()):
            raise AssertionError(f'{name}: sample counts differ from the independent reference')
        actual = result[name].to_numpy()
        matches = np.isclose(actual, means[:, i], rtol=rtol, atol=atol, equal_nan=True)
        if not matches.all():
            raise AssertionError(f'{name}: mean differs at {list(index[~matches][:5])}')
        finite = np.isfinite(actual) & np.isfinite(means[:, i])
        report.append({'variable': name, 'windows_checked': len(times),
                       'valid_windows': int((counts[:, i] > 0).sum()),
                       'max_absolute_difference': float(np.max(np.abs(actual[finite] - means[finite, i])))
                       if finite.any() else None,
                       'counts_equal': True, 'means_equal_within_roundoff': True})
    return pd.DataFrame(report)


def write_l1(frame, city, series, *, root=PROCESSED_ROOT, kind='fluxes'):
    """Write one complete CSV after a successful exact round-trip check."""
    validate_frame(frame, 'L1')
    if frame.empty:
        raise ValueError('Refusing to write an empty L1 series')
    path = series_file(city, series, root=root, kind=kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.part')
    stream = temporary.open('x', newline='', encoding='utf-8')
    try:
        with stream:
            frame.to_csv(stream, index_label='datetime_utc', na_rep='')
        pd.testing.assert_frame_equal(read_l0(temporary), frame.rename_axis('datetime_utc'),
                                      check_exact=True, check_freq=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def build_l1_products(dataset_root, output_root, *, cities=CITIES, verify_radiation=True):
    """Build a fresh set of single-source products and diagnostics in a staging directory."""
    dataset_root, output_root = Path(dataset_root).resolve(), Path(output_root).resolve()
    reports = output_root.parent / 'reports' / 'L1'
    if (output_root/'L1').exists() or reports.exists():
        raise FileExistsError('Use a fresh staging directory for the L1 build')
    sources = source_configs(dataset_root, cities)
    before = source_state(sources)
    grids, summary, checks = {}, [], []
    for source in sorted(sources, key=lambda s: s.kind != 'fluxes'):
        path = source.output_path(output_root)
        site_reports = reports / source.city / source.site
        site_reports.mkdir(parents=True, exist_ok=True)
        details = {}
        if source.kind == 'fluxes':
            frame, omitted, grid = prepare_flux(read_l0(source.path), source)
            grids[source] = grid
            details['off_grid_excluded'] = len(omitted)
            if not omitted.empty:
                omitted.to_csv(site_reports / f'{path.stem}.off_grid_excluded.csv', index_label='datetime_utc')
            frame = apply_flux_qc(frame)
            details.update({f'{name}_failed': int(frame[name].eq(0).sum()) for name in FLAG_COLUMNS})
        else:
            grid = select_grid(source, grids)
            if source.kind == 'meteo':
                original = read_l0(source.path)
                frame, diagnostics = prepare_meteo(original, grid)
                matched = frame.index.intersection(original.index)
                pd.testing.assert_frame_equal(frame.loc[matched], original.loc[matched],
                                              check_exact=True, check_dtype=False, check_freq=False)
                if not frame.loc[~frame.index.isin(original.index)].isna().all().all():
                    raise AssertionError('Meteorology contains data without an exact source timestamp')
                details['exact_meteo_matches'] = len(matched)
            else:
                frame, diagnostics = prepare_radiation(source.path, grid)
                counts = diagnostics['radiation_rows']
                details.update(native_radiation_rows=int(counts.sum()), windows_with_radiation=int(counts.gt(0).sum()))
                if verify_radiation:
                    check = verify_radiation_csv(source.path, frame.index, grid.interval, frame,
                                                 diagnostics, list(frame.columns))
                    checks.append(check.assign(city=source.city, site=source.site, series=path.stem))
            diagnostics.to_csv(site_reports / f'{path.stem}.alignment.csv', index_label='datetime_utc')

        write_l1(frame, source.city, source.output_series, root=output_root, kind=source.kind)
        summary.append({'city': source.city, 'site': source.site, 'series': path.stem, 'kind': source.kind,
                        'source': source.path.relative_to(dataset_root).as_posix(),
                        'file': path.relative_to(output_root).as_posix(),
                        'interval_minutes': grid.interval.total_seconds() / 60,
                        'grid_phase_seconds': grid.phase.total_seconds(),
                        'rows': len(frame), 'columns': len(frame.columns), 'first_time_utc': str(frame.index[0]),
                        'last_time_utc': str(frame.index[-1]), **details})
        print(f'{path.name}: {len(frame)} rows, step {grid.interval}', flush=True)

    if source_state(source_configs(dataset_root, cities)) != before:
        raise RuntimeError('L0 changed while L1 was being built')
    summary = pd.DataFrame(summary)
    summary.to_csv(reports/'L1.summary.csv', index=False)
    for city, table in summary.groupby('city', sort=False):
        table.to_csv(reports/city/'L1.summary.csv', index=False)
    if checks:
        pd.concat(checks, ignore_index=True).to_csv(reports/'L1.radiation_verification.csv', index=False)
    return summary, before
