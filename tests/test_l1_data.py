"""Checks for independent L1 products, native coverage and time grid rules."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flux_qc import FLAG_COLUMNS, apply_flux_qc
from l0_data import read_city, series_file, write_l0
from l1_data import (FluxGrid, L1Source, aggregate_radiation_csv, build_l1_products,
                     prepare_flux, prepare_meteo, radiation_diagnostics, read_l0,
                     source_configs, verify_radiation_csv, write_l1)
from l1_publish import publish_l1, stage_l1


def frame(times, **columns):
    return pd.DataFrame(columns, index=pd.DatetimeIndex(times, name='datetime_utc'))


class L1Tests(unittest.TestCase):
    def test_flux_only_reader_ignores_other_products(self):
        root_name = ROOT/'_temp'
        times = ['2024-01-31 23:30', '2024-02-01 00:00', '2024-03-01 00:00']
        expected = frame(times, Qh=[np.nan, 1.2345678901234567, -2.], Tair=[270., 271., 272.])
        with tempfile.TemporaryDirectory(dir=root_name) as name:
            root = Path(name)/'processed'
            path = write_l1(expected, 'Moscow', 'MSU_2m', root=root)
            self.assertEqual(path.relative_to(root).as_posix(), 'L1/Moscow/MSU-MO-2m_fluxes.csv')
            write_l1(frame(times, Tair=[280., 281., 282.]), 'Moscow', 'MSU-MO-2m', root=root, kind='meteo')
            actual = read_city('Moscow', level='L1', root=root)
            self.assertEqual(list(actual), ['MSU-MO-2m'])
            pd.testing.assert_frame_equal(actual['MSU-MO-2m'], expected, check_exact=True)
            with self.assertRaises(ValueError):
                write_l1(expected.iloc[:0], 'Moscow', 'MSU_2m', root=root)
            pd.testing.assert_frame_equal(read_l0(path), expected, check_exact=True)

    def test_zvenigorod_off_grid_rows_excluded_without_rounding(self):
        flux = frame(['2020-11-24 10:20', '2020-11-24 10:30', '2020-11-24 11:00', '2020-11-24 11:03'],
                     Qh=[np.nan, np.nan, 2., 3.], Wind=[np.nan, np.nan, 4., 5.])
        original = flux.copy()
        source = L1Source('Moscow', 'fluxes', Path('IAP-Zvenigorod_56m'))
        selected, excluded, grid = prepare_flux(flux, source)
        pd.testing.assert_frame_equal(selected, flux.iloc[1:3], check_exact=True)
        pd.testing.assert_frame_equal(excluded, flux.iloc[[0,3]], check_exact=True)
        pd.testing.assert_frame_equal(flux, original, check_exact=True)
        self.assertEqual(grid.interval, pd.Timedelta('30min'))

    def test_flux_grid_step_and_phase_with_gaps(self):
        f = frame(['2024-01-01 00:10', '2024-01-01 00:40', '2024-01-01 01:40'], Qh=[1.,2.,3.])
        selected, dropped, grid = prepare_flux(f, L1Source('Moscow', 'fluxes', Path('MSU-MO_2m')))
        pd.testing.assert_frame_equal(selected, f, check_exact=True)
        self.assertTrue(dropped.empty)
        self.assertEqual(grid.interval, pd.Timedelta('30min'))
        self.assertEqual(grid.phase, pd.Timedelta('10min'))
        self.assertEqual(grid.labels(pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-01 01:15')).tolist(),
                         list(pd.to_datetime(['2024-01-01 00:10', '2024-01-01 00:40', '2024-01-01 01:10'])))

    def test_off_grid_and_duplicate_flux_labels_are_rejected(self):
        source = L1Source('Moscow', 'fluxes', Path('MSU-MO_2m'))
        for labels in (['00:00', '00:00', '00:30'], ['00:00', '00:30', '01:01']):
            f = frame([f'2024-01-01 {label}' for label in labels], Qh=[1.,2.,3.])
            with self.assertRaises(ValueError):
                prepare_flux(f, source)

    def test_meteo_only_at_exact_labels_and_inside_its_own_period(self):
        f = frame(['2024-01-01 00:10', '2024-01-01 01:10', '2024-01-01 01:20'], Tair=[275., 276., 277.])
        grid = FluxGrid(pd.Timedelta('30min'), pd.Timedelta('10min'))
        result, diagnostic = prepare_meteo(f, grid)
        self.assertEqual(result.columns.tolist(), ['Tair'])
        pd.testing.assert_frame_equal(result, f.reindex(result.index), check_exact=True)
        self.assertEqual(result.index[-1], pd.Timestamp('2024-01-01 01:10'))
        self.assertEqual(diagnostic.meteo_timestamp_present.tolist(), [1,0,1])

    def test_sources_without_matching_flux_heights_and_standalone_sensors(self):
        flux = frame(['2024-01-01 00:20', '2024-01-01 00:40'], Qh=[1.,2.], Tair=[270.,271.])
        meteo = frame(['2024-01-01 00:00', '2024-01-01 00:40', '2024-01-01 01:20'], Tair=[280.,281.,282.])
        radiation = frame(['2023-12-31 23:59:30', '2024-01-01 00:00', '2024-01-01 01:00:30'],
                          SWdown=[20.,40.,60.])
        with tempfile.TemporaryDirectory(dir=ROOT/'_temp') as name:
            root = Path(name)/'processed'
            target = Path(name)/'candidate'/'processed'
            write_l0(flux, 'Tomsk', 'fluxes', 'TFN-IMCES_30m', root=root)
            write_l0(meteo, 'Tomsk', 'meteo', 'TFN-IMCES_100m', root=root)
            write_l0(meteo, 'Tomsk', 'meteo', 'TFN-IMCES_AMIIS', root=root)
            write_l0(radiation, 'Tomsk', 'radiation', 'TFN-IMCES_AMIIS', root=root)
            write_l0(radiation, 'Tomsk', 'auxiliary', 'TFN-IMCES_AMIIS', root=root)
            summary, _ = build_l1_products(root, target, cities=['Tomsk'])
            expected = {'TFN-IMCES-30m_fluxes.csv', 'TFN-IMCES-100m_meteo.csv',
                        'TFN-IMCES-meteo.csv', 'TFN-IMCES-radiation.csv'}
            self.assertEqual({p.name for p in (target/'L1/Tomsk').iterdir()}, expected)
            self.assertTrue(summary.interval_minutes.eq(20).all())
            out = read_l0(target/'L1/Tomsk/TFN-IMCES-30m_fluxes.csv')
            pd.testing.assert_frame_equal(out[flux.columns], flux, check_exact=True)
            self.assertEqual(tuple(out.columns[len(flux.columns):]), FLAG_COLUMNS)
            for file in ('TFN-IMCES-100m_meteo.csv', 'TFN-IMCES-meteo.csv'):
                out = read_l0(target/'L1/Tomsk'/file)
                pd.testing.assert_frame_equal(out, meteo.reindex(out.index), check_exact=True)
                self.assertEqual(out.index[[0,-1]].tolist(), meteo.index[[0,-1]].tolist())
            out = read_l0(target/'L1/Tomsk/TFN-IMCES-radiation.csv')
            self.assertEqual(out.columns.tolist(), ['SWdown'])
            self.assertEqual(out.SWdown.dropna().tolist(), [30., 60.])
            self.assertEqual(out.index[[0,-1]].tolist(), list(pd.to_datetime(['2024-01-01', '2024-01-01 01:20'], format='mixed')))

    def test_discovery_excludes_lod_and_archival_zvenigorod(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'_temp') as name:
            root = Path(name)/'processed'
            flux = frame(['2024-01-01 00:00', '2024-01-01 00:30'], Qh=[1.,2.])
            for series in ('MSU-MO_2m', 'IAP-Zvenigorod_56m', 'IAP-Zvenigorod_56m_SNC', 'LOD_31m'):
                write_l0(flux, 'Moscow', 'fluxes', series, root=root)
            write_l0(flux, 'Moscow', 'meteo', 'LOD_31m', root=root)
            names = {s.path.name for s in source_configs(root, ['Moscow'])}
            self.assertEqual(names, {'MSU-MO_2m', 'IAP-Zvenigorod_56m'})

    def test_radiation_end_labels_and_flux_gaps(self):
        period = pd.DatetimeIndex(['2024-01-01 00:30', '2024-01-01 01:30'], name='datetime_utc')
        native = frame(['2024-01-01 00:00', '2024-01-01 00:15', '2024-01-01 00:30', '2024-01-01 01:00',
                        '2024-01-01 01:15', '2024-01-01 01:30'], SWdown=[999., 10., 30., 999., np.nan, 40.])
        with tempfile.TemporaryDirectory(dir=ROOT/'_temp') as name:
            path=Path(name)/'radiation.csv'
            native.to_csv(path)
            data, counts, rows = aggregate_radiation_csv(path, period, '30min', ['SWdown'])
            self.assertEqual(data.SWdown.tolist(), [20.,40.])
            self.assertEqual(counts.SWdown.tolist(), [2,1])
            self.assertEqual(rows.tolist(), [2,2])

    def test_three_second_samples_and_month_boundary(self):
        index = pd.date_range('2024-02-29 23:30', '2024-03-01 00:30', freq='3s', name='datetime_utc')
        target = pd.DatetimeIndex(['2024-03-01 00:00', '2024-03-01 00:30'], name=index.name)
        radiation = pd.DataFrame({'SWdown': np.arange(1201, dtype=float),
                                  'LWdown': 100 + np.arange(1201, dtype=float)}, index=index)
        radiation.iloc[123, 1] = np.nan
        with tempfile.TemporaryDirectory(dir=ROOT/'_temp') as name:
            path = Path(name) / 'radiation.csv'
            radiation.to_csv(path)
            for chunksize in [137, 601]:
                actual, counts, rows = aggregate_radiation_csv(path, target, '30min', list(radiation), chunksize)
                self.assertEqual(actual.SWdown.tolist(), [300.5,900.5])
                self.assertEqual(rows.tolist(), [600,600])
                self.assertEqual(counts.LWdown.tolist(), [599,600])
                verification = verify_radiation_csv(path, target, '30min', actual,
                                                    radiation_diagnostics(counts, rows), list(radiation), chunksize=113)
                self.assertTrue(verification.means_equal_within_roundoff.all())

    def test_duplicate_radiation_across_chunks_is_rejected(self):
        native = frame(['2024-01-01 00:15', '2024-01-01 00:15'], SWdown=[10., 20.])
        target = pd.DatetimeIndex(['2024-01-01 00:30'],name='datetime_utc')
        with tempfile.TemporaryDirectory(dir=ROOT/'_temp') as name:
            path = Path(name) / 'radiation.csv'
            native.to_csv(path)
            with self.assertRaises(ValueError):
                aggregate_radiation_csv(path, target, '30min', ['SWdown'], chunksize=1)

    def test_publish_replaces_old_l1_and_keeps_l0_l2(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'_temp') as name:
            base = Path(name)
            root, run = base/'processed', base/'run'
            flux = frame(['2024-01-01 00:00','2024-01-01 00:30'],Qh=[1.,2.])
            for city, station in [('Moscow','MSU-MO_2m'), ('Tomsk','TFN-IMCES_30m'), ('Beijing','IAP-Beijing_47m')]:
                write_l0(flux, city, 'fluxes', station, root=root)
            old = root/'L1/Moscow/MSU-MO-2m.csv'
            old.parent.mkdir(parents=True)
            old.write_bytes(b'combined CSV')
            l2 = root/'L2/sentinel.csv'
            l2.parent.mkdir()
            l2.write_bytes(b'existing L2')
            l0 = {p:p.read_bytes() for p in (root/'L0').rglob('*.csv')}
            stage_l1(run, root)
            self.assertEqual(old.read_bytes(), b'combined CSV')
            publish_l1(run, root)
            self.assertFalse(old.exists())
            self.assertEqual((run/'backup/processed/L1/Moscow/MSU-MO-2m.csv').read_bytes(), b'combined CSV')
            saved = read_l0(series_file('Moscow','MSU-MO-2m',root=root))
            pd.testing.assert_frame_equal(saved[flux.columns], flux, check_exact=True)
            self.assertEqual(tuple(saved.columns[len(flux.columns):]), FLAG_COLUMNS)
            self.assertEqual(l2.read_bytes(), b'existing L2')
            self.assertEqual({p:p.read_bytes() for p in l0}, l0)


class FluxQCTests(unittest.TestCase):
    def test_hard_limits_cd_and_hourly_expansion_keep_original_values(self):
        index = pd.date_range('2024-01-01', periods=6, freq='20min', name='datetime_utc')
        raw = frame(index, Qh=[np.nan, 10., 700., 10., 10., 10.], Qtau=[np.nan, .1, .1, .1, .1, .1],
                    Ustar=[np.nan, .5, .5, .5, 2., .5], Wind=[np.nan, 1., 1., 1., 1., 1.])
        checked = apply_flux_qc(raw)
        pd.testing.assert_frame_equal(checked[raw.columns], raw, check_exact=True)
        self.assertEqual(checked.qc_tot.tolist(), [1,1,0,1,0,1])
        self.assertEqual(checked.qc_tot_noCd.tolist(), [1,1,0,1,1,1])
        self.assertEqual(checked.qc_tot_noCd_ext.tolist(), [1,0,0,0,1,1])
        self.assertEqual(checked.Cd.iloc[4], 4.)
        self.assertEqual(tuple(checked.columns[-4:]), FLAG_COLUMNS)

    def test_statistical_outlier_inside_physical_limits_is_flagged(self):
        index = pd.date_range('2024-01-01', periods=21*48, freq='30min', name='datetime_utc')
        raw = frame(index, Qh=10.)
        raw.loc['2024-01-10 12:00', 'Qh'] = 400.
        checked = apply_flux_qc(raw)
        self.assertEqual(checked.index[checked.qc_tot.eq(0)].tolist(), [pd.Timestamp('2024-01-10 12:00')])
        self.assertEqual(checked.Qh.max(), 400.)


if __name__ == '__main__':
    unittest.main()
