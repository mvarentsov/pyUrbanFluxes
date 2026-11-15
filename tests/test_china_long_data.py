"""Checks for the new formats: end times, independent sensors and duplicates."""

from pathlib import Path
import csv
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import china_long_data as china


class ChinaLongTests(unittest.TestCase):
    def test_meteorology_midnight_missing_and_two_sensors(self):
        with tempfile.TemporaryDirectory(dir=REPO / '_temp') as name:
            path = Path(name) / '240131-30.txt'
            path.write_text('H(m) Vnw Vse D RH T\n80 1.2 2.3 330 999 -5.0\n'
                            'date= 240131\nhour= 23:30 - 24:00\n', encoding='utf-8')
            frame = china.to_utc(china.read_meteorology(path, 'Beijing'))
            self.assertEqual(frame.index[0], pd.Timestamp('2024-01-31 16:00:00'))
            self.assertEqual(frame.Wind_nw.iloc[0], 1.2)
            self.assertEqual(frame.Wind_se.iloc[0], 2.3)
            self.assertTrue(pd.isna(frame.RHair.iloc[0]))
            self.assertEqual(frame.Tair.iloc[0], -5)

    def test_meteorology_period_and_date_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO / '_temp') as name:
            path = Path(name) / '240201-60.txt'
            for date, period in [('240202', '0:00 - 1:00'), ('240201', '0:00 - 0:30')]:
                path.write_text(f'80 1 2 330 80 10\ndate= {date}\nhour= {period}\n', encoding='utf-8')
                with self.assertRaises(ValueError):
                    china.read_meteorology(path, 'Xianghe')

    def test_same_observation_deduplicated_but_conflict_not_averaged(self):
        index = pd.DatetimeIndex(['2024-01-01', '2024-01-01'], name='datetime_utc')
        frame = pd.DataFrame({'a': [1, 1], 'b': [np.nan, np.nan]}, index=index)
        result, removed = china.unique_observations(frame, 'test')
        self.assertEqual((len(result), removed), (1, 1))
        frame.iloc[1, 0] = 2
        with self.assertRaises(ValueError):
            china.unique_observations(frame, 'test')

    def test_flux_uses_short_notebook_definitions(self):
        for city in ['Beijing', 'Xianghe']:
            index = pd.DatetimeIndex(['2024-01-01 00:00:00'])
            frame = pd.DataFrame({name: [1.0] for name in china.FLUX_INPUT[city]}, index=index)
            if city == 'Beijing':
                frame['air_density'] = 1.2
                frame['u*'] = 2.0
                frame['air_temperature'] = 280.0
                frame['air_pressure'] = 100000.0
                frame['u_var'] = 9.0
            else:
                frame['rho_d_Avg'] = 1.2
                frame['u_star_R'] = 2.0
                frame['wTs_Cov_R_F_SND'] = 0.1
                frame['Cp'] = 1000.0
                frame['rho_a_Avg'] = 1.25
                frame['wH2O_Cov_R_F'] = 0.02
            actual = china.flux_frame(city, frame)
            self.assertAlmostEqual(actual.Qtau.iloc[0], 4.8)
            self.assertEqual(actual.index[0], pd.Timestamp('2023-12-31 16:00:00'))
            self.assertEqual(list(actual.columns), list(china.FLUX_MAP[city].values()))
            if city == 'Beijing':
                self.assertEqual(actual.Tair.iloc[0], 280)
                self.assertEqual(actual.Pair.iloc[0], 100000)
                self.assertEqual(actual.Wind_U_std.iloc[0], 3)
            else:
                self.assertEqual(actual.Qh.iloc[0], 125)
                self.assertAlmostEqual(actual.Qle.iloc[0], 50.02)

    def test_uppercase_and_embedded_level_folder(self):
        self.assertEqual(china.level_from_path(Path('delivery/80M/a.txt')), '80m')
        self.assertEqual(china.level_from_path(Path('delivery/2024-02-29    32m/a.dat')), '32m')

    def test_empty_stub_does_not_replace_real_observation_or_remove_a_gap(self):
        index = pd.DatetimeIndex(['2024-01-01', '2024-01-01', '2024-01-02'], name='datetime_utc')
        frame = pd.DataFrame({'a': [np.nan, 5, np.nan], 'b': [np.nan, 6, np.nan]}, index=index)
        result, removed = china.unique_observations(frame, 'stubs')
        self.assertEqual(removed, 1)
        self.assertEqual(result.a.iloc[0], 5)
        self.assertTrue(result.iloc[1].isna().all())
        self.assertEqual(result.attrs['empty_duplicate_placeholders_removed'], 1)

    def test_cumulative_radiation_reexports_preserve_only_new_native_rows(self):
        with tempfile.TemporaryDirectory(dir=REPO / '_temp') as name:
            root = Path(name)
            directory = root / 'Xianghe sample Radiation' / '2024-01-31 32m'
            directory.mkdir(parents=True)
            columns = list(china.RADIATION_MAP['Xianghe'])
            for filename, count in [('a.dat', 2), ('a_1.dat', 3), ('a_2.dat', 4)]:
                with (directory / filename).open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['TOA5', 'station'])
                    writer.writerow(['TIMESTAMP', *columns])
                    writer.writerow(['TS', *['W/m2'] * len(columns)])
                    writer.writerow(['', *['Smp'] * len(columns)])
                    for i in range(count):
                        writer.writerow([f'2024-01-01 00:00:{i * 3:02d}', *[i + 0.5] * len(columns)])
            outputs, sources = china.export_radiation('Xianghe', root / 'output', root)
            frame = pd.read_csv(root / 'output' / outputs[0]['file'], index_col=0, parse_dates=[0])
            self.assertEqual(len(frame), 4)
            self.assertEqual(outputs[0]['duplicate_rows_removed'], 5)
            self.assertEqual(frame.SWdown.tolist(), [0.5, 1.5, 2.5, 3.5])
            self.assertTrue(frame.index.is_unique)
            self.assertEqual(frame.index.to_series().diff().dropna().unique().tolist(), [pd.Timedelta(seconds=3)])
            self.assertEqual(len(sources), 3)


if __name__ == '__main__':
    unittest.main()
