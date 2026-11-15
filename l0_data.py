"""Locations shared by the five L0 data preparation notebooks."""

import glob
from pathlib import Path


DATA_ROOT = Path(r"D:\Data\IFA\Urban_fluxes_NE")
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
CHINA_SHORT_ROOT = RAW_ROOT / "Beijing" / "short DS"
CHINA_LONG_ROOT = RAW_ROOT / "Beijing" / "long dataset"
MSU_OBS_PATH = Path(r"D:\Data\Meteo\MSK\meteo_data\msu\data_msu_fixed.csv")

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
