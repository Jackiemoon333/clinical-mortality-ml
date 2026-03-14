from pathlib import Path

# Project root
BASE_DIR = Path.cwd().parent

# Data folders
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Database file
EICU_DB = RAW_DATA_DIR / "eicu_v2_0_1.sqlite3"