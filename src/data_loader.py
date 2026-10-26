"""Load and validate the raw flight dataset."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "flights.csv"


def load_data(path=DEFAULT_DATA_PATH):
    """Load flight data from a CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def validate_data(df):
    """Check that essential columns exist."""
    required = {"year", "month", "day", "hour", "minute",
                "carrier", "origin", "dest", "distance",
                "arr_delay", "time_hour"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return True
