"""Data loading and time-index preparation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_RAW_PATH = Path("data/raw/de_hourly_power_and_weather.csv")
DEFAULT_PROCESSED_PATH = Path("data/processed/de_hourly_power_and_weather_prepared.csv")
TIME_COLUMNS = ("ds", "timestamp", "utc_timestamp")
REQUIRED_COLUMNS = {
    "Consumption",
    "Wind",
    "Solar",
    "Wind+Solar",
    "temperature",
    "radiation_direct_horizontal",
    "radiation_diffuse_horizontal",
}


def find_time_column(columns: Iterable[str]) -> str:
    """Return the first supported time column found in a dataframe."""
    for column in TIME_COLUMNS:
        if column in columns:
            return column
    raise ValueError(f"No supported time column found. Expected one of: {TIME_COLUMNS}")


def prepare_time_series_frame(
    data: pd.DataFrame,
    required_columns: set[str] | None = REQUIRED_COLUMNS,
) -> pd.DataFrame:
    """Normalize a raw dataframe to sorted hourly time-series format with a ds column."""
    frame = data.copy()
    time_col = find_time_column(frame.columns)
    if time_col != "ds":
        frame = frame.rename(columns={time_col: "ds"})

    frame["ds"] = pd.to_datetime(frame["ds"], utc=True).dt.tz_convert(None)
    frame = frame.sort_values("ds").drop_duplicates(subset="ds").reset_index(drop=True)

    if required_columns is not None:
        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    return frame


def load_power_weather_data(
    processed_path: str | Path = DEFAULT_PROCESSED_PATH,
    raw_path: str | Path = DEFAULT_RAW_PATH,
) -> pd.DataFrame:
    """Load the prepared dataset when available, otherwise fall back to raw data."""
    processed_path = Path(processed_path)
    raw_path = Path(raw_path)

    if processed_path.exists():
        data = pd.read_csv(processed_path)
    elif raw_path.exists():
        data = pd.read_csv(raw_path)
    else:
        raise FileNotFoundError(f"Neither {processed_path} nor {raw_path} exists")

    return prepare_time_series_frame(data)


def save_prepared_data(data: pd.DataFrame, output_path: str | Path = DEFAULT_PROCESSED_PATH) -> Path:
    """Save prepared data using the timestamp column name used by the notebooks."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = data.copy()
    if "ds" in frame.columns:
        frame = frame.rename(columns={"ds": "timestamp"})
    frame.to_csv(output_path, index=False)
    return output_path
