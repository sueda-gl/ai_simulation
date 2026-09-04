"""Pure timestamp arithmetic behind every export's period / timestamp columns.

Moved verbatim (B5) out of `app/utils/timestamp_utils.py`, which now keeps only
the three *session readers* (`get_simulation_base_time`, `get_duration_hours`,
`get_periods`) and delegates the arithmetic here. Nothing in this module imports
Streamlit or reads session state: `base_time`, `duration_hours` and `periods`
are always passed in explicitly, so the exact strings and period numbers an
export writes can be computed (and tested) outside a Streamlit script run.

Behaviour is unchanged, including the two long-standing conventions the owner
deferred and asked to be carried over as they are:

* the base time is *today's midnight* (`midnight_of(datetime.now())`), so a
  re-export on a later day moves every timestamp string; and
* period numbering is `int(ts // duration_hours) + 1`, i.e. 1-indexed with a
  NaN/None/negative timestamp and a non-positive duration both collapsing to
  period 1.

USAGE:
    from app.reports.timestamps import TimestampConverter

    converter = TimestampConverter(base_time, duration_hours, periods)
    result = converter.convert(timestamp_hours)
    # result = {
    #     'datetime': datetime object,
    #     'formatted': 'DD/MM/YYYY HH:MM' string,
    #     'date': date object,
    #     'time': time object,
    #     'period': int,
    #     'timestamp_hours': float (original)
    # }

The UI keeps importing `TimestampConverter` from `app.utils.timestamp_utils`,
where a thin subclass fills the three arguments in from session state.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Union

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Default timestamp format for Excel exports (DD/MM/YYYY HH:MM)
DEFAULT_TIMESTAMP_FORMAT = '%d/%m/%Y %H:%M'

# Default duration per period in hours (used as fallback)
DEFAULT_DURATION_HOURS = 2.0

# Default number of periods (used as fallback)
DEFAULT_PERIODS = 15


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def midnight_of(now: datetime) -> datetime:
    """
    Anchor a wall-clock time to the midnight that starts its day.

    This is the base time every export's timestamp column is relative to; the
    caller passes `datetime.now()` (see `get_simulation_base_time`).

    Args:
        now: Wall-clock time to anchor

    Returns:
        datetime: Midnight of that day (00:00:00.000000)

    Example:
        >>> midnight_of(datetime(2025, 12, 18, 14, 37, 5))
        datetime.datetime(2025, 12, 18, 0, 0)
    """
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def calculate_period(timestamp_hours: float, duration_hours: float) -> int:
    """
    Calculate the period number for a given timestamp.

    Period numbering starts at 1 (not 0).
    Period boundaries are at multiples of duration_hours.

    Args:
        timestamp_hours: Time in hours from simulation start
        duration_hours: Duration per period in hours

    Returns:
        int: Period number (1-indexed)

    Example:
        >>> calculate_period(0.0, 2.0)   # Returns 1 (first period)
        >>> calculate_period(1.5, 2.0)   # Returns 1 (still first period)
        >>> calculate_period(2.0, 2.0)   # Returns 2 (second period starts)
        >>> calculate_period(5.5, 2.0)   # Returns 3 (third period)
    """
    if pd.isna(timestamp_hours) or timestamp_hours is None:
        return 1  # Default to period 1 for missing timestamps

    if duration_hours <= 0:
        return 1

    # Period 1 starts at timestamp_hours=0
    # Period 2 starts at timestamp_hours=duration_hours
    # etc.
    if timestamp_hours < 0:
        return 1

    return int(timestamp_hours // duration_hours) + 1


def timestamp_hours_to_datetime(
    timestamp_hours: float,
    base_time: datetime
) -> datetime:
    """
    Convert timestamp_hours to a datetime object.

    Args:
        timestamp_hours: Time in hours from simulation start
        base_time: Base datetime to add hours to

    Returns:
        datetime: Absolute datetime

    Example:
        >>> dt = timestamp_hours_to_datetime(2.5, base_time)
        >>> print(dt)  # 2025-12-18 02:30:00
    """
    if pd.isna(timestamp_hours) or timestamp_hours is None:
        timestamp_hours = 0.0

    return base_time + timedelta(hours=float(timestamp_hours))


def timestamp_hours_to_formatted_string(
    timestamp_hours: float,
    base_time: datetime,
    format_str: str = DEFAULT_TIMESTAMP_FORMAT
) -> str:
    """
    Convert timestamp_hours to a formatted string.

    Args:
        timestamp_hours: Time in hours from simulation start
        base_time: Base datetime to add hours to
        format_str: strftime format string (default: '%d/%m/%Y %H:%M')

    Returns:
        str: Formatted timestamp string

    Example:
        >>> ts = timestamp_hours_to_formatted_string(2.5, base_time)
        >>> print(ts)  # '18/12/2025 02:30'
    """
    dt = timestamp_hours_to_datetime(timestamp_hours, base_time)
    return dt.strftime(format_str)


def convert_timestamp(
    timestamp_hours: float,
    base_time: datetime,
    duration_hours: float = DEFAULT_DURATION_HOURS,
    format_str: str = DEFAULT_TIMESTAMP_FORMAT
) -> Dict[str, Any]:
    """
    Convert timestamp_hours to all commonly needed formats.

    This is a convenience function that returns all timestamp representations
    in a single call, useful when multiple formats are needed.

    Args:
        timestamp_hours: Time in hours from simulation start
        base_time: Base datetime
        duration_hours: Duration per period
        format_str: strftime format string

    Returns:
        dict: All timestamp representations:
            - 'datetime': datetime object
            - 'formatted': formatted string (e.g., '18/12/2025 02:30')
            - 'date': date object
            - 'time': time object
            - 'period': int (1-indexed)
            - 'timestamp_hours': float (original value)

    Example:
        >>> result = convert_timestamp(2.5, base_time)
        >>> print(result['period'])      # 2
        >>> print(result['formatted'])   # '18/12/2025 02:30'
    """
    # Handle missing/invalid timestamp
    if pd.isna(timestamp_hours) or timestamp_hours is None:
        dt = base_time
        period = 1
        original = np.nan
    else:
        dt = timestamp_hours_to_datetime(timestamp_hours, base_time)
        period = calculate_period(timestamp_hours, duration_hours)
        original = float(timestamp_hours)

    return {
        'datetime': dt,
        'formatted': dt.strftime(format_str),
        'date': dt.date(),
        'time': dt.time(),
        'period': period,
        'timestamp_hours': original
    }


# ============================================================================
# TIMESTAMP CONVERTER CLASS
# ============================================================================

class TimestampConverter:
    """
    A reusable timestamp converter that caches configuration values.

    Use this class when converting multiple timestamps in a loop to avoid
    repeated calls to get configuration values.

    Example:
        converter = TimestampConverter(base_time, duration_hours, periods)

        for request in purchase_requests:
            timestamp_hours = request.get('timestamp_hours')
            result = converter.convert(timestamp_hours)

            # Use result['datetime'], result['period'], result['formatted'], etc.

    Attributes:
        base_time: Cached base datetime
        duration_hours: Cached duration per period
        periods: Cached number of periods
        format_str: Timestamp format string
    """

    def __init__(
        self,
        base_time: datetime,
        duration_hours: float = DEFAULT_DURATION_HOURS,
        periods: int = DEFAULT_PERIODS,
        format_str: str = DEFAULT_TIMESTAMP_FORMAT
    ):
        """
        Initialize the converter with explicit configuration values.

        Args:
            base_time: Base datetime every timestamp is relative to
            duration_hours: Duration per period
            periods: Number of periods
            format_str: Timestamp format string
        """
        self.base_time = base_time
        self.duration_hours = duration_hours
        self.periods = periods
        self.format_str = format_str

    def convert(self, timestamp_hours: Union[float, None]) -> Dict[str, Any]:
        """
        Convert timestamp_hours to all formats using cached configuration.

        Args:
            timestamp_hours: Time in hours from simulation start

        Returns:
            dict: All timestamp representations (see convert_timestamp)
        """
        return convert_timestamp(
            timestamp_hours,
            base_time=self.base_time,
            duration_hours=self.duration_hours,
            format_str=self.format_str
        )

    def to_datetime(self, timestamp_hours: Union[float, None]) -> datetime:
        """Convert to datetime using cached base_time."""
        return timestamp_hours_to_datetime(timestamp_hours, self.base_time)

    def to_formatted(self, timestamp_hours: Union[float, None]) -> str:
        """Convert to formatted string using cached values."""
        return timestamp_hours_to_formatted_string(
            timestamp_hours,
            self.base_time,
            self.format_str
        )

    def to_period(self, timestamp_hours: Union[float, None]) -> int:
        """Calculate period using cached duration_hours."""
        return calculate_period(timestamp_hours, self.duration_hours)

    def get_term_duration(self) -> float:
        """Get total term duration in hours."""
        return self.duration_hours * self.periods
