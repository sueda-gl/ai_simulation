# app/utils/timestamp_utils.py
"""
Session-state readers for the timestamp handling shared by all Excel exports.

The arithmetic itself lives in `app.reports.timestamps`, which is Streamlit-free
and takes `base_time` / `duration_hours` / `periods` explicitly (B5). This module
is the UI-side seam: it reads those three values out of `st.session_state` and
delegates. Everything it exports keeps the signature and the behaviour it had
before the split, so the pages can go on importing from here.

USAGE:
    from app.utils.timestamp_utils import TimestampConverter

    # Create a converter instance (caches base time and duration for consistency)
    converter = TimestampConverter()

    # Convert timestamp_hours to various formats
    result = converter.convert(timestamp_hours)
    # result = {
    #     'datetime': datetime object,
    #     'formatted': 'DD/MM/YYYY HH:MM' string,
    #     'date': date object,
    #     'time': time object,
    #     'period': int,
    #     'timestamp_hours': float (original)
    # }

    # Or use individual functions:
    from app.utils.timestamp_utils import (
        get_simulation_base_time,
        get_duration_hours,
        calculate_period,
        timestamp_hours_to_datetime,
        timestamp_hours_to_formatted_string
    )
"""

from datetime import datetime
from typing import Optional, Dict, Any
import streamlit as st

from app.reports import timestamps as _ts
from app.reports.timestamps import (
    DEFAULT_TIMESTAMP_FORMAT,
    DEFAULT_DURATION_HOURS,
    DEFAULT_PERIODS,
)


# ============================================================================
# SESSION READERS
# ============================================================================

def get_simulation_base_time() -> datetime:
    """
    Get the base time for timestamp calculations.

    Uses midnight of current day for consistency across all exports.
    This ensures timestamps are relative to a known starting point.

    Returns:
        datetime: Midnight of current day (00:00:00.000000)

    Example:
        >>> base = get_simulation_base_time()
        >>> print(base)  # 2025-12-18 00:00:00
    """
    return _ts.midnight_of(datetime.now())


def get_duration_hours() -> float:
    """
    Get the duration in hours per period from simulation configuration.

    Checks multiple sources in order:
    1. st.session_state.sim_params.duration_hours
    2. DEFAULT_DURATION_HOURS (2.0)

    Returns:
        float: Duration per period in hours

    Example:
        >>> duration = get_duration_hours()  # Returns 2.0 (or configured value)
    """
    # sim_params object (Page 1 parameters)
    if hasattr(st.session_state, 'sim_params'):
        if hasattr(st.session_state.sim_params, 'duration_hours'):
            return float(st.session_state.sim_params.duration_hours)

    # Fallback to default
    return DEFAULT_DURATION_HOURS


def get_periods() -> int:
    """
    Get the number of periods from simulation configuration.

    Checks multiple sources in order:
    1. st.session_state.sim_params.periods
    2. DEFAULT_PERIODS (15)

    Returns:
        int: Number of periods in the simulation
    """
    # sim_params object (Page 1 parameters)
    if hasattr(st.session_state, 'sim_params'):
        if hasattr(st.session_state.sim_params, 'periods'):
            return int(st.session_state.sim_params.periods)

    # Fallback to default
    return DEFAULT_PERIODS


# ============================================================================
# DELEGATING WRAPPERS (fill the session values in, then call the pure code)
# ============================================================================

def calculate_period(timestamp_hours: float, duration_hours: Optional[float] = None) -> int:
    """
    Calculate the period number for a given timestamp.

    Period numbering starts at 1 (not 0).
    Period boundaries are at multiples of duration_hours.

    Args:
        timestamp_hours: Time in hours from simulation start
        duration_hours: Duration per period in hours (if None, fetched from config)

    Returns:
        int: Period number (1-indexed)

    Example:
        >>> calculate_period(0.0, 2.0)   # Returns 1 (first period)
        >>> calculate_period(1.5, 2.0)   # Returns 1 (still first period)
        >>> calculate_period(2.0, 2.0)   # Returns 2 (second period starts)
        >>> calculate_period(5.5, 2.0)   # Returns 3 (third period)
    """
    if duration_hours is None:
        duration_hours = get_duration_hours()

    return _ts.calculate_period(timestamp_hours, duration_hours)


def timestamp_hours_to_datetime(
    timestamp_hours: float,
    base_time: Optional[datetime] = None
) -> datetime:
    """
    Convert timestamp_hours to a datetime object.

    Args:
        timestamp_hours: Time in hours from simulation start
        base_time: Base datetime to add hours to (if None, uses midnight today)

    Returns:
        datetime: Absolute datetime

    Example:
        >>> dt = timestamp_hours_to_datetime(2.5)
        >>> print(dt)  # 2025-12-18 02:30:00
    """
    if base_time is None:
        base_time = get_simulation_base_time()

    return _ts.timestamp_hours_to_datetime(timestamp_hours, base_time)


def timestamp_hours_to_formatted_string(
    timestamp_hours: float,
    base_time: Optional[datetime] = None,
    format_str: str = DEFAULT_TIMESTAMP_FORMAT
) -> str:
    """
    Convert timestamp_hours to a formatted string.

    Args:
        timestamp_hours: Time in hours from simulation start
        base_time: Base datetime to add hours to (if None, uses midnight today)
        format_str: strftime format string (default: '%d/%m/%Y %H:%M')

    Returns:
        str: Formatted timestamp string

    Example:
        >>> ts = timestamp_hours_to_formatted_string(2.5)
        >>> print(ts)  # '18/12/2025 02:30'
    """
    if base_time is None:
        base_time = get_simulation_base_time()

    return _ts.timestamp_hours_to_formatted_string(timestamp_hours, base_time, format_str)


def convert_timestamp(
    timestamp_hours: float,
    base_time: Optional[datetime] = None,
    duration_hours: Optional[float] = None,
    format_str: str = DEFAULT_TIMESTAMP_FORMAT
) -> Dict[str, Any]:
    """
    Convert timestamp_hours to all commonly needed formats.

    This is a convenience function that returns all timestamp representations
    in a single call, useful when multiple formats are needed.

    Args:
        timestamp_hours: Time in hours from simulation start
        base_time: Base datetime (if None, uses midnight today)
        duration_hours: Duration per period (if None, fetched from config)
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
        >>> result = convert_timestamp(2.5)
        >>> print(result['period'])      # 2
        >>> print(result['formatted'])   # '18/12/2025 02:30'
    """
    if base_time is None:
        base_time = get_simulation_base_time()

    if duration_hours is None:
        duration_hours = get_duration_hours()

    return _ts.convert_timestamp(
        timestamp_hours,
        base_time=base_time,
        duration_hours=duration_hours,
        format_str=format_str
    )


# ============================================================================
# TIMESTAMP CONVERTER CLASS
# ============================================================================

class TimestampConverter(_ts.TimestampConverter):
    """
    A reusable timestamp converter that caches configuration values.

    The conversion methods (`convert`, `to_datetime`, `to_formatted`,
    `to_period`, `get_term_duration`) come unchanged from
    `app.reports.timestamps.TimestampConverter`; this subclass only supplies the
    defaults, which is the part that has to read session state.

    Use this class when converting multiple timestamps in a loop to avoid
    repeated calls to get configuration values.

    Example:
        converter = TimestampConverter()

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
        base_time: Optional[datetime] = None,
        duration_hours: Optional[float] = None,
        periods: Optional[int] = None,
        format_str: str = DEFAULT_TIMESTAMP_FORMAT
    ):
        """
        Initialize the converter with optional custom values.

        Args:
            base_time: Custom base datetime (default: midnight today)
            duration_hours: Custom duration per period (default: from config)
            periods: Custom number of periods (default: from config)
            format_str: Custom timestamp format string
        """
        super().__init__(
            base_time=base_time if base_time is not None else get_simulation_base_time(),
            duration_hours=duration_hours if duration_hours is not None else get_duration_hours(),
            periods=periods if periods is not None else get_periods(),
            format_str=format_str,
        )
