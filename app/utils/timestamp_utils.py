# app/utils/timestamp_utils.py
"""
Centralized timestamp utilities for consistent timestamp handling across all Excel exports.

This module provides a single source of truth for:
- Base time calculation (when simulation starts)
- Period calculation from timestamp_hours
- Timestamp formatting (datetime objects and formatted strings)
- Duration hours retrieval from simulation parameters

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

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np
import streamlit as st


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
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


def get_duration_hours() -> float:
    """
    Get the duration in hours per period from simulation configuration.
    
    Checks multiple sources in order:
    1. st.session_state.simulation_params['simulation']['duration_hours']
    2. st.session_state.sim_params.duration_hours
    3. DEFAULT_DURATION_HOURS (2.0)
    
    Returns:
        float: Duration per period in hours
    
    Example:
        >>> duration = get_duration_hours()  # Returns 2.0 (or configured value)
    """
    # Try simulation_params dict first (preferred)
    if hasattr(st.session_state, 'simulation_params'):
        sim_params = st.session_state.simulation_params.get('simulation', {})
        if 'duration_hours' in sim_params:
            return float(sim_params['duration_hours'])
    
    # Try sim_params object (legacy)
    if hasattr(st.session_state, 'sim_params'):
        if hasattr(st.session_state.sim_params, 'duration_hours'):
            return float(st.session_state.sim_params.duration_hours)
    
    # Fallback to default
    return DEFAULT_DURATION_HOURS


def get_periods() -> int:
    """
    Get the number of periods from simulation configuration.
    
    Checks multiple sources in order:
    1. st.session_state.simulation_params['simulation']['periods']
    2. st.session_state.sim_params.periods
    3. DEFAULT_PERIODS (15)
    
    Returns:
        int: Number of periods in the simulation
    """
    # Try simulation_params dict first (preferred)
    if hasattr(st.session_state, 'simulation_params'):
        sim_params = st.session_state.simulation_params.get('simulation', {})
        if 'periods' in sim_params:
            return int(sim_params['periods'])
    
    # Try sim_params object (legacy)
    if hasattr(st.session_state, 'sim_params'):
        if hasattr(st.session_state.sim_params, 'periods'):
            return int(st.session_state.sim_params.periods)
    
    # Fallback to default
    return DEFAULT_PERIODS


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
    if pd.isna(timestamp_hours) or timestamp_hours is None:
        return 1  # Default to period 1 for missing timestamps
    
    if duration_hours is None:
        duration_hours = get_duration_hours()
    
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
    if pd.isna(timestamp_hours) or timestamp_hours is None:
        timestamp_hours = 0.0
    
    if base_time is None:
        base_time = get_simulation_base_time()
    
    return base_time + timedelta(hours=float(timestamp_hours))


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
    dt = timestamp_hours_to_datetime(timestamp_hours, base_time)
    return dt.strftime(format_str)


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
        self.base_time = base_time if base_time is not None else get_simulation_base_time()
        self.duration_hours = duration_hours if duration_hours is not None else get_duration_hours()
        self.periods = periods if periods is not None else get_periods()
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


# ============================================================================
# PRICE FORMATTING UTILITIES
# ============================================================================

def format_price(price: Union[float, int, None], decimal_places: int = 2) -> Union[float, str]:
    """
    Format a price to a specified number of decimal places.
    
    This provides consistent price formatting across all Excel exports.
    
    Args:
        price: The price value to format (can be float, int, or None)
        decimal_places: Number of decimal places (default: 2)
    
    Returns:
        float: Formatted price as float with specified decimal places
        str: 'N/A' if price is None or NaN
    
    Example:
        >>> format_price(137.5625)   # Returns 137.56
        >>> format_price(100.0)      # Returns 100.00
        >>> format_price(None)       # Returns 'N/A'
        >>> format_price(np.nan)     # Returns 'N/A'
    """
    if price is None or (isinstance(price, float) and pd.isna(price)):
        return 'N/A'
    
    try:
        return round(float(price), decimal_places)
    except (ValueError, TypeError):
        return 'N/A'


# ============================================================================
# UTILITY FUNCTIONS FOR BATCH PROCESSING
# ============================================================================

def add_timestamp_columns(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp_hours',
    converter: Optional[TimestampConverter] = None
) -> pd.DataFrame:
    """
    Add standardized timestamp columns to a DataFrame.
    
    Adds the following columns:
    - {timestamp_col}_datetime: datetime objects
    - {timestamp_col}_formatted: formatted strings
    - {timestamp_col}_date: date objects
    - {timestamp_col}_time: time objects
    - period: period numbers
    
    Args:
        df: DataFrame with timestamp_hours column
        timestamp_col: Name of the timestamp column
        converter: Optional TimestampConverter instance
    
    Returns:
        DataFrame with added columns
    """
    if timestamp_col not in df.columns:
        return df
    
    if converter is None:
        converter = TimestampConverter()
    
    df = df.copy()
    
    # Vectorized conversion (more efficient for large DataFrames)
    results = df[timestamp_col].apply(converter.convert)
    
    df[f'{timestamp_col}_datetime'] = results.apply(lambda x: x['datetime'])
    df[f'{timestamp_col}_formatted'] = results.apply(lambda x: x['formatted'])
    df[f'{timestamp_col}_date'] = results.apply(lambda x: x['date'])
    df[f'{timestamp_col}_time'] = results.apply(lambda x: x['time'])
    df['period'] = results.apply(lambda x: x['period'])
    
    return df
