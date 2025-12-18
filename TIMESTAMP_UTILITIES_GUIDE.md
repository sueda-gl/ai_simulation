# Timestamp Utilities Guide

This document explains the centralized timestamp handling system introduced to ensure consistency across all Excel exports.

## Problem Solved

Previously, timestamp handling was inconsistent across different Excel export files:

| Issue | Before | After |
|-------|--------|-------|
| **Base Time** | Some used `datetime.now()`, others used midnight | All use midnight of current day |
| **Format Output** | String vs datetime vs separate date/time | All use formatted string 'DD/MM/YYYY HH:MM' |
| **Column Names** | 'timestamp', 'Purchase Date/Time', 'Request Date & Time' | All use `'Purchase Timestamp'` |
| **Period Calculation** | Different defaults (1.0 vs 2.0 duration_hours) | Centralized `get_duration_hours()` |
| **Fallback Handling** | Inconsistent (1 vs np.nan for missing data) | Consistent period=1 for missing timestamps |

## Standardized Column Names

All Excel exports now use consistent column names:

| Column | Description | Format |
|--------|-------------|--------|
| `Purchase Timestamp` | When the purchase was made | 'DD/MM/YYYY HH:MM' (string) |
| `Period` | Which period the purchase falls in | Integer (1-indexed) |

## New Centralized Module

**Location:** `app/utils/timestamp_utils.py`

### Quick Usage

```python
from app.utils.timestamp_utils import TimestampConverter

# Create converter (caches config values for efficiency)
ts_converter = TimestampConverter()

# Convert timestamp_hours to all formats
timestamp_hours = request.get('timestamp_hours', np.nan)
ts_result = ts_converter.convert(timestamp_hours)

# Access different formats
period = ts_result['period']           # int: Period number (1-indexed)
datetime_obj = ts_result['datetime']   # datetime: Full datetime object
formatted = ts_result['formatted']     # str: 'DD/MM/YYYY HH:MM'
date_obj = ts_result['date']           # date: Date component only
time_obj = ts_result['time']           # time: Time component only
original = ts_result['timestamp_hours'] # float: Original value (or np.nan)
```

### Individual Functions

For simpler use cases, you can also use individual functions:

```python
from app.utils.timestamp_utils import (
    get_simulation_base_time,    # Returns midnight of current day
    get_duration_hours,          # Gets duration from config (default: 2.0)
    get_periods,                 # Gets number of periods from config (default: 15)
    calculate_period,            # Calculates period from timestamp_hours
    timestamp_hours_to_datetime, # Converts to datetime object
    timestamp_hours_to_formatted_string,  # Converts to 'DD/MM/YYYY HH:MM'
    convert_timestamp,           # Returns dict with all formats
    format_price                 # Formats prices to 2 decimal places
)

# Examples
base_time = get_simulation_base_time()  # 2025-12-18 00:00:00
duration = get_duration_hours()          # 2.0 (from config)
period = calculate_period(5.5)           # 3 (with 2-hour periods)
dt = timestamp_hours_to_datetime(2.5)    # 2025-12-18 02:30:00
ts = timestamp_hours_to_formatted_string(2.5)  # '18/12/2025 02:30'
```

## Files Updated

The following files now use the centralized timestamp utilities:

1. **`app/pages/results/components/export_section.py`**
   - Function: `_build_transaction_level_dataframe()`

2. **`app/pages/results/visualizations/transaction_viz.py`**
   - Function: `_build_purchase_vs_bid_export()`

3. **`app/pages/results/visualizations/donation_viz.py`**
   - Function: `_build_donation_transaction_export()`

4. **`app/pages/results/visualizations/vendor_viz.py`**
   - Function: `_build_purchase_request_export()`
   - Function: `render_vendor_selection()`

5. **`app/pages/results/visualizations/purchasing_viz.py`**
   - Function: `render_purchasing_quantity()` (transaction export section)

## Price Formatting

The module also provides a price formatting utility for consistent display across exports:

```python
from app.utils.timestamp_utils import format_price

# Format prices to 2 decimal places
format_price(137.5625)   # Returns 137.56
format_price(100.0)      # Returns 100.0
format_price(None)       # Returns 'N/A'
format_price(np.nan)     # Returns 'N/A'

# Custom decimal places
format_price(137.5625, decimal_places=3)  # Returns 137.563
```

## Configuration Sources

The utilities check configuration values in this order:

1. `st.session_state.simulation_params['simulation']` (preferred - dict)
2. `st.session_state.sim_params` (legacy - object)
3. Default values (fallback):
   - `duration_hours`: 2.0
   - `periods`: 15

## Timestamp Format

All timestamps use the standardized format: **`DD/MM/YYYY HH:MM`**

Example: `18/12/2025 02:30`

## Period Calculation

Periods are 1-indexed (start at 1, not 0):

| timestamp_hours | duration_hours | Period |
|-----------------|----------------|--------|
| 0.0             | 2.0            | 1      |
| 1.5             | 2.0            | 1      |
| 2.0             | 2.0            | 2      |
| 5.5             | 2.0            | 3      |
| 29.9            | 2.0            | 15     |

## Adding New Excel Exports

When creating a new Excel export with timestamps, follow this pattern:

```python
from app.utils.timestamp_utils import TimestampConverter

def build_export_data(df):
    """Build export data with timestamps."""
    records = []
    
    # Create converter once (caches config values)
    ts_converter = TimestampConverter()
    
    for idx, row in df.iterrows():
        purchase_requests = row.get('purchase_requests', [])
        
        for request in purchase_requests:
            # Get timestamp and convert
            timestamp_hours = request.get('timestamp_hours', np.nan)
            ts_result = ts_converter.convert(timestamp_hours)
            
            record = {
                'Period': ts_result['period'],
                'Timestamp': ts_result['formatted'],
                'Purchase Date': ts_result['date'],
                'Purchase Time': ts_result['time'],
                # ... other fields
            }
            records.append(record)
    
    return records
```

## Testing

To verify timestamps are consistent across exports:

1. Run a simulation
2. Download multiple Excel files (e.g., Transaction-Level, Donation, Vendor Selection)
3. Compare timestamps for the same transaction_id across files
4. All timestamps should match exactly

## Note on File Naming

The `datetime.now()` calls used for file naming (e.g., `simulation_agent_level_20251218_143052.xlsx`) are **not** part of the timestamp utilities. These are intentionally different as they capture the download time, not the simulation time.
