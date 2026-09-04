# app/reports/bidding.py
"""Pure builders behind the bid-value tables and Excel export.

Moved verbatim out of `app/pages/results/visualizations/bidding_viz.py`: same
rounding, same column order, same sheet names, same openpyxl formatting. The
only changes are mechanical -- the vendors list and the timestamp configuration
come in as arguments instead of being read from `st.session_state`, and the
block that used to write straight into `st.download_button` now returns the
xlsx bytes the page hands to it.

Nothing in this module imports Streamlit or touches session state.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from app.reports.timestamps import TimestampConverter
from app.reports.xlsx import apply_bid_price_formatting, to_xlsx_bytes

# `apply_bid_price_formatting` is the `_apply_price_formatting_bid` that used to
# live in bidding_viz.py, moved verbatim into app/reports/xlsx.py.
apply_price_formatting_bid = apply_bid_price_formatting


def collect_bid_values(df):
    """Every numeric `bid_value` across all agents' purchase requests."""
    all_bids = []

    for idx, row in df.iterrows():
        requests = row.get('purchase_requests', [])
        if isinstance(requests, list):
            for req in requests:
                if isinstance(req, dict):
                    bid_val = req.get('bid_value')
                    # Only include actual numeric bid values (not "N/A")
                    if bid_val != 'N/A' and bid_val is not None:
                        try:
                            all_bids.append(float(bid_val))
                        except (ValueError, TypeError):
                            pass

    return all_bids


def bid_stats_frame(all_bids) -> pd.DataFrame:
    """The '📊 Statistics' table beside the bid-value histogram."""
    return pd.DataFrame({
        'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        'Value': [
            f"{len(all_bids):,}",
            f"€{np.mean(all_bids):.2f}",
            f"€{np.median(all_bids):.2f}",
            f"€{np.std(all_bids):.2f}",
            f"€{min(all_bids):.2f}",
            f"€{max(all_bids):.2f}"
        ]
    })


def build_bid_value_export(df, vendors_data, base_time, duration_hours, periods):
    """
    Build transaction-level export data for bid transactions only.

    Returns a list of transaction records with fields:
    - Transaction ID
    - Agent ID
    - Honesty_Humility
    - Assigned Allowance Level
    - Study Program
    - Group_experiment
    - TWT+Sospeso [=AW2+AX2]{Periods 1+2}
    - income
    - Vendor ID
    - Vendor Price
    - Bid Value
    - Purchase Timestamp
    - Period

    Note: This export only includes BID transactions (typically from Regular customers
    who chose to bid rather than use Purchase Now).
    """
    bid_records = []

    if 'purchase_requests' not in df.columns:
        return bid_records

    # Use centralized timestamp converter
    ts_converter = TimestampConverter(base_time, duration_hours, periods)

    # Build vendor lookup
    vendor_lookup = {}
    if vendors_data:
        for vendor in vendors_data:
            vendor_id = vendor.get('vendor_id')
            if vendor_id is not None:
                vendor_lookup[vendor_id] = vendor
                vendor_lookup[str(vendor_id)] = vendor

    for idx, row in df.iterrows():
        # Get agent information
        agent_id = row.get('agent_id', idx + 1)

        # Agent traits
        honesty_humility = row.get('Honesty_Humility', np.nan)
        allowance_level = row.get('Assigned Allowance Level', np.nan)
        study_program = row.get('Study Program', '')

        # Group_experiment with fallbacks
        group_experiment = row.get('Group_experiment', '')
        if group_experiment == '' or pd.isna(group_experiment):
            group_experiment = row.get('group', '')
        if group_experiment == '' or pd.isna(group_experiment):
            group_experiment = row.get('group_experiment', '')
        if pd.isna(group_experiment):
            group_experiment = ''

        twt_sospeso = row.get('TWT+Sospeso [=AW2+AX2]{Periods 1+2}', np.nan)

        # Income
        income = np.nan
        if 'income' in row and pd.notna(row['income']):
            income = round(row['income'], 2)
        elif 'actual_allowance' in row and pd.notna(row['actual_allowance']):
            income = round(row['actual_allowance'], 2)

        # Get purchase requests
        purchase_requests = row.get('purchase_requests', [])
        if not isinstance(purchase_requests, list):
            continue

        # Process each purchase request - only include BID transactions
        for req_idx, request in enumerate(purchase_requests):
            if not isinstance(request, dict):
                continue

            # Check if this is a BID transaction
            platform_price = request.get('platformPrice', request.get('platform_price', ''))
            bid_value = request.get('bid_value', 'N/A')

            # Only include actual BID transactions with valid bid values
            if platform_price != 'BID' or bid_value == 'N/A' or bid_value is None:
                continue

            try:
                bid_value_numeric = float(bid_value)
            except (ValueError, TypeError):
                continue

            # Get transaction ID
            transaction_id = request.get('transaction_id', f"A{agent_id}_R{req_idx+1}")

            # Get vendor info
            vendor_id = request.get('vendorID', request.get('vendor_id'))
            vendor_price = np.nan

            if vendor_id is not None:
                lookup_key = vendor_id
                if isinstance(vendor_id, float) and vendor_id.is_integer():
                    lookup_key = int(vendor_id)

                if lookup_key in vendor_lookup:
                    vendor_price = vendor_lookup[lookup_key].get('price', np.nan)
                elif str(lookup_key) in vendor_lookup:
                    vendor_price = vendor_lookup[str(lookup_key)].get('price', np.nan)

            # Get timestamp
            timestamp_hours = request.get('timestamp_hours', np.nan)
            ts_result = ts_converter.convert(timestamp_hours)
            period = ts_result['period']
            request_datetime = ts_result['datetime']

            # Build record
            record = {
                'Transaction ID': transaction_id,
                'Agent ID': agent_id,
                'Honesty_Humility': honesty_humility,
                'Assigned Allowance Level': allowance_level,
                'Study Program': study_program,
                'Group_experiment': group_experiment,
                'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': twt_sospeso,
                'income': income,
                'Vendor ID': f"Vendor {int(vendor_id)}" if vendor_id is not None and not pd.isna(vendor_id) else '',
                'Vendor Price': vendor_price,
                'Bid Value': bid_value_numeric,
                'Purchase Timestamp': ts_result['formatted'],
                'Period': period,
                '_sort_datetime': request_datetime  # Hidden for sorting
            }

            bid_records.append(record)

    # Sort by timestamp
    if bid_records:
        bid_records.sort(key=lambda x: x['_sort_datetime'] if isinstance(x['_sort_datetime'], datetime) else datetime.min)

        # Remove hidden sort column
        for record in bid_records:
            record.pop('_sort_datetime', None)

    return bid_records


def bid_values_xlsx(bid_df: pd.DataFrame) -> bytes:
    """'Total' sheet plus one sheet per period, with the bid price formatting."""
    sheets = {'Total': bid_df}

    # Additional sheets by Period
    if 'Period' in bid_df.columns:
        periods = sorted(bid_df['Period'].dropna().unique())
        for period_val in periods:
            period_df = bid_df[bid_df['Period'] == period_val]
            sheet_name = f'Period {int(period_val)}'
            sheets[sheet_name] = period_df

    return to_xlsx_bytes(sheets, apply_bid_price_formatting)


def bidding_range(platform_markup, price_range, example_vendor_price):
    """The illustrative bidding range: (baseline, min bid, max bid)."""
    baseline_price = (1 + platform_markup) * example_vendor_price  # Pc = (1+m) × vendor_price
    min_bid_price = (1 - price_range) * baseline_price      # Pmb = (1-r) × Pc
    max_bid_price = (1 + price_range) * baseline_price      # Ppn = (1+r) × Pc

    return baseline_price, min_bid_price, max_bid_price
