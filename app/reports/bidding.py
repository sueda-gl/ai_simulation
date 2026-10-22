# app/reports/bidding.py
"""Pure builders behind the bid-value tables and Excel export.

The tables and workbook bytes `app/pages/results/visualizations/bidding_viz.py`
renders and offers for download. The vendors list and the timestamp
configuration come in as arguments; nothing in this module imports Streamlit or
touches session state.
"""
import numpy as np
import pandas as pd

from app.reports._common import (
    group_experiment_skipping_blanks, money, money_or_missing, purchase_requests_of,
    request_time, sort_chronologically, vendors_by_id_and_str,
)
from app.reports.timestamps import TimestampConverter
from app.reports.xlsx import apply_bid_price_formatting, period_sheets, to_xlsx_bytes


def collect_bid_values(df):
    """Every numeric `bid_value` across all agents' purchase requests."""
    all_bids = []

    for idx, row in df.iterrows():
        for _, req in purchase_requests_of(row):
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

    # Build vendor lookup (int and string keys)
    vendor_lookup = vendors_by_id_and_str(vendors_data)

    for idx, row in df.iterrows():
        # Get agent information
        agent_id = row.get('agent_id', idx + 1)

        # Agent traits
        honesty_humility = row.get('Honesty_Humility', np.nan)
        allowance_level = row.get('Assigned Allowance Level', np.nan)
        study_program = row.get('Study Program', '')

        # Group_experiment with fallbacks
        group_experiment = group_experiment_skipping_blanks(row)

        twt_sospeso = row.get('TWT+Sospeso [=AW2+AX2]{Periods 1+2}', np.nan)

        # Income
        income = np.nan
        if 'income' in row and pd.notna(row['income']):
            income = round(row['income'], 2)
        elif 'actual_allowance' in row and pd.notna(row['actual_allowance']):
            income = round(row['actual_allowance'], 2)

        # Process each purchase request - only include BID transactions
        for req_idx, request in purchase_requests_of(row):
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
            ts_result, period, request_datetime = request_time(ts_converter, request)

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
                # money columns are stored rounded to two decimals
                'Vendor Price': money_or_missing(vendor_price),
                'Bid Value': money(bid_value_numeric),
                'Purchase Timestamp': ts_result['formatted'],
                'Period': period,
                '_sort_datetime': request_datetime  # Hidden for sorting
            }

            bid_records.append(record)

    # Sort by timestamp (drops the hidden sort column)
    sort_chronologically(bid_records)

    return bid_records


def bid_values_xlsx(bid_df: pd.DataFrame) -> bytes:
    """'Total' sheet plus one sheet per period, with the bid price formatting."""
    return to_xlsx_bytes(period_sheets(bid_df), apply_bid_price_formatting)


def bidding_range(platform_markup, price_range, example_vendor_price):
    """The illustrative bidding range: (baseline, min bid, max bid)."""
    baseline_price = (1 + platform_markup) * example_vendor_price  # Pc = (1+m) × vendor_price
    min_bid_price = (1 - price_range) * baseline_price      # Pmb = (1-r) × Pc
    max_bid_price = (1 + price_range) * baseline_price      # Ppn = (1+r) × Pc

    return baseline_price, min_bid_price, max_bid_price
