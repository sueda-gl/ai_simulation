# app/reports/donation.py
"""Pure builders behind the donation results exports (Decisions 3 and 13).

Everything here was moved VERBATIM out of
`app/pages/results/visualizations/donation_viz.py`: same rounding, same column
order, same sheet names, same openpyxl number formats. The only edits are the
ones the report-module convention asks for:

* `st.session_state.vendors` became the explicit `vendors` argument of
  :func:`build_donation_transaction_export`;
* the base time / period duration / period count its `TimestampConverter` used
  to read out of session state are explicit arguments, so this module stays
  Streamlit-free and builds the (pure) converter itself;
* the `st.*` calls were replaced by return values.

No Streamlit, no session state: pandas / numpy / openpyxl only.

The workbook plumbing (`to_xlsx_bytes`) and the donation number formatter
live in `app/reports/xlsx.py` (B1); this module imports them rather than
keeping its own copy.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from app.reports.timestamps import TimestampConverter
from app.reports.xlsx import apply_donation_price_formatting, to_xlsx_bytes

# The pricing constants the transaction export was written with. They are
# hard-coded defaults, NOT the Page-1 simulation parameters: this export never
# read them (see the original comment at donation_viz.py "defaults; this export
# never read the Page-1 parameters"). Kept as keyword defaults so the values are
# named and overridable, and deliberately not passed by the page.
DEFAULT_MARKET_PRICE = 100.0
DEFAULT_PLATFORM_MARKUP = 0.1
DEFAULT_PRICE_RANGE = 0.25

# The trait columns the donation_default agent-level export offers, in order.
DONATION_DEFAULT_TRAIT_COLUMNS = [
    'Assigned Allowance Level',
    'Group_experiment',
    'Honesty_Humility',
    'Study Program',
    'TWT+Sospeso [=AW2+AX2]{Periods 1+2}',
]


# ---------------------------------------------------------------------------
# donation_default: agent-level export + on-screen statistics
# ---------------------------------------------------------------------------
def donation_default_stats_frame(numeric_data: pd.Series) -> pd.DataFrame:
    """The '📈 Statistics' table shown beside the donation_default histogram."""
    stats = numeric_data.describe()
    return pd.DataFrame({
        'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
        'Donation Rate': [f"{stats[key]:.2%}" for key in ['mean', 'std', 'min', 'max', '50%', '25%', '75%']]
    })


def donation_default_export_columns(df: pd.DataFrame, decision_name: str) -> list:
    """The columns the donation_default export offers, in the order it used."""
    # Build export dataframe with requested columns
    export_columns = []

    # Add Agent ID first
    if 'agent_id' in df.columns:
        export_columns.append('agent_id')

    # Add trait columns that exist in the dataframe
    for col in DONATION_DEFAULT_TRAIT_COLUMNS:
        if col in df.columns:
            export_columns.append(col)

    # Add donation_default column
    if decision_name in df.columns:
        export_columns.append(decision_name)

    return export_columns


def build_donation_default_export(df: pd.DataFrame, export_columns: list) -> pd.DataFrame:
    """The 'Donation Results' sheet: selected columns, `agent_id` -> `Agent ID`."""
    export_df = df[export_columns].copy()

    # Rename agent_id to 'Agent ID' for clarity
    if 'agent_id' in export_df.columns:
        export_df = export_df.rename(columns={'agent_id': 'Agent ID'})

    return export_df


def build_donation_default_xlsx(export_df: pd.DataFrame) -> bytes:
    """The `donation_default_results_{ts}.xlsx` workbook: one 'Donation Results' sheet."""
    return to_xlsx_bytes({'Donation Results': export_df},
                         formatter=apply_donation_price_formatting)


# ---------------------------------------------------------------------------
# final_donation_rate: transaction-level export
# ---------------------------------------------------------------------------
def build_donation_transaction_export(
    df,
    vendors=None,
    *,
    base_time,
    duration_hours,
    periods,
    market_price: float = DEFAULT_MARKET_PRICE,
    platform_markup: float = DEFAULT_PLATFORM_MARKUP,
    price_range: float = DEFAULT_PRICE_RANGE,
):
    """
    Build transaction-level data for all customer types with donation information.

    Returns a list of transaction records with fields:
    - Transaction ID
    - Agent ID
    - Honesty_Humility (agent trait)
    - Assigned Allowance Level
    - Study Program (agent trait)
    - Group_experiment
    - TWT+Sospeso [=AW2+AX2]{Periods 1+2} (agent trait)
    - income (agent trait)
    - Customer Type (Regular, Fixed, Discount)
    - Income Category
    - Purchase Request Type (PN/Bid/Fixed/Discount)
    - Purchase Timestamp (DD/MM/YYYY HH:MM format)
    - Period
    - Customer Price (PN/Bid only, N/A for Fixed/Discount since actual price is unknown)
    - Default Donation Rate
    - Final Donation Rate

    Note: 'Donation Paid' and 'Total Paid by Customer' columns were removed because
    the actual price is unknown for Fixed/Discount customers, making these calculations
    misleading. The donation rate (percentage) is the meaningful decision output.

    `vendors` is the vendor list the page reads off session state; `base_time`,
    `duration_hours` and `periods` are the three session values the page's
    `TimestampConverter` used to read for itself (`get_simulation_base_time()`,
    `get_duration_hours()`, `get_periods()`), so the converter can be built here.
    """
    transaction_records = []

    if 'purchase_requests' not in df.columns:
        return transaction_records

    # Use centralized timestamp converter for consistent handling
    ts_converter = TimestampConverter(base_time, duration_hours, periods)

    # Build vendor lookup dictionary for quick access
    vendor_lookup = {}
    if vendors:
        for vendor in vendors:
            vendor_id = vendor.get('vendor_id')
            if vendor_id is not None:
                vendor_lookup[vendor_id] = vendor
                vendor_lookup[str(vendor_id)] = vendor

    # Calculate standard prices (legacy fallback using market_price)
    baseline_price = (1 + platform_markup) * market_price
    pn_price = (1 + price_range) * baseline_price  # PN price = max bid price
    discount_price = market_price * 0.7  # Assume 30% discount
    fixed_price = market_price  # Fixed price = market price

    for idx, row in df.iterrows():
        # Get agent information
        agent_id = row.get('agent_id', idx + 1)
        allowance_level = row.get('Assigned Allowance Level', np.nan)

        # ====================================================================
        # AGENT TRAITS: Extract standard trait columns (consistent with Disclose Income)
        # ====================================================================
        honesty_humility = row.get('Honesty_Humility', np.nan)
        study_program = row.get('Study Program', np.nan)
        twt_sospeso = row.get('TWT+Sospeso [=AW2+AX2]{Periods 1+2}', np.nan)
        income_value = row.get('income', np.nan)

        # Group_experiment with fallbacks (handle various column naming conventions)
        group_experiment = row.get('Group_experiment', '')
        if group_experiment == '' or pd.isna(group_experiment):
            group_experiment = row.get('group', '')
        if group_experiment == '' or pd.isna(group_experiment):
            group_experiment = row.get('group_experiment', '')
        if pd.isna(group_experiment):
            group_experiment = ''
        # ====================================================================

        income_category_raw = row.get('income_category', np.nan)
        # Use 'N/A' for empty/missing income_category (e.g., regular customers who didn't disclose income)
        if pd.isna(income_category_raw) or income_category_raw == '' or income_category_raw is None:
            income_category = 'N/A'
        else:
            income_category = income_category_raw

        # Get AGENT-LEVEL donation rates (used as fallback only)
        agent_default_rate = row.get('donation_default', np.nan)
        agent_final_rate = row.get('final_donation_rate', agent_default_rate)

        # Convert to numeric for fallback
        try:
            agent_default_rate = float(agent_default_rate) if not pd.isna(agent_default_rate) else 0.10
        except (ValueError, TypeError):
            agent_default_rate = 0.10

        try:
            agent_final_rate = float(agent_final_rate) if not pd.isna(agent_final_rate) else agent_default_rate
        except (ValueError, TypeError):
            agent_final_rate = agent_default_rate

        # Get purchase requests
        purchase_requests = row.get('purchase_requests', [])
        if not isinstance(purchase_requests, list):
            continue

        # Process each purchase request
        for req_idx, request in enumerate(purchase_requests):
            if not isinstance(request, dict):
                continue

            # Get customer type from request
            customer_type = request.get('customer_type', request.get('customerType', 'regular'))
            if isinstance(customer_type, str):
                customer_type_display = customer_type.capitalize()
            else:
                customer_type_display = 'Regular'

            # Get timestamp and convert using centralized utilities
            timestamp_hours = request.get('timestamp_hours', np.nan)
            ts_result = ts_converter.convert(timestamp_hours)

            period = ts_result['period']
            request_datetime = ts_result['datetime']
            purchase_date = ts_result['date']
            purchase_time = ts_result['time']

            # Determine Purchase Request Type and Customer Price
            platform_price = request.get('platformPrice', request.get('platform_price', ''))
            bid_value = request.get('bid_value', 'N/A')
            transaction_id = request.get('transaction_id')

            # Lookup specific vendor price
            vendor_id = request.get('vendorID', request.get('vendor_id'))

            # Normalize vendor_id for lookup (handle float 1.0 -> int 1)
            lookup_key = vendor_id
            if isinstance(vendor_id, float) and vendor_id.is_integer():
                lookup_key = int(vendor_id)

            vendor_price = None
            if lookup_key is not None:
                if lookup_key in vendor_lookup:
                    vendor_price = vendor_lookup[lookup_key].get('price')
                elif str(lookup_key) in vendor_lookup:
                    vendor_price = vendor_lookup[str(lookup_key)].get('price')

            # Recalculate PN price based on actual vendor price if available
            current_pn_price = pn_price  # Default to market-based
            current_fixed_price = fixed_price
            current_discount_price = discount_price

            if vendor_price is not None:
                v_baseline = (1 + platform_markup) * vendor_price
                current_pn_price = (1 + price_range) * v_baseline
                current_fixed_price = vendor_price
                current_discount_price = vendor_price * 0.7

            if platform_price == 'DISCOUNT' or customer_type.lower() == 'discount':
                purchase_request_type = 'Discount'
                customer_price = current_discount_price
            elif platform_price == 'FIXED' or customer_type.lower() == 'fixed':
                purchase_request_type = 'Fixed'
                customer_price = current_fixed_price
            elif platform_price == 'PN':
                purchase_request_type = 'PN'
                customer_price = current_pn_price
            elif platform_price == 'BID' and bid_value != 'N/A':
                purchase_request_type = 'Bid'
                try:
                    customer_price = float(bid_value)
                except (ValueError, TypeError):
                    customer_price = current_pn_price
            else:
                # Default to PN for regular customers
                purchase_request_type = 'PN' if customer_type.lower() == 'regular' else customer_type_display
                customer_price = current_pn_price

            # ====================================================================
            # NEW: Get REQUEST-SPECIFIC donation rate (priority over agent-level)
            # ====================================================================
            # Check if this request has its own donation rate
            request_donation_rate = request.get('final_donation_rate', None)

            # Use request-level if available, otherwise fall back to agent-level
            if request_donation_rate is not None:
                try:
                    final_donation_rate = float(request_donation_rate)
                except (ValueError, TypeError):
                    final_donation_rate = agent_final_rate
            else:
                # No request-level rate, use agent-level fallback
                final_donation_rate = agent_final_rate

            # Ensure valid range
            final_donation_rate = np.clip(final_donation_rate, 0.0, 1.0) if not pd.isna(final_donation_rate) else agent_final_rate
            # ====================================================================

            # Only show price for PN and BID customers (we don't know actual price for Fixed/Discount)
            # Excel formatting will handle 2-decimal display (no rounding of actual values)
            # Use np.nan instead of 'N/A' string to avoid mixed types in the column (Arrow compatibility)
            if purchase_request_type in ['PN', 'Bid']:
                display_customer_price = customer_price
            else:
                display_customer_price = np.nan

            # Build record with standardized timestamp column
            # NOTE: Removed 'Donation Paid' and 'Total Paid by Customer' columns because:
            # - For Fixed/Discount customers, we don't know the actual price paid
            # - Donation rate (percentage) is the meaningful decision output
            # - Actual monetary values would be misleading without knowing the true price
            record = {
                'Transaction ID': transaction_id,
                'Agent ID': agent_id,
                'Honesty_Humility': honesty_humility,
                'Assigned Allowance Level': allowance_level,
                'Study Program': study_program,
                'Group_experiment': group_experiment,
                'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': twt_sospeso,
                'income': income_value,
                'Customer Type': customer_type_display,
                'Income Category': income_category,
                'Purchase Request Type': purchase_request_type,
                'Purchase Timestamp': ts_result['formatted'],
                'Period': period,
                'Customer Price': display_customer_price,
                'Default Donation Rate': agent_default_rate,
                'Final Donation Rate': final_donation_rate,
                '_sort_datetime': request_datetime  # Hidden column for sorting
            }

            transaction_records.append(record)

    # Sort all records by timestamp in chronological order
    if transaction_records:
        transaction_records.sort(key=lambda x: x['_sort_datetime'] if isinstance(x['_sort_datetime'], datetime) else datetime.min)

        # Remove the hidden sorting column before returning
        for record in transaction_records:
            record.pop('_sort_datetime', None)

    return transaction_records


def donation_transactions_frame(transaction_records: list) -> pd.DataFrame:
    """The transaction records as the DataFrame the page previews and exports."""
    # Create DataFrame from records
    transactions_df = pd.DataFrame(transaction_records)

    # Sort by Period and Agent ID
    transactions_df = transactions_df.sort_values(['Period', 'Agent ID'])

    return transactions_df


def build_donation_transactions_xlsx(transactions_df: pd.DataFrame) -> bytes:
    """The `donation_transactions_{ts}.xlsx` workbook: 'Total' + one sheet per period."""
    # Sheet 1: Total (all periods combined)
    sheets = {'Total': transactions_df}

    # Additional sheets: One per Period
    periods = sorted(transactions_df['Period'].dropna().unique())
    for period_val in periods:
        period_df = transactions_df[transactions_df['Period'] == period_val]
        sheet_name = f'Period {int(period_val)}'
        sheets[sheet_name] = period_df

    return to_xlsx_bytes(sheets, formatter=apply_donation_price_formatting)
