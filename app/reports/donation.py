# app/reports/donation.py
"""Pure builders behind the donation results exports (Decisions 3 and 13).

The sheet frames and workbook bytes behind
`app/pages/results/visualizations/donation_viz.py`. The vendor list is the
explicit `vendors` argument of :func:`build_donation_transaction_export`, and
the base time / period duration / period count of its `TimestampConverter` are
explicit arguments, so this module stays Streamlit-free and builds the (pure)
converter itself.

No Streamlit, no session state: pandas / numpy / openpyxl only.

The workbook plumbing (`to_xlsx_bytes`) and the donation number formatter
live in `app/reports/xlsx.py`; this module imports them rather than
keeping its own copy.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from app.reports.timestamps import TimestampConverter
from app.reports.xlsx import apply_donation_price_formatting, period_sheets, to_xlsx_bytes
from src.contract.columns import rename_consensus_columns

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


def donation_rate_summary(numeric_data: pd.Series) -> dict:
    """Mean, std, min and max of the donation rates (the metric row)."""
    return {'mean': numeric_data.mean(), 'std': numeric_data.std(),
            'min': numeric_data.min(), 'max': numeric_data.max()}


def donation_distribution_stats(donation_data: pd.Series) -> dict:
    """The 'Distribution Stats' list beside the final donation rate histogram:
    min, 25th / 50th / 75th percentiles, max and range."""
    return {'min': donation_data.min(),
            'q25': donation_data.quantile(0.25),
            'q50': donation_data.quantile(0.50),
            'q75': donation_data.quantile(0.75),
            'max': donation_data.max(),
            'range': donation_data.max() - donation_data.min()}


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
    `duration_hours` and `periods` are the three values the page reads (the
    run's start via `RunContext.base_time`, and `sim_params`), so the converter
    can be built here.
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

    # Customer Price rule (COOPECON AI Agent Documentation 2026-10-07): Purchase Now =
    # (1 + price_range) x (1 + platform_markup) x vendor_price, Bid = the bid value,
    # Fixed / Discount = N/A. The market price is the fallback when no vendor is named.
    baseline_price = (1 + platform_markup) * market_price
    pn_price = (1 + price_range) * baseline_price

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

            # PN price from the vendor's own price when the request names a vendor
            current_pn_price = pn_price  # market-price fallback
            if vendor_price is not None:
                v_baseline = (1 + platform_markup) * vendor_price
                current_pn_price = (1 + price_range) * v_baseline

            if platform_price == 'DISCOUNT' or customer_type.lower() == 'discount':
                purchase_request_type = 'Discount'
                customer_price = np.nan  # set by the platform algorithm; exported as N/A
            elif platform_price == 'FIXED' or customer_type.lower() == 'fixed':
                purchase_request_type = 'Fixed'
                customer_price = np.nan  # set by the platform algorithm; exported as N/A
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
            # Stored rounded to two decimals, like every money column of every export
            # Use np.nan instead of 'N/A' string to avoid mixed types in the column (Arrow compatibility)
            if purchase_request_type in ['PN', 'Bid']:
                display_customer_price = float(f"{customer_price:.2f}")
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


def donation_transactions_summary(transactions_df: pd.DataFrame):
    """The summary row above the transaction-level download: (distinct agents,
    distinct periods, average final donation rate)."""
    num_agents = transactions_df['Agent ID'].nunique()
    num_periods = transactions_df['Period'].nunique()
    avg_donation = transactions_df['Final Donation Rate'].mean()
    return num_agents, num_periods, avg_donation


def build_donation_transactions_xlsx(transactions_df: pd.DataFrame) -> bytes:
    """The `donation_transactions_{ts}.xlsx` workbook: 'Total' + one sheet per period."""
    # this frame always has a Period column (no Total-only fallback)
    sheets = period_sheets(transactions_df, missing_ok=False)
    return to_xlsx_bytes(sheets, formatter=apply_donation_price_formatting)


# ---------------------------------------------------------------------------
# donation-only run: the sheets of the results page's export section
# (`app/pages/results/components/export_section.py`)
# ---------------------------------------------------------------------------

# The trait columns the export section keeps for a donation-only run, in order.
DONATION_ONLY_EXPORT_TRAIT_COLUMNS = ['Honesty_Humility', 'Assigned Allowance Level', 'Study Program',
                                      'Group_experiment', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']


def donation_compare_all_sheets(results_dict) -> dict:
    """Compare-all export: one sheet per population mode ('Copula', 'ResSpec',
    'ResBase') with the agent id, the trait columns (shortened names) and one
    donation column per income mode. A population mode without results has no
    sheet, so the dict may be empty."""
    trait_columns = DONATION_ONLY_EXPORT_TRAIT_COLUMNS
    population_modes = [
        ('copula', 'Copula'),
        ('research_spec', 'ResSpec'),
        ('research_baseline', 'ResBase')
    ]
    
    sheets_data = {}  # Store DataFrames for each sheet
    
    for pop_key, pop_prefix in population_modes:
        # Find the DataFrames for this population mode
        cat_key = f"{pop_key}_categorical"
        cont_key = f"{pop_key}_continuous"
        
        cat_df = results_dict.get(cat_key)
        cont_df = results_dict.get(cont_key)
        
        # Use whichever DataFrame is available for traits (they have the same agents)
        base_df = cat_df if cat_df is not None and not cat_df.empty else cont_df
        
        if base_df is None or base_df.empty:
            continue
        
        # Build DataFrame for this population mode
        sheet_data = {}
        
        # Add Agent ID
        if 'agent_id' in base_df.columns:
            sheet_data['Agent_ID'] = base_df['agent_id'].values
        else:
            sheet_data['Agent_ID'] = list(range(1, len(base_df) + 1))
        
        # Add trait columns
        for trait in trait_columns:
            if trait in base_df.columns:
                # Use shorter column names for readability
                short_trait = trait.replace('Assigned Allowance Level', 'Income_Level')
                short_trait = short_trait.replace('TWT+Sospeso [=AW2+AX2]{Periods 1+2}', 'TWT_Sospeso')
                short_trait = short_trait.replace('Group_experiment', 'Group')
                short_trait = short_trait.replace('Study Program', 'Study_Program')
                sheet_data[short_trait] = base_df[trait].values
        
        # Add donation columns for each income mode
        if cat_df is not None and not cat_df.empty and 'donation_default' in cat_df.columns:
            sheet_data['donation_Categorical'] = cat_df['donation_default'].values
        
        if cont_df is not None and not cont_df.empty and 'donation_default' in cont_df.columns:
            sheet_data['donation_Continuous'] = cont_df['donation_default'].values
        
        # Create DataFrame for this sheet
        sheet_df = pd.DataFrame(sheet_data)
        sheets_data[pop_prefix] = sheet_df

    return sheets_data


def donation_all_configs_frame(results_dict) -> pd.DataFrame:
    """Same-population export of several configurations (e.g. Compare both): the
    first configuration's trait columns, 'Agent ID' first, then one
    `donation_default_<Config>` column per configuration (same agents, so the
    rows line up)."""
    trait_columns = DONATION_ONLY_EXPORT_TRAIT_COLUMNS
    first_config_df = next(iter(results_dict.values()))
    available_traits = [col for col in trait_columns if col in first_config_df.columns]
    combined_df = first_config_df[available_traits].copy()
    
    # Add agent_id if it exists
    if 'agent_id' in first_config_df.columns:
        combined_df['Agent ID'] = first_config_df['agent_id'].values
    
    # Add donation_default from each configuration
    for config_key, config_df in results_dict.items():
        if not config_df.empty and 'donation_default' in config_df.columns:
            config_suffix = config_key.replace('_', ' ').title().replace(' ', '_')
            new_col_name = f"donation_default_{config_suffix}"
            combined_df[new_col_name] = config_df['donation_default'].values
    
    # Reorder columns to put Agent ID first
    if 'Agent ID' in combined_df.columns:
        cols = ['Agent ID'] + [col for col in combined_df.columns if col != 'Agent ID']
        combined_df = combined_df[cols]

    return combined_df


def donation_single_export_frame(df) -> pd.DataFrame:
    """Single-configuration export: the (already donation-only) frame with
    Decision 4's 'consensus' columns named 'integrated' and `agent_id` renamed to
    'Agent ID', first."""
    # (Decision 4 columns say "integrated", not "consensus")
    df_export = rename_consensus_columns(df.copy())
    
    # Rename agent_id to 'Agent ID' for clarity
    if 'agent_id' in df_export.columns:
        df_export = df_export.rename(columns={'agent_id': 'Agent ID'})
    
    # Reorder columns to put Agent ID first
    if 'Agent ID' in df_export.columns:
        cols = ['Agent ID'] + [col for col in df_export.columns if col != 'Agent ID']
        df_export = df_export[cols]

    return df_export
