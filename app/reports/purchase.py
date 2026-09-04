"""Pure (Streamlit-free) report builders for the transaction decisions.

Everything the transaction visualisations write to a file or compute as a
statistic, minus the Decision 4 model sheets (those live in `app.reports.rtd`):

* Decision 3 - "purchase now vs bid": the request-level export and the PN/BID
  request statistics.
* Decision 4 in DEFAULT (unselected) mode - the priority-lists export and the
  priority statistics shown alongside it.
* Decision 5 - "rejected transaction option": the result value counts.

Moved verbatim from `app/pages/results/visualizations/transaction_viz.py`, which
now imports them back and keeps only the Streamlit rendering. Nothing here
imports Streamlit or reads session state: the pricing parameters, the vendor
list and the `TimestampConverter` are passed in explicitly.
"""
from collections import Counter

import numpy as np
import pandas as pd

from app.reports.xlsx import apply_transaction_price_formatting, to_xlsx_bytes


def build_purchase_vs_bid_export(
    df,
    *,
    ts_converter,
    market_price: float = 100.0,
    platform_markup: float = 0.1,
    price_range: float = 0.25,
    vendors=None,
):
    """
    Build transaction-level export data for regular customers showing purchase vs bid decisions.
    
    Returns a list of transaction records with fields:
    - Agent ID
    - Assigned Allowance Level
    - Group_experiment
    - Customer Type (Regular, Fixed, Discount)
    - Income Category
    - Purchase Request Type (PN/Bid)
    - timestamp (DD/MM/YYYY HH:MM format)
    - Period
    - Customer Price (based on PN price or bid value, using vendor's actual price)
    
    Records are sorted by timestamp in chronological order.
    """
    from datetime import datetime, timedelta
    
    transaction_records = []
    
    if 'purchase_requests' not in df.columns:
        return transaction_records
    
    # Pricing parameters and the vendor list come from the caller (the page
    # reads them off the session's sim_params / vendors).
    vendors_data = vendors

    # Build vendor lookup dictionary for quick access
    vendor_lookup = {}
    if vendors_data:
        for vendor in vendors_data:
            vendor_id = vendor.get('vendor_id')
            if vendor_id is not None:
                # Store with both int and string keys to ensure lookup works
                vendor_lookup[vendor_id] = vendor
                vendor_lookup[str(vendor_id)] = vendor
    
    # Centralized timestamp converter, built by the caller (it reads its base
    # time, period duration and period count off session state)
    
    for idx, row in df.iterrows():
        # Get agent information
        agent_id = row.get('agent_id', idx + 1)
        
        # Agent Traits (matching disclose income export)
        # Honesty_Humility
        honesty_humility = ''
        if 'Honesty_Humility' in row and pd.notna(row['Honesty_Humility']):
            honesty_humility = round(row['Honesty_Humility'], 2)
        
        allowance_level = row.get('Assigned Allowance Level', np.nan)
        
        # Study Program
        study_program = row.get('Study Program', '')
        
        # Group_experiment (with fallbacks)
        group_experiment = ''
        if 'Group_experiment' in row and pd.notna(row['Group_experiment']):
            group_experiment = row['Group_experiment']
        elif 'group' in row and pd.notna(row['group']):
            group_experiment = row['group']
        elif 'group_experiment' in row and pd.notna(row['group_experiment']):
            group_experiment = row['group_experiment']
        
        # TWT+Sospeso
        twt_sospeso = ''
        if 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}' in row and pd.notna(row['TWT+Sospeso [=AW2+AX2]{Periods 1+2}']):
            twt_sospeso = round(row['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'], 2)
        
        # Income
        income = ''
        if 'income' in row and pd.notna(row['income']):
            income = round(row['income'], 2)
        elif 'actual_allowance' in row and pd.notna(row['actual_allowance']):
            income = round(row['actual_allowance'], 2)
        
        # Income category - Regular customers don't have this (assigned in Decision 6 only for Discount/Fixed)
        income_category_raw = row.get('income_category', np.nan)
        # Use 'N/A' for empty/missing income_category (e.g., regular customers who didn't disclose income)
        if pd.isna(income_category_raw) or income_category_raw == '' or income_category_raw is None:
            income_category = 'N/A'
        else:
            income_category = income_category_raw
        
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
            
            # Only include regular customers for this export
            if customer_type.lower() != 'regular':
                continue
            
            # Get timestamp and convert using centralized utilities
            timestamp_hours = request.get('timestamp_hours', np.nan)
            ts_result = ts_converter.convert(timestamp_hours)
            
            period = ts_result['period']
            timestamp_str = ts_result['formatted']
            sort_key = ts_result['timestamp_hours'] if not pd.isna(ts_result['timestamp_hours']) else 0.0
            
            # Determine Purchase Request Type and Customer Price
            platform_price = request.get('platformPrice', request.get('platform_price', ''))
            bid_value = request.get('bid_value', 'N/A')
            
            # Get vendor price for this request's vendor
            vendor_id = request.get('vendorID', request.get('vendor_id'))
            
            # Normalize vendor_id for lookup (handle float 1.0 -> int 1)
            lookup_key = vendor_id
            if isinstance(vendor_id, float) and vendor_id.is_integer():
                lookup_key = int(vendor_id)
                
            vendor_price = None
            if lookup_key is not None:
                # Try direct lookup first
                if lookup_key in vendor_lookup:
                    vendor_price = vendor_lookup[lookup_key].get('price')
                # Try string lookup if not found
                elif str(lookup_key) in vendor_lookup:
                    vendor_price = vendor_lookup[str(lookup_key)].get('price')
            
            # Get Transaction ID (pre-assigned by central system)
            transaction_id = request.get('transaction_id')
            
            # Calculate customer price based on vendor's actual price
            # Formula: Customer Price (PN) = (1 + price_range) × (1 + platform_markup) × vendor_price
            if vendor_price is not None:
                baseline_price = (1 + platform_markup) * vendor_price
                pn_price = (1 + price_range) * baseline_price
            else:
                # Fallback to market_price if vendor price not available
                baseline_price = (1 + platform_markup) * market_price
                pn_price = (1 + price_range) * baseline_price
            
            # Only include PN and BID for regular customers
            if platform_price == 'PN':
                purchase_request_type = 'PN'
                customer_price = pn_price  # PN uses calculated price based on vendor
            elif platform_price == 'BID' and bid_value != 'N/A':
                purchase_request_type = 'Bid'
                try:
                    customer_price = float(bid_value)
                except (ValueError, TypeError):
                    customer_price = pn_price
            else:
                # Skip if not PN or BID
                continue
            
            # Show price for both PN and BID customers
            # Format to 2 decimal places for display
            display_customer_price = float(f"{customer_price:.2f}")
            
            # Build record
            record = {
                'Purchase Request ID': transaction_id,  # Placeholder, will be updated after sorting
                'Agent ID': agent_id,
                'Honesty_Humility': honesty_humility,
                'Assigned Allowance Level': allowance_level,
                'Study Program': study_program,
                'Group_experiment': group_experiment,
                'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': twt_sospeso,
                'income': income,
                'Customer Type': customer_type_display,
                'Income Category': income_category,
                'Purchase Request Type': purchase_request_type,
                'Vendor': vendor_id,
                'Vendor Price': vendor_price,
                'Purchase Timestamp': timestamp_str,
                'Period': period,
                'Customer Price': display_customer_price,
                '_sort_key': sort_key  # Hidden column for sorting
            }
            
            transaction_records.append(record)
    
    # Sort all records by timestamp in chronological order
    if transaction_records:
        transaction_records.sort(key=lambda x: x.get('_sort_key', 0.0))
        
        # Assign unique Purchase Request IDs based on sorted order
        for idx, record in enumerate(transaction_records):
            record['Purchase Request ID'] = idx + 1
            record.pop('_sort_key', None)
    
    return transaction_records


def purchase_vs_bid_xlsx_bytes(export_df):
    """Purchase-now-vs-bid workbook bytes: `Total` plus one sheet per period."""
    # Sheet 1: Total (all data)
    sheets = {'Total': export_df}

    # Additional sheets by Period
    if 'Period' in export_df.columns:
        periods = sorted(export_df['Period'].dropna().unique())
        for period in periods:
            sheets[f'Period {int(period)}'] = export_df[export_df['Period'] == period]

    # 2-decimal formatting on every sheet
    return to_xlsx_bytes(sheets, apply_transaction_price_formatting)


def purchase_vs_bid_request_counts(df):
    """PN/BID counts over every purchase request (regular customers only).

    Returns (Counter of platformPrice values, total number of counted requests).
    """
    # Collect all purchase decisions from all requests
    regular_requests = []

    for idx, row in df.iterrows():
        requests = row.get('purchase_requests', [])
        if isinstance(requests, list):
            for req in requests:
                if isinstance(req, dict):
                    platform_price = req.get('platformPrice')

                    # Count only PN and BID for regular customers
                    if platform_price in ['PN', 'BID']:
                        regular_requests.append(platform_price)

    # Count regular customer choices
    regular_counts = Counter(regular_requests)
    return regular_counts, len(regular_requests)


def purchase_vs_bid_breakdown_frame(pn_count, bid_count, total_regular_requests):
    """Request-level choices table shown next to the donut chart."""
    return pd.DataFrame({
        'Choice': ['Purchase Now (PN)', 'Bid (BID)'],
        'Requests': [pn_count, bid_count],
        'Percentage': [
            f"{pn_count/total_regular_requests*100:.1f}%",
            f"{bid_count/total_regular_requests*100:.1f}%"
        ]
    })


def priority_lists_xlsx_bytes(export_df):
    """Priority-lists workbook bytes: one `Priority Lists` sheet."""
    # 2-decimal formatting (for any numeric columns that may exist)
    return to_xlsx_bytes({'Priority Lists': export_df},
                         apply_transaction_price_formatting)


def priority_list_agent_count(decision_data):
    """How many agents carry a priority LIST (rather than a single value)."""
    # Count how many agents have lists vs single values
    return decision_data.apply(lambda x: isinstance(x, list)).sum()


def priority_option_agent_counts(decision_data):
    """Number of agents whose priority list contains each option (unique per agent)."""
    # Count how many agents have each option in their priority list
    option_agent_counts = Counter()
    for agent_list in decision_data:
        if isinstance(agent_list, list):
            # Count unique options per agent (not duplicates)
            for opt in set(agent_list):
                option_agent_counts[opt] += 1
        else:
            option_agent_counts[agent_list] += 1
    return option_agent_counts


def priority_length_breakdown_lines(decision_data):
    """Lines of the form `{n} options: {k} agents ({p}%)`, one per list length."""
    list_lengths = decision_data.apply(lambda x: len(x) if isinstance(x, list) else 1)
    length_counts = list_lengths.value_counts().sort_index()

    return [
        f"{int(length)} options: {count:,} agents ({(count / len(decision_data)) * 100:.1f}%)"
        for length, count in length_counts.items()]


def priority_first_choice_counts(decision_data):
    """Value counts of each agent's FIRST priority option."""
    first_choices = decision_data.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
    return first_choices.value_counts()


def rejected_option_value_counts(decision_data):
    """(value counts of the chosen option, most common option) for Decision 5."""
    value_counts = decision_data.value_counts()
    current_option = value_counts.index[0] if len(value_counts) > 0 else "forgo_transaction"
    return value_counts, current_option


def prepare_priority_lists_export(df: pd.DataFrame, decision_data) -> pd.DataFrame:
    """
    Prepare rejected transaction defaults priority lists for Excel export.
    
    Creates columns: Agent ID, Honesty_Humility, Assigned Allowance Level, Study Program,
    Group_experiment, TWT+Sospeso [=AW2+AX2]{Periods 1+2}, income,
    Priority 1, Priority 2, Priority 3, Priority 4, Priority 5
    
    Priority columns contain option numbers (1, 2, 3, 4, 5) or N/A if not selected.
    
    Args:
        df: Full results DataFrame with agent data
        decision_data: Series containing priority lists
        
    Returns:
        DataFrame formatted for Excel export
    """
    # Map option codes to numbers
    option_to_number = {
        "higher_price_category": 1,
        "lower_pn_vendor": 2,
        "current_vendor_pn": 3,
        "place_bid": 4,
        "forgo_transaction": 5
    }
    
    # Create export dataframe
    export_df = pd.DataFrame()
    
    # Agent ID
    if 'agent_id' in df.columns:
        export_df['Agent ID'] = df['agent_id']
    elif 'index' in df.columns:
        export_df['Agent ID'] = df['index'] + 1  # Convert 0-based to 1-based
    else:
        export_df['Agent ID'] = range(1, len(df) + 1)
    
    # Honesty_Humility
    if 'Honesty_Humility' in df.columns:
        export_df['Honesty_Humility'] = df['Honesty_Humility'].round(2)
    else:
        export_df['Honesty_Humility'] = ''
    
    # Assigned Allowance Level
    if 'Assigned Allowance Level' in df.columns:
        export_df['Assigned Allowance Level'] = df['Assigned Allowance Level']
    elif 'income_category' in df.columns:
        export_df['Assigned Allowance Level'] = df['income_category']
    else:
        export_df['Assigned Allowance Level'] = ''
    
    # Study Program
    if 'Study Program' in df.columns:
        export_df['Study Program'] = df['Study Program']
    else:
        export_df['Study Program'] = ''
    
    # Group_experiment
    if 'Group_experiment' in df.columns:
        export_df['Group_experiment'] = df['Group_experiment']
    elif 'group' in df.columns:
        export_df['Group_experiment'] = df['group']
    elif 'group_experiment' in df.columns:
        export_df['Group_experiment'] = df['group_experiment']
    else:
        export_df['Group_experiment'] = ''
    
    # TWT+Sospeso
    if 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}' in df.columns:
        export_df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'] = df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'].round(2)
    else:
        export_df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'] = ''
    
    # Income
    if 'income' in df.columns:
        export_df['income'] = df['income'].round(2)
    elif 'actual_allowance' in df.columns:
        export_df['income'] = df['actual_allowance'].round(2)
    else:
        export_df['income'] = ''
    
    # Priority columns (1-5)
    for priority_pos in range(1, 6):
        column_name = f'Priority {priority_pos}'
        priority_values = []
        
        for agent_list in decision_data:
            if isinstance(agent_list, list):
                # Check if agent has this priority position
                if len(agent_list) >= priority_pos:
                    option_code = agent_list[priority_pos - 1]
                    option_number = option_to_number.get(option_code, 'N/A')
                    priority_values.append(option_number)
                else:
                    # Agent doesn't have this many priorities
                    priority_values.append('N/A')
            else:
                # Single value (legacy format) - only for priority 1
                if priority_pos == 1:
                    option_number = option_to_number.get(agent_list, 'N/A')
                    priority_values.append(option_number)
                else:
                    priority_values.append('N/A')
        
        export_df[column_name] = priority_values
    
    return export_df
