# app/reports/vendor.py
"""Pure builders behind the vendor-decision tables and Excel exports.

Every function here was moved verbatim out of
`app/pages/results/visualizations/vendor_viz.py`: same rounding, same column
order, same sheet names, same openpyxl formatting. The only changes are
mechanical -- values that used to be read from `st.session_state` (the pricing
parameters, the vendors list, the configured price bounds) are now explicit
arguments, and the blocks that used to write straight into `st.dataframe` /
`st.download_button` now return the DataFrame rows / xlsx bytes the page hands
to those widgets.

Nothing in this module imports Streamlit or touches session state.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from app.reports.xlsx import apply_vendor_price_formatting, to_xlsx_bytes
from app.reports.timestamps import TimestampConverter
from src.vendor_attribute_generator import calculate_vendor_score_with_breakdown


# `apply_vendor_price_formatting` is the `_apply_price_formatting_vendor` that
# used to live in vendor_viz.py, moved verbatim into app/reports/xlsx.py.
apply_price_formatting_vendor = apply_vendor_price_formatting


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def calculate_vendor_score(vendor, weights, proximity, all_vendors,
                           price_min_config=None, price_max_config=None):
    """
    Calculate vendor integrated composite score.

    This is a thin wrapper around the centralized calculate_vendor_score_with_breakdown()
    function from vendor_attribute_generator.py. All scoring logic is maintained in one place.

    Args:
        vendor: Vendor dict with attributes
        weights: Dict of weights for each attribute
        proximity: Proximity score for this agent-vendor pair
        all_vendors: List of all vendors (for fallback price normalization)
        price_min_config: Configured minimum price bound (from vendor_price_min)
        price_max_config: Configured maximum price bound (from vendor_price_max)

    Returns:
        float: Composite score
    """
    result = calculate_vendor_score_with_breakdown(
        vendor=vendor,
        weights=weights,
        proximity=proximity,
        all_vendors=all_vendors,
        price_min_config=price_min_config,
        price_max_config=price_max_config
    )
    return result['integrated_score']


# ---------------------------------------------------------------------------
# purchase-request level export
# ---------------------------------------------------------------------------
def build_purchase_request_export(df, vendors_data, base_time, duration_hours, periods,
                                  platform_markup=0.1, price_range=0.25,
                                  price_min_config=None, price_max_config=None):
    """
    Build purchase request-level export data from simulation results.

    Args:
        df: DataFrame with simulation results
        vendors_data: List of vendor dictionaries (or None if not available)
        base_time: Base datetime every timestamp string is relative to
        duration_hours: Duration per period (drives the Period column)
        periods: Number of periods in the run
        platform_markup: Platform markup used for the customer price formula
        price_range: Price range parameter used for the customer price formula
        price_min_config: Configured minimum price bound (from vendor_price_min)
        price_max_config: Configured maximum price bound (from vendor_price_max)

    Returns:
        List of dicts with purchase request level data
    """
    purchase_request_records = []

    # Check if we have purchase_requests column
    if 'purchase_requests' not in df.columns:
        return []

    # Use centralized timestamp converter for consistent handling
    ts_converter = TimestampConverter(base_time, duration_hours, periods)

    # Build vendor lookup dictionary for quick access
    vendor_lookup = {}
    if vendors_data:
        for vendor in vendors_data:
            vendor_id = vendor.get('vendor_id')
            vendor_lookup[vendor_id] = vendor

    # Iterate through each agent
    for idx, row in df.iterrows():
        # Get agent-level data
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

        # Get vendor proximity scores for this agent
        proximity_scores = row.get('vendor_proximity_scores', {})
        if not isinstance(proximity_scores, dict):
            proximity_scores = {}

        # Get vendor choice weights for score calculation
        vendor_weights = row.get('vendor_choice_weights', {})
        if not isinstance(vendor_weights, dict):
            vendor_weights = {
                'price': 0.25,
                'quality': 0.25,
                'proximity': 0.25,
                'sustainability': 0.25
            }

        # Get purchase requests for this agent
        purchase_requests = row.get('purchase_requests', [])
        if not isinstance(purchase_requests, list):
            continue

        # Process each purchase request
        for req_idx, request in enumerate(purchase_requests):
            if not isinstance(request, dict):
                continue

            # Extract request data
            # Use global transaction_id assigned by simulation.py (snake_case), with fallback
            transaction_id = request.get('transaction_id', request.get('transactionID', request.get('request_id', f"T{agent_id}_{req_idx+1}")))
            vendor_id = request.get('vendorID', np.nan)

            # Get timestamp and convert using centralized utilities
            timestamp_hours = request.get('timestamp_hours', np.nan)
            ts_result = ts_converter.convert(timestamp_hours)

            period = ts_result['period']
            request_datetime = ts_result['datetime']

            # Determine customer type from request or agent
            customer_type = request.get('customer_type', request.get('customerType', 'Regular'))
            # Capitalize first letter
            if isinstance(customer_type, str):
                customer_type = customer_type.capitalize()

            # Get vendor attributes
            vendor_price = np.nan
            vendor_quality = np.nan
            vendor_sustainability = np.nan
            vendor_proximity = np.nan
            vendor_integrated_score = np.nan

            if not pd.isna(vendor_id) and vendor_id in vendor_lookup:
                vendor = vendor_lookup[vendor_id]
                vendor_price = vendor.get('price', np.nan)
                vendor_quality = vendor.get('quality', np.nan)
                vendor_sustainability = vendor.get('sustainability', np.nan)

                # Get proximity for this agent-vendor pair
                vendor_proximity = proximity_scores.get(str(int(vendor_id)), np.nan)

                # Calculate vendor integrated score
                if not pd.isna(vendor_price) and not pd.isna(vendor_quality) and \
                   not pd.isna(vendor_sustainability) and not pd.isna(vendor_proximity):
                    vendor_integrated_score = calculate_vendor_score(
                        vendor, vendor_weights, vendor_proximity, vendors_data,
                        price_min_config=price_min_config,
                        price_max_config=price_max_config
                    )

            # Get platform price type for this request
            platform_price = request.get('platformPrice', request.get('platform_price', ''))
            bid_value = request.get('bid_value', 'N/A')

            # Calculate customer paid price based on vendor's actual price and pricing formula
            # Formula: Customer Price (PN) = (1 + price_range) × (1 + platform_markup) × vendor_price
            customer_paid_price = np.nan
            if not pd.isna(vendor_price):
                if platform_price == 'PN':
                    # PN price: apply both platform markup and price range
                    baseline_price = (1 + platform_markup) * vendor_price
                    customer_paid_price = (1 + price_range) * baseline_price
                elif platform_price == 'BID' and bid_value != 'N/A':
                    # BID: customer pays their bid value
                    try:
                        customer_paid_price = float(bid_value)
                    except (ValueError, TypeError):
                        customer_paid_price = np.nan

            # Format for display - show 2 decimal places for both PN and BID
            # For Fixed/Discount customers, show 'N/A' as their pricing uses a different mechanism
            if (platform_price == 'PN' or platform_price == 'BID') and not pd.isna(customer_paid_price):
                display_customer_paid_price = float(f"{customer_paid_price:.2f}")
            else:
                # N/A for Fixed/Discount customers (income-based pricing not calculated here)
                display_customer_paid_price = 'N/A'

            # Determine Purchase Type (PN, Bid, Fixed, Discount)
            if platform_price == 'PN':
                purchase_type = 'PN'
            elif platform_price == 'BID':
                purchase_type = 'Bid'
            elif platform_price == 'FIXED' or customer_type.lower() == 'fixed':
                purchase_type = 'Fixed'
            elif platform_price == 'DISCOUNT' or customer_type.lower() == 'discount':
                purchase_type = 'Discount'
            else:
                # Fallback based on customer type
                purchase_type = customer_type if customer_type else 'Unknown'

            # Build record (include hidden sort key)
            record = {
                'Purchase Request ID': transaction_id,  # Will be reassigned after sorting
                'Agent ID': agent_id,
                'Honesty_Humility': honesty_humility,
                'Assigned Allowance Level': allowance_level,
                'Study Program': study_program,
                'Group_experiment': group_experiment,
                'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': twt_sospeso,
                'income': income,
                'Customer Type': customer_type,
                'Purchase Type': purchase_type,
                'Purchase Timestamp': ts_result['formatted'],
                'Period': period,
                'Selected Vendor': f"Vendor {int(vendor_id)}" if not pd.isna(vendor_id) else np.nan,
                'Vendor Price': vendor_price,
                'Quality': vendor_quality,
                'Sustainability': vendor_sustainability,
                'Proximity': vendor_proximity,
                'Vendor Integrated Score': vendor_integrated_score,
                'Customer Paid Price': display_customer_paid_price,
                '_sort_datetime': request_datetime  # Hidden sort key
            }

            purchase_request_records.append(record)

    # Sort records by timestamp in chronological order
    if purchase_request_records:
        purchase_request_records.sort(key=lambda x: x.get('_sort_datetime', datetime.min))

        # Assign unique Purchase Request IDs (1, 2, 3, ...) based on chronological order
        # This ensures each request has a unique ID that reflects its position in the timeline
        for idx, record in enumerate(purchase_request_records):
            record['Purchase Request ID'] = idx + 1
            record.pop('_sort_datetime', None)  # Remove temporary sorting column

    return purchase_request_records


def purchase_requests_xlsx(pr_df: pd.DataFrame) -> bytes:
    """Total sheet + one sheet per Period, with the vendor price formatting."""
    sheets = {'Total': pr_df}

    # Additional sheets by Period
    if 'Period' in pr_df.columns:
        periods = sorted(pr_df['Period'].unique())
        for period in periods:
            period_df = pr_df[pr_df['Period'] == period]
            sheet_name = f'Period {period}'
            sheets[sheet_name] = period_df

    return to_xlsx_bytes(sheets, apply_price_formatting_vendor)


# ---------------------------------------------------------------------------
# vendor choice weights export
# ---------------------------------------------------------------------------
def build_vendor_choice_weights_export(df, decision_data):
    """Agent-level vendor choice weights rows (moved verbatim from the page)."""
    export_data = []

    for idx, row in df.iterrows():
        # Start with basic info
        row_data = {}

        # Add Agent ID
        if 'agent_id' in df.columns:
            row_data['Agent ID'] = row['agent_id']
        else:
            row_data['Agent ID'] = idx + 1

        # Add Honesty_Humility
        if 'Honesty_Humility' in df.columns:
            row_data['Honesty_Humility'] = round(row['Honesty_Humility'], 2) if pd.notna(row['Honesty_Humility']) else ''
        else:
            row_data['Honesty_Humility'] = ''

        # Add Assigned Allowance Level
        if 'Assigned Allowance Level' in df.columns:
            row_data['Assigned Allowance Level'] = row['Assigned Allowance Level']
        elif 'income_category' in df.columns:
            row_data['Assigned Allowance Level'] = row['income_category']
        else:
            row_data['Assigned Allowance Level'] = ''

        # Add Study Program
        if 'Study Program' in df.columns:
            row_data['Study Program'] = row['Study Program']
        else:
            row_data['Study Program'] = ''

        # Add Group_experiment
        if 'Group_experiment' in df.columns:
            row_data['Group_experiment'] = row['Group_experiment']
        elif 'group' in df.columns:
            row_data['Group_experiment'] = row['group']
        elif 'group_experiment' in df.columns:
            row_data['Group_experiment'] = row['group_experiment']
        else:
            row_data['Group_experiment'] = ''

        # Add TWT+Sospeso
        if 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}' in df.columns:
            row_data['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'] = round(row['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'], 2) if pd.notna(row['TWT+Sospeso [=AW2+AX2]{Periods 1+2}']) else ''
        else:
            row_data['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'] = ''

        # Add income
        if 'income' in df.columns:
            row_data['income'] = round(row['income'], 2) if pd.notna(row['income']) else ''
        elif 'actual_allowance' in df.columns:
            row_data['income'] = round(row['actual_allowance'], 2) if pd.notna(row['actual_allowance']) else ''
        else:
            row_data['income'] = ''

        # Extract weights from the decision_data (which is a dict)
        weights = decision_data.iloc[idx]

        if isinstance(weights, dict):
            # Add each weight as a numeric value (e.g., 0.25 instead of "25%")
            row_data['Price'] = weights.get('price', 0.0)
            row_data['Quality'] = weights.get('quality', 0.0)
            row_data['Proximity'] = weights.get('proximity', 0.0)
            row_data['Sustainability'] = weights.get('sustainability', 0.0)
        else:
            # Fallback if weights aren't in expected format
            row_data['Price'] = 0.0
            row_data['Quality'] = 0.0
            row_data['Proximity'] = 0.0
            row_data['Sustainability'] = 0.0

        export_data.append(row_data)

    return export_data


def vendor_choice_weights_xlsx(export_df: pd.DataFrame) -> bytes:
    return to_xlsx_bytes({'Vendor Choice Weights': export_df}, apply_price_formatting_vendor)


# ---------------------------------------------------------------------------
# vendor selection statistics
# ---------------------------------------------------------------------------
def count_vendor_requests(df):
    """Purchase-request and completed-transaction counts, overall and per vendor."""
    total_purchase_requests = 0
    total_transactions_completed = 0
    vendor_pr_counts = {}  # Purchase requests per vendor
    vendor_tx_counts = {}  # Transactions per vendor

    if 'purchase_requests' in df.columns:
        for idx, row in df.iterrows():
            requests = row.get('purchase_requests', [])
            if isinstance(requests, list):
                total_purchase_requests += len(requests)
                # Count completed transactions per vendor
                for req in requests:
                    if isinstance(req, dict):
                        vendor_id = req.get('vendorID')
                        if not pd.isna(vendor_id):
                            # Count purchase request per vendor
                            vendor_pr_counts[vendor_id] = vendor_pr_counts.get(vendor_id, 0) + 1

                            # Count transaction if completed
                            completed = req.get('transactionCompleted', req.get('completed', req.get('transaction_completed', True)))
                            if completed or completed == 1:
                                total_transactions_completed += 1
                                vendor_tx_counts[vendor_id] = vendor_tx_counts.get(vendor_id, 0) + 1

    return total_purchase_requests, total_transactions_completed, vendor_pr_counts, vendor_tx_counts


def count_requests_per_vendor(df):
    """Per-vendor purchase-request / transaction counts used by the breakdown tables."""
    vendor_purchase_requests = {}
    vendor_transactions = {}

    if 'purchase_requests' in df.columns:
        for idx, row in df.iterrows():
            requests = row.get('purchase_requests', [])
            if isinstance(requests, list):
                for req in requests:
                    if isinstance(req, dict):
                        vendor_id = req.get('vendorID')
                        if not pd.isna(vendor_id):
                            # Count purchase request
                            vendor_purchase_requests[vendor_id] = vendor_purchase_requests.get(vendor_id, 0) + 1

                            # Count transaction if completed
                            completed = req.get('transactionCompleted', req.get('completed', req.get('transaction_completed', True)))
                            if completed or completed == 1:
                                vendor_transactions[vendor_id] = vendor_transactions.get(vendor_id, 0) + 1

    return vendor_purchase_requests, vendor_transactions


def sorted_vendor_ids(vendors_data, vendor_counts):
    """All vendor IDs (configured + actually selected), sorted."""
    all_vendor_ids = set()
    if vendors_data:
        all_vendor_ids = {int(v.get('vendor_id')) for v in vendors_data}

    # Add any vendors that were selected (in case configuration doesn't match results)
    if len(vendor_counts) > 0:
        all_vendor_ids.update([int(vid) for vid in vendor_counts.index])

    return sorted(list(all_vendor_ids))


def build_selection_breakdown(vendor_ids, vendor_counts, vendor_purchase_requests,
                              vendor_transactions, agents_with_selection,
                              total_vendor_purchase_requests):
    """Rows of the '📈 Selection Breakdown' table."""
    breakdown_data = []
    for vid in vendor_ids:
        # Get agent count
        agent_count = 0
        if vid in vendor_counts.index:
            agent_count = vendor_counts[vid]
        elif float(vid) in vendor_counts.index:
            agent_count = vendor_counts[float(vid)]

        # Get request count
        pr_count = vendor_purchase_requests.get(vid, 0)
        if pr_count == 0:
            pr_count = vendor_purchase_requests.get(float(vid), 0)

        # Get transaction count
        tx_count = vendor_transactions.get(vid, 0)
        if tx_count == 0:
            tx_count = vendor_transactions.get(float(vid), 0)

        # Calculate completion rate for this vendor (% of requests completed)
        completion_rate = f"{(tx_count/pr_count)*100:.1f}%" if pr_count > 0 else "0.0%"

        breakdown_data.append({
            'Vendor': f"Vendor {int(vid)}",
            'Agents': int(agent_count),
            '% Agents': f"{(agent_count/agents_with_selection)*100:.1f}%" if agents_with_selection > 0 else "0.0%",
            'Purchase Requests': int(pr_count),
            '% Requests': f"{(pr_count/total_vendor_purchase_requests)*100:.1f}%" if total_vendor_purchase_requests > 0 else "0.0%",
            'Transactions': int(tx_count),
            '% Completed': completion_rate
        })

    return breakdown_data


# ---------------------------------------------------------------------------
# per-period vendor breakdown
# ---------------------------------------------------------------------------
def collect_period_data(df, duration_hours):
    """{period: {vendor_id: {'agents': set(), 'requests': count, 'transactions': count}}}"""
    period_data = {}

    for idx, row in df.iterrows():
        requests = row.get('purchase_requests', [])
        agent_id = row.get('agent_id', idx + 1)

        if isinstance(requests, list):
            for req in requests:
                if isinstance(req, dict):
                    vendor_id = req.get('vendorID')

                    # Get period from timestamp_hours or period field
                    timestamp_hours = req.get('timestamp_hours', np.nan)
                    if not pd.isna(timestamp_hours):
                        # FIXED: Use actual duration_hours instead of hardcoded 24
                        period = int(timestamp_hours // duration_hours) + 1 if timestamp_hours >= 0 else np.nan
                    else:
                        period = req.get('period', np.nan)

                    if not pd.isna(vendor_id) and not pd.isna(period):
                        # Normalize vendor_id and period
                        try:
                            vendor_id_key = int(vendor_id)
                            period_key = int(period)
                        except (ValueError, TypeError):
                            continue

                        # Initialize period if not exists
                        if period_key not in period_data:
                            period_data[period_key] = {}

                        # Initialize vendor if not exists for this period
                        if vendor_id_key not in period_data[period_key]:
                            period_data[period_key][vendor_id_key] = {
                                'agents': set(),
                                'requests': 0,
                                'transactions': 0
                            }

                        # Add agent to set (for unique count)
                        period_data[period_key][vendor_id_key]['agents'].add(agent_id)

                        # Count purchase request
                        period_data[period_key][vendor_id_key]['requests'] += 1

                        # Count transaction if completed
                        completed = req.get('transactionCompleted', req.get('completed', req.get('transaction_completed', True)))
                        if completed or completed == 1:
                            period_data[period_key][vendor_id_key]['transactions'] += 1

    return period_data


def period_totals(period_data, sorted_periods):
    """Totals across ALL periods: (unique agents, requests, transactions)."""
    all_agents_across_periods = set()
    total_requests_all_periods = 0
    total_transactions_all_periods = 0

    for period in sorted_periods:
        period_vendors = period_data[period]
        all_agents_across_periods.update(set().union(*[v['agents'] for v in period_vendors.values()]))
        total_requests_all_periods += sum(v['requests'] for v in period_vendors.values())
        total_transactions_all_periods += sum(v['transactions'] for v in period_vendors.values())

    return all_agents_across_periods, total_requests_all_periods, total_transactions_all_periods


def build_period_breakdown(period_data, sorted_periods, vendor_ids):
    """Rows of the combined 'Vendor Selection Breakdown by Period' table."""
    all_periods_breakdown_data = []

    for period in sorted_periods:
        # Get vendors for this period (already normalized to int keys)
        period_vendors = period_data[period]

        # Calculate totals for this period (for percentage calculations)
        # Only sum up ACTUAL data from period_vendors
        total_agents_period = len(set().union(*[v['agents'] for v in period_vendors.values()]))
        total_requests_period = sum(v['requests'] for v in period_vendors.values())

        # Iterate through ALL vendors (including those with 0 selections)
        for vid in vendor_ids:
            # Check if vendor exists in this period's data
            if vid in period_vendors:
                vendor_stats = period_vendors[vid]
                agent_count = len(vendor_stats['agents'])
                request_count = vendor_stats['requests']
                transaction_count = vendor_stats['transactions']
            else:
                # Zero values for unselected vendor
                agent_count = 0
                request_count = 0
                transaction_count = 0

            # Calculate completion rate for this vendor in this period (% of requests completed)
            period_completion_rate = f"{(transaction_count/request_count)*100:.1f}%" if request_count > 0 else "0.0%"

            all_periods_breakdown_data.append({
                'Period': int(period),
                'Vendor': f"Vendor {vid}",
                'Agents': int(agent_count),
                '% Agents': f"{(agent_count/total_agents_period)*100:.1f}%" if total_agents_period > 0 else "0.0%",
                'Purchase Requests': int(request_count),
                '% Requests': f"{(request_count/total_requests_period)*100:.1f}%" if total_requests_period > 0 else "0.0%",
                'Transactions': int(transaction_count),
                '% Completed': period_completion_rate
            })

    return all_periods_breakdown_data


def vendor_selection_breakdown_xlsx(combined_breakdown_df: pd.DataFrame) -> bytes:
    return to_xlsx_bytes({'Vendor Breakdown': combined_breakdown_df}, apply_price_formatting_vendor)


# ---------------------------------------------------------------------------
# vendor attribute / proximity / score tables
# ---------------------------------------------------------------------------
def proximity_statistics(df):
    """(avg, min, max, std) proximity per vendor across all agents."""
    avg_proximity_per_vendor = {}
    min_proximity_per_vendor = {}
    max_proximity_per_vendor = {}
    std_proximity_per_vendor = {}

    if 'vendor_proximity_scores' in df.columns:
        # Extract all proximity scores
        all_proximity_scores = df['vendor_proximity_scores'].dropna()

        if len(all_proximity_scores) > 0:
            # Initialize accumulators
            proximity_lists = {}

            for proximity_dict in all_proximity_scores:
                if isinstance(proximity_dict, dict):
                    for vendor_key, proximity_value in proximity_dict.items():
                        # vendor_key is a string like "1", "2", etc.
                        if vendor_key not in proximity_lists:
                            proximity_lists[vendor_key] = []
                        proximity_lists[vendor_key].append(float(proximity_value))

            # Calculate statistics
            for vendor_key in proximity_lists:
                scores = proximity_lists[vendor_key]
                if len(scores) > 0:
                    vendor_id = int(vendor_key)
                    avg_proximity_per_vendor[vendor_id] = np.mean(scores)
                    min_proximity_per_vendor[vendor_id] = np.min(scores)
                    max_proximity_per_vendor[vendor_id] = np.max(scores)
                    std_proximity_per_vendor[vendor_id] = np.std(scores)

    return (avg_proximity_per_vendor, min_proximity_per_vendor,
            max_proximity_per_vendor, std_proximity_per_vendor)


def average_vendor_scores(df, vendors_data, price_min_config=None, price_max_config=None):
    """Integrated score per vendor, averaged across all agents."""
    vendor_integrated_scores = {}

    if 'vendor_choice_weights' in df.columns and 'vendor_proximity_scores' in df.columns:
        for vendor in vendors_data:
            vendor_id = vendor.get('vendor_id')
            scores = []

            # Calculate score for each agent
            for idx, row in df.iterrows():
                weights = row.get('vendor_choice_weights', {})
                proximity_scores = row.get('vendor_proximity_scores', {})

                if isinstance(weights, dict) and isinstance(proximity_scores, dict):
                    proximity = proximity_scores.get(str(vendor_id), 50.0)
                    score = calculate_vendor_score(
                        vendor, weights, proximity, vendors_data,
                        price_min_config=price_min_config,
                        price_max_config=price_max_config
                    )
                    scores.append(score)

            # Average score across all agents
            if scores:
                vendor_integrated_scores[vendor_id] = np.mean(scores)

    return vendor_integrated_scores


def build_vendor_attributes_table(vendors_data, vendor_counts, vendor_purchase_requests,
                                  vendor_transactions, agents_with_selection,
                                  total_vendor_purchase_requests, total_vendor_transactions,
                                  avg_proximity_per_vendor, vendor_integrated_scores):
    """Rows of the '📋 Vendor Attributes & Selection Results' table."""
    vendor_table_data = []

    for idx, vendor in enumerate(vendors_data, 1):
        vendor_id = vendor.get('vendor_id', idx)

        # Get counts for this vendor (from earlier calculations)
        agent_count = 0
        if vendor_id in vendor_counts.index:
            agent_count = int(vendor_counts[vendor_id])

        pr_count = vendor_purchase_requests.get(vendor_id, 0)
        tx_count = vendor_transactions.get(vendor_id, 0)

        # Get proximity statistics for this vendor
        avg_proximity = avg_proximity_per_vendor.get(vendor_id, None)
        proximity_avg_display = f"{avg_proximity:.1f}" if avg_proximity is not None else "N/A"

        # Get integrated score
        integrated_score = vendor_integrated_scores.get(vendor_id, None)
        integrated_score_display = f"{integrated_score:.3f}" if integrated_score is not None else "N/A"

        # Get quantity information - check if per-period data exists
        quantity_per_period = vendor.get('quantity_offered_per_period', {})
        if quantity_per_period and isinstance(quantity_per_period, dict):
            # Calculate total quantity across all periods
            total_quantity = sum(quantity_per_period.values())
            # Show average quantity (calculated from per-period values)
            avg_quantity = vendor.get('quantity_offered', 100)
            # Show ONLY the average quantity (clean display)
            quantity_display = str(avg_quantity)
            total_quantity_display = total_quantity
        else:
            # Legacy: single quantity value (assume 1 period)
            quantity_display = str(vendor.get('quantity_offered', 100))
            total_quantity_display = vendor.get('quantity_offered', 100)

        vendor_table_data.append({
            'Vendor ID': f"Vendor {vendor_id}",
            'Price ($)': f"${vendor.get('price', 0):.2f}",
            'Average Quantity Per Period': quantity_display,
            'Total Quantity': total_quantity_display,
            'Quality': vendor.get('quality', 'N/A'),
            'Sustainability': vendor.get('sustainability', 'N/A'),
            'Average Proximity': proximity_avg_display,
            'Integrated Score': integrated_score_display,
            'Agents': agent_count,
            '% Agents': f"{(agent_count / agents_with_selection * 100) if agents_with_selection > 0 else 0:.1f}%",
            'Purchase Requests': pr_count,
            '% Purchase Requests': f"{(pr_count / total_vendor_purchase_requests * 100) if total_vendor_purchase_requests > 0 else 0:.1f}%",
            'Transactions': tx_count,
            '% Transactions': f"{(tx_count / total_vendor_transactions * 100) if total_vendor_transactions > 0 else 0:.1f}%"
        })

    return vendor_table_data


def build_vendor_period_details(vendors_data, avg_proximity_per_vendor, vendor_integrated_scores):
    """Rows of the per-period vendor detail export."""
    vendor_period_data = []

    for vendor in vendors_data:
        vendor_id = vendor.get('vendor_id')
        price = vendor.get('price', 0)
        quality = vendor.get('quality', 0)
        sustainability = vendor.get('sustainability', 0)

        # Get metrics
        avg_prox = avg_proximity_per_vendor.get(vendor_id, np.nan)
        int_score = vendor_integrated_scores.get(vendor_id, np.nan)

        # Get periods
        quantity_per_period = vendor.get('quantity_offered_per_period', {})
        if quantity_per_period and isinstance(quantity_per_period, dict):
            for period, qty in sorted(quantity_per_period.items()):
                vendor_period_data.append({
                    'Vendor ID': f"Vendor {vendor_id}",
                    'Period': int(period),
                    'Quantity Offered': qty,
                    'Price': price,
                    'Quality': quality,
                    'Sustainability': sustainability,
                    'Average Proximity': avg_prox,
                    'Integrated Score': int_score
                })
        else:
            # Single period default (Period 1)
            qty = vendor.get('quantity_offered', 100)
            vendor_period_data.append({
                'Vendor ID': f"Vendor {vendor_id}",
                'Period': 1,
                'Quantity Offered': qty,
                'Price': price,
                'Quality': quality,
                'Sustainability': sustainability,
                'Average Proximity': avg_prox,
                'Integrated Score': int_score
            })

    return vendor_period_data


def vendor_period_details_xlsx(vendor_period_df: pd.DataFrame) -> bytes:
    return to_xlsx_bytes({'Vendor Details Per Period': vendor_period_df},
                         apply_price_formatting_vendor)


def build_vendor_score_breakdown(df, vendors_data, avg_proximity_per_vendor,
                                 price_min_config=None, price_max_config=None):
    """Rows of the '🔍 Vendor Score Breakdown (Average Across All Agents)' table."""
    score_breakdown_data = []

    # Get average weights across all agents
    all_weights = df['vendor_choice_weights'].dropna()
    if len(all_weights) > 0:
        avg_weights = {
            'price': np.mean([w.get('price', 0) for w in all_weights if isinstance(w, dict)]),
            'quality': np.mean([w.get('quality', 0) for w in all_weights if isinstance(w, dict)]),
            'proximity': np.mean([w.get('proximity', 0) for w in all_weights if isinstance(w, dict)]),
            'sustainability': np.mean([w.get('sustainability', 0) for w in all_weights if isinstance(w, dict)])
        }

        for vendor in vendors_data:
            vendor_id = vendor.get('vendor_id')
            price = vendor.get('price', 0)
            quality = vendor.get('quality', 3)
            sustainability = vendor.get('sustainability', 3)
            avg_proximity = avg_proximity_per_vendor.get(vendor_id, 50.0)

            # Use centralized scoring function for consistency
            score_result = calculate_vendor_score_with_breakdown(
                vendor=vendor,
                weights=avg_weights,
                proximity=avg_proximity,
                all_vendors=vendors_data,
                price_min_config=price_min_config,
                price_max_config=price_max_config
            )

            score_breakdown_data.append({
                'Vendor': f"Vendor {vendor_id}",
                'Price ($)': f"${price:.2f}",
                'Norm Price': f"{score_result['norm_price']:.3f}",
                'Price Weight': f"{score_result['weight_price']:.2f}",
                'Price Component': f"{score_result['weighted_price']:.3f}",
                'Quality (1-5)': quality,
                'Norm Quality': f"{score_result['norm_quality']:.3f}",
                'Quality Weight': f"{score_result['weight_quality']:.2f}",
                'Quality Component': f"{score_result['weighted_quality']:.3f}",
                'Sustainability (1-5)': sustainability,
                'Norm Sustain': f"{score_result['norm_sustainability']:.3f}",
                'Sustain Weight': f"{score_result['weight_sustainability']:.2f}",
                'Sustain Component': f"{score_result['weighted_sustainability']:.3f}",
                'Avg Proximity': f"{avg_proximity:.1f}",
                'Norm Proximity': f"{score_result['norm_proximity']:.3f}",
                'Proximity Weight': f"{score_result['weight_proximity']:.2f}",
                'Proximity Component': f"{score_result['weighted_proximity']:.3f}",
                'Final Score': f"{score_result['integrated_score']:.3f}"
            })

    return score_breakdown_data


def vendor_score_breakdown_xlsx(score_df: pd.DataFrame) -> bytes:
    return to_xlsx_bytes({'Vendor Score Breakdown': score_df}, apply_price_formatting_vendor)


def build_proximity_matrix(df):
    """Rows of the complete Agent-Vendor proximity matrix."""
    proximity_matrix_data = []
    for idx in range(len(df)):
        row_data = {}

        # Add Agent ID
        if 'agent_id' in df.columns:
            row_data['Agent ID'] = df.iloc[idx]['agent_id']
        else:
            row_data['Agent ID'] = idx + 1

        # Add proximity scores for each vendor
        scores = df.iloc[idx]['vendor_proximity_scores']
        if isinstance(scores, dict):
            for v_id in sorted(scores.keys(), key=lambda x: int(x)):
                row_data[f'Vendor {v_id} Proximity'] = scores[v_id]

        proximity_matrix_data.append(row_data)

    return proximity_matrix_data


def proximity_matrix_xlsx(proximity_df: pd.DataFrame) -> bytes:
    return to_xlsx_bytes({'Agent-Vendor Proximity': proximity_df}, apply_price_formatting_vendor)
