# app/reports/purchasing.py
"""Pure builders behind the purchasing-decision tables and Excel exports.

Every function here was moved verbatim out of
`app/pages/results/visualizations/purchasing_viz.py`: same rounding, same
column order, same sheet names, same `pd.cut` binning. The only changes are
mechanical -- the simulation parameters (`periods`, `duration_hours`) and the
timestamp anchoring the export strings are now explicit arguments instead of
`st.session_state` reads, and the blocks that used to write straight into
`st.dataframe` / `st.download_button` now return the DataFrame / xlsx bytes the
page hands to those widgets.

Nothing in this module imports Streamlit or touches session state.
"""
from collections import Counter

import pandas as pd

from app.reports.timestamps import TimestampConverter
from app.reports.xlsx import to_xlsx_bytes


# ---------------------------------------------------------------------------
# purchasing quantity: distribution statistics
# ---------------------------------------------------------------------------
def quantity_stats_frame(decision_data, periods) -> pd.DataFrame:
    """The '📈 Statistics' table next to the purchase-quantity histogram."""
    stats = decision_data.describe()

    return pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
        'Purchase Requests per Term': [
            f"{stats['mean']:.2f}",
            f"{stats['std']:.2f}",
            f"{int(stats['min'])}",
            f"{int(stats['max'])}",
            f"{stats['50%']:.2f}",
            f"{stats['25%']:.2f}",
            f"{stats['75%']:.2f}"
        ],
        'Purchase Requests per Period': [
            f"{stats['mean']/periods:.2f}",
            f"{stats['std']/periods:.2f}",
            f"{int(stats['min'])/periods:.2f}",
            f"{int(stats['max'])/periods:.2f}",
            f"{stats['50%']/periods:.2f}",
            f"{stats['25%']/periods:.2f}",
            f"{stats['75%']/periods:.2f}"
        ]
    })


def count_requests_by_customer_type(df):
    """Purchase requests per customer type (title-cased Regular/Fixed/Discount)."""
    customer_type_counts = {'Regular': 0, 'Fixed': 0, 'Discount': 0}

    for idx, row in df.iterrows():
        purchase_requests = row.get('purchase_requests', [])
        if isinstance(purchase_requests, list):
            for req in purchase_requests:
                if isinstance(req, dict):
                    customer_type = req.get('customer_type', 'regular')
                    # Normalize to title case
                    if isinstance(customer_type, str):
                        customer_type = customer_type.capitalize()

                    if customer_type in customer_type_counts:
                        customer_type_counts[customer_type] += 1

    return customer_type_counts


def customer_type_stats_frame(customer_type_counts, total_purchases_by_type) -> pd.DataFrame:
    """The customer-type '📊 Statistics' table, including its TOTAL row."""
    type_stats = []
    for ctype, count in customer_type_counts.items():
        percentage = (count / total_purchases_by_type * 100) if total_purchases_by_type > 0 else 0
        type_stats.append({
            'Customer Type': ctype,
            'Purchase Requests': f"{count:,}",
            'Percentage': f"{percentage:.1f}%"
        })

    # Add total row
    type_stats.append({
        'Customer Type': 'TOTAL',
        'Purchase Requests': f"{total_purchases_by_type:,}",
        'Percentage': '100.0%'
    })

    return pd.DataFrame(type_stats)


def quantities_by_customer_type(df, target_type):
    """Extract purchasing quantities for agents of a specific customer type"""
    quantities = []
    for idx, row in df.iterrows():
        # Get customer type - try direct column first, then purchase_requests as fallback
        customer_type = ''

        # Priority 1: Check if customer_type is directly available in the dataframe
        if 'customer_type' in row and pd.notna(row['customer_type']) and str(row['customer_type']).strip():
            customer_type = str(row['customer_type']).capitalize()
        else:
            # Priority 2: Extract from purchase_requests if available
            purchase_requests = row.get('purchase_requests', [])
            if isinstance(purchase_requests, list) and len(purchase_requests) > 0:
                # Check customer type from first request (all requests have same customer type)
                first_req = purchase_requests[0]
                if isinstance(first_req, dict):
                    customer_type = first_req.get('customer_type', 'regular')
                    if isinstance(customer_type, str):
                        customer_type = customer_type.capitalize()

        # If this agent matches the target customer type, include their quantity
        if customer_type == target_type:
            qty = row.get('purchasing_quantity', 0)
            quantities.append(qty)

    return pd.Series(quantities) if quantities else pd.Series([0])


def customer_type_quantity_stats_frame(type_quantities, periods) -> pd.DataFrame:
    """Per-customer-type version of the purchase-quantity statistics table."""
    type_stats_desc = type_quantities.describe()

    return pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
        'Purchase Requests per Term': [
            f"{type_stats_desc['mean']:.2f}",
            f"{type_stats_desc['std']:.2f}",
            f"{int(type_stats_desc['min'])}",
            f"{int(type_stats_desc['max'])}",
            f"{type_stats_desc['50%']:.2f}",
            f"{type_stats_desc['25%']:.2f}",
            f"{type_stats_desc['75%']:.2f}"
        ],
        'Purchase Requests per Period': [
            f"{type_stats_desc['mean']/periods:.2f}",
            f"{type_stats_desc['std']/periods:.2f}",
            f"{int(type_stats_desc['min'])/periods:.2f}",
            f"{int(type_stats_desc['max'])/periods:.2f}",
            f"{type_stats_desc['50%']/periods:.2f}",
            f"{type_stats_desc['25%']/periods:.2f}",
            f"{type_stats_desc['75%']/periods:.2f}"
        ]
    })


def income_category_stats_frame(df_with_category) -> pd.DataFrame:
    """The 'Requests by Income Category' table."""
    category_stats = df_with_category.groupby('income_category')['purchasing_quantity'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()

    # Sort by category number (ascending = lowest income first)
    category_stats = category_stats.sort_values('income_category')

    category_stats.columns = ['Category', 'Agents', 'Mean Qty', 'Std Dev', 'Min', 'Max']
    category_stats['Mean Qty'] = category_stats['Mean Qty'].round(2)
    category_stats['Std Dev'] = category_stats['Std Dev'].round(2)

    return category_stats


# ---------------------------------------------------------------------------
# purchasing quantity: per-period request tables
# ---------------------------------------------------------------------------
def collect_purchase_timestamps(df):
    """All purchase timestamps plus the (unused-by-the-page) per-agent timeline."""
    all_timestamps = []
    agent_timelines = []

    for idx, requests in enumerate(df['purchase_requests']):
        if isinstance(requests, list) and len(requests) > 0:
            agent_id = df.iloc[idx].get('agent_id', idx + 1)
            for req in requests:
                if isinstance(req, dict) and 'timestamp_hours' in req:
                    timestamp = req['timestamp_hours']
                    all_timestamps.append(timestamp)
                    agent_timelines.append({
                        'agent_id': agent_id,
                        'timestamp': timestamp
                    })

    return all_timestamps, agent_timelines


def period_bins_and_labels(periods, duration_hours, term_duration):
    """The right-inclusive `pd.cut` bins (and P1..Pn labels) the period chart uses."""
    period_bins = []
    period_labels = []
    for i in range(periods):
        start = i * duration_hours
        end = (i + 1) * duration_hours
        period_labels.append(f"P{i+1}")
        period_bins.append(start)
    period_bins.append(term_duration)

    return period_bins, period_labels


def counts_per_period(timestamps, period_bins, period_labels):
    """Count purchases per period with the page's `pd.cut(..., include_lowest=True)`."""
    period_counts = pd.cut(timestamps, bins=period_bins, labels=period_labels, include_lowest=True)
    return [sum(period_counts == label) for label in period_labels]


def period_details_frame(period_labels, requests_per_period, completed_per_period) -> pd.DataFrame:
    """The 'Period Details' table (per period plus the TOTAL row)."""
    stats_rows = []
    for i, label in enumerate(period_labels):
        requests = requests_per_period[i]
        completed = completed_per_period[i]
        pct_completed = (completed / requests * 100) if requests > 0 else 100.0

        stats_rows.append({
            'Period': label,
            'Purchase Requests': requests,
            'Purchases Completed': completed,
            '% Completed': f"{pct_completed:.1f}%"
        })

    # Add TOTAL row
    total_requests = sum(requests_per_period)
    total_completed = sum(completed_per_period)
    total_pct = (total_completed / total_requests * 100) if total_requests > 0 else 100.0

    stats_rows.append({
        'Period': 'TOTAL',
        'Purchase Requests': total_requests,
        'Purchases Completed': total_completed,
        '% Completed': f"{total_pct:.1f}%"
    })

    return pd.DataFrame(stats_rows)


def timestamps_by_customer_type(df):
    """Purchase timestamps split by (title-cased) customer type."""
    timestamps_by_type = {'Regular': [], 'Fixed': [], 'Discount': []}

    for idx, requests in enumerate(df['purchase_requests']):
        if isinstance(requests, list) and len(requests) > 0:
            for req in requests:
                if isinstance(req, dict) and 'timestamp_hours' in req:
                    # Get customer_type from request (lowercase: discount, fixed, regular)
                    customer_type = req.get('customer_type', 'regular')
                    if isinstance(customer_type, str):
                        # Normalize to title case for grouping
                        customer_type = customer_type.capitalize()

                    if customer_type in timestamps_by_type:
                        timestamps_by_type[customer_type].append(req['timestamp_hours'])

    return timestamps_by_type


# ---------------------------------------------------------------------------
# transaction-level export
# ---------------------------------------------------------------------------
def build_transaction_export(df, base_time, duration_hours, periods):
    """Flatten `purchase_requests` to the transaction-level export DataFrame."""
    # Flatten purchase_requests to transaction-level DataFrame
    transactions = []
    # Use centralized timestamp converter for consistent handling
    ts_converter = TimestampConverter(base_time, duration_hours, periods)

    for idx, row in df.iterrows():
        purchase_requests = row.get('purchase_requests', [])
        if isinstance(purchase_requests, list):
            for req in purchase_requests:
                if isinstance(req, dict):
                    # Get timestamp_hours and convert using centralized utilities
                    timestamp_hours = req.get('timestamp_hours', 0.0)
                    ts_result = ts_converter.convert(timestamp_hours)

                    period = ts_result['period']
                    timestamp_str = ts_result['formatted']

                    transactions.append({
                        'transaction_id': req.get('transaction_id'),
                        'Agent ID': req.get('customer_id', idx + 1),
                        'vendorID': req.get('vendorID', 1),
                        'platformProductID': req.get('platformProductID', 1),
                        'purchase type': req.get('platformPrice', 'N/A'),
                        'purchase_bid_value': req.get('bid_value', 'N/A'),
                        'Purchase Timestamp': timestamp_str,
                        'Period': period,
                        'timestamp_hours': timestamp_hours  # Keep for sorting
                    })

    if len(transactions) == 0:
        return None

    transactions_df = pd.DataFrame(transactions)

    # CRITICAL: Sort by timestamp across ALL customers
    transactions_df['timestamp_hours'] = pd.to_numeric(transactions_df['timestamp_hours'], errors='coerce')
    transactions_df = transactions_df.sort_values(
        by='timestamp_hours',
        ascending=True,
        na_position='last'
    ).reset_index(drop=True).copy()

    # Handle transaction_id
    # If IDs were pre-assigned (central system), use them. Otherwise generate them.
    if 'transaction_id' in transactions_df.columns and not transactions_df['transaction_id'].isnull().all():
        # Move transaction_id to first column
        cols = ['transaction_id'] + [c for c in transactions_df.columns if c != 'transaction_id']
        transactions_df = transactions_df[cols]
    else:
        # Fallback: Generate sequential IDs if missing
        if 'transaction_id' in transactions_df.columns:
            transactions_df = transactions_df.drop(columns=['transaction_id'])
        transactions_df.insert(0, 'transaction_id', range(1, len(transactions_df) + 1))

    # Drop timestamp_hours column before display/export
    transactions_df = transactions_df.drop(columns=['timestamp_hours'])

    return transactions_df


def purchasing_transactions_xlsx(transactions_df: pd.DataFrame) -> bytes:
    return to_xlsx_bytes({'Transactions': transactions_df})


# ---------------------------------------------------------------------------
# agent-level purchasing export
# ---------------------------------------------------------------------------
def build_agent_level_purchases(df, periods, duration_hours):
    """Agent-level purchasing rows: one 'Total' record plus one per period."""
    agent_level_data = []

    for idx, row in df.iterrows():
        agent_id = row.get('agent_id', idx + 1)

        # Agent Traits (matching disclose income export)
        # Honesty_Humility
        honesty_humility = ''
        if 'Honesty_Humility' in row and pd.notna(row['Honesty_Humility']):
            honesty_humility = round(row['Honesty_Humility'], 2)

        allowance_level = row.get('Assigned Allowance Level', '')

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

        income_category_raw = row.get('income_category', '')
        # Handle None/NaN/empty - display as 'N/A' for Regular customers who don't have income categories
        if pd.isna(income_category_raw) or income_category_raw == '' or income_category_raw is None:
            income_category = 'N/A'
        else:
            income_category = income_category_raw

        # Get customer type - try direct column first, then purchase_requests as fallback
        customer_type = ''

        # Priority 1: Check if customer_type is directly available in the dataframe
        if 'customer_type' in row and pd.notna(row['customer_type']) and str(row['customer_type']).strip():
            customer_type = str(row['customer_type']).capitalize()
        else:
            # Priority 2: Extract from purchase_requests if available
            purchase_requests = row.get('purchase_requests', [])
            if isinstance(purchase_requests, list) and len(purchase_requests) > 0:
                first_req = purchase_requests[0]
                if isinstance(first_req, dict):
                    customer_type = first_req.get('customer_type', '')
                    if isinstance(customer_type, str):
                        customer_type = customer_type.capitalize()

        # Get purchase_requests for counting
        purchase_requests = row.get('purchase_requests', [])

        # Total counts
        total_requests = len(purchase_requests) if isinstance(purchase_requests, list) else 0
        total_completed = total_requests  # All requests are completed (100%)
        pct_completed = 100.0 if total_requests > 0 else 0.0

        # Add overall record
        agent_level_data.append({
            'Agent ID': agent_id,
            'Honesty_Humility': honesty_humility,
            'Assigned Allowance Level': allowance_level,
            'Study Program': study_program,
            'Group_experiment': group_experiment,
            'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': twt_sospeso,
            'income': income,
            'Customer Type': customer_type,
            'Income Category': income_category,
            'Count of Purchase Requests': total_requests,
            'Count of Completed Transactions': total_completed,
            '% Completed Transactions': f"{pct_completed:.1f}%",
            'Period': 'Total'
        })

        # Breakdown by period
        if isinstance(purchase_requests, list):
            # Count requests per period for this agent
            period_counts = {f"P{i+1}": 0 for i in range(periods)}

            for req in purchase_requests:
                if isinstance(req, dict) and 'timestamp_hours' in req:
                    timestamp = req['timestamp_hours']
                    # Determine which period this request belongs to
                    period_idx = int(timestamp // duration_hours)
                    if 0 <= period_idx < periods:
                        period_label = f"P{period_idx + 1}"
                        period_counts[period_label] += 1

            # Add one record per period for this agent
            for period_label, count in period_counts.items():
                completed = count  # All requests are completed
                pct = 100.0 if count > 0 else 0.0

                agent_level_data.append({
                    'Agent ID': agent_id,
                    'Honesty_Humility': honesty_humility,
                    'Assigned Allowance Level': allowance_level,
                    'Study Program': study_program,
                    'Group_experiment': group_experiment,
                    'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': twt_sospeso,
                    'income': income,
                    'Customer Type': customer_type,
                    'Income Category': income_category,
                    'Count of Purchase Requests': count,
                    'Count of Completed Transactions': completed,
                    '% Completed Transactions': f"{pct:.1f}%",
                    'Period': period_label
                })

    return agent_level_data


def agent_level_purchases_xlsx(agent_df: pd.DataFrame, periods) -> bytes:
    """'Total' sheet plus one sheet per non-empty period (P1..Pn)."""
    sheets = {}

    # Sheet 1: Total (all agents, total across all periods)
    total_df = agent_df[agent_df['Period'] == 'Total'].drop(columns=['Period'])
    sheets['Total'] = total_df

    # Additional sheets: One per Period
    period_labels = [f"P{i+1}" for i in range(periods)]
    for period_label in period_labels:
        period_df = agent_df[agent_df['Period'] == period_label].drop(columns=['Period'])
        if len(period_df) > 0:
            sheets[period_label] = period_df

    return to_xlsx_bytes(sheets)


# ---------------------------------------------------------------------------
# purchasing frequency
# ---------------------------------------------------------------------------
def collect_timestamps(df):
    """Every `timestamp_hours` across all agents' purchase requests."""
    all_timestamps = []

    for idx, row in df.iterrows():
        requests = row.get('purchase_requests', [])
        if isinstance(requests, list):
            for req in requests:
                if isinstance(req, dict):
                    if 'timestamp_hours' in req:
                        all_timestamps.append(req['timestamp_hours'])

    return all_timestamps


def count_requests_by_customer_type_lower(df):
    """Request counts keyed by the raw lower-cased customer type."""
    customer_type_counts = Counter()

    for idx, row in df.iterrows():
        requests = row.get('purchase_requests', [])
        if isinstance(requests, list):
            for req in requests:
                if isinstance(req, dict):
                    # Get customer_type from request (lowercase: discount, fixed, regular)
                    customer_type = req.get('customer_type', 'regular')
                    if isinstance(customer_type, str):
                        # Normalize to lowercase for counting
                        customer_type = customer_type.lower()
                        customer_type_counts[customer_type] += 1

    return customer_type_counts


def top_agents_by_quantity(df, n=20):
    """Indices of the up-to-n agents with the most purchases (timeline chart)."""
    agent_purchase_counts = df.groupby(df.index)['purchasing_quantity'].first().sort_values(ascending=False)
    return agent_purchase_counts.head(n).index.tolist()


def build_agent_timeline(df, sample_agents):
    """Rows behind the 'Sample Agent Purchase Schedules' scatter."""
    timeline_data = []
    for idx in sample_agents:
        requests = df.iloc[idx]['purchase_requests']
        agent_id = df.iloc[idx].get('agent_id', idx + 1)
        quantity = df.iloc[idx].get('purchasing_quantity', 0)

        if isinstance(requests, list):
            for req in requests:
                if isinstance(req, dict) and 'timestamp_hours' in req:
                    timeline_data.append({
                        'Agent': f"Agent {agent_id} ({quantity} items)",
                        'Time': req['timestamp_hours'],
                        'Purchase': 1
                    })

    return timeline_data
