"""Small helpers the report builders share.

Each helper here is a block that two or more builders carried as identical
copies; those builders now call it instead. Where two copies differed in a
detail (a default value, a fallback label, the order of a multiplication) they
were NOT merged: each builder keeps its own code, so no export changes. The
helpers are written exactly as the copies they replace, including the money
rule: a money value is computed from the exact inputs first and only then
stored as ``float(f"{x:.2f}")``.

Nothing in this module imports Streamlit or reads session state.
"""
import ast
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# money and rounding
# ---------------------------------------------------------------------------
def money(value):
    """A money value as every export stores it: two decimals, ``float(f"{value:.2f}")``.

    The caller computes ``value`` from the exact inputs first; this is the only
    rounding step."""
    return float(f"{value:.2f}")


def money_or_missing(value):
    """``money(value)``, or ``value`` itself, unchanged, when it is missing (NaN/None)."""
    return money(value) if not pd.isna(value) else value


def round2_or_nan(value):
    """``round(value, 2)``, or NaN when the value is missing."""
    return round(value, 2) if not pd.isna(value) else np.nan


def purchase_now_price(price, platform_markup, price_range):
    """The Purchase Now price of `price`: (1 + price_range) x (1 + platform_markup) x price,
    multiplied in this order -- the baseline price (1 + platform_markup) x price first."""
    baseline_price = (1 + platform_markup) * price
    return (1 + price_range) * baseline_price


# ---------------------------------------------------------------------------
# one agent row of the results frame
# ---------------------------------------------------------------------------
def purchase_requests_of(row):
    """(index, request) for each dict in the row's `purchase_requests` list.

    Nothing when the row's value is not a list; the index counts every list
    item, so it is the request's position in the agent's list."""
    requests = row.get('purchase_requests', [])
    if not isinstance(requests, list):
        return
    for index, request in enumerate(requests):
        if isinstance(request, dict):
            yield index, request


def rounded_or_blank(row, column):
    """``round(row[column], 2)`` when the row has the column and the value is not
    missing, else ''."""
    if column in row and pd.notna(row[column]):
        return round(row[column], 2)
    return ''


def income_rounded_or_blank(row):
    """The row's income rounded to two decimals: 'income', else 'actual_allowance'
    (each only when present and not missing), else ''."""
    if 'income' in row and pd.notna(row['income']):
        return round(row['income'], 2)
    if 'actual_allowance' in row and pd.notna(row['actual_allowance']):
        return round(row['actual_allowance'], 2)
    return ''


def group_experiment_or_blank(row):
    """The first of 'Group_experiment', 'group', 'group_experiment' the row has
    with a non-missing value, else ''."""
    for column in ('Group_experiment', 'group', 'group_experiment'):
        if column in row and pd.notna(row[column]):
            return row[column]
    return ''


def group_experiment_skipping_blanks(row):
    """'Group_experiment', falling back to 'group' and then 'group_experiment'
    while the value so far is '' or missing; '' when it is still missing."""
    group_experiment = row.get('Group_experiment', '')
    if group_experiment == '' or pd.isna(group_experiment):
        group_experiment = row.get('group', '')
    if group_experiment == '' or pd.isna(group_experiment):
        group_experiment = row.get('group_experiment', '')
    if pd.isna(group_experiment):
        group_experiment = ''
    return group_experiment


def income_category_or_na(row):
    """The row's income category, 'N/A' when it is missing or empty (e.g. a
    regular customer who did not disclose income)."""
    income_category_raw = row.get('income_category', np.nan)
    if pd.isna(income_category_raw) or income_category_raw == '' or income_category_raw is None:
        return 'N/A'
    return income_category_raw


def income_disclosed_flag(row):
    """Decision 1 as 1/0: 1 when 'disclose_income' is 'Y', else 0."""
    return 1 if row.get('disclose_income', '') == 'Y' else 0


def documents_disclosed_code(row):
    """Decision 2's three-state outcome: 1 (Y = disclosed), 0 (N = asked but
    declined) or 'NA' (never eligible). N and NA are NOT collapsed to 0, so the
    qualified-disclosure denominator stays recoverable."""
    disclose_documents_raw = row.get('disclose_documents', '')
    return 1 if disclose_documents_raw == 'Y' else (0 if disclose_documents_raw == 'N' else 'NA')


def rejected_default_priorities(row):
    """Decision 4's default list as five priority values (positions 1-5), 'N/A'
    beyond the list's length. A string that looks like a list is parsed first;
    any other non-list value counts as a one-item list (none when falsy)."""
    rejected_defaults = row.get('rejected_transaction_defaults', '')
    if isinstance(rejected_defaults, str) and rejected_defaults.startswith('['):
        try:
            rejected_defaults = ast.literal_eval(rejected_defaults)
        except:  # noqa: E722 -- as the builders always had it: any failure means no list
            rejected_defaults = []
    elif not isinstance(rejected_defaults, list):
        rejected_defaults = [rejected_defaults] if rejected_defaults else []

    choices = []
    for priority_num in range(1, 6):
        if isinstance(rejected_defaults, list) and len(rejected_defaults) >= priority_num:
            choices.append(rejected_defaults[priority_num - 1])
        else:
            choices.append('N/A')
    return choices


def vendor_weights_or_equal(row):
    """The agent's vendor choice weights; equal weights (0.25 each) when the row
    carries no dict."""
    vendor_weights = row.get('vendor_choice_weights', {})
    if not isinstance(vendor_weights, dict):
        vendor_weights = {
            'price': 0.25,
            'quality': 0.25,
            'proximity': 0.25,
            'sustainability': 0.25
        }
    return vendor_weights


def proximity_scores_or_empty(row):
    """The agent's vendor proximity scores ({vendor id string: score}); {} when the
    row carries no dict."""
    proximity_scores = row.get('vendor_proximity_scores', {})
    if not isinstance(proximity_scores, dict):
        proximity_scores = {}
    return proximity_scores


# ---------------------------------------------------------------------------
# one purchase request
# ---------------------------------------------------------------------------
def request_time(ts_converter, request):
    """(converted, period, datetime) of the request's 'timestamp_hours' (NaN when
    it has none), through the run's `TimestampConverter`; `converted` is the
    converter's full result (its 'formatted' string, ...)."""
    ts_result = ts_converter.convert(request.get('timestamp_hours', np.nan))
    return ts_result, ts_result['period'], ts_result['datetime']


def request_customer_type(request):
    """(customer type, display name) of a request: 'customer_type', else
    'customerType', else 'regular'; the display name is it capitalised, or
    'Regular' when it is not a string."""
    customer_type = request.get('customer_type', request.get('customerType', 'regular'))
    if isinstance(customer_type, str):
        customer_type_display = customer_type.capitalize()
    else:
        customer_type_display = 'Regular'
    return customer_type, customer_type_display


# ---------------------------------------------------------------------------
# vendors
# ---------------------------------------------------------------------------
def vendors_by_id(vendors):
    """{vendor_id: vendor} over the vendor list (empty without vendors)."""
    vendor_lookup = {}
    if vendors:
        for vendor in vendors:
            vendor_id = vendor.get('vendor_id')
            vendor_lookup[vendor_id] = vendor
    return vendor_lookup


def vendors_by_id_and_str(vendors):
    """{vendor_id: vendor, str(vendor_id): vendor} over the vendors that have an
    id, so a lookup works with either key type."""
    vendor_lookup = {}
    if vendors:
        for vendor in vendors:
            vendor_id = vendor.get('vendor_id')
            if vendor_id is not None:
                vendor_lookup[vendor_id] = vendor
                vendor_lookup[str(vendor_id)] = vendor
    return vendor_lookup


def vendor_price_from(vendor_lookup, vendor_id):
    """The 'price' of the vendor `vendor_id` names in a `vendors_by_id_and_str`
    lookup (a float id like 1.0 is looked up as 1, then as its string); None when
    no vendor matches."""
    lookup_key = vendor_id
    if isinstance(vendor_id, float) and vendor_id.is_integer():
        lookup_key = int(vendor_id)

    vendor_price = None
    if lookup_key is not None:
        if lookup_key in vendor_lookup:
            vendor_price = vendor_lookup[lookup_key].get('price')
        elif str(lookup_key) in vendor_lookup:
            vendor_price = vendor_lookup[str(lookup_key)].get('price')
    return vendor_price


def request_vendor(vendor_lookup, vendor_id, proximity_scores):
    """(vendor, price, quality, sustainability, proximity) of the vendor a request
    names, from a `vendors_by_id` lookup and the agent's proximity scores;
    (None, NaN, NaN, NaN, NaN) when the id is missing or names no vendor."""
    if not pd.isna(vendor_id) and vendor_id in vendor_lookup:
        vendor = vendor_lookup[vendor_id]
        return (vendor,
                vendor.get('price', np.nan),
                vendor.get('quality', np.nan),
                vendor.get('sustainability', np.nan),
                proximity_scores.get(str(int(vendor_id)), np.nan))
    return None, np.nan, np.nan, np.nan, np.nan


def all_present(*values):
    """True when none of `values` is missing (pd.isna) -- the condition for
    computing a vendor score."""
    return not any(pd.isna(value) for value in values)


# ---------------------------------------------------------------------------
# record lists
# ---------------------------------------------------------------------------
def number_chronologically(records):
    """In place: sort by the hidden '_sort_datetime' key, number the records
    1, 2, 3, ... in 'Purchase Request ID', and drop the hidden key."""
    if records:
        records.sort(key=lambda x: x.get('_sort_datetime', datetime.min))
        for idx, record in enumerate(records):
            record['Purchase Request ID'] = idx + 1
            record.pop('_sort_datetime', None)


def sort_chronologically(records):
    """In place: sort by the hidden '_sort_datetime' key (a non-datetime value
    sorts as datetime.min) and drop the hidden key."""
    if records:
        records.sort(key=lambda x: x['_sort_datetime'] if isinstance(x['_sort_datetime'], datetime) else datetime.min)
        for record in records:
            record.pop('_sort_datetime', None)


# ---------------------------------------------------------------------------
# export frames: columns, names, order
# ---------------------------------------------------------------------------
def agent_id_column(df):
    """The 'Agent ID' column of a frame-wise export: 'agent_id', else 'index' + 1
    (0-based to 1-based), else 1..n."""
    if 'agent_id' in df.columns:
        return df['agent_id']
    if 'index' in df.columns:
        return df['index'] + 1
    return range(1, len(df) + 1)


def column_or_blank(df, *names):
    """df[name] for the first of `names` the frame has, else '' (an empty cell in
    every row)."""
    for name in names:
        if name in df.columns:
            return df[name]
    return ''


def rounded_column_or_blank(df, *names):
    """df[name].round(2) for the first of `names` the frame has, else ''."""
    for name in names:
        if name in df.columns:
            return df[name].round(2)
    return ''


def copy_present_columns(out, df, pairs):
    """out[dst] = df[src] for each (src, dst) pair whose `src` the frame has."""
    for src, dst in pairs:
        if src in df.columns:
            out[dst] = df[src]


def agent_id_renamed(df):
    """The frame with 'agent_id' renamed to 'Agent ID' (unchanged without it)."""
    if 'agent_id' in df.columns:
        df = df.rename(columns={'agent_id': 'Agent ID'})
    return df


def column_first(df, column):
    """The frame with `column` moved to the front, the other columns in order."""
    return df[[column] + [c for c in df.columns if c != column]]


# Result key -> name of that configuration in a workbook holding several
# configurations (a "Compare all" sheet name, or a sheet-name prefix), in the
# order such a workbook writes them.
CONFIG_SHEET_NAMES = (
    ('copula_categorical', 'Copula_Cat'),
    ('copula_continuous', 'Copula_Cont'),
    ('research_spec_categorical', 'ResSpec_Cat'),
    ('research_spec_continuous', 'ResSpec_Cont'),
    ('research_baseline_categorical', 'ResBase_Cat'),
    ('research_baseline_continuous', 'ResBase_Cont'),
)
