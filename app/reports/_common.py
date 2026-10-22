"""Small helpers the report builders share.

Each helper here is a block that two or more builders carried as identical
copies; those builders now call it instead. The money rule: a money value is
computed from the exact inputs first and only then stored as
``float(f"{x:.2f}")``.

The one rule for a missing value (docs/README.md section 9): a cell whose value
the run did not produce is exported as ``'N/A'``; nothing is substituted for it.
No column falls back to a different quantity (the income never to the 12-200
credit allowance, an allowance level never to an income category), a request
that names no vendor has no vendor cells, an agent without a Decision 5 dict has
no weights and no vendor score, and a Y/N outcome that is missing is ``'N/A'``
(1 for Y, 0 for N). The helpers below carry that rule; ``NA`` is the label.

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


def purchase_now_price(price, platform_markup, price_range):
    """The Purchase Now price of `price`: (1 + price_range) x (1 + platform_markup) x price,
    multiplied in this order -- the baseline price (1 + platform_markup) x price first.
    Every sheet computes it here, so every sheet stores the same cents (for a
    two-decimal price the other order can differ by one cent: 1.40 -> 1.92 vs 1.93)."""
    baseline_price = (1 + platform_markup) * price
    return (1 + price_range) * baseline_price


# ---------------------------------------------------------------------------
# missing values: 'N/A', never a substitute
# ---------------------------------------------------------------------------
NA = 'N/A'


def is_missing(value):
    """True when a cell value is missing: None, NaN / NaT, or the empty string."""
    if isinstance(value, str):
        return value == ''
    if isinstance(value, (list, tuple, dict, set)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def or_na(value):
    """`value`, or 'N/A' when it is missing."""
    return NA if is_missing(value) else value


def value_or_na(row, column):
    """row[column] when the row has the column and the value is not missing, else 'N/A'."""
    return or_na(row[column]) if column in row else NA


def rounded_value_or_na(row, column):
    """``round(row[column], 2)`` when the row has the column and the value is not
    missing, else 'N/A'."""
    value = value_or_na(row, column)
    return value if value == NA else round(value, 2)


def series_or_na(series):
    """The series with every missing (NaN / None / '') cell as 'N/A'; the series
    itself, dtype untouched, when nothing is missing."""
    missing = series.isna() | (series == '')
    if missing.any():
        return series.astype(object).where(~missing, NA)
    return series


def column_or_na(df, *names):
    """df[name] for the first of `names` (spellings of ONE quantity) the frame has,
    each missing cell as 'N/A' (`series_or_na`); 'N/A' in every row when the frame
    has none of them."""
    for name in names:
        if name in df.columns:
            return series_or_na(df[name])
    return NA


def rounded_column_or_na(df, name):
    """df[name].round(2) with each missing cell as 'N/A'; 'N/A' in every row when
    the frame has no such column."""
    if name not in df.columns:
        return NA
    return series_or_na(df[name].round(2))


def mean_or_na(values):
    """The mean of the values that are present; 'N/A' when none is. Accumulated
    in order with plain additions (not the compensated `sum`), as the builders
    always did, so a present value's mean is bit-identical."""
    total, count = 0, 0
    for value in values:
        if not is_missing(value):
            total += float(value)
            count += 1
    return total / count if count else NA


def money_or_na(value):
    """`money(value)`, or 'N/A' when the value is missing."""
    return NA if is_missing(value) else money(value)


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


def income_or_na(row):
    """The row's 'income' (the dollar value the run assigned) rounded to two
    decimals; 'N/A' when it is missing. Never the 12-200 credit allowance
    ('actual_allowance'), which is a different quantity."""
    return rounded_value_or_na(row, 'income')


# 'Group_experiment' and 'Study Program' are the only names the engine produces
# (config/trait_requirements.yaml; src/decisions/donation_default.py reads them).
def group_experiment_or_na(row):
    """The row's 'Group_experiment'; 'N/A' when it is absent, missing or ''."""
    return value_or_na(row, 'Group_experiment')


def study_program_or_na(row):
    """The row's 'Study Program'; 'N/A' when it is absent, missing or ''."""
    return value_or_na(row, 'Study Program')


def income_category_or_na(row):
    """The row's income category, 'N/A' when it is missing or empty (e.g. a
    regular customer who did not disclose income)."""
    income_category_raw = row.get('income_category', np.nan)
    if pd.isna(income_category_raw) or income_category_raw == '' or income_category_raw is None:
        return 'N/A'
    return income_category_raw


def yes_no_code(value):
    """A Y/N outcome as 1 (Y) / 0 (N); 'N/A' for anything else -- a missing value,
    or Decision 2's never-eligible 'NA'. N and the never-eligible state are NOT
    collapsed to 0, so the qualified-disclosure denominator stays recoverable."""
    return 1 if value == 'Y' else (0 if value == 'N' else NA)


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


def vendor_weights_or_none(row):
    """The agent's Decision 5 dict ('vendor_choice_weights'); None when the row
    carries no dict -- then every weight cell and the vendor score are 'N/A'."""
    vendor_weights = row.get('vendor_choice_weights', None)
    return vendor_weights if isinstance(vendor_weights, dict) else None


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
    'customerType'; the display name is it capitalised. A request without one
    gives ('', 'N/A') -- nothing is substituted, and '' matches no type."""
    customer_type = request.get('customer_type', request.get('customerType', ''))
    if isinstance(customer_type, str) and customer_type != '':
        return customer_type, customer_type.capitalize()
    return '', NA


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


def vendor_from(vendor_lookup, vendor_id):
    """The vendor `vendor_id` names in a `vendors_by_id_and_str` lookup (a float id
    like 1.0 is looked up as 1, then as its string); None when the id is missing
    or names no vendor in the list."""
    if vendor_id is None or (isinstance(vendor_id, float) and np.isnan(vendor_id)):
        return None
    lookup_key = vendor_id
    if isinstance(vendor_id, float) and vendor_id.is_integer():
        lookup_key = int(vendor_id)
    if lookup_key in vendor_lookup:
        return vendor_lookup[lookup_key]
    if str(lookup_key) in vendor_lookup:
        return vendor_lookup[str(lookup_key)]
    return None


def vendor_price_from(vendor_lookup, vendor_id):
    """The 'price' of the vendor `vendor_id` names (`vendor_from`); None when no
    vendor matches."""
    vendor = vendor_from(vendor_lookup, vendor_id)
    return vendor.get('price') if vendor is not None else None


def vendor_label(vendor, vendor_id):
    """'Vendor <id>' for a request whose id names `vendor` (a vendor of the run);
    'N/A' when the id is missing or names no vendor (vendor is None)."""
    return f"Vendor {int(vendor_id)}" if vendor is not None else NA


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
