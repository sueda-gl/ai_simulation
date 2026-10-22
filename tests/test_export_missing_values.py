"""The one export rule for a missing value (docs/README.md section 9, app/reports/_common.py):
a cell the run did not produce is exported as 'N/A'; nothing is substituted for it.

A small hand-built results frame (three agents, two vendors) carries one case of each
kind of missing value; every export sheet is built from it and the cells are checked:

* group / study programme: read from the one column the engine produces
  ('Group_experiment', 'Study Program'); '' or NaN -> 'N/A';
* income: the dollar value the run assigned; missing -> 'N/A', never the 12-200
  credit allowance ('actual_allowance'); an allowance level never falls back to an
  income category or an income;
* a request whose vendor id is missing or names no vendor of the run: every vendor
  cell 'N/A';
* an agent without a Decision 5 dict: every weight cell and the vendor score 'N/A'
  (an agent with the dict is untouched);
* a Y/N outcome: 1 / 0, 'N/A' when missing -- Decision 2's never-eligible agents
  included, so Y, N and never-eligible stay distinguishable.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.reports import (
    build_agent_level_dataframe, build_agent_level_purchases, build_bid_value_export,
    build_donation_default_export, build_donation_transaction_export,
    build_purchase_request_export, build_purchase_vs_bid_export, build_transaction_export,
    build_transaction_level_dataframe, build_vendor_choice_weights_export,
    donation_default_export_columns, prepare_disclose_documents_excel_data,
    prepare_disclose_income_excel_data, prepare_disclosure_excel_data,
    prepare_priority_lists_export,
)
from app.reports.timestamps import TimestampConverter

NA = 'N/A'
BASE_TIME = datetime(2025, 1, 15)
PRICES = dict(market_price=100.0, platform_markup=0.1, price_range=0.25)
BOUNDS = dict(price_min_config=50.0, price_max_config=150.0)
TWT = 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}'

VENDORS = [{'vendor_id': 1, 'price': 1.40, 'quality': 4, 'sustainability': 3, 'quantity_offered': 100},
           {'vendor_id': 2, 'price': 80.0, 'quality': 2, 'sustainability': 5, 'quantity_offered': 100}]
WEIGHTS = {'price': 0.4, 'quality': 0.3, 'proximity': 0.2, 'sustainability': 0.1}
PROXIMITY = {'1': 60.0, '2': 40.0}


def _request(rid, agent, vendor, platform, bid='N/A', hours=0.1):
    return {'request_id': rid, 'quantity': 1, 'timestamp_hours': hours, 'customer_id': agent,
            'customer_type': 'regular', 'vendorID': vendor, 'platformPrice': platform,
            'bid_value': bid, 'final_donation_rate': 0.1, 'transaction_id': rid}


def _agent(agent_id, requests, **overrides):
    row = {
        'agent_id': agent_id, 'Honesty_Humility': 3.1, 'Assigned Allowance Level': 2.0,
        'Study Program': 'IM', 'Group_experiment': 'HighSub', TWT: 1.5,
        'income': 10000.0, 'actual_allowance': 32.0, 'income_category': np.nan,
        'disclose_income': 'Y', 'disclose_documents': 'N', 'customer_type': 'regular',
        'donation_default': 0.1, 'rejected_transaction_defaults': ['place_bid', 'forgo_transaction'],
        'vendor_choice_weights': WEIGHTS, 'vendor_proximity_scores': PROXIMITY,
        'purchasing_quantity': len(requests), 'final_donation_rate': 0.1, 'preferred_vendor': 1,
        'purchase_requests': requests,
    }
    row.update(overrides)
    return row


def frame():
    """Agent 1 complete; agent 2 with a blank group, study programme, income and no
    Decision 5 dict (never eligible for Decision 2); agent 3 with missing Y/N outcomes
    and requests naming no vendor (None) and a vendor not in the run (99)."""
    return pd.DataFrame([
        _agent(1, [_request(1, 1, 1, 'PN', hours=0.1), _request(2, 1, 2, 'BID', '90.0', hours=0.2)]),
        _agent(2, [_request(3, 2, 1, 'PN', hours=0.3)], **{
            'Group_experiment': '', 'Study Program': np.nan, 'income': np.nan,
            'actual_allowance': 72.0, 'vendor_choice_weights': np.nan,
            'disclose_income': 'N', 'disclose_documents': 'NA'}),
        _agent(3, [_request(4, 3, None, 'PN', hours=0.4), _request(5, 3, 99, 'BID', '95.0', hours=0.5)],
               **{'disclose_income': np.nan, 'disclose_documents': np.nan}),
    ])


def sheets(df, vendors=VENDORS):
    """Every export sheet that carries agent traits, vendor cells, weights or Y/N outcomes."""
    df = df.copy()
    df.attrs['vendors'] = vendors
    times = dict(base_time=BASE_TIME, duration_hours=1.0, periods=1)
    return {
        'agent_level': build_agent_level_dataframe(
            df, vendors_data=vendors, vendor_price_min=50.0, vendor_price_max=150.0),
        'transaction_level': build_transaction_level_dataframe(
            df, vendors_data=vendors, **times, **PRICES, vendor_price_min=50.0, vendor_price_max=150.0),
        'donation_default': build_donation_default_export(
            df, donation_default_export_columns(df, 'donation_default')),
        'donation_transactions': pd.DataFrame(build_donation_transaction_export(
            df, vendors=vendors, **times, **PRICES)),
        'purchase_vs_bid': pd.DataFrame(build_purchase_vs_bid_export(
            df, ts_converter=TimestampConverter(BASE_TIME, 1.0, 1), vendors=vendors, **PRICES)),
        'purchase_requests': pd.DataFrame(build_purchase_request_export(
            df, vendors, BASE_TIME, 1.0, 1, platform_markup=0.1, price_range=0.25, **BOUNDS)),
        'bid_values': pd.DataFrame(build_bid_value_export(df, vendors, BASE_TIME, 1.0, 1)),
        'purchasing_transactions': build_transaction_export(df, BASE_TIME, 1.0, 1),
        'agent_level_purchases': pd.DataFrame(build_agent_level_purchases(df, 1, 1.0)),
        'disclose_income_sheet': prepare_disclose_income_excel_data(df),
        'disclose_documents_sheet': prepare_disclose_documents_excel_data(df),
        'disclosure_sheet': prepare_disclosure_excel_data(df),
        'priority_lists': prepare_priority_lists_export(df, df['rejected_transaction_defaults']),
        'vendor_choice_weights': pd.DataFrame(build_vendor_choice_weights_export(
            df, df['vendor_choice_weights'])),
    }


def rows_of(sheet, agent_id):
    """The sheet's rows of one agent, in the sheet's order."""
    return sheet[sheet['Agent ID'] == agent_id]


def cell(sheet, agent_id, column, position=0):
    return rows_of(sheet, agent_id)[column].iloc[position]


@pytest.fixture(scope='module')
def out():
    return sheets(frame())


TRAIT_SHEETS = ['agent_level', 'transaction_level', 'donation_default', 'donation_transactions',
                'purchase_vs_bid', 'purchase_requests', 'agent_level_purchases', 'disclosure_sheet',
                'priority_lists', 'vendor_choice_weights']


# ------------------------------------------------------------ group / study programme
@pytest.mark.parametrize('name', TRAIT_SHEETS)
def test_blank_group_and_study_programme_are_na(out, name):
    sheet = out[name]
    assert cell(sheet, 2, 'Group_experiment') == NA      # '' in the run
    assert cell(sheet, 2, 'Study Program') == NA         # NaN in the run
    assert cell(sheet, 1, 'Group_experiment') == 'HighSub'
    assert cell(sheet, 1, 'Study Program') == 'IM'


def test_bid_sheet_reads_the_group_and_study_programme_the_same_way():
    df = frame()
    df.loc[0, 'Group_experiment'] = np.nan
    df.loc[0, 'Study Program'] = ''
    bids = sheets(df)['bid_values']
    assert cell(bids, 1, 'Group_experiment') == NA and cell(bids, 1, 'Study Program') == NA


# --------------------------------------------------------------------------- income
INCOME_SHEETS = ['agent_level', 'transaction_level', 'donation_transactions', 'purchase_vs_bid',
                 'purchase_requests', 'agent_level_purchases', 'disclose_income_sheet',
                 'disclosure_sheet', 'priority_lists', 'vendor_choice_weights']


@pytest.mark.parametrize('name', INCOME_SHEETS)
def test_missing_income_is_na_never_the_allowance(out, name):
    sheet = out[name]
    assert cell(sheet, 2, 'income') == NA                # not actual_allowance (72)
    assert float(cell(sheet, 1, 'income')) == 10000.0


def test_missing_income_in_the_documents_sheet_and_its_derived_column(out):
    dd = out['disclose_documents_sheet']
    assert cell(dd, 2, 'Income') == NA and cell(dd, 2, 'PersonalIncentive') == NA
    assert float(cell(dd, 1, 'Income')) == 10000.0


def test_a_run_without_an_income_column_exports_na_everywhere():
    out = sheets(frame().drop(columns=['income']))
    for name in INCOME_SHEETS:
        assert (rows_of(out[name], 1)['income'] == NA).all(), name
    assert (out['bid_values']['income'] == NA).all()


def test_an_allowance_level_never_falls_back_to_another_quantity():
    out = sheets(frame().drop(columns=['Assigned Allowance Level']))
    for name in ('disclose_income_sheet', 'disclose_documents_sheet', 'disclosure_sheet',
                 'priority_lists', 'vendor_choice_weights'):
        # not actual_allowance (32 / 72), not income_category, not income
        assert (out[name]['Assigned Allowance Level'] == NA).all(), name
    kept = sheets(frame())
    assert float(cell(kept['disclosure_sheet'], 1, 'Assigned Allowance Level')) == 2.0


def test_income_category_stays_na_when_the_run_assigned_none(out):
    assert cell(out['agent_level'], 1, 'income_category') == NA
    assert cell(out['agent_level_purchases'], 1, 'Income Category') == NA


# --------------------------------------------------------- a request without a vendor
def test_request_naming_no_vendor_has_na_in_every_vendor_cell(out):
    tl = rows_of(out['transaction_level'], 3)           # request 4: no id; request 5: vendor 99
    for column in ('Vendor ID', 'Vendor Price', 'Vendor Quality', 'Vendor Sustainability',
                   'Vendor Proximity', 'Standardized Vendor Price Score',
                   'Standardized Vendor Quality Score', 'Standardized Vendor Sustainability Score',
                   'Standardized Vendor Proximity Score', 'Vendor Integrated Score'):
        assert (tl[column] == NA).all(), column
    pr = rows_of(out['purchase_requests'], 3)
    for column in ('Selected Vendor', 'Vendor Price', 'Quality', 'Sustainability', 'Proximity',
                   'Vendor Integrated Score'):
        assert (pr[column] == NA).all(), column
    pvb = rows_of(out['purchase_vs_bid'], 3)
    assert (pvb['Vendor'] == NA).all() and (pvb['Vendor Price'] == NA).all()
    bids = rows_of(out['bid_values'], 3)                 # the vendor-99 bid
    assert len(bids) == 1 and (bids['Vendor ID'] == NA).all() and (bids['Vendor Price'] == NA).all()
    pt = out['purchasing_transactions']
    assert pt.loc[pt['transaction_id'] == 4, 'vendorID'].iloc[0] == NA


def test_request_naming_a_vendor_keeps_its_vendor_cells(out):
    tl = out['transaction_level']
    assert cell(tl, 1, 'Vendor ID') == 'Vendor 1' and cell(tl, 1, 'Vendor Price') == 1.40
    assert cell(tl, 1, 'Vendor Quality') == 4 and cell(tl, 1, 'Vendor Proximity') == 60.0
    assert isinstance(cell(tl, 1, 'Vendor Integrated Score'), float)
    assert cell(out['purchase_requests'], 1, 'Selected Vendor') == 'Vendor 1'
    assert cell(out['bid_values'], 1, 'Vendor ID') == 'Vendor 2'


# ------------------------------------------------------------------ Decision 5 weights
def test_agent_without_the_decision_5_dict_has_na_weights_and_score(out):
    al = out['agent_level']
    for column in ('weight_price', 'weight_quality', 'weight_proximity', 'weight_sustainability',
                   'avg_vendor_score'):
        assert cell(al, 2, column) == NA, column
    tl = out['transaction_level']
    for column in ('weight_price', 'weight_quality', 'weight_proximity', 'weight_sustainability',
                   'Vendor Integrated Score'):
        assert cell(tl, 2, column) == NA, column
    # the standardized components do not depend on the weights
    assert isinstance(cell(tl, 2, 'Standardized Vendor Price Score'), float)
    assert cell(out['purchase_requests'], 2, 'Vendor Integrated Score') == NA
    vw = out['vendor_choice_weights']
    for column in ('Price', 'Quality', 'Proximity', 'Sustainability'):
        assert cell(vw, 2, column) == NA, column


def test_agent_with_the_decision_5_dict_is_unchanged(out):
    assert cell(out['agent_level'], 1, 'weight_price') == 0.4
    assert cell(out['transaction_level'], 1, 'weight_sustainability') == 0.1
    assert cell(out['vendor_choice_weights'], 1, 'Proximity') == 0.2
    assert isinstance(cell(out['agent_level'], 1, 'avg_vendor_score'), float)


# ------------------------------------------------------------------- Y/N outcomes
def test_yes_no_outcomes_are_1_0_or_na(out):
    for name, di, dd in (('agent_level', 'disclose_income', 'disclose_documents'),
                         ('transaction_level', 'disclose_income', 'disclose_documents'),
                         ('disclosure_sheet', 'disclose_income', 'disclose_documents')):
        sheet = out[name]
        assert (cell(sheet, 1, di), cell(sheet, 1, dd)) == (1, 0), name       # Y, N
        assert (cell(sheet, 2, di), cell(sheet, 2, dd)) == (0, NA), name      # N, never eligible
        assert (cell(sheet, 3, di), cell(sheet, 3, dd)) == (NA, NA), name     # missing
    di_sheet = out['disclose_income_sheet']['Disclose Income (Y=1)'].tolist()
    assert di_sheet == [1, 0, NA]
    dd_sheet = out['disclose_documents_sheet']
    assert dd_sheet['Disclose Income (Y=1)'].tolist() == [1, 0, NA]
    assert dd_sheet['Disclose Documents (Y=1)'].tolist() == [0, NA, NA]


def test_the_never_eligible_label_is_na_not_the_engine_spelling(out):
    assert 'NA' not in set(out['agent_level']['disclose_documents'])
    assert 'NA' not in set(out['transaction_level']['disclose_documents'])


# ------------------------------------------------ the rule has no exceptions (leftovers)
def test_purchase_now_request_naming_no_vendor_has_no_customer_price(out):
    """No market-price substitute: the PN request of agent 3 names no vendor."""
    tl = rows_of(out['transaction_level'], 3)
    assert tl.loc[tl['Purchase Request Type'] == 'Purchase Now', 'Customer Price'].tolist() == [NA]
    assert tl.loc[tl['Purchase Request Type'] == 'Bid', 'Customer Price'].tolist() == [95.0]   # the bid value
    dt = rows_of(out['donation_transactions'], 3)
    assert dt.loc[dt['Purchase Request Type'] == 'PN', 'Customer Price'].tolist() == [NA]
    pvb = rows_of(out['purchase_vs_bid'], 3)
    assert pvb.loc[pvb['Purchase Request Type'] == 'PN', 'Customer Price'].tolist() == [NA]
    assert cell(out['purchase_requests'], 3, 'Customer Paid Price') == NA
    # a PN request naming a vendor keeps its price (1.40 -> 1.93 in every sheet)
    assert cell(out['transaction_level'], 1, 'Customer Price') == 1.93
    assert cell(out['donation_transactions'], 1, 'Customer Price') == 1.93
    assert cell(out['purchase_vs_bid'], 1, 'Customer Price') == 1.93


def test_fixed_and_discount_prices_are_na_in_the_donation_sheet_too():
    df = frame()
    df.at[0, 'purchase_requests'][0]['platformPrice'] = 'FIXED'
    df.at[0, 'purchase_requests'][0]['customer_type'] = 'fixed'
    dt = rows_of(sheets(df)['donation_transactions'], 1)
    assert dt.loc[dt['Purchase Request Type'] == 'Fixed', 'Customer Price'].tolist() == [NA]


def test_agent_without_a_scorable_vendor_has_na_averages():
    df = frame()
    df.at[0, 'purchase_requests'] = []
    df.loc[0, 'purchasing_quantity'] = 0
    df.loc[0, 'preferred_vendor'] = np.nan
    al = sheets(df)['agent_level']
    for column in ('avg_vendor_price', 'avg_vendor_quality', 'avg_vendor_sustainability',
                   'avg_vendor_proximity', 'avg_vendor_score', 'most_selected_vendor'):
        assert cell(al, 1, column) == NA, column
    # agent 3's requests name no vendor of the run: nothing to average either
    for column in ('avg_vendor_price', 'avg_vendor_score'):
        assert cell(al, 3, column) == NA, column


def test_blank_honesty_humility_and_twt_are_na_in_every_sheet():
    df = frame()
    df.loc[1, 'Honesty_Humility'] = np.nan
    df.loc[1, TWT] = np.nan
    out = sheets(df)
    for name in TRAIT_SHEETS + ['bid_values', 'disclose_income_sheet', 'disclose_documents_sheet']:
        sheet = out[name]
        if not len(rows_of(sheet, 2)):          # agent 2 places no bid
            continue
        if 'Honesty_Humility' in sheet.columns:
            assert cell(sheet, 2, 'Honesty_Humility') == NA, name
            assert float(cell(sheet, 1, 'Honesty_Humility')) == 3.1, name
        twt = TWT if TWT in sheet.columns else 'TWT+Sospeso'
        if twt in sheet.columns:
            assert cell(sheet, 2, twt) == NA, name


def test_product_id_is_na_because_the_engine_writes_none(out):
    """src/decisions/purchasing_quantity.py puts no platformProductID on a request; the
    sheet no longer substitutes 1 for it."""
    assert (out['purchasing_transactions']['platformProductID'] == NA).all()


def test_missing_donation_rates_are_na_not_zero():
    df = frame()
    df.loc[0, 'donation_default'] = np.nan
    df.loc[0, 'final_donation_rate'] = np.nan
    for request in df.at[0, 'purchase_requests']:
        del request['final_donation_rate']
    out = sheets(df)
    assert (rows_of(out['transaction_level'], 1)['final_donation_rate'] == NA).all()
    dt = rows_of(out['donation_transactions'], 1)
    assert (dt['Default Donation Rate'] == NA).all() and (dt['Final Donation Rate'] == NA).all()
    assert cell(out['agent_level'], 1, 'donation_default') == NA
    assert cell(out['transaction_level'], 2, 'final_donation_rate') == 0.1   # a run value stays


def test_disclosure_sheet_defaults_are_not_substituted():
    df = frame().drop(columns=['Honesty_Humility'])
    di = sheets(df)['disclose_income_sheet']
    assert (di['WOPB'] == NA).all() and (di['WPB'] == NA).all() and (di['Intercept'] == NA).all()
    assert (di['Honesty_Humility'] == NA).all() and (di['Religious'] == NA).all()
