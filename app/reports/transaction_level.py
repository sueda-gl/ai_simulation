"""Transaction-level export: one row per purchase request.

`build_transaction_level_dataframe` builds the Transaction-Level sheet the
results page (`app/pages/results/components/export_section.py`) exports. The
pricing parameters and the base time / period duration / period count of its
`TimestampConverter` are parameters; nothing here imports Streamlit or touches
session state.
"""
import numpy as np
import pandas as pd

from app.reports._common import (
    all_present, documents_disclosed_code, income_category_or_na, income_disclosed_flag,
    money, money_or_missing, number_chronologically, proximity_scores_or_empty,
    purchase_requests_of, rejected_default_priorities, request_time, request_vendor,
    round2_or_nan, vendor_weights_or_equal, vendors_by_id,
)
from app.reports.timestamps import TimestampConverter
from src.vendor_attribute_generator import calculate_vendor_score_with_breakdown


def build_transaction_level_dataframe(
    df,
    vendors_data=None,
    *,
    base_time,
    periods,
    market_price: float = 100.0,
    platform_markup: float = 0.1,
    price_range: float = 0.25,
    duration_hours: float = 1.0,
    vendor_price_min: float = 50.0,
    vendor_price_max: float = 150.0,
):
    """
    Build transaction-level DataFrame with one row per purchase request.

    Includes:
    - Purchase Request ID and timing
    - Agent reference (ID and traits)
    - Vendor selection and attributes
    - Purchase decision (PN/BID)
    - Pricing and donation information

    Args:
        df: Original simulation results DataFrame
        vendors_data: List of vendor dictionaries (optional)
        base_time: Base datetime every timestamp string is relative to (the page
            passes the run's start, ``RunContext.base_time``).
        periods: Number of periods in the run (was ``sim_params.periods``).
        market_price: Market price (was ``sim_params.market_price``)
        platform_markup: Platform markup (was ``sim_params.platform_markup``)
        price_range: Purchase-Now price range (was ``sim_params.price_range``)
        duration_hours: Hours per period (was ``sim_params.duration_hours``);
            drives the Period column through the timestamp converter.
        vendor_price_min: Configured lower price bound for vendor-score
            normalisation (was ``sim_params.vendor_price_min``)
        vendor_price_max: Configured upper price bound for vendor-score
            normalisation (was ``sim_params.vendor_price_max``)

    Returns:
        pd.DataFrame: Transaction-level data
    """
    transaction_records = []
    
    # Timestamp/period arithmetic: the page used to build this converter from
    # session state and pass it in; it now passes the three values it read
    # (base time, period duration, period count) and the converter is built here.
    ts_converter = TimestampConverter(base_time, duration_hours, periods)
    
    # Configured price bounds for consistent normalization. The page reads the
    # pricing parameters off st.session_state.sim_params and passes the VALUES in.
    price_min_config = vendor_price_min
    price_max_config = vendor_price_max
    
    # Calculate standard prices
    # Customer Price rule (COOPECON AI Agent Documentation 2026-10-07, Transaction-Level
    # export): Purchase Now = (1 + price_range) x (1 + platform_markup) x vendor_price;
    # Bid = the bid value; Fixed / Discount = N/A (the platform algorithm sets them).
    # The market price is the fallback for a request that names no vendor.
    market_pn_price = (1 + price_range) * (1 + platform_markup) * market_price
    
    # Build vendor lookup
    vendor_lookup = vendors_by_id(vendors_data)
    
    for idx, row in df.iterrows():
        # Get agent-level data
        agent_id = row.get('agent_id', idx + 1)
        honesty_humility = row.get('Honesty_Humility', np.nan)
        allowance_level = row.get('Assigned Allowance Level', np.nan)
        study_program = row.get('Study Program', '')
        group_experiment = row.get('Group_experiment', '')
        twt_sospeso = row.get('TWT+Sospeso [=AW2+AX2]{Periods 1+2}', np.nan)
        customer_type = row.get('customer_type', '')

        # Get income (try 'income' first, fallback to 'actual_allowance')
        income = row.get('income', row.get('actual_allowance', np.nan))

        # Income category - use 'N/A' for empty/missing values (e.g., regular customers who didn't disclose income)
        income_category = income_category_or_na(row)
        agent_donation_default = row.get('donation_default', np.nan)

        # Decision 1: Income disclosed (convert Y/N to 1/0)
        income_disclosed = income_disclosed_flag(row)

        # Decision 2: Documents disclosed (1 / 0 / 'NA': N and NA are not collapsed)
        documents_disclosed = documents_disclosed_code(row)

        # Decision 4: Rejected Transaction Defaults - 5 priority columns
        priority_choices = rejected_default_priorities(row)

        # Decision 5: Vendor Choice Weights (also used for the vendor score below)
        vendor_weights = vendor_weights_or_equal(row)
        weight_price = vendor_weights.get('price', np.nan)
        weight_quality = vendor_weights.get('quality', np.nan)
        weight_proximity = vendor_weights.get('proximity', np.nan)
        weight_sustainability = vendor_weights.get('sustainability', np.nan)

        # Decision 11: Rejected Transaction Option
        rejected_transaction_option = row.get('rejected_transaction_option', '')
        
        # Get vendor proximity scores for this agent
        proximity_scores = proximity_scores_or_empty(row)
        
        # Process each purchase request
        for req_idx, request in purchase_requests_of(row):
            # Transaction identification
            request_id = request.get('request_id', req_idx + 1)
            # Use global transaction_id if available (assigned by simulation.py), otherwise fallback
            transaction_id = request.get('transaction_id', f"A{agent_id}_R{request_id}")
            
            # Timing - use centralized timestamp converter
            ts_result, period, request_datetime = request_time(ts_converter, request)
            
            # Vendor information - use centralized scoring function
            vendor_id = request.get('vendorID', np.nan)
            vendor_price_score = np.nan
            vendor_quality_score = np.nan
            vendor_sustainability_score = np.nan
            vendor_proximity_score = np.nan
            vendor_integrated_score = np.nan

            vendor, vendor_price, vendor_quality, vendor_sustainability, vendor_proximity = \
                request_vendor(vendor_lookup, vendor_id, proximity_scores)
            # Use centralized scoring function for consistency
            if vendor is not None and all_present(
                    vendor_price, vendor_quality, vendor_sustainability, vendor_proximity):
                score_result = calculate_vendor_score_with_breakdown(
                    vendor=vendor,
                    weights=vendor_weights,
                    proximity=vendor_proximity,
                    all_vendors=list(vendor_lookup.values()),
                    price_min_config=price_min_config,
                    price_max_config=price_max_config
                )

                vendor_price_score = score_result['norm_price']
                vendor_quality_score = score_result['norm_quality']
                vendor_sustainability_score = score_result['norm_sustainability']
                vendor_proximity_score = score_result['norm_proximity']
                vendor_integrated_score = score_result['integrated_score']
            
            pn_price = ((1 + price_range) * (1 + platform_markup) * vendor_price
                        if not pd.isna(vendor_price) else market_pn_price)

            # Purchase decision and pricing
            platform_price = request.get('platformPrice', '')
            bid_value = request.get('bid_value', 'N/A')
            
            # Determine purchase request type and customer price
            if platform_price == 'DISCOUNT':
                purchase_request_type = 'Discount'
                customer_price = np.nan  # set by the platform algorithm; exported as N/A
            elif platform_price == 'FIXED':
                purchase_request_type = 'Fixed'
                customer_price = np.nan  # set by the platform algorithm; exported as N/A
            elif platform_price == 'PN':
                purchase_request_type = 'Purchase Now'
                customer_price = pn_price
            elif platform_price == 'BID':
                purchase_request_type = 'Bid'
                try:
                    # For BID transactions, Customer Price is the bid amount if successful
                    bid_val_numeric = float(bid_value) if bid_value != 'N/A' else pn_price
                    customer_price = bid_val_numeric
                except (ValueError, TypeError):
                    customer_price = pn_price
            else:
                purchase_request_type = customer_type.capitalize() if customer_type else 'Regular'
                customer_price = pn_price
            
            # Display customer price (stored rounded to two decimals, like every
            # money column of every export):
            # - For Purchase Now: pn_price
            # - For Bid: bid_value (if numeric)
            # - For Discount/Fixed: 'N/A' (actual price calculated by separate algorithm)
            if purchase_request_type in ['Purchase Now', 'Bid']:
                display_customer_price = money(customer_price)
            else:
                display_customer_price = 'N/A'

            # Bid value as stored: two decimals; N/A for non-Bid transactions
            if purchase_request_type == 'Bid':
                try:
                    display_bid_value = money(float(bid_value))
                except (ValueError, TypeError):
                    display_bid_value = bid_value
            else:
                display_bid_value = 'N/A'
            display_vendor_price = money_or_missing(vendor_price)
            
            # Donation information
            # Priority: request-level > agent-level
            final_donation_rate = request.get('final_donation_rate', agent_donation_default)
            try:
                final_donation_rate = float(final_donation_rate) if not pd.isna(final_donation_rate) else 0.0
            except (ValueError, TypeError):
                final_donation_rate = 0.0
            
            # Build transaction record - organized by decision sequence
            # Column names standardized to match Agent-Level export
            transaction_record = {
                # ===== IDENTIFICATION =====
                'Purchase Request ID': transaction_id,
                'Agent ID': agent_id,

                # ===== AGENT TRAITS =====
                'Honesty_Humility': honesty_humility,
                'Assigned Allowance Level': allowance_level,
                'Study Program': study_program,
                'Group_experiment': group_experiment,
                'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': twt_sospeso,

                # ===== INCOME & CUSTOMER INFO (matched to Agent-Level names) =====
                'income': round2_or_nan(income),
                'income_category': income_category,

                # ===== DECISION 1: Disclose Income (1/0 format, matched to Agent-Level) =====
                'disclose_income': income_disclosed,

                # ===== DECISION 2: Disclose Documents (1/0 format, matched to Agent-Level) =====
                'disclose_documents': documents_disclosed,

                # ===== CUSTOMER TYPE (after disclosure decisions) =====
                'customer_type': customer_type.capitalize() if customer_type else '',

                # ===== DECISION 3: Donation Default (matched to Agent-Level name) =====
                'donation_default': agent_donation_default,

                # ===== DECISION 4: Rejected Transaction Defaults (5 Priority Options) =====
                'rejected_transaction_1_choice': priority_choices[0],
                'rejected_transaction_2_choice': priority_choices[1],
                'rejected_transaction_3_choice': priority_choices[2],
                'rejected_transaction_4_choice': priority_choices[3],
                'rejected_transaction_5_choice': priority_choices[4],

                # ===== DECISION 5: Vendor Choice Weights =====
                'weight_price': weight_price,
                'weight_quality': weight_quality,
                'weight_proximity': weight_proximity,
                'weight_sustainability': weight_sustainability,

                # ===== DECISION 6 & 7: Timing =====
                'Period': period,
                'Purchase Timestamp': ts_result['formatted'],
                '_sort_datetime': request_datetime,  # Hidden sort key

                # ===== DECISION 8: Vendor Selection =====
                'Vendor ID': f"Vendor {int(vendor_id)}" if not pd.isna(vendor_id) else '',
                'Vendor Price': display_vendor_price,
                'Vendor Quality': vendor_quality,
                'Vendor Sustainability': vendor_sustainability,
                'Vendor Proximity': vendor_proximity,

                # ===== Standardized Vendor Scores =====
                'Standardized Vendor Price Score': vendor_price_score,
                'Standardized Vendor Quality Score': vendor_quality_score,
                'Standardized Vendor Sustainability Score': vendor_sustainability_score,
                'Standardized Vendor Proximity Score': vendor_proximity_score,
                'Vendor Integrated Score': vendor_integrated_score,

                # ===== DECISION 9 & 10: Purchase Decision & Pricing =====
                'Purchase Request Type': purchase_request_type,
                'Bid Value': display_bid_value,
                'Customer Price': display_customer_price,

                # ===== DECISION 11: Rejected Transaction Option =====
                'rejected_transaction_option': rejected_transaction_option,

                # ===== TRANSACTION OUTCOME =====
                'Transaction completed': 1,  # All requests are completed by default

                # ===== DECISION 13: Final Donation Rate (matched to Agent-Level name) =====
                'final_donation_rate': final_donation_rate,
            }
            
            transaction_records.append(transaction_record)
    
    # Sort by timestamp, assign unique Purchase Request IDs in that order, drop the sort key
    number_chronologically(transaction_records)
    
    return pd.DataFrame(transaction_records)
