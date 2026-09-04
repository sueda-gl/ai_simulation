"""Transaction-level export: one row per purchase request.

`build_transaction_level_dataframe` is `_build_transaction_level_dataframe`
moved verbatim out of `app/pages/results/components/export_section.py`. The
only edits are to its head: the pricing parameters it used to read off
`st.session_state.sim_params`, and the base time / period duration / period
count its `TimestampConverter` used to read from session state, are now
parameters. Nothing here imports Streamlit or touches session state; every
value, rounding and column order is exactly what the page produced.
"""
from datetime import datetime

import numpy as np
import pandas as pd

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
            reads it from ``app.utils.timestamp_utils.get_simulation_base_time``).
        periods: Number of periods in the run (was ``get_periods()``).
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
    baseline_price = (1 + platform_markup) * market_price
    pn_price = (1 + price_range) * baseline_price
    discount_price = market_price * 0.7
    fixed_price = market_price
    
    # Build vendor lookup
    vendor_lookup = {}
    if vendors_data:
        for vendor in vendors_data:
            vendor_id = vendor.get('vendor_id')
            vendor_lookup[vendor_id] = vendor
    
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
        income_category_raw = row.get('income_category', np.nan)
        income_category = 'N/A' if (pd.isna(income_category_raw) or income_category_raw == '' or income_category_raw is None) else income_category_raw
        agent_donation_default = row.get('donation_default', np.nan)

        # Decision 1: Income disclosed (convert Y/N to 1/0)
        disclose_income_raw = row.get('disclose_income', '')
        income_disclosed = 1 if disclose_income_raw == 'Y' else 0

        # Decision 2: Documents disclosed. Preserve the three-state outcome:
        # Y=disclosed (1), N=asked-but-declined (0), NA=never-eligible. Do NOT collapse N and NA to 0.
        disclose_documents_raw = row.get('disclose_documents', '')
        documents_disclosed = (
            1 if disclose_documents_raw == 'Y' else (0 if disclose_documents_raw == 'N' else 'NA')
        )

        # Decision 4: Rejected Transaction Defaults - 5 priority columns
        rejected_defaults = row.get('rejected_transaction_defaults', '')
        if isinstance(rejected_defaults, str) and rejected_defaults.startswith('['):
            try:
                import ast
                rejected_defaults = ast.literal_eval(rejected_defaults)
            except:
                rejected_defaults = []
        elif not isinstance(rejected_defaults, list):
            rejected_defaults = [rejected_defaults] if rejected_defaults else []

        # Create 5 priority values
        priority_choices = []
        for priority_num in range(1, 6):
            if isinstance(rejected_defaults, list) and len(rejected_defaults) >= priority_num:
                priority_choices.append(rejected_defaults[priority_num - 1])
            else:
                priority_choices.append('N/A')

        # Decision 5: Vendor Choice Weights
        vendor_weights = row.get('vendor_choice_weights', {})
        if not isinstance(vendor_weights, dict):
            vendor_weights = {'price': 0.25, 'quality': 0.25, 'proximity': 0.25, 'sustainability': 0.25}
        weight_price = vendor_weights.get('price', np.nan)
        weight_quality = vendor_weights.get('quality', np.nan)
        weight_proximity = vendor_weights.get('proximity', np.nan)
        weight_sustainability = vendor_weights.get('sustainability', np.nan)

        # Decision 11: Rejected Transaction Option
        rejected_transaction_option = row.get('rejected_transaction_option', '')
        
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
        
        # Get purchase requests
        purchase_requests = row.get('purchase_requests', [])
        if not isinstance(purchase_requests, list):
            continue
        
        # Process each purchase request
        for req_idx, request in enumerate(purchase_requests):
            if not isinstance(request, dict):
                continue
            
            # Transaction identification
            request_id = request.get('request_id', req_idx + 1)
            # Use global transaction_id if available (assigned by simulation.py), otherwise fallback
            transaction_id = request.get('transaction_id', f"A{agent_id}_R{request_id}")
            
            # Timing - use centralized timestamp converter
            timestamp_hours = request.get('timestamp_hours', np.nan)
            ts_result = ts_converter.convert(timestamp_hours)
            period = ts_result['period']
            request_datetime = ts_result['datetime']
            purchase_date = ts_result['date']
            purchase_time = ts_result['time']
            
            # Vendor information - use centralized scoring function
            vendor_id = request.get('vendorID', np.nan)
            vendor_price = np.nan
            vendor_quality = np.nan
            vendor_sustainability = np.nan
            vendor_proximity = np.nan
            vendor_price_score = np.nan
            vendor_quality_score = np.nan
            vendor_sustainability_score = np.nan
            vendor_proximity_score = np.nan
            vendor_integrated_score = np.nan

            if not pd.isna(vendor_id) and vendor_id in vendor_lookup:
                vendor = vendor_lookup[vendor_id]
                vendor_price = vendor.get('price', np.nan)
                vendor_quality = vendor.get('quality', np.nan)
                vendor_sustainability = vendor.get('sustainability', np.nan)
                vendor_proximity = proximity_scores.get(str(int(vendor_id)), np.nan)
                
                # Use centralized scoring function for consistency
                if not pd.isna(vendor_price) and not pd.isna(vendor_quality) and \
                   not pd.isna(vendor_sustainability) and not pd.isna(vendor_proximity):
                    
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
            
            # Purchase decision and pricing
            platform_price = request.get('platformPrice', '')
            bid_value = request.get('bid_value', 'N/A')
            
            # Determine purchase request type and customer price
            # Customer Type can be: 'Regular', 'Fixed', or 'Bid'
            # Note: 'PN' (Purchase Now) is treated as a sub-type of 'Regular' or 'Bid' but for high-level Customer Type
            # we classify it based on the platform price mechanism.
            
            customer_type_str = 'Regular' # Default
            
            if platform_price == 'DISCOUNT':
                purchase_request_type = 'Discount'
                customer_price = discount_price
                customer_type_str = 'Regular' # Discount is a type of regular price
            elif platform_price == 'FIXED':
                purchase_request_type = 'Fixed'
                customer_price = fixed_price
                customer_type_str = 'Fixed'
            elif platform_price == 'PN':
                purchase_request_type = 'Purchase Now'
                customer_price = pn_price
                # PN is typically available in Bid scenarios or as a specific option, 
                # but if it stands alone or is the chosen option, it's a fixed price purchase.
                # However, user requested mapping: Regular, Bid, Fixed.
                # If PN is a "buy it now" option in a bid, it might be considered 'Bid' context or 'Fixed' price execution.
                # Let's look at how customer_type is derived in the simulation.
                # If customer_type variable exists, use it.
                if customer_type:
                     customer_type_str = customer_type.capitalize()
                else:
                     customer_type_str = 'Regular' 
            elif platform_price == 'BID':
                purchase_request_type = 'Bid'
                customer_type_str = 'Bid'
                try:
                    # For BID transactions, Customer Price is the bid amount if successful
                    bid_val_numeric = float(bid_value) if bid_value != 'N/A' else pn_price
                    customer_price = bid_val_numeric
                except (ValueError, TypeError):
                    customer_price = pn_price
            else:
                purchase_request_type = customer_type.capitalize() if customer_type else 'Regular'
                customer_price = pn_price
                customer_type_str = 'Regular'
            
            # Display customer price:
            # - For Purchase Now: pn_price
            # - For Bid: bid_value (if numeric)
            # - For Discount/Fixed: 'N/A' (actual price calculated by separate algorithm)
            display_customer_price = customer_price if purchase_request_type in ['Purchase Now', 'Bid'] else 'N/A'
            
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
                'income': round(income, 2) if not pd.isna(income) else np.nan,
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
                'Vendor Price': vendor_price,
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
                'Bid Value': bid_value if purchase_request_type == 'Bid' else 'N/A',  # N/A for non-Bid transactions
                'Customer Price': display_customer_price,

                # ===== DECISION 11: Rejected Transaction Option =====
                'rejected_transaction_option': rejected_transaction_option,

                # ===== TRANSACTION OUTCOME =====
                'Transaction completed': 1,  # All requests are completed by default

                # ===== DECISION 13: Final Donation Rate (matched to Agent-Level name) =====
                'final_donation_rate': final_donation_rate,
            }
            
            transaction_records.append(transaction_record)
    
    # Sort by timestamp
    if transaction_records:
        transaction_records.sort(key=lambda x: x.get('_sort_datetime', datetime.min))
        
        # Assign unique Purchase Request IDs based on sorted order and remove sort key
        for idx, record in enumerate(transaction_records):
            record['Purchase Request ID'] = idx + 1
            record.pop('_sort_datetime', None)  # Remove hidden sort key
            if 'Transaction ID' in record:
                del record['Transaction ID']
    
    return pd.DataFrame(transaction_records)
