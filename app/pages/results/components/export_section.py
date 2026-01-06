import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
from app.models import initialize_session_state
from app.utils.timestamp_utils import TimestampConverter, get_duration_hours
from src.vendor_attribute_generator import calculate_vendor_score_with_breakdown


def _apply_price_formatting(writer, sheet_name: str, df: pd.DataFrame):
    """
    Apply Excel number formatting to price-related columns to display 2 decimal places.
    
    This formats the DISPLAY only - the underlying values retain full precision.
    
    Args:
        writer: ExcelWriter object
        sheet_name: Name of the sheet to format
        df: DataFrame being written (to identify column positions)
    """
    from openpyxl.styles import numbers
    
    # Define columns that should display with 2 decimal places
    price_columns = [
        # Vendor attributes
        'Vendor Price', 'avg_vendor_price', 'price',
        # Customer prices
        'Customer Price', 'customer_price',
        # Bid values
        'Bid Value', 'bid_value',
        # Scores (normalized 0-1, but show 2 decimals for clarity)
        'Standardized Vendor Price Score', 'Standardized Vendor Quality Score',
        'Standardized Vendor Sustainability Score', 'Standardized Vendor Proximity Score',
        'Vendor Integrated Score', 'avg_vendor_score',
        # Donation rates
        'donation_default', 'Agent Donation Default', 'Final Donation Rate',
        'final_donation_rate', 'avg_vendor_quality', 'avg_vendor_sustainability',
        'avg_vendor_proximity', 'Vendor Proximity', 'Vendor Quality', 'Vendor Sustainability',
        # Income
        'income', 'Income',
        # Vendor choice weights
        'weight_price', 'weight_quality', 'weight_proximity', 'weight_sustainability',
    ]
    
    workbook = writer.book
    worksheet = workbook[sheet_name]
    
    # Get column indices for price columns
    for col_idx, col_name in enumerate(df.columns, start=1):
        if col_name in price_columns:
            # Apply number format to entire column (skip header row)
            for row_idx in range(2, len(df) + 2):  # Start from row 2 (after header)
                cell = worksheet.cell(row=row_idx, column=col_idx)
                if isinstance(cell.value, (int, float)) and cell.value is not None:
                    cell.number_format = '0.00'


def _build_compare_all_wide_format(results_dict, trait_columns):
    """
    Build a wide-format DataFrame for "Compare all" mode where each population mode
    has its own set of columns with correct agent traits and donation rates.
    
    Structure:
    | Copula_Agent_ID | Copula_HH | ... | Copula_donation_Cat | Copula_donation_Cont | ResSpec_Agent_ID | ResSpec_HH | ... |
    
    Each row contains data for 3 DIFFERENT agents (one from each population mode),
    but each agent's traits correctly match their donation rates.
    
    Args:
        results_dict: Dictionary with keys like 'copula_categorical', 'research_spec_continuous', etc.
        trait_columns: List of trait column names to include
        
    Returns:
        pd.DataFrame: Wide-format DataFrame with all population modes side-by-side
    """
    # Define population modes and their prefixes
    population_modes = [
        ('copula', 'Copula'),
        ('research_spec', 'ResSpec'),
        ('research_baseline', 'ResBase')
    ]
    
    income_modes = ['categorical', 'continuous']
    
    # Build a DataFrame for each population mode
    population_dfs = []
    
    for pop_key, pop_prefix in population_modes:
        # Find the DataFrames for this population mode
        cat_key = f"{pop_key}_categorical"
        cont_key = f"{pop_key}_continuous"
        
        cat_df = results_dict.get(cat_key)
        cont_df = results_dict.get(cont_key)
        
        # Use whichever DataFrame is available for traits (they have the same agents)
        base_df = cat_df if cat_df is not None else cont_df
        
        if base_df is None or base_df.empty:
            continue
        
        # Create DataFrame for this population mode
        pop_data = {}
        
        # Add Agent ID with prefix
        if 'agent_id' in base_df.columns:
            pop_data[f'{pop_prefix}_Agent_ID'] = base_df['agent_id'].values
        else:
            pop_data[f'{pop_prefix}_Agent_ID'] = list(range(1, len(base_df) + 1))
        
        # Add trait columns with prefix
        for trait in trait_columns:
            if trait in base_df.columns:
                # Use shorter column names for readability
                short_trait = trait.replace('Assigned Allowance Level', 'Income_Level')
                short_trait = short_trait.replace('TWT+Sospeso [=AW2+AX2]{Periods 1+2}', 'TWT_Sospeso')
                short_trait = short_trait.replace('Group_experiment', 'Group')
                short_trait = short_trait.replace('Study Program', 'Study_Program')
                pop_data[f'{pop_prefix}_{short_trait}'] = base_df[trait].values
        
        # Add donation columns for each income mode
        if cat_df is not None and not cat_df.empty and 'donation_default' in cat_df.columns:
            pop_data[f'{pop_prefix}_donation_Categorical'] = cat_df['donation_default'].values
        
        if cont_df is not None and not cont_df.empty and 'donation_default' in cont_df.columns:
            pop_data[f'{pop_prefix}_donation_Continuous'] = cont_df['donation_default'].values
        
        # Create DataFrame for this population mode
        pop_df = pd.DataFrame(pop_data)
        population_dfs.append(pop_df)
    
    # Combine all population DataFrames horizontally (side-by-side)
    if population_dfs:
        combined_df = pd.concat(population_dfs, axis=1)
    else:
        combined_df = pd.DataFrame()
    
    return combined_df


def _is_compare_all_mode(results_dict):
    """
    Check if results_dict contains configurations from multiple population modes.
    
    Returns True if we have configurations from different population types
    (copula, research_spec, research_baseline).
    """
    if results_dict is None or len(results_dict) <= 1:
        return False
    
    keys = list(results_dict.keys())
    
    has_copula = any(k.startswith('copula') for k in keys)
    has_research_spec = any(k.startswith('research_spec') for k in keys)
    has_research_baseline = any(k.startswith('research_baseline') for k in keys)
    
    # It's "Compare all" if we have configurations from at least 2 different population modes
    population_count = sum([has_copula, has_research_spec, has_research_baseline])
    return population_count >= 2


def _build_agent_level_dataframe(df, vendors_data=None, simulation_params=None):
    """
    Build agent-level DataFrame with one row per agent.
    
    Includes:
    - Agent ID and traits
    - All agent-level decisions
    - Summary statistics from transactions
    - Average vendor proximity, price, quality, sustainability, and score
    
    Args:
        df: Original simulation results DataFrame
        vendors_data: List of vendor dictionaries (optional)
        simulation_params: Simulation parameters (optional)
        
    Returns:
        pd.DataFrame: Agent-level data
    """
    agent_records = []
    
    # Pre-calculate static vendor averages (Price, Quality, Sustainability)
    avg_vendor_price_global = np.nan
    avg_vendor_quality_global = np.nan
    avg_vendor_sustainability_global = np.nan
    
    # Get configuration for score normalization
    price_min_config = 50.0
    price_max_config = 150.0
    
    # Try to get params from argument or session state
    sim_params = {}
    if simulation_params:
        sim_params = simulation_params.get('simulation', {})
    elif hasattr(st.session_state, 'simulation_params'):
        sim_params = st.session_state.simulation_params.get('simulation', {})
    elif hasattr(st.session_state, 'sim_params'):
        # Fallback to direct object access
        sim_params = {
            'vendor_price_min': getattr(st.session_state.sim_params, 'vendor_price_min', 50.0),
            'vendor_price_max': getattr(st.session_state.sim_params, 'vendor_price_max', 150.0)
        }
        
    if sim_params:
        price_min_config = sim_params.get('vendor_price_min', 50.0)
        price_max_config = sim_params.get('vendor_price_max', 150.0)
    
    if vendors_data:
        prices = [float(v.get('price', np.nan)) for v in vendors_data if v.get('price') is not None]
        qualities = [float(v.get('quality', np.nan)) for v in vendors_data if v.get('quality') is not None]
        susts = [float(v.get('sustainability', np.nan)) for v in vendors_data if v.get('sustainability') is not None]
        
        if prices: avg_vendor_price_global = np.mean(prices)
        if qualities: avg_vendor_quality_global = np.mean(qualities)
        if susts: avg_vendor_sustainability_global = np.mean(susts)
    
    for idx, row in df.iterrows():
        agent_id = row.get('agent_id', idx + 1)
        
        # Start with agent ID and traits
        agent_record = {
            'Agent ID': agent_id,
            'Honesty_Humility': row.get('Honesty_Humility', np.nan),
            'Assigned Allowance Level': row.get('Assigned Allowance Level', np.nan),
            'Study Program': row.get('Study Program', ''),
            'Group_experiment': row.get('Group_experiment', ''),
            'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': row.get('TWT+Sospeso [=AW2+AX2]{Periods 1+2}', np.nan),
        }
        
        # Income and Income Category (before Disclose Income)
        # Income: Try to get from multiple sources
        income = row.get('income', np.nan)
        # If income is NaN or not present, try actual_allowance as fallback
        if pd.isna(income) or income is None:
            income = row.get('actual_allowance', np.nan)
        
        agent_record['income'] = round(income, 2) if not pd.isna(income) else np.nan
        agent_record['income_category'] = row.get('income_category', np.nan)

        # Decision 1: Disclose Income (convert Y/N to 1/0 for consistency)
        disclose_income_raw = row.get('disclose_income', '')
        agent_record['disclose_income'] = 1 if disclose_income_raw == 'Y' else 0
        
        # Decision 2: Disclose Documents & Customer Type (convert Y/N to 1/0 for consistency)
        disclose_documents_raw = row.get('disclose_documents', '')
        agent_record['disclose_documents'] = 1 if disclose_documents_raw == 'Y' else 0
        agent_record['customer_type'] = row.get('customer_type', '')
        
        # Decision 3: Donation Default (exclude raw/intermediate columns)
        agent_record['donation_default'] = row.get('donation_default', np.nan)
        # NOTE: donation_default_raw_pos is intentionally excluded
        
        # Decision 4: Rejected Transaction Defaults - Split list into 5 priority columns
        rejected_defaults = row.get('rejected_transaction_defaults', '')
        
        # Parse if it's a string representation of a list
        if isinstance(rejected_defaults, str) and rejected_defaults.startswith('['):
            try:
                import ast
                rejected_defaults = ast.literal_eval(rejected_defaults)
            except:
                rejected_defaults = []
        elif not isinstance(rejected_defaults, list):
            rejected_defaults = [rejected_defaults] if rejected_defaults else []
        
        # Create 5 priority columns
        for priority_num in range(1, 6):
            if isinstance(rejected_defaults, list) and len(rejected_defaults) >= priority_num:
                agent_record[f'rejected_transaction_{priority_num}_choice'] = rejected_defaults[priority_num - 1]
            else:
                agent_record[f'rejected_transaction_{priority_num}_choice'] = 'N/A'
        
        # Decision 5: Vendor Choice Weights (flatten dict to columns)
        vendor_weights = row.get('vendor_choice_weights', {})
        if isinstance(vendor_weights, dict):
            agent_record['weight_price'] = vendor_weights.get('price', np.nan)
            agent_record['weight_quality'] = vendor_weights.get('quality', np.nan)
            agent_record['weight_proximity'] = vendor_weights.get('proximity', np.nan)
            agent_record['weight_sustainability'] = vendor_weights.get('sustainability', np.nan)
        else:
            agent_record['weight_price'] = np.nan
            agent_record['weight_quality'] = np.nan
            agent_record['weight_proximity'] = np.nan
            agent_record['weight_sustainability'] = np.nan
        
        # Decision 6: Purchasing Quantity (agent-level)
        # Split purchasing_quantity into two columns
        total_requests = row.get('purchasing_quantity', 0)
        agent_record['purchase_requests'] = total_requests  # Count of requests made
        agent_record['completed_transactions'] = total_requests  # Consistent with purchase requests
        
        # Decision 7: Purchasing Frequency
        agent_record['purchasing_frequency'] = row.get('purchasing_frequency', np.nan)
        
        # Decision 8: Vendor Selection (agent-level)
        # Note: This represents the highest scored vendor on average, not a fixed choice
        # In reality, vendor selection varies by product/request
        agent_record['most_selected_vendor'] = row.get('preferred_vendor', np.nan)
        
        # Vendor proximity scores
        proximity_scores = row.get('vendor_proximity_scores', {})
        if not isinstance(proximity_scores, dict):
            proximity_scores = {}
            
        # Get purchase requests for weighted averages
        purchase_requests = row.get('purchase_requests', [])
        if not isinstance(purchase_requests, list):
            purchase_requests = []
            
        # Calculate Weighted Averages based on Purchase Requests
        # If requests exist, average is weighted by the number of requests to each vendor
        # If no requests, fall back to the "most selected vendor" (preferred vendor)
        
        sum_price = 0
        sum_quality = 0
        sum_sust = 0
        sum_prox = 0
        sum_score = 0
        count_requests = 0
        
        # Helper to get vendor by ID
        def get_vendor_by_id(vid):
            if vendors_data:
                for v in vendors_data:
                    if str(v.get('vendor_id')) == str(vid):
                        return v
            return None

        # 1. Try to calculate from actual requests
        if purchase_requests:
            for req in purchase_requests:
                if isinstance(req, dict):
                    v_id = req.get('vendorID')
                    vendor = get_vendor_by_id(v_id)
                    
                    if vendor:
                        count_requests += 1
                        
                        # Attributes
                        v_price = vendor.get('price', np.nan)
                        v_quality = vendor.get('quality', np.nan)
                        v_sust = vendor.get('sustainability', np.nan)
                        
                        # Proximity (specific to this agent-vendor pair)
                        v_prox = np.nan
                        if v_id is not None:
                            v_prox = proximity_scores.get(str(int(v_id)), np.nan)
                            if pd.isna(v_prox) and str(v_id) in proximity_scores:
                                v_prox = proximity_scores[str(v_id)]
                        
                        # Add to sums (handle NaNs by skipping or treating as 0? skipping attribute specific sums)
                        if not pd.isna(v_price): sum_price += v_price
                        if not pd.isna(v_quality): sum_quality += v_quality
                        if not pd.isna(v_sust): sum_sust += v_sust
                        if not pd.isna(v_prox): sum_prox += float(v_prox)
                        
                        # Calculate Score for this specific transaction
                        if not (pd.isna(v_price) or pd.isna(v_quality) or pd.isna(v_sust) or pd.isna(v_prox)):
                            # Normalize
                            if price_max_config > price_min_config:
                                clamped_price = max(price_min_config, min(v_price, price_max_config))
                                norm_price = 1 - ((clamped_price - price_min_config) / (price_max_config - price_min_config))
                            else:
                                norm_price = 0.5
                                
                            norm_quality = (v_quality - 1) / 4 if v_quality >= 1 else 0
                            norm_sust = (v_sust - 1) / 4 if v_sust >= 1 else 0
                            norm_prox = float(v_prox) / 100
                            
                            score = (
                                vendor_weights.get('price', 0.25) * norm_price +
                                vendor_weights.get('quality', 0.25) * norm_quality +
                                vendor_weights.get('proximity', 0.25) * norm_prox +
                                vendor_weights.get('sustainability', 0.25) * norm_sust
                            )
                            sum_score += score

        # 2. Assign Averages
        if count_requests > 0:
            agent_record['avg_vendor_proximity'] = sum_prox / count_requests
            agent_record['avg_vendor_price'] = sum_price / count_requests
            agent_record['avg_vendor_quality'] = sum_quality / count_requests
            agent_record['avg_vendor_sustainability'] = sum_sust / count_requests
            agent_record['avg_vendor_score'] = sum_score / count_requests
        else:
            # Fallback: Use preferred vendor (most_selected_vendor) if available
            # This handles agents with 0 quantity who still have a preference
            pref_vendor_id = row.get('preferred_vendor')
            vendor = get_vendor_by_id(pref_vendor_id)
            
            if vendor:
                v_price = vendor.get('price', np.nan)
                v_quality = vendor.get('quality', np.nan)
                v_sust = vendor.get('sustainability', np.nan)
                
                v_prox = np.nan
                if pref_vendor_id is not None:
                    v_prox = proximity_scores.get(str(int(pref_vendor_id)), np.nan)
                
                agent_record['avg_vendor_price'] = v_price
                agent_record['avg_vendor_quality'] = v_quality
                agent_record['avg_vendor_sustainability'] = v_sust
                agent_record['avg_vendor_proximity'] = float(v_prox) if not pd.isna(v_prox) else np.nan
                
                # Calculate single score
                if not (pd.isna(v_price) or pd.isna(v_quality) or pd.isna(v_sust) or pd.isna(v_prox)):
                    if price_max_config > price_min_config:
                        clamped_price = max(price_min_config, min(v_price, price_max_config))
                        norm_price = 1 - ((clamped_price - price_min_config) / (price_max_config - price_min_config))
                    else:
                        norm_price = 0.5
                    
                    norm_quality = (v_quality - 1) / 4 if v_quality >= 1 else 0
                    norm_sust = (v_sust - 1) / 4 if v_sust >= 1 else 0
                    norm_prox = float(v_prox) / 100
                    
                    score = (
                        vendor_weights.get('price', 0.25) * norm_price +
                        vendor_weights.get('quality', 0.25) * norm_quality +
                        vendor_weights.get('proximity', 0.25) * norm_prox +
                        vendor_weights.get('sustainability', 0.25) * norm_sust
                    )
                    agent_record['avg_vendor_score'] = score
                else:
                    agent_record['avg_vendor_score'] = np.nan
            else:
                # No requests and no preferred vendor found - set to NaN
                agent_record['avg_vendor_proximity'] = np.nan
                agent_record['avg_vendor_price'] = np.nan
                agent_record['avg_vendor_quality'] = np.nan
                agent_record['avg_vendor_sustainability'] = np.nan
                agent_record['avg_vendor_score'] = np.nan
        
        # Decision 11: Rejected Transaction Option
        agent_record['rejected_transaction_option'] = row.get('rejected_transaction_option', '')
        
        # Decision 13: Final Donation Rate
        agent_record['final_donation_rate'] = row.get('final_donation_rate', np.nan)
        
        agent_records.append(agent_record)
    
    return pd.DataFrame(agent_records)


def _build_transaction_level_dataframe(df, vendors_data=None, simulation_params=None):
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
        simulation_params: Simulation parameters for pricing calculations
        
    Returns:
        pd.DataFrame: Transaction-level data
    """
    transaction_records = []
    
    # Get pricing parameters
    market_price = 100.0
    platform_markup = 0.1
    price_range = 0.25
    duration_hours = 1.0
    # Get configured price bounds for consistent normalization
    price_min_config = 50.0
    price_max_config = 150.0
    
    if simulation_params:
        sim_params = simulation_params.get('simulation', {})
        market_price = sim_params.get('market_price', 100.0)
        platform_markup = sim_params.get('platform_markup', 0.1)
        price_range = sim_params.get('price_range', 0.25)
        duration_hours = sim_params.get('duration_hours', 1.0)
        price_min_config = sim_params.get('vendor_price_min', 50.0)
        price_max_config = sim_params.get('vendor_price_max', 150.0)
    elif hasattr(st.session_state, 'simulation_params'):
        sim_params = st.session_state.simulation_params.get('simulation', {})
        market_price = sim_params.get('market_price', 100.0)
        platform_markup = sim_params.get('platform_markup', 0.1)
        price_range = sim_params.get('price_range', 0.25)
        duration_hours = sim_params.get('duration_hours', 1.0)
        price_min_config = sim_params.get('vendor_price_min', 50.0)
        price_max_config = sim_params.get('vendor_price_max', 150.0)
    elif hasattr(st.session_state, 'sim_params'):
        market_price = getattr(st.session_state.sim_params, 'market_price', 100.0)
        platform_markup = getattr(st.session_state.sim_params, 'platform_markup', 0.1)
        price_min_config = getattr(st.session_state.sim_params, 'vendor_price_min', 50.0)
        price_max_config = getattr(st.session_state.sim_params, 'vendor_price_max', 150.0)
        price_range = getattr(st.session_state.sim_params, 'price_range', 0.25)
        duration_hours = getattr(st.session_state.sim_params, 'duration_hours', 1.0)
    
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
    
    # Use centralized timestamp converter for consistent handling
    ts_converter = TimestampConverter()
    
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

        income_category = row.get('income_category', np.nan)
        agent_donation_default = row.get('donation_default', np.nan)

        # Decision 1: Income disclosed (convert Y/N to 1/0)
        disclose_income_raw = row.get('disclose_income', '')
        income_disclosed = 1 if disclose_income_raw == 'Y' else 0

        # Decision 2: Documents disclosed (convert Y/N/NA to 1/0)
        disclose_documents_raw = row.get('disclose_documents', '')
        documents_disclosed = 1 if disclose_documents_raw == 'Y' else 0

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
            # - For Discount/Fixed: np.nan (unknown) - use np.nan instead of 'N/A' for Arrow compatibility
            display_customer_price = customer_price if purchase_request_type in ['Purchase Now', 'Bid'] else np.nan
            
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
                'customer_type': customer_type.capitalize() if customer_type else '',

                # ===== DECISION 1: Disclose Income (1/0 format, matched to Agent-Level) =====
                'disclose_income': income_disclosed,

                # ===== DECISION 2: Disclose Documents (1/0 format, matched to Agent-Level) =====
                'disclose_documents': documents_disclosed,

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
                'Bid Value': bid_value if purchase_request_type == 'Bid' else np.nan,  # Use np.nan for Arrow compatibility
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


def render_export_section(df, results_dict=None, using_selected_config=False):
    """Render the export/download section (simplified)"""
    # Remove 'raw', 'index', 'consumption_frequency', 'actual_allowance', 'income', 'customer_type', and 'enriched_requests_count' columns before any processing
    # Use exact column name matching to avoid filtering out 'disclose_income' when we only want to exclude 'income'
    columns_to_exclude = ['raw', 'index', 'consumption_frequency', 'enriched_requests_count']
    
    if df is not None:
        df = df[[col for col in df.columns if col not in columns_to_exclude]]
    if results_dict is not None:
        results_dict = {
            key: config_df[[col for col in config_df.columns if col not in columns_to_exclude]]
            for key, config_df in results_dict.items()
        }

    st.subheader("💾 Export Results")
    
    # Check if this is a donation-only run (special simplified export)
    trait_columns = ['Honesty_Humility', 'Assigned Allowance Level', 'Study Program', 
                     'Group_experiment', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
    
    is_donation_only_run = (
        hasattr(st.session_state, 'custom_decisions') and 
        st.session_state.custom_decisions == ['donation_default'] and
        hasattr(st.session_state, 'default_decisions') and
        len(st.session_state.default_decisions) == 0
    )
    
    if is_donation_only_run:
        # DONATION-ONLY EXPORT: Simplified version with just donation and traits
        # Filter main df to only include donation columns
        columns_to_keep = [col for col in df.columns 
                          if (col == 'donation_default' or col in trait_columns or col == 'agent_id')]
        df = df[columns_to_keep]
        
        # Filter results_dict to only include donation columns
        if results_dict is not None:
            results_dict = {
                key: config_df[[col for col in config_df.columns 
                               if (col == 'donation_default' or col in trait_columns or col == 'agent_id')]]
                for key, config_df in results_dict.items()
            }
        
        # Check if we have multiple configurations to compare
        export_all_configs = results_dict is not None and len(results_dict) > 1 and not using_selected_config
        
        # Check if this is "Compare all" mode (different population modes with different agents)
        is_compare_all = _is_compare_all_mode(results_dict)
        
        if export_all_configs and is_compare_all:
            st.markdown(f"""
            **Donation Default Results Export (Compare All Mode - {len(results_dict)} Configurations):**
            - Each population mode (Copula, Research Spec, Research Baseline) has its own columns
            - Agent ID, traits, and donation rates for Categorical and Continuous income modes
            - **Note:** Each row contains 3 different agents (one per population mode), each with their correct traits
            """)
        elif export_all_configs:
            st.markdown(f"""
            **Donation Default Results Export (All {len(results_dict)} Configurations):**
            - Agent ID, trait columns, and donation_default rate for each configuration
            - All configurations combined in one file for comparison
            """)
        else:
            st.markdown("""
            **Donation Default Results Export:**
            - Agent ID, trait columns, and donation_default rate
            """)
        
        try:
            buffer = BytesIO()
            
            if export_all_configs and is_compare_all:
                # COMPARE ALL MODE: Each population mode gets its own sheet
                # Each sheet contains traits + both categorical and continuous donation columns
                
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
                
                if not sheets_data:
                    st.warning("⚠️ No data available for export")
                    return
                
                # Write each population mode to its own sheet
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    for sheet_name, sheet_df in sheets_data.items():
                        sheet_df.to_excel(writer, index=False, sheet_name=sheet_name)
                        # Apply 2-decimal formatting to price/rate columns
                        _apply_price_formatting(writer, sheet_name, sheet_df)
                
                # Show metrics
                total_sheets = len(sheets_data)
                first_sheet_df = next(iter(sheets_data.values()))
                n_agents = len(first_sheet_df)
                n_columns = len(first_sheet_df.columns)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sheets", total_sheets)
                with col2:
                    st.metric("Agents per Sheet", n_agents)
                with col3:
                    st.metric("Columns per Sheet", n_columns)
                
                excel_label = f"📊 Download Donation Excel ({total_sheets} Sheets)"
                excel_filename = f"donation_compare_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                
                st.download_button(
                    label=excel_label,
                    data=buffer.getvalue(),
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Each population mode has its own sheet with traits and donation rates for both income modes"
                )
                
                # Show preview with explanation
                with st.expander("📋 Preview Donation Data (first 5 rows per sheet)"):
                    st.info("""
                    **Sheet Structure:**
                    - **Copula**: Synthetic agents generated from copula
                    - **ResSpec**: Original 280 participants (random sample)
                    - **ResBase**: Original 280 participants (sequential order)
                    
                    Each sheet contains Agent ID, traits, and donation rates for both Categorical and Continuous income modes.
                    """)
                    
                    for sheet_name, sheet_df in sheets_data.items():
                        st.markdown(f"**{sheet_name} Sheet:**")
                        st.dataframe(sheet_df.head(), use_container_width=True)
                        st.caption(f"Columns: {', '.join(sheet_df.columns)}")
            
            elif export_all_configs:
                # SAME POPULATION MODE: Multiple income modes with same agents
                # Safe to combine by row since agents are identical
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
                
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    combined_df.to_excel(writer, index=False, sheet_name='All Configurations')
                    # Apply 2-decimal formatting to price/rate columns
                    _apply_price_formatting(writer, 'All Configurations', combined_df)
                
                st.metric("Total Agents", len(combined_df))
                
                excel_label = f"📊 Download Donation Excel (All {len(results_dict)} Configs)"
                excel_filename = f"donation_all_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                
                st.download_button(
                    label=excel_label,
                    data=buffer.getvalue(),
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help=f"Donation results with {len(results_dict)} configurations for comparison"
                )
                
                # Show preview
                with st.expander("📋 Preview Donation Data (first 5 rows)"):
                    st.dataframe(combined_df.head(), use_container_width=True)
                    st.caption(f"**Columns ({len(combined_df.columns)})**: {', '.join(combined_df.columns[:10])}{'...' if len(combined_df.columns) > 10 else ''}")
            
            else:
                # SINGLE CONFIG: Simple export with just one configuration
                df_export = df.copy()
                
                # Rename agent_id to 'Agent ID' for clarity
                if 'agent_id' in df_export.columns:
                    df_export = df_export.rename(columns={'agent_id': 'Agent ID'})
                
                # Reorder columns to put Agent ID first
                if 'Agent ID' in df_export.columns:
                    cols = ['Agent ID'] + [col for col in df_export.columns if col != 'Agent ID']
                    df_export = df_export[cols]
                
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_export.to_excel(writer, index=False, sheet_name='Donation Results')
                    # Apply 2-decimal formatting to price/rate columns
                    _apply_price_formatting(writer, 'Donation Results', df_export)
                
                st.metric("Total Agents", len(df_export))
                
                excel_filename = f"donation_default_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                st.download_button(
                    label="📊 Download Donation Default Excel",
                    data=buffer.getvalue(),
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Donation results with agent traits and donation rates"
                )
                
                # Show preview
                with st.expander("📋 Preview Donation Data (first 5 rows)"):
                    st.dataframe(df_export.head(), use_container_width=True)
                    st.caption(f"**Columns ({len(df_export.columns)})**: {', '.join(df_export.columns)}")
        
        except ImportError:
            st.caption("⚠️ Excel export requires openpyxl")
    
    else:
        # FULL TWO-LEVEL EXPORT: For all other simulations
        st.markdown("""
        **Two separate Excel files are available for download:**
        - **Agent-Level Excel**: One row per agent with all agent-level decisions and summary statistics
        - **Transaction-Level Excel**: One row per purchase request with detailed transaction information
        """)
        
        # Get vendor data if available
        vendors_data = None
        if hasattr(df, 'attrs') and 'vendors' in df.attrs:
            vendors_data = df.attrs['vendors']
        
        # Get simulation parameters if available
        simulation_params = None
        if hasattr(st.session_state, 'simulation_params'):
            simulation_params = st.session_state.simulation_params
        
        try:
            # Build agent-level and transaction-level DataFrames
            agent_df = _build_agent_level_dataframe(df, vendors_data=vendors_data, simulation_params=simulation_params)
            transaction_df = _build_transaction_level_dataframe(df, vendors_data=vendors_data, simulation_params=simulation_params)
            
            # Show summary statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Agents", len(agent_df))
                st.caption("Rows in Agent-Level file")
            with col2:
                st.metric("Total Transactions", len(transaction_df))
                st.caption("Rows in Transaction-Level file")
            
            # Create two separate Excel files
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Agent-Level Excel
            agent_buffer = BytesIO()
            with pd.ExcelWriter(agent_buffer, engine='openpyxl') as writer:
                agent_df.to_excel(writer, index=False, sheet_name='Agent Level')
                # Apply 2-decimal formatting to price columns
                _apply_price_formatting(writer, 'Agent Level', agent_df)
            
            # Transaction-Level Excel
            transaction_buffer = BytesIO()
            with pd.ExcelWriter(transaction_buffer, engine='openpyxl') as writer:
                transaction_df.to_excel(writer, index=False, sheet_name='Transaction Level')
                # Apply 2-decimal formatting to price columns
                _apply_price_formatting(writer, 'Transaction Level', transaction_df)
            
            # Download buttons for separate files
            st.markdown("### 📥 Download Files")
            col1, col2 = st.columns(2)
            
            with col1:
                agent_filename = f"simulation_agent_level_{timestamp}.xlsx"
                st.download_button(
                    label="📊 Download Agent-Level Excel",
                    data=agent_buffer.getvalue(),
                    file_name=agent_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help=f"Agent-level data: {len(agent_df)} agents × {len(agent_df.columns)} columns"
                )
            
            with col2:
                transaction_filename = f"simulation_transaction_level_{timestamp}.xlsx"
                st.download_button(
                    label="📊 Download Transaction-Level Excel",
                    data=transaction_buffer.getvalue(),
                    file_name=transaction_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help=f"Transaction-level data: {len(transaction_df)} transactions × {len(transaction_df.columns)} columns"
                )
            
            # Show preview of what's in each file
            st.markdown("### 📋 Data Preview")
            
            with st.expander("👥 Preview Agent-Level Data (first 5 rows)"):
                st.dataframe(agent_df.head(), use_container_width=True)
                st.caption(f"**Columns ({len(agent_df.columns)})**: {', '.join(agent_df.columns[:15])}{'...' if len(agent_df.columns) > 15 else ''}")
            
            with st.expander("🔄 Preview Transaction-Level Data (first 5 rows)"):
                st.dataframe(transaction_df.head(), use_container_width=True)
                st.caption(f"**Columns ({len(transaction_df.columns)})**: {', '.join(transaction_df.columns[:15])}{'...' if len(transaction_df.columns) > 15 else ''}")
            
        except Exception as e:
            st.error(f"Error creating Excel export: {str(e)}")
            st.caption("⚠️ Please ensure all required data is available. If the problem persists, contact support.")
            import traceback
            st.caption(f"Error details: {traceback.format_exc()}")
            
            # Fallback: show raw data
            with st.expander("🔍 View Raw Data (for debugging)"):
                st.dataframe(df, use_container_width=True)

    if st.button("🔄 Clear Results"):
        # Clear all session state to reset the entire application
        keys_to_delete = [key for key in st.session_state.keys()]
        for key in keys_to_delete:
            del st.session_state[key]
        
        # Reinitialize session state with default values
        initialize_session_state()
        
        # Stay on results page to show "no results" message
        st.session_state.page = 'results'
        
        # Force page reload
        st.rerun()
