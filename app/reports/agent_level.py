"""Agent-level export: one row per agent.

`build_agent_level_dataframe` is `_build_agent_level_dataframe` moved verbatim
out of `app/pages/results/components/export_section.py`. The only edits are to
its head: the two vendor-price bounds it used to read off
`st.session_state.sim_params` are now parameters, so nothing here imports
Streamlit or touches session state. Every value, rounding and column order is
exactly what the page produced.
"""
import numpy as np
import pandas as pd


def build_agent_level_dataframe(
    df,
    vendors_data=None,
    *,
    vendor_price_min: float = 50.0,
    vendor_price_max: float = 150.0,
):
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
        vendor_price_min: Configured lower price bound used to normalise the
            vendor price score (was ``sim_params.vendor_price_min``)
        vendor_price_max: Configured upper price bound used to normalise the
            vendor price score (was ``sim_params.vendor_price_max``)

    Returns:
        pd.DataFrame: Agent-level data
    """
    agent_records = []
    
    # Pre-calculate static vendor averages (Price, Quality, Sustainability)
    avg_vendor_price_global = np.nan
    avg_vendor_quality_global = np.nan
    avg_vendor_sustainability_global = np.nan
    # Configuration for score normalization. The page reads these off
    # st.session_state.sim_params and passes the VALUES in.
    price_min_config = vendor_price_min
    price_max_config = vendor_price_max
    
    if vendors_data:
        prices = [float(v.get('price', np.nan)) for v in vendors_data if v.get('price') is not None]
        qualities = [float(v.get('quality', np.nan)) for v in vendors_data if v.get('quality') is not None]
        susts = [float(v.get('sustainability', np.nan)) for v in vendors_data if v.get('sustainability') is not None]
        
        if prices: avg_vendor_price_global = np.mean(prices)
        if qualities: avg_vendor_quality_global = np.mean(qualities)
        if susts: avg_vendor_sustainability_global = np.mean(susts)
    
    for idx, row in df.iterrows():
        agent_id = row.get('agent_id', idx + 1)
        
        # Start with agent ID and all personality traits
        agent_record = {
            'Agent ID': agent_id,
            'Agreeable': row.get('Agreeable', np.nan),
            'Openness': row.get('OpennessBig5', np.nan),
            'Honesty_Humility': row.get('Honesty_Humility', np.nan),
            'Extraversion': row.get('ExtraversionBig5', np.nan),
            'Neuroticism': row.get('NeuroticismBig5', np.nan),
            'ReligiousAffiliation': row.get('ReligiousAffiliation', np.nan),
            'ReligiousService': row.get('ReligiousService', np.nan),
            'Assigned Allowance Level': row.get('Assigned Allowance Level', np.nan),
            'Study Program': row.get('Study Program', ''),
            'Group_experiment': row.get('Group_experiment', ''),
            'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': row.get('TWT+Sospeso [=AW2+AX2]{Periods 1+2}', np.nan),
        }
        
        # Religious composite (from decision output, or compute fallback)
        if 'disclose_income_religious_composite' in row and pd.notna(row.get('disclose_income_religious_composite')):
            agent_record['Religious'] = row.get('disclose_income_religious_composite')
        else:
            # Fallback: compute from raw values if available
            ra = row.get('ReligiousAffiliation', np.nan)
            rs = row.get('ReligiousService', np.nan)
            if pd.notna(ra) and pd.notna(rs):
                rs_scaled = rs / 4.0  # Scale to 0-1 (assuming max=4)
                agent_record['Religious'] = (ra + rs_scaled) / 2
            else:
                agent_record['Religious'] = np.nan
        
        # Income and Income Category (before Disclose Income)
        # Income: Try to get from multiple sources
        income = row.get('income', np.nan)
        # If income is NaN or not present, try actual_allowance as fallback
        if pd.isna(income) or income is None:
            income = row.get('actual_allowance', np.nan)
        
        agent_record['income'] = round(income, 2) if not pd.isna(income) else np.nan
        # Income category - use 'N/A' for empty/missing values (e.g., regular customers who didn't disclose income)
        income_category_raw = row.get('income_category', np.nan)
        agent_record['income_category'] = 'N/A' if (pd.isna(income_category_raw) or income_category_raw == '' or income_category_raw is None) else income_category_raw

        # Disclose Income calculated columns
        # Check if disclose_income was actually calculated (not defaulted) by checking for calculated columns
        disclose_income_was_calculated = 'disclose_income_di' in row and pd.notna(row.get('disclose_income_di'))
        
        # I-High (income_high indicator)
        if disclose_income_was_calculated and 'disclose_income_income_high' in row and pd.notna(row.get('disclose_income_income_high')):
            agent_record['I-High'] = row.get('disclose_income_income_high')
        elif disclose_income_was_calculated:
            # Fallback: compute from Assigned Allowance Level
            level = row.get('Assigned Allowance Level', np.nan)
            if pd.notna(level):
                agent_record['I-High'] = 1 if level > 3 else 0
            else:
                agent_record['I-High'] = 'N/A'
        else:
            agent_record['I-High'] = 'N/A'  # Defaulted - not calculated
        
        # Configuration Values (Weights and Intercept) - N/A if defaulted
        if disclose_income_was_calculated:
            agent_record['WOPB'] = row.get('disclose_income_wopb', np.nan)
            agent_record['WPB'] = row.get('disclose_income_wpb', np.nan)
            agent_record['Intercept'] = row.get('disclose_income_intercept', np.nan)
        else:
            agent_record['WOPB'] = 'N/A'
            agent_record['WPB'] = 'N/A'
            agent_record['Intercept'] = 'N/A'
        
        # Calculated Values (PB_i and DI_i) - N/A if defaulted
        if disclose_income_was_calculated:
            agent_record['PB_i'] = row.get('disclose_income_anchored_pb', np.nan)
            agent_record['DI_i'] = row.get('disclose_income_di', row.get('disclose_income_raw', np.nan))
        else:
            agent_record['PB_i'] = 'N/A'
            agent_record['DI_i'] = 'N/A'

        # Decision 1: Disclose Income (convert Y/N to 1/0 for consistency)
        disclose_income_raw = row.get('disclose_income', '')
        agent_record['disclose_income'] = 1 if disclose_income_raw == 'Y' else 0
        
        # Decision 2: Disclose Documents & Customer Type
        # Three-state outcome: Y=disclosed (1), N=asked-but-declined (0), NA=never-eligible (gated out).
        # Preserve NA so the qualified-disclosure denominator is recoverable (do NOT collapse N and NA to 0).
        disclose_documents_raw = row.get('disclose_documents', '')
        agent_record['disclose_documents'] = (
            1 if disclose_documents_raw == 'Y' else (0 if disclose_documents_raw == 'N' else 'NA')
        )
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

        # Decision 4 MODEL outputs (five trait-based sub-decision elements + the
        # Section-6 rank aggregation; present only when Decision 4 ran as a custom
        # decision). The complete-simulation results page shows only the integrated
        # default list, so this workbook must carry EVERY element variable (professor
        # 2026-09): the raw inputs, the z-scores the equations use, each element's
        # score, intermediates, deterministic/final segment, its option list, and the
        # integration diagnostics. Lists are exported as 'a > b > c' option-number
        # strings; the dedicated Decision 4 Excel has the per-element sheets.
        if 'rtd_choice_length' in row:
            # -- inputs the Decision 4 equations use (traits are already in the record
            #    above; these add the ones only Decision 4 needs) --
            agent_record['rtd_Conscientiousness'] = row.get('ConscientiousnessBig5', np.nan)
            agent_record['rtd_Education'] = row.get('Education', np.nan)
            agent_record['rtd_stdactions'] = row.get('stdactions', np.nan)
            agent_record['rtd_income_mode'] = row.get('rtd_income_mode', 'N/A')
            # -- z-scores actually multiplied by the coefficients (Stata names) --
            for src, export_name in [
                    ('rtd_z_extraversion', 'rtd_z_extraversionbig5'),
                    ('rtd_z_agreeable', 'rtd_z_agreeable'),
                    ('rtd_z_neuroticism', 'rtd_z_neuroticismbig5'),
                    ('rtd_z_conscientiousness', 'rtd_z_conscientiousnessbig5'),
                    ('rtd_z_openness', 'rtd_z_opennessbig5'),
                    ('rtd_reducation', 'rtd_reducation'),
                    ('rtd_z_income', 'rtd_z_net_income'),
                    ('rtd_z_stdactions', 'rtd_z_stdactions')]:
                if src in row:
                    agent_record[export_name] = row.get(src, np.nan)
            # -- 1. Options List Length (Tendency to Plan) --
            agent_record['rtd_weighted_ttp'] = row.get('rtd_weighted_ttp', np.nan)
            agent_record['rtd_weighted_ttp06'] = row.get('rtd_weighted_ttp06', np.nan)
            agent_record['rtd_choice_length_deterministic'] = row.get(
                'rtd_choice_length_deterministic', np.nan)
            agent_record['rtd_choice_length'] = row.get('rtd_choice_length', np.nan)
            # -- 2-5. Ranking elements --
            for src in ('rtd_flex_ivw', 'rtd_flex_z_ivw'):
                if src in row:
                    agent_record[src] = row.get(src, np.nan)
            for mech_col, export_name in [('loyalty', 'rtd_loyalty'), ('wtp', 'rtd_wtp'),
                                          ('rt', 'rtd_risk_taking'), ('flex', 'rtd_flexibility')]:
                if f'rtd_{mech_col}_segment' not in row:
                    continue
                agent_record[f'{export_name}_score'] = row.get(f'rtd_{mech_col}_score', np.nan)
                agent_record[f'{export_name}_z'] = row.get(f'rtd_{mech_col}_z', np.nan)
                agent_record[f'{export_name}_segment_deterministic'] = row.get(
                    f'rtd_{mech_col}_segment_deterministic', np.nan)
                agent_record[f'{export_name}_segment'] = row.get(f'rtd_{mech_col}_segment', np.nan)
                ranking = row.get(f'rtd_{mech_col}_ranking', None)
                agent_record[f'{export_name}_ranking'] = (
                    ' > '.join(str(o) for o in ranking) if isinstance(ranking, list) else 'N/A'
                )
            # -- 6. Section-6 rank aggregation: the integrated default list (option
            # numbers, after the list-length and Option-5 truncation), the integrated
            # ranking it was cut from, and the tie-break diagnostics. Headers say
            # "integrated", never "consensus" (professor 2026-09-17; Lavie #14).
            if 'rtd_default_list' in row:
                default_list = row.get('rtd_default_list', None)
                agent_record['rtd_default_list'] = (
                    ' > '.join(str(o) for o in default_list) if isinstance(default_list, list) else 'N/A'
                )
                consensus = row.get('rtd_consensus_ranking', None)
                # Exported as rtd_integrated_ranking (professor 2026-09-17: "integrated",
                # not "consensus", in the Excel outputs); the model columns keep their names.
                agent_record['rtd_integrated_ranking'] = (
                    ' > '.join(str(o) for o in consensus) if isinstance(consensus, list) else 'N/A'
                )
                agent_record['rtd_integrated_kemeny_status'] = row.get('rtd_consensus_kemeny_status', 'N/A')
                agent_record['rtd_integrated_n_kemeny_optimal'] = row.get('rtd_consensus_n_kemeny_optimal', np.nan)
                agent_record['rtd_integrated_is_kemeny_optimal'] = row.get('rtd_consensus_is_kemeny_optimal', 'N/A')
                agent_record['rtd_integrated_settled_by'] = row.get('rtd_consensus_settled_by', 'N/A')
                agent_record['rtd_integrated_truncated_by'] = row.get('rtd_consensus_truncated_by', 'N/A')
                agent_record['rtd_default_list_length'] = row.get('rtd_default_list_length', np.nan)


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
