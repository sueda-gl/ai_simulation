# app/pages/decision_execution.py
"""
Decision execution functions for running individual and combined simulations.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from app.simulation import run_simulation_from_sidebar
from app.models import ALL_DECISIONS

# Default values for unselected decisions
DEFAULT_DECISION_VALUES = {
    "donation_default": 0.10,  # 10%
    "disclose_income": {
        "type": "random_probability",
        "probability_y": 0.5,  # 50% chance of Y (disclosing)
        "options": ["Y", "N"],
        "description": "Probability of disclosing income for Fixed status"
    },
    "disclose_documents": {
        "type": "random_probability", 
        "probability_y": 0.5,  # 50% chance of Y (disclosing)
        "options": ["Y", "N"],
        "description": "Probability of disclosing documents for Discount status"
    },
    "rejected_transaction_defaults": "SIMULATION_MODE_DEPENDENT",  # Option 5 for snapshot, real-time for live
    "vendor_choice_weights": {
        "price": 0.25,
        "quality": 0.25,
        "proximity": 0.25,
        "sustainability": 0.25
    },
    "consumption_quantity": "RANDOM_WITHIN_LIMIT",  # Random within consumption limit
    "consumption_frequency": "CALCULATED",  # Consumption quantity / Period duration
    "vendor_selection": "deterministic",  # Deterministic based on weights
    "purchase_vs_bid": {
        "type": "random_probability",
        "probability_y": 0.5,  # 50% chance of purchase (vs bid)
        "options": ["purchase", "bid"],
        "description": "Probability of purchasing immediately vs bidding"
    },
    "bid_value": "RANDOM_WITHIN_RANGE",  # Random within bidding price range
    "rejected_transaction_option": "forgo_transaction",  # Option 5 (forgo transaction)
    "rejected_bid_value": "NA",  # Not relevant given Option 5
    "final_donation_rate": 0.10  # Keep default 10%
}

# Description text for display purposes
DEFAULT_DECISION_DESCRIPTIONS = {
    "donation_default": "10%",
    "disclose_income": "configurable probability Y/N (default 50% each)", 
    "disclose_documents": "configurable probability Y/N for qualified users (default 50% each)",
    "rejected_transaction_defaults": "Default behavior for handling rejected transactions will be applied",
    "vendor_choice_weights": "equal weight of 25% to Price, Quality, Proximity, and Sustainability",
    "consumption_quantity": "random within consumption limit",
    "consumption_frequency": "Consumption quantity divided by Period duration",
    "vendor_selection": "deterministic based on vendor choice weights",
    "purchase_vs_bid": "configurable probability purchase/bid (default 50% each)",
    "bid_value": "random within bidding price range",
    "rejected_transaction_option": "Default option for handling rejected transactions will be used",
    "rejected_bid_value": "Default handling for rejected bid values will be applied",
    "final_donation_rate": "Default donation rate will be maintained"
}


def get_actual_default_value(decision_name, sim_params=None):
    """
    Get the actual default value for a decision, handling random generation where needed.
    This function returns values that can be used directly by the simulation.
    """
    import random
    import streamlit as st
    
    base_value = DEFAULT_DECISION_VALUES.get(decision_name)
    
    # NEW: Handle parametric random decisions with configurable probabilities
    if isinstance(base_value, dict) and base_value.get("type") == "random_probability":
        # Get probability from session state or use default
        prob_key = f"{decision_name}_probability_y"
        probability_y = st.session_state.get(prob_key, base_value.get("probability_y", 0.5))
        
        options = base_value.get("options", ["Y", "N"])
        
        # Weighted random choice
        if random.random() < probability_y:
            return options[0]  # First option (Y or purchase)
        else:
            return options[1]  # Second option (N or bid)
    
    # Handle rejected_transaction_defaults based on simulation execution mode
    elif decision_name == "rejected_transaction_defaults":
        # Check simulation execution mode from session state
        if hasattr(st.session_state, 'sim_params') and hasattr(st.session_state.sim_params, 'simulation_execution_mode'):
            if st.session_state.sim_params.simulation_execution_mode == "snapshot":
                # Snapshot mode: assume Option 5 (forgo transaction) for all
                return "forgo_transaction"
            else:
                # Live mode: users will be asked in real time
                return "REAL_TIME_DECISION"
        else:
            # Default to snapshot behavior if mode is unknown
            return "forgo_transaction"
    
    # Handle random within consumption limit
    elif base_value == "RANDOM_WITHIN_LIMIT":
        # This needs to be handled per agent based on their income category
        # Return a placeholder that the simulation will interpret
        return "RANDOM_WITHIN_LIMIT"
    
    # Handle calculated consumption frequency
    elif base_value == "CALCULATED":
        # This will be calculated based on consumption quantity / period duration
        # Return a placeholder that the simulation will interpret
        return "CALCULATED"
    
    # Handle random bid value within range
    elif base_value == "RANDOM_WITHIN_RANGE":
        # This needs market price and bidding range from sim_params
        # Return a placeholder that the simulation will interpret
        return "RANDOM_WITHIN_RANGE"
    
    # For all other values (numbers, dictionaries, strings), return as-is
    else:
        return base_value


def run_individual_decision(decision_name):
    """Run a single decision simulation"""
    with st.spinner(f"Running {decision_name} simulation..."):
        try:
            # Clear any selected configuration for donation_default to allow full comparison
            if decision_name == "donation_default" and hasattr(st.session_state, 'selected_donation_config'):
                st.info("🔄 Clearing selected configuration to show all comparison variants")
                delattr(st.session_state, 'selected_donation_config')
                
                # Also restore original session state values if they were overridden
                if hasattr(st.session_state, '_original_population_mode'):
                    st.session_state.population_mode = st.session_state._original_population_mode
                    st.session_state.income_spec_mode = st.session_state._original_income_spec_mode
                    delattr(st.session_state, '_original_population_mode')
                    delattr(st.session_state, '_original_income_spec_mode')
                    st.info("🔄 Restored original UI settings for comparison")
            
            # Temporarily modify selected decisions
            original_decisions = st.session_state.decision_params.selected_decisions.copy()
            st.session_state.decision_params.selected_decisions = [decision_name]
            
            # If this is donation_default, collect and apply coefficient parameters
            if decision_name == "donation_default":
                # Collect regression coefficients from session state
                coeffs = {
                    'intercept': st.session_state.get('donation_coeff_intercept', 1.22985660120368),
                    'beta_group': {
                        'MidSub': st.session_state.get('donation_coeff_midsub', 0.856140306694656),
                        'NoSub': st.session_state.get('donation_coeff_nosub', -0.926633374153906),
                        'FullSub': st.session_state.get('donation_coeff_fullsub', 0.0)
                    },
                    'beta_income_q': {
                        'Q1': st.session_state.get('donation_coeff_q1', -0.520290427509808),
                        'Q2': st.session_state.get('donation_coeff_q2', 3.754612744416796),
                        'Q3': st.session_state.get('donation_coeff_q3', 4.001714810873598),
                        'Q4_Q5': st.session_state.get('donation_coeff_q45', 0.0)
                    },
                    'beta_income_linear': st.session_state.get('donation_coeff_linear', 0.0256),
                    'beta_study': {
                        'Incoming': st.session_state.get('donation_coeff_incoming', -6.920193024391676),
                        'Law5yr': st.session_state.get('donation_coeff_law', -2.081331674770856),
                        'UG3yr': st.session_state.get('donation_coeff_ug', -2.139093511519692),
                        'Grad2yr': st.session_state.get('donation_coeff_grad', 0.0)
                    },
                    'beta_hh': st.session_state.get('donation_coeff_hh', 0.634001208840808),
                    'income_mode': st.session_state.get('income_spec_mode', 'categorical')
                }
                
                # Store the coefficients in decision_params for the simulation
                if not hasattr(st.session_state, 'custom_coefficients'):
                    st.session_state.custom_coefficients = {}
                st.session_state.custom_coefficients['donation_default'] = coeffs
            
            # Set state variables correctly for individual runs
            # This ensures the results display shows only the executed decision
            st.session_state.custom_decisions = [decision_name]  # Only this decision was run with custom parameters
            st.session_state.default_decisions = []  # No decisions used default values (since only one was run)
            
            # Run simulation
            run_simulation_from_sidebar()
            
            # Store in individual results
            if st.session_state.simulation_results:
                if 'individual_results' not in st.session_state:
                    st.session_state.individual_results = {}
                
                st.session_state.individual_results[decision_name] = st.session_state.simulation_results
                st.success(f"✅ {decision_name} simulation complete!")
                
                # Show preview of results
                results = next(iter(st.session_state.simulation_results.values()))
                if results is not None and not results.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Agents Simulated", f"{len(results):,}")
                    with col2:
                        if decision_name == "donation_default":
                            donation_col = 'donation_default_raw' if 'donation_default_raw' in results.columns else 'donation_default'
                            if donation_col in results.columns:
                                st.metric("Average Donation Rate", f"{results[donation_col].mean():.1%}")
            
            # Restore original decisions
            st.session_state.decision_params.selected_decisions = original_decisions
            
        except Exception as e:
            st.error(f"❌ Error running {decision_name}: {str(e)}")
            import traceback
            st.text(traceback.format_exc())


def run_combined_simulation(selected_decisions):
    """Run complete simulation with selected decisions using custom parameters and unselected decisions using defaults"""
    
    # Store information about selected vs default decisions
    unselected_decisions = [d for d in ALL_DECISIONS if d not in selected_decisions]
    
    with st.spinner(f"Running complete simulation: {len(selected_decisions)} custom + {len(unselected_decisions)} default decisions..."):
        try:
            # Store original selected decisions
            original_decisions = st.session_state.decision_params.selected_decisions.copy()
            
            # Set to run ALL decisions (this ensures complete simulation)
            st.session_state.decision_params.selected_decisions = ALL_DECISIONS
            
            # Store metadata about which decisions use custom vs default parameters
            st.session_state.custom_decisions = selected_decisions
            st.session_state.default_decisions = unselected_decisions
            
            # Run simulation with all decisions
            run_simulation_from_sidebar()
            
            # Restore original selected decisions
            st.session_state.decision_params.selected_decisions = original_decisions
            
            # Show completion message
            if st.session_state.simulation_results:
                st.success(f"✅ Complete simulation finished!")
                st.info(f"📊 **{len(selected_decisions)} decisions** used your custom parameters")
                st.info(f"🔧 **{len(unselected_decisions)} decisions** used default values")
                
                # Show preview
                results = next(iter(st.session_state.simulation_results.values()))
                if results is not None and not results.empty:
                    st.metric("Total Agents Simulated", f"{len(results):,}")
                    
        except Exception as e:
            st.error(f"❌ Error running complete simulation: {str(e)}")
            import traceback
            st.text(traceback.format_exc())


# ==================== CONFIGURATION SELECTION SYSTEM ====================

def save_selected_configuration(result_key, result_df):
    """Save the selected configuration for later use in combined simulations"""
    
    # Extract configuration details from the result key
    config_details = extract_configuration_details(result_key)
    
    # Get current coefficient values from session state
    coefficients = get_current_coefficients()
    
    # Get current stochastic parameters
    stochastic_params = get_current_stochastic_params()
    
    # Calculate key metrics from the result
    metrics = calculate_result_metrics(result_df)
    
    # Create complete configuration object
    config = {
        'result_key': result_key,
        'population_mode': config_details['population_mode'],
        'income_spec_mode': config_details['income_spec_mode'],
        'coefficients': coefficients,
        'stochastic_params': stochastic_params,
        'metrics': metrics,
        'selected_timestamp': datetime.now(),
        'total_agents': len(result_df),
        'source': 'individual_donation_run'
    }
    
    # Store in session state
    st.session_state.selected_donation_config = config
    
    return config


def extract_configuration_details(result_key):
    """Extract population and income mode from result key"""
    
    # Population mode detection
    if 'copula' in result_key:
        population_mode = 'Copula (synthetic)'
    elif 'research_spec' in result_key or 'documentation' in result_key:
        population_mode = 'Research Specification'
    elif 'baseline' in result_key:
        population_mode = 'Research Baseline'
    else:
        # For single-mode results, use current session state
        population_mode = st.session_state.get('population_mode', 'Copula (synthetic)')
    
    # Income mode detection
    if 'categorical' in result_key:
        income_spec_mode = 'categorical only'
    elif 'continuous' in result_key:
        income_spec_mode = 'continuous only'
    else:
        # For single-mode results, use current session state
        income_spec_mode = st.session_state.get('income_spec_mode', 'categorical only')
    
    return {
        'population_mode': population_mode,
        'income_spec_mode': income_spec_mode
    }


def get_current_coefficients():
    """Collect all current coefficient values from session state"""
    return {
        'intercept': st.session_state.get('donation_coeff_intercept', 1.22985660120368),
        'beta_group': {
            'MidSub': st.session_state.get('donation_coeff_midsub', 0.856140306694656),
            'NoSub': st.session_state.get('donation_coeff_nosub', -0.926633374153906),
            'FullSub': st.session_state.get('donation_coeff_fullsub', 0.0)
        },
        'beta_income_q': {
            'Q1': st.session_state.get('donation_coeff_q1', -0.520290427509808),
            'Q2': st.session_state.get('donation_coeff_q2', 3.754612744416796),
            'Q3': st.session_state.get('donation_coeff_q3', 4.001714810873598),
            'Q4_Q5': st.session_state.get('donation_coeff_q45', 0.0)
        },
        'beta_income_linear': st.session_state.get('donation_coeff_linear', 0.0256),
        'beta_study': {
            'Incoming': st.session_state.get('donation_coeff_incoming', -6.920193024391676),
            'Law5yr': st.session_state.get('donation_coeff_law', -2.081331674770856),
            'UG3yr': st.session_state.get('donation_coeff_ug', -2.139093511519692),
            'Grad2yr': st.session_state.get('donation_coeff_grad', 0.0)
        },
        'beta_hh': st.session_state.get('donation_coeff_hh', 0.634001208840808)
    }


def get_current_stochastic_params():
    """Collect current stochastic parameters from session state"""
    return {
        'stochastic': {
            'sigma_value': st.session_state.get('sigma_value_ui', 9.8995),
            'sigma_coefficient': st.session_state.get('sigma_coefficient', 1.0),
            'sigma_in_copula': st.session_state.get('sigma_in_copula', False),
            'sigma_in_research': st.session_state.get('sigma_in_research', True),
            'raw_output': st.session_state.get('raw_draw_mode', False)
        },
        'anchor_weights': {
            'observed': st.session_state.get('anchor_observed_weight', 0.75),
            'predicted': 1 - st.session_state.get('anchor_observed_weight', 0.75)
        }
    }


def calculate_result_metrics(result_df):
    """Calculate key metrics from result DataFrame"""
    
    # Determine which donation column to use
    donation_col = 'donation_default'
    if 'donation_default_raw' in result_df.columns and st.session_state.get('raw_draw_mode', False):
        donation_col = 'donation_default_raw'
    
    metrics = {
        'mean_donation': result_df[donation_col].mean(),
        'std_donation': result_df[donation_col].std(),
        'median_donation': result_df[donation_col].median(),
        'min_donation': result_df[donation_col].min(),
        'max_donation': result_df[donation_col].max(),
        'q25_donation': result_df[donation_col].quantile(0.25),
        'q75_donation': result_df[donation_col].quantile(0.75),
        'donation_column_used': donation_col
    }
    
    return metrics


def format_result_name(result_key):
    """Format result key into human-readable name"""
    
    name_mapping = {
        'copula_categorical': '🔬 Copula + Categorical Income',
        'copula_continuous': '🔬 Copula + Continuous Income',
        'research_spec_categorical': '📊 Research Spec + Categorical Income',
        'research_spec_continuous': '📊 Research Spec + Continuous Income',
        'research_baseline_categorical': '📈 Research Baseline + Categorical Income',
        'research_baseline_continuous': '📈 Research Baseline + Continuous Income',
        'categorical': '💰 Categorical Income Mode',
        'continuous': '📈 Continuous Income Mode',
        'copula': '🔬 Copula Population',
        'documentation': '📊 Research Specification',
        'baseline': '📈 Research Baseline'
    }
    
    return name_mapping.get(result_key, f"📊 {result_key.replace('_', ' ').title()}")


def is_configuration_selected(result_key):
    """Check if a specific configuration is currently selected"""
    
    if not hasattr(st.session_state, 'selected_donation_config'):
        return False
    
    return st.session_state.selected_donation_config.get('result_key') == result_key


def clear_selected_configuration():
    """Clear the currently selected configuration"""
    
    if hasattr(st.session_state, 'selected_donation_config'):
        del st.session_state.selected_donation_config
