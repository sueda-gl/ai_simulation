# app/pages/decision_execution.py
"""
Decision execution functions for running individual and combined simulations.
"""
import streamlit as st
from app.simulation import run_simulation_from_sidebar
from app.models import ALL_DECISIONS

# Default values for unselected decisions
DEFAULT_DECISION_VALUES = {
    "donation_default": 0.10,  # 10%
    "disclose_income": "RANDOM_Y_N",  # Will be randomly chosen as Y or N
    "disclose_documents": "RANDOM_Y_N_THRESHOLD",  # Random Y/N for those qualified
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
    "purchase_vs_bid": "RANDOM_CHOICE",  # Random purchase or bid
    "bid_value": "RANDOM_WITHIN_RANGE",  # Random within bidding price range
    "rejected_transaction_option": "forgo_transaction",  # Option 5 (forgo transaction)
    "rejected_bid_value": "NA",  # Not relevant given Option 5
    "final_donation_rate": 0.10  # Keep default 10%
}

# Description text for display purposes
DEFAULT_DECISION_DESCRIPTIONS = {
    "donation_default": "10%",
    "disclose_income": "random Y/N", 
    "disclose_documents": "random Y/N for those qualified (income below threshold), while granting the discount to all those who submitted document",
    "rejected_transaction_defaults": "Snapshot mode: forgo all transactions. Live mode: Users will be asked in real time",
    "vendor_choice_weights": "equal weight of 25% to Price, Quality, Proximity, and Sustainability",
    "consumption_quantity": "random within consumption limit",
    "consumption_frequency": "Consumption quantity divided by Period duration",
    "vendor_selection": "deterministic based on vendor choice weights",
    "purchase_vs_bid": "random choice",
    "bid_value": "random within bidding price range",
    "rejected_transaction_option": "forgo transaction",
    "rejected_bid_value": "Not relevant given choice of Option 5",
    "final_donation_rate": "keep default 10%"
}


def get_actual_default_value(decision_name, sim_params=None):
    """
    Get the actual default value for a decision, handling random generation where needed.
    This function returns values that can be used directly by the simulation.
    """
    import random
    import streamlit as st
    
    base_value = DEFAULT_DECISION_VALUES.get(decision_name)
    
    # Handle rejected_transaction_defaults based on simulation execution mode
    if decision_name == "rejected_transaction_defaults":
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
    
    # Handle random Y/N decisions
    elif base_value == "RANDOM_Y_N":
        return random.choice(["Y", "N"])
    
    # Handle random Y/N with threshold check (for disclose_documents)
    elif base_value == "RANDOM_Y_N_THRESHOLD":
        # This would need agent-specific income to properly implement
        # For now, return random Y/N
        return random.choice(["Y", "N"])
    
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
    
    # Handle random purchase vs bid choice
    elif base_value == "RANDOM_CHOICE":
        return random.choice(["purchase", "bid"])
    
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
            # Temporarily modify selected decisions
            original_decisions = st.session_state.decision_params.selected_decisions.copy()
            st.session_state.decision_params.selected_decisions = [decision_name]
            
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
