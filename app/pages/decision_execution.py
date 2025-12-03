# app/pages/decision_execution.py
"""
Decision execution functions for running individual and combined simulations.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from app.simulation import run_simulation_from_sidebar
from app.models import ALL_DECISIONS


def format_decision_title(decision_name, include_number=False):
    """Format decision name for display, with special handling for specific decisions
    
    Args:
        decision_name: The decision name to format
        include_number: If True, prepend decision number (e.g., "1. Decision Name")
    
    Returns:
        Formatted decision title string
    """
    from app.models import ALL_DECISIONS
    
    # Get decision number from ALL_DECISIONS
    decision_number = None
    if include_number and decision_name in ALL_DECISIONS:
        decision_number = ALL_DECISIONS.index(decision_name) + 1
    
    # Format the title
    if decision_name == "purchase_vs_bid":
        title = "Purchase Now Vs Bid"
    elif decision_name == "purchasing_quantity":
        title = "Purchase Request Quantity"
    elif decision_name == "purchasing_frequency":
        title = "Purchase Request Frequency"
    else:
        title = decision_name.replace('_', ' ').title()
    
    # Add number prefix if requested
    if decision_number is not None:
        return f"{decision_number}. {title}"
    return title


def can_run_complete_simulation():
    """
    Determine if complete simulation can run based on configuration state.
    
    This prevents running all decisions when multiple donation configurations would be generated,
    unless the user has explicitly selected one configuration to use.
    
    Returns:
        tuple: (can_run: bool, reason: str, config_count: int)
            - can_run: Whether complete simulation is allowed
            - reason: Human-readable explanation
            - config_count: Number of configurations that would be generated
    """
    # Check if multiple configurations will be generated for donation_default
    population_mode = st.session_state.get('population_mode', 'Copula (synthetic)')
    income_spec_mode = st.session_state.get('income_spec_mode', 'categorical only')
    
    # Count how many configurations will be generated
    # Population modes: "Compare all" generates 3, others generate 1
    population_count = 3 if population_mode == "Compare all" else 1
    
    # Income modes: "Compare both" generates 2, others generate 1
    income_count = 2 if income_spec_mode == "Compare both" else 1
    
    total_configs = population_count * income_count
    
    # Case 1: Only one configuration - always allow
    if total_configs == 1:
        return (True, "Single configuration", 1)
    
    # Case 2: Multiple configurations - check if one is selected
    has_selected_config = hasattr(st.session_state, 'selected_donation_config')
    
    if has_selected_config:
        # User has selected a specific configuration - allow with that config
        config = st.session_state.selected_donation_config
        config_name = f"{config['population_mode']} + {config['income_spec_mode']}"
        return (True, f"Using selected configuration: {config_name}", total_configs)
    else:
        # Multiple configs but none selected - block complete simulation
        return (False, f"Multiple donation configurations ({total_configs}) detected - please select one first", total_configs)


def render_simulation_buttons(decision_name, selected_decisions):
    """
    Render both individual and complete simulation buttons for a decision tab.
    
    This provides a consistent interface across all decision tabs, allowing users to:
    1. Run only the current decision (for testing/validation)
    2. Run the complete simulation with all 13 decisions
    
    Args:
        decision_name: Name of the current decision (e.g., "donation_default")
        selected_decisions: List of all selected decisions from session state
    """
    st.markdown("---")
    st.markdown('<h3 class="section-header">🚀 Simulation Options</h3>', unsafe_allow_html=True)
    
    # Safety check: ensure selected_decisions is a list
    if selected_decisions is None or not isinstance(selected_decisions, list):
        selected_decisions = []
    
    # Calculate unselected decisions for informational purposes
    unselected_decisions = [d for d in ALL_DECISIONS if d not in selected_decisions]
    
    # Display context in two columns
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"**🔬 Test Run**\n\nTest only {format_decision_title(decision_name)} with current parameters")
        st.caption("Quick validation of this decision's configuration")
    with col_info2:
        st.info(f"**🎯 Complete Simulation**\n\nRun all {len(ALL_DECISIONS)} decisions end-to-end")
        if len(unselected_decisions) > 0:
            st.caption(f"{len(selected_decisions)} custom + {len(unselected_decisions)} defaults")
        else:
            st.caption("All decisions use custom parameters")
    
    # Render action buttons in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            f"🔬 Run {format_decision_title(decision_name)} Only", 
            type="primary", 
            use_container_width=True,
            key=f"run_{decision_name}_only_btn",
            help=f"Execute only {format_decision_title(decision_name)} to test and validate your parameters"
        ):
            run_individual_decision(decision_name)
    
    with col2:
        # Show detailed breakdown in expander
        with st.expander("📊 View Complete Simulation Configuration", expanded=False):
            st.markdown("**What will run in Complete Simulation:**")
            
            if len(selected_decisions) > 0:
                st.markdown(f"**✅ Custom Parameters ({len(selected_decisions)} decisions):**")
                for i, dec in enumerate(selected_decisions, 1):
                    icon = "🎯" if dec == decision_name else "✓"
                    label = " **(current tab)**" if dec == decision_name else ""
                    st.caption(f"{icon} {format_decision_title(dec, include_number=True)}{label}")
            
            if len(unselected_decisions) > 0:
                st.markdown(f"\n**🔧 Default Values ({len(unselected_decisions)} decisions):**")
                for i, dec in enumerate(unselected_decisions, 1):
                    st.caption(f"{format_decision_title(dec, include_number=True)}")
        
        # Check if complete simulation can run (validation for multiple configurations)
        can_run, reason, config_count = can_run_complete_simulation()
        
        if not can_run:
            # Disabled button with explanation
            st.button(
                "🎯 Run Complete Simulation", 
                type="primary",
                use_container_width=True,
                disabled=True,
                key=f"run_complete_from_{decision_name}_btn_disabled",
                help="Multiple configurations detected - select one first"
            )
            
            # Show helpful warning message
            st.warning(f"""
⚠️ **Complete Simulation Blocked**

{reason}

**To run all decisions:**

1. Click **🔬 Run {format_decision_title(decision_name)} Only** above
2. Review the {config_count} result configurations
3. **Select one configuration** from the results
4. Return here and click **Run Complete Simulation**

This ensures all decisions use consistent settings.
            """)
            
        else:
            # Enabled button - can proceed
            # Show info about selected config if applicable
            if config_count > 1 and hasattr(st.session_state, 'selected_donation_config'):
                config = st.session_state.selected_donation_config
                st.success(f"✅ Using: {config['population_mode']} + {config['income_spec_mode']}")
            
            if st.button(
                "🎯 Run Complete Simulation", 
                type="primary",
                use_container_width=True,
                key=f"run_complete_from_{decision_name}_btn",
                help=f"Run all {len(ALL_DECISIONS)} decisions with current configuration"
            ):
                # Show confirmation info before running
                with st.spinner("🔄 Preparing complete simulation..."):
                    st.info(f"""
                    **🚀 Starting complete end-to-end simulation**
                    
                    - Running all {len(ALL_DECISIONS)} decisions in sequence
                    - Current decision ({format_decision_title(decision_name)}) will use your configured parameters above
                    - {len(selected_decisions)} decisions total with custom parameters
                    - {len(unselected_decisions)} decisions using default values
                    """)
                    
                    # Execute the combined simulation
                    run_combined_simulation(selected_decisions)

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
        "description": "Probability of disclosing documents (applies only to agents qualified for discount: income < threshold)"
    },
    "rejected_transaction_defaults": {
        "type": "prioritized_selection",
        "priority_template": ["forgo_transaction"],  # Default: all agents use Option 5 only
        "options": [
            ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
            ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"), 
            ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"),
            ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
            ("forgo_transaction", "Option 5: Forgo the purchase request")
        ],
        "description": "Each agent gets a prioritized list. If Option 5 is included, it must be last."
    },
    "vendor_choice_weights": {
        "type": "checkbox_selection",
        "default_selection": ["price", "quality", "proximity", "sustainability"],
        "parameters": {
            "price": {"name": "Price", "description": "the product price offered to the customer"},
            "quality": {"name": "Quality", "description": "product quality based on customer ratings"},
            "proximity": {"name": "Proximity", "description": "the proximity of vendor to customer"},
            "sustainability": {"name": "Sustainability", "description": "vendor sustainability rating"}
        }
    },
    "purchasing_quantity": "RANDOM_WITHIN_LIMIT",  # Random within purchasing limit
    "purchasing_frequency": "CALCULATED",  # Consumption quantity / Period duration
    "vendor_selection": "deterministic",  # Deterministic based on highest weighted vendor-product score
    "purchase_vs_bid": {
        "type": "random_probability",
        "probability_y": 0.5,  # 50% chance of Purchase Now (vs bid)
        "options": ["Purchase Now", "bid"],
        "description": "Probability of Purchase Now vs bidding (applies only to REGULAR customers - those who did not disclose income)"
    },
    "bid_value": "RANDOM_WITHIN_RANGE",  # Random within bidding price range
    "rejected_transaction_option": {
        "type": "radio_selection",
        "default_option": "forgo_transaction", 
        "options": [
            ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
            ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"),
            ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"), 
            ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
            ("forgo_transaction", "Option 5: Forgo the purchase request")
        ]
    },
    "rejected_bid_value": "NA",  # Not relevant given Option 5
    "final_donation_rate": 0.10  # Keep default 10%
}

# Description text for display purposes
DEFAULT_DECISION_DESCRIPTIONS = {
    "donation_default": "10%",
    "disclose_income": "configurable probability Y/N (default 50% each)", 
    "disclose_documents": "configurable probability Y/N (applies only to agents with income < discount threshold, default 50% each)",
    "rejected_transaction_defaults": "Selected option for handling rejected transactions will be applied to all agents",
    "vendor_choice_weights": "equal weight distribution among selected parameters (Price, Quality, Proximity, Sustainability)",
    "purchasing_quantity": "random within purchasing limit",
    "purchasing_frequency": "Consumption quantity divided by Period duration",
    "vendor_selection": "deterministic based on highest weighted vendor-product score",
    "purchase_vs_bid": "configurable probability Purchase Now/bid for REGULAR customers only (default 50% each)",
    "bid_value": "random within bidding price range (only for REGULAR customers who chose to bid)",
    "rejected_transaction_option": "Selected specific option for transaction rejection handling will be used",
    "rejected_bid_value": "Default handling for rejected bid values will be applied",
    "final_donation_rate": "Default donation rate will be maintained"
}


def get_actual_default_value(decision_name, sim_params=None):
    """
    Get the actual default value for a decision, handling random generation where needed.
    This function returns values that can be used directly by the simulation.
    
    Priority order:
    1. Pre-configured default from Page 2 Overview tab ({decision_name}_default_*)
    2. Post-simulation adjustment from Results page ({decision_name}_*)
    3. Hard-coded default from DEFAULT_DECISION_VALUES
    """
    import random
    import streamlit as st
    
    base_value = DEFAULT_DECISION_VALUES.get(decision_name)
    
    # NEW: Handle parametric random decisions with configurable probabilities
    if isinstance(base_value, dict) and base_value.get("type") == "random_probability":
        # Priority 1: Check for pre-configured default from Overview tab
        pre_config_key = f"{decision_name}_default_probability_y"
        # Priority 2: Check for post-simulation adjustment from Results page
        post_sim_key = f"{decision_name}_probability_y"
        # Priority 3: Use hard-coded default
        
        probability_y = st.session_state.get(
            pre_config_key, 
            st.session_state.get(
                post_sim_key, 
                base_value.get("probability_y", 0.5)
            )
        )
        
        options = base_value.get("options", ["Y", "N"])
        
        # Weighted random choice
        if random.random() < probability_y:
            return options[0]  # First option (Y or purchase)
        else:
            return options[1]  # Second option (N or bid)
    
    # Handle prioritized selection decisions (rejected_transaction_defaults)
    elif isinstance(base_value, dict) and base_value.get("type") == "prioritized_selection":
        # Priority 1: Check for configured priority template from Overview tab
        pre_config_key = f"{decision_name}_priority_template"
        
        # Priority 2: Use hard-coded default template
        priority_template = st.session_state.get(
            pre_config_key,
            base_value.get("priority_template", ["forgo_transaction"])
        )
        return priority_template
    
    # Handle radio selection decisions (rejected transaction options)
    elif isinstance(base_value, dict) and base_value.get("type") == "radio_selection":
        # Priority 1: Check for pre-configured default from Overview tab
        pre_config_key = f"{decision_name}_default_selection"
        
        # Priority 2: Check for post-simulation adjustments (legacy keys)
        if decision_name == "rejected_transaction_defaults":
            post_sim_key = "rejected_transaction_defaults_option"
        elif decision_name == "rejected_transaction_option":
            post_sim_key = "rejected_transaction_option_selection"
        else:
            post_sim_key = f"{decision_name}_selection"
        
        # Priority 3: Use hard-coded default
        selected_value = st.session_state.get(
            pre_config_key,
            st.session_state.get(
                post_sim_key,
                base_value.get("default_option", "forgo_transaction")
            )
        )
        return selected_value
    
    # Handle checkbox selection decisions (vendor choice weights)
    elif isinstance(base_value, dict) and base_value.get("type") == "checkbox_selection":
        # Priority 1: Check for pre-configured default from Overview tab
        pre_config_key = f"{decision_name}_default_params"
        # Priority 2: Check for post-simulation adjustment
        post_sim_key = "vendor_choice_weights_selection"
        
        selected_params = st.session_state.get(
            pre_config_key,
            st.session_state.get(
                post_sim_key,
                base_value.get("default_selection", [])
            )
        )
        
        # Calculate equal weights for selected parameters
        if len(selected_params) > 0:
            weight_per_param = 1.0 / len(selected_params)
            weights = {}
            
            # Set weights for all parameters
            for param_key in base_value.get("parameters", {}).keys():
                if param_key in selected_params:
                    weights[param_key] = weight_per_param
                else:
                    weights[param_key] = 0.0
            
            return weights
        else:
            # Fallback to equal weights if nothing selected
            params = list(base_value.get("parameters", {}).keys())
            if params:
                weight_per_param = 1.0 / len(params)
                return {param: weight_per_param for param in params}
            else:
                return {"price": 0.25, "quality": 0.25, "proximity": 0.25, "sustainability": 0.25}
    
    # Handle numeric defaults (donation_default, final_donation_rate, etc.)
    elif isinstance(base_value, (int, float)):
        # Priority 1: Check for pre-configured default from Overview tab
        pre_config_key = f"{decision_name}_default_value"
        # Priority 2: Check for post-simulation adjustment (specific keys)
        post_sim_key = f"{decision_name}_config"
        
        # Priority 3: Use hard-coded default
        return st.session_state.get(
            pre_config_key,
            st.session_state.get(
                post_sim_key,
                base_value
            )
        )
    
    # Handle random within purchasing limit
    elif base_value == "RANDOM_WITHIN_LIMIT":
        # This needs to be handled per agent based on their income category
        # Return a placeholder that the simulation will interpret
        return "RANDOM_WITHIN_LIMIT"
    
    # Handle calculated purchasing frequency
    elif base_value == "CALCULATED":
        # This will be calculated based on purchasing quantity / period duration
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
            # Store original state values to restore later (without modifying session state immediately)
            original_decisions = st.session_state.decision_params.selected_decisions.copy()
            original_custom_decisions = getattr(st.session_state, 'custom_decisions', [])
            original_default_decisions = getattr(st.session_state, 'default_decisions', [])
            
            # Clear any selected configuration for donation_default to allow full comparison
            had_selected_config = False
            original_population_mode = None
            original_income_spec_mode = None
            
            if decision_name == "donation_default" and hasattr(st.session_state, 'selected_donation_config'):
                st.info("🔄 Clearing selected configuration to show all comparison variants")
                had_selected_config = True
                delattr(st.session_state, 'selected_donation_config')
                
                # Store original values without immediate state modification
                if hasattr(st.session_state, '_original_population_mode'):
                    original_population_mode = st.session_state.population_mode
                    original_income_spec_mode = st.session_state.income_spec_mode
                    st.session_state.population_mode = st.session_state._original_population_mode
                    st.session_state.income_spec_mode = st.session_state._original_income_spec_mode
                    delattr(st.session_state, '_original_population_mode')
                    delattr(st.session_state, '_original_income_spec_mode')
                    st.info("🔄 Restored original UI settings for comparison")
            
            # Modify selected decisions for simulation
            st.session_state.decision_params.selected_decisions = [decision_name]
            
            # If this is donation_default, collect and apply coefficient parameters
            if decision_name == "donation_default":
                # Collect regression coefficients from YAML-loaded session state
                coeffs = get_current_coefficients()
                coeffs['income_mode'] = st.session_state.get('income_spec_mode', 'categorical')
                
                # Store the coefficients in decision_params for the simulation
                if not hasattr(st.session_state, 'custom_coefficients'):
                    st.session_state.custom_coefficients = {}
                st.session_state.custom_coefficients['donation_default'] = coeffs
            
            # Set state variables correctly for individual runs
            # This ensures the results display shows only the executed decision
            st.session_state.custom_decisions = [decision_name]  # Only this decision was run with custom parameters
            st.session_state.default_decisions = []  # No decisions used default values (since only one was run)
            
            # Run simulation - check if Monte Carlo or Single Run mode
            if st.session_state.sim_params.simulation_mode == "Monte-Carlo Study":
                # Run Monte Carlo study
                from app.simulation import run_monte_carlo_study
                mc_summary, mc_detailed, output_log = run_monte_carlo_study()
                if mc_summary is not None:
                    st.session_state.mc_results = {
                        'summary': mc_summary,
                        'detailed': mc_detailed,
                        'log': output_log
                    }
                    st.session_state.simulation_results = None
                    st.success(f"✅ Monte Carlo study for {decision_name} complete!")
                    
                    # Show Monte Carlo preview
                    if 'donation_default' in mc_summary['decision'].values:
                        donation_row = mc_summary[mc_summary['decision'] == 'donation_default'].iloc[0]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Donation Rate", f"{donation_row['mean']:.2%}")
                        with col2:
                            st.metric("Std Deviation", f"{donation_row['std']:.2%}")
                        with col3:
                            st.metric("Number of Runs", int(donation_row['runs']))
                    
                    # Navigate to results page
                    st.session_state.page = 'results'
                    st.info("🔄 Redirecting to Results page to view full Monte Carlo analysis...")
                    st.rerun()
                else:
                    st.error("❌ Monte Carlo simulation returned no results")
            else:
                # Run single simulation
                run_simulation_from_sidebar()
            
            # Store in individual results (only for single run mode)
            if st.session_state.sim_params.simulation_mode != "Monte-Carlo Study" and st.session_state.simulation_results:
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
                            if 'donation_default' in results.columns:
                                st.metric("Average Donation Rate", f"{results['donation_default'].mean():.2%}")
            
            # Restore all original state in one operation to minimize reruns
            st.session_state.decision_params.selected_decisions = original_decisions
            st.session_state.custom_decisions = original_custom_decisions
            st.session_state.default_decisions = original_default_decisions
            
        except Exception as e:
            st.error(f"❌ Error running {decision_name}: {str(e)}")
            import traceback
            st.text(traceback.format_exc())


def run_combined_simulation(selected_decisions):
    """Run complete simulation with selected decisions using custom parameters and unselected decisions using defaults"""
    
    # Store information about selected vs default decisions
    unselected_decisions = [d for d in ALL_DECISIONS if d not in selected_decisions]
    
    # Check if using a selected configuration (for multiple config scenarios)
    using_selected_config = hasattr(st.session_state, 'selected_donation_config')
    original_population_mode = None
    original_income_spec = None
    
    if using_selected_config:
        # Store original values to restore later
        original_population_mode = st.session_state.population_mode
        original_income_spec = st.session_state.income_spec_mode
        
        # Apply selected configuration
        config = st.session_state.selected_donation_config
        st.session_state.population_mode = config['population_mode']
        st.session_state.income_spec_mode = config['income_spec_mode']
        
        # Mark that we're using selected config (for results display)
        st.session_state._using_selected_config = True
        
        st.info(f"🎯 **Using selected configuration**: {config['population_mode']} + {config['income_spec_mode']}")
    
    # Create appropriate spinner message
    if len(selected_decisions) == 0:
        spinner_msg = f"Running complete simulation: All {len(ALL_DECISIONS)} decisions with default values..."
    elif len(unselected_decisions) == 0:
        spinner_msg = f"Running complete simulation: All {len(selected_decisions)} decisions with custom parameters..."
    else:
        spinner_msg = f"Running complete simulation: {len(selected_decisions)} custom + {len(unselected_decisions)} default decisions..."
    
    with st.spinner(spinner_msg):
        try:
            # Store original selected decisions
            original_decisions = st.session_state.decision_params.selected_decisions.copy()
            
            # Set to run ALL decisions (this ensures complete simulation)
            st.session_state.decision_params.selected_decisions = ALL_DECISIONS
            
            # Store metadata about which decisions use custom vs default parameters
            st.session_state.custom_decisions = selected_decisions
            st.session_state.default_decisions = unselected_decisions
            
            # Run simulation with all decisions - check if Monte Carlo or Single Run mode
            if st.session_state.sim_params.simulation_mode == "Monte-Carlo Study":
                # Run Monte Carlo study
                from app.simulation import run_monte_carlo_study
                mc_summary, mc_detailed, output_log = run_monte_carlo_study()
                if mc_summary is not None:
                    st.session_state.mc_results = {
                        'summary': mc_summary,
                        'detailed': mc_detailed,
                        'log': output_log
                    }
                    st.session_state.simulation_results = None
                    st.success(f"✅ Monte Carlo complete simulation finished!")
                    
                    # Show Monte Carlo summary
                    st.info(f"📊 Completed {st.session_state.n_runs} Monte Carlo runs with {len(ALL_DECISIONS)} decisions")
                    
                    # Navigate to results page
                    st.session_state.page = 'results'
                    st.info("🔄 Redirecting to Results page to view full Monte Carlo analysis...")
                    
                    # Restore state before rerun
                    st.session_state.decision_params.selected_decisions = original_decisions
                    if using_selected_config and original_population_mode is not None:
                        st.session_state.population_mode = original_population_mode
                        st.session_state.income_spec_mode = original_income_spec
                    
                    st.rerun()
                else:
                    st.error("❌ Monte Carlo simulation returned no results")
            else:
                # Run single simulation
                run_simulation_from_sidebar()
            
            # Restore original selected decisions
            st.session_state.decision_params.selected_decisions = original_decisions
            
            # Restore original population/income modes if we changed them
            if using_selected_config and original_population_mode is not None:
                st.session_state.population_mode = original_population_mode
                st.session_state.income_spec_mode = original_income_spec
            
            # Show completion message
            if st.session_state.simulation_results:
                st.success(f"✅ Complete simulation finished!")
                
                # Provide clear messaging based on configuration
                if len(selected_decisions) == 0:
                    st.info(f"🔧 **All {len(ALL_DECISIONS)} decisions** used default values")
                elif len(unselected_decisions) == 0:
                    st.info(f"📊 **All {len(selected_decisions)} decisions** used your custom parameters")
                else:
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
            
            # Restore original modes even on error
            if using_selected_config and original_population_mode is not None:
                st.session_state.population_mode = original_population_mode
                st.session_state.income_spec_mode = original_income_spec


# ==================== CONFIGURATION SELECTION SYSTEM ====================

def save_selected_configuration(result_key, result_df):
    """Save the selected configuration for later use in combined simulations
    
    IMPORTANT: When saving a configuration, we must use the coefficients that match
    the income mode of the selected result (not the current session state mode).
    This is critical in "Compare both" mode where multiple results exist with different
    coefficient sets.
    """
    
    # Extract configuration details from the result key
    config_details = extract_configuration_details(result_key)
    
    # Get coefficient values that match the income mode of the selected result
    # This ensures we save the RIGHT coefficients for the selected configuration
    coefficients = get_current_coefficients(income_mode=config_details['income_spec_mode'])
    
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


def get_current_coefficients(income_mode=None):
    """Collect all current coefficient values from session state
    
    IMPORTANT: Ensures coefficients are loaded from YAML first.
    YAML is the SINGLE source of truth - no fallback values.
    
    Args:
        income_mode: Optional income mode string (e.g., 'categorical only', 'continuous only').
                     If provided, uses mode-specific coefficients. Otherwise uses main session state.
    """
    # Ensure coefficients are loaded from YAML
    from app.models import load_donation_coefficients_from_yaml
    if 'donation_coeff_intercept' not in st.session_state:
        load_donation_coefficients_from_yaml()
    
    # Determine which coefficient set to use based on income mode
    # Mode-specific suffixes: _cat for categorical, _cont for continuous
    if income_mode and 'continuous' in income_mode.lower():
        suffix = '_cont'
    elif income_mode and 'categorical' in income_mode.lower():
        suffix = '_cat'
    else:
        suffix = None  # Use main session state variables
    
    # Helper to get coefficient with optional suffix
    def get_coeff(name, default=None):
        if suffix:
            suffixed_key = f'donation_coeff_{name}{suffix}'
            if suffixed_key in st.session_state:
                return st.session_state[suffixed_key]
        # Fallback to main session state variable
        main_key = f'donation_coeff_{name}'
        return st.session_state.get(main_key, default)
    
    # Return coefficients from session state - NO FALLBACK VALUES for critical ones
    return {
        'intercept': get_coeff('intercept'),
        'beta_group': {
            'MidSub': get_coeff('midsub'),
            'NoSub': get_coeff('nosub'),
            'FullSub': get_coeff('fullsub')
        },
        'beta_income_q': {
            'Q1': get_coeff('q1', 0.0),
            'Q2': get_coeff('q2', 0.0),
            'Q3': get_coeff('q3', 0.0),
            'Q4': get_coeff('q4', 0.0),
            'Q5': get_coeff('q5', st.session_state.get('donation_coeff_q45', 0.0))  # Support both Q5 and legacy Q4_Q5
        },
        'beta_income_linear': get_coeff('linear', 0.0),
        'beta_study': {
            'Incoming': get_coeff('incoming'),
            'Law5yr': get_coeff('law'),
            'UG3yr': get_coeff('ug'),
            'Grad2yr': get_coeff('grad')
        },
        'beta_hh': get_coeff('hh')
    }


def get_current_stochastic_params():
    """Collect current stochastic parameters from session state"""
    return {
        'stochastic': {
            'sigma_value': st.session_state.get('sigma_value_ui', 9.8995),
            'sigma_coefficient': st.session_state.get('sigma_coefficient', 1.0),
            'sigma_in_copula': st.session_state.get('sigma_in_copula', False),
            'sigma_in_research': st.session_state.get('sigma_in_research', True)
        },
        'anchor_weights': {
            'observed': st.session_state.get('anchor_observed_weight', 0.75),
            'predicted': 1 - st.session_state.get('anchor_observed_weight', 0.75)
        }
    }


def calculate_result_metrics(result_df):
    """Calculate key metrics from result DataFrame - always uses truncated donation_default"""
    
    # Always use truncated donation_default for consistency
    donation_col = 'donation_default'
    
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
