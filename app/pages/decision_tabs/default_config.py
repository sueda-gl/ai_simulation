# app/pages/decision_tabs/default_config.py
"""
UI components for configuring default decision parameters before simulation.
These parameters apply to decisions that are NOT selected for custom configuration.
"""
import streamlit as st
from app.pages.decision_execution import DEFAULT_DECISION_VALUES


def render_default_decisions_config(selected_decisions, all_decisions):
    """
    Render configuration UI for default (unselected) decisions.
    
    Args:
        selected_decisions: List of decisions selected for custom configuration
        all_decisions: List of all available decisions
    """
    # Get unselected decisions (these will use default values)
    unselected_decisions = [d for d in all_decisions if d not in selected_decisions]
    
    if not unselected_decisions:
        # All decisions are custom configured
        st.info("✅ **All decisions are selected for custom configuration** - No defaults needed")
        return
    
    st.markdown('<h3 class="section-header">🔧 Configure Default Decision Parameters</h3>', unsafe_allow_html=True)
    st.caption(f"Configure default behavior for {len(unselected_decisions)} unselected decision(s). These settings will be used in the complete simulation.")
    
    # Render each decision individually without grouping
    for decision in unselected_decisions:
        render_decision_default_config(decision)
        if decision != unselected_decisions[-1]:  # Don't add separator after last decision
            st.markdown("---")
    
    # Reset all defaults button
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("💡 All default parameters are pre-configured with sensible values. Adjust as needed.")
    with col2:
        if st.button("🔄 Reset All Defaults", help="Reset all default parameters to system defaults", key="reset_all_defaults"):
            reset_all_default_parameters(unselected_decisions)
            st.success("✅ All defaults reset to system values")
            st.rerun()


def render_decision_default_config(decision_name):
    """Render configuration UI for a specific default decision"""
    
    # Special handling for purchase_vs_bid to show "Purchase Now" instead of "Purchase"
    if decision_name == "purchase_vs_bid":
        decision_title = "Purchase Now Vs Bid"
    else:
        decision_title = decision_name.replace('_', ' ').title()
    
    default_value = DEFAULT_DECISION_VALUES.get(decision_name)
    
    st.markdown(f"**🎯 {decision_title}**")
    
    # Handle different types of default decisions
    if isinstance(default_value, dict) and default_value.get("type") == "random_probability":
        render_probability_default_config(decision_name, default_value)
    
    elif isinstance(default_value, dict) and default_value.get("type") == "radio_selection":
        render_radio_default_config(decision_name, default_value)
    
    elif isinstance(default_value, dict) and default_value.get("type") == "checkbox_selection":
        render_checkbox_default_config(decision_name, default_value)
    
    elif isinstance(default_value, (int, float)):
        render_numeric_default_config(decision_name, default_value)
    
    elif isinstance(default_value, str):
        render_placeholder_default_config(decision_name, default_value)
    
    else:
        st.caption(f"Default: {default_value}")


def render_probability_default_config(decision_name, default_value):
    """Render UI for probability-based default decisions (Y/N, Purchase Now/bid)"""
    
    options = default_value.get("options", ["Y", "N"])
    description = default_value.get("description", "Probability configuration")
    default_probability = default_value.get("probability_y", 0.5)
    
    # Session state key for this decision's probability
    prob_key = f"{decision_name}_default_probability_y"
    
    # Note: Initialization now happens at app startup in initialize_default_decision_parameters()
    # This ensures values persist even when widgets are conditionally rendered
    
    # Special handling for purchase_vs_bid - show it only applies to regular customers
    if decision_name == "purchase_vs_bid":
        st.caption("⚠️ **Note**: This decision only applies to **REGULAR customers** (those who did not disclose income)")
        st.caption(description)
        
        # Show customer type distribution if simulation results exist
        if 'simulation_results' in st.session_state and st.session_state.simulation_results:
            results_dict = st.session_state.simulation_results
            first_result = next(iter(results_dict.values()))
            
            if first_result is not None and not first_result.empty and 'customer_type' in first_result.columns:
                # Analyze customer types
                from src.decisions.income_utils import analyze_customer_types
                stats = analyze_customer_types(first_result)
                
                # Show distribution
                type_col1, type_col2, type_col3 = st.columns(3)
                with type_col1:
                    st.metric("Regular Customers", 
                             f"{stats['regular']['count']:,}", 
                             f"{stats['regular']['percentage']:.1f}%",
                             help="Only these customers make Purchase Now vs Bid choice")
                with type_col2:
                    st.metric("Fixed Customers", 
                             f"{stats['fixed']['count']:,}",
                             f"{stats['fixed']['percentage']:.1f}%",
                             help="Use fixed pricing only")
                with type_col3:
                    st.metric("Discount Customers", 
                             f"{stats['discount']['count']:,}",
                             f"{stats['discount']['percentage']:.1f}%",
                             help="Use discount pricing")
                
                st.caption(f"💡 The probability below applies to {stats['regular']['count']:,} regular customers ({stats['regular']['percentage']:.1f}% of total)")
    else:
        st.caption(description)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Customize label for purchase_vs_bid
        if decision_name == "purchase_vs_bid":
            slider_label = f"P({options[0]}) - Probability for REGULAR customers only"
            slider_help = f"Probability that REGULAR customers will choose {options[0]} vs {options[1]}"
        else:
            slider_label = f"P({options[0]}) - Probability of {options[0]}"
            slider_help = f"Probability that agents will choose {options[0]} vs {options[1]}"
        
        # Widget uses ONLY key parameter - Streamlit automatically syncs with session_state[prob_key]
        probability = st.slider(
            slider_label,
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get(prob_key, default_probability),
            step=0.01,
            help=slider_help,
            key=prob_key  # Streamlit manages value automatically via session state
        )
    
    with col2:
        st.metric("Ratio", f"{probability:.0%} : {1-probability:.0%}")
        st.caption(f"{options[0]} : {options[1]}")
    
    with col3:
        st.metric("Default", f"{default_probability:.0%}")
        if probability != default_probability:
            st.caption("⚙️ Modified")
        else:
            st.caption("✓ Default")


def render_radio_default_config(decision_name, default_value):
    """Render UI for radio selection default decisions"""
    
    options = default_value.get("options", [])
    default_option = default_value.get("default_option", "")
    
    # Session state key for this decision's selection
    selection_key = f"{decision_name}_default_selection"
    
    # Note: Initialization now happens at app startup in initialize_default_decision_parameters()
    # This ensures values persist even when widgets are conditionally rendered
    
    # Create option names mapping
    option_names = dict(options) if options else {}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Compute default index based on current session state or configured default option
        option_codes = [opt[0] for opt in options]
        default_selected = st.session_state.get(selection_key, default_option)
        try:
            default_index = option_codes.index(default_selected) if default_selected in option_codes else 0
        except Exception:
            default_index = 0
        
        # Widget uses BOTH an explicit index (first render) and key (subsequent sync)
        selected = st.radio(
            "Default Option",
            options=option_codes,
            format_func=lambda x: option_names.get(x, x),
            index=default_index,
            help="Choose the default option for this decision",
            key=selection_key  # Streamlit manages value automatically via session state
        )
    
    with col2:
        st.metric("Selected", option_names.get(selected, selected).split(":")[0] if ":" in option_names.get(selected, selected) else option_names.get(selected, selected)[:20])
        if selected != default_option:
            st.caption("⚙️ Modified")
        else:
            st.caption("✓ Default")


def render_checkbox_default_config(decision_name, default_value):
    """Render UI for checkbox selection default decisions (e.g., vendor_choice_weights)"""
    
    parameters = default_value.get("parameters", {})
    default_selection = default_value.get("default_selection", [])
    
    # Session state key for this decision's selection
    selection_key = f"{decision_name}_default_params"
    
    # Note: Initialization now happens at app startup in initialize_default_decision_parameters()
    # This ensures values persist even when widgets are conditionally rendered
    
    st.caption("Select which parameters should be included (equal weight distribution)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_params = []
        
        for param_key, param_info in parameters.items():
            checkbox_key = f"{decision_name}_default_param_{param_key}"
            
            # Create checkbox - it will use the value from session state automatically
            is_selected = st.checkbox(
                f"{param_info['name']} - {param_info['description']}",
                value=st.session_state.get(checkbox_key, param_key in default_selection),
                key=checkbox_key
            )
            
            if is_selected:
                selected_params.append(param_key)
        
        # Update main selection state to reflect current checkbox states
        st.session_state[selection_key] = selected_params
    
    with col2:
        if selected_params:
            weight_per_param = 1.0 / len(selected_params)
            st.metric("Parameters", f"{len(selected_params)}/{len(parameters)}")
            st.metric("Weight Each", f"{weight_per_param:.1%}")
            
            if set(selected_params) != set(default_selection):
                st.caption("⚙️ Modified")
            else:
                st.caption("✓ Default")
        else:
            st.warning("⚠️ No parameters selected")


def render_numeric_default_config(decision_name, default_value):
    """Render UI for numeric default decisions"""
    
    # Session state key for this decision's value
    value_key = f"{decision_name}_default_value"
    
    # Note: Initialization now happens at app startup in initialize_default_decision_parameters()
    # This ensures values persist even when widgets are conditionally rendered
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Determine if this is a percentage (between 0 and 1)
        if 0 <= default_value <= 1:
            # Widget uses ONLY key parameter - Streamlit automatically manages the value
            value = st.slider(
                "Default Value",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get(value_key, default_value),
                step=0.01,
                format="%.2f",
                help="Set the default value for this decision",
                key=value_key  # Streamlit manages value automatically via session state
            )
            st.caption(f"Percentage: {value:.1%}")
        else:
            # Widget uses ONLY key parameter - Streamlit automatically manages the value
            value = st.number_input(
                "Default Value",
                min_value=0.0,
                value=float(st.session_state.get(value_key, default_value)),
                step=0.1,
                help="Set the default value for this decision",
                key=value_key  # Streamlit manages value automatically via session state
            )
    
    with col2:
        st.metric("Current", f"{value:.2f}")
        if value != default_value:
            st.caption("⚙️ Modified")
        else:
            st.caption("✓ Default")


def render_placeholder_default_config(decision_name, default_value):
    """Render info for placeholder/computed default decisions"""
    
    descriptions = {
        "RANDOM_WITHIN_LIMIT": "Random value within consumption limit (computed per agent based on income category)",
        "CALCULATED": "Calculated as: Consumption quantity ÷ Period duration",
        "RANDOM_WITHIN_RANGE": "Random bid amount within bidding price range (computed based on market parameters)",
        "deterministic": "Deterministic selection based on highest weighted vendor-product score",
        "NA": "Not applicable - this decision is not relevant given other parameter choices"
    }
    
    description = descriptions.get(default_value, f"Default behavior: {default_value}")
    
    st.info(f"ℹ️ **Computed During Simulation**")
    st.caption(description)
    st.caption("💡 No pre-configuration needed - this value is automatically calculated only for agents who ended up placing a bid")


def reset_all_default_parameters(unselected_decisions):
    """Reset all default parameters to system defaults"""
    
    for decision_name in unselected_decisions:
        # Clear all session state keys for this decision
        keys_to_clear = [
            f"{decision_name}_default_probability_y",
            f"{decision_name}_default_selection",
            f"{decision_name}_default_params",
            f"{decision_name}_default_value"
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Clear checkbox states for vendor_choice_weights
        default_value = DEFAULT_DECISION_VALUES.get(decision_name)
        if isinstance(default_value, dict) and default_value.get("type") == "checkbox_selection":
            parameters = default_value.get("parameters", {})
            for param_key in parameters.keys():
                checkbox_key = f"{decision_name}_default_param_{param_key}"
                if checkbox_key in st.session_state:
                    del st.session_state[checkbox_key]


