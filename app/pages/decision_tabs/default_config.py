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
    
    # Initialize if not exists
    if prob_key not in st.session_state:
        st.session_state[prob_key] = default_probability
    
    st.caption(description)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        probability = st.slider(
            f"P({options[0]}) - Probability of {options[0]} vs bidding",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state[prob_key],
            step=0.01,
            help=f"Probability that agents will choose {options[0]} vs {options[1]}",
            key=prob_key
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
    
    # Initialize if not exists
    if selection_key not in st.session_state:
        st.session_state[selection_key] = default_option
    
    # Create option names mapping
    option_names = dict(options) if options else {}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected = st.radio(
            "Default Option",
            options=[opt[0] for opt in options],
            format_func=lambda x: option_names.get(x, x),
            index=[opt[0] for opt in options].index(st.session_state[selection_key]) if st.session_state[selection_key] in [opt[0] for opt in options] else 0,
            help="Choose the default option for this decision",
            key=selection_key
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
    
    # Initialize main selection key if not exists
    if selection_key not in st.session_state:
        st.session_state[selection_key] = default_selection.copy()
    
    # Initialize all checkbox keys before any widgets are created
    # This ensures checkboxes start with the correct state on first render
    for param_key in parameters.keys():
        checkbox_key = f"{decision_name}_default_param_{param_key}"
        if checkbox_key not in st.session_state:
            # Initialize based on current selection state
            st.session_state[checkbox_key] = param_key in st.session_state[selection_key]
    
    st.caption("Select which parameters should be included (equal weight distribution)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_params = []
        
        for param_key, param_info in parameters.items():
            checkbox_key = f"{decision_name}_default_param_{param_key}"
            
            # Create checkbox - it will use the value from session state automatically
            is_selected = st.checkbox(
                f"{param_info['name']} - {param_info['description']}",
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
    
    # Initialize if not exists
    if value_key not in st.session_state:
        st.session_state[value_key] = default_value
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Determine if this is a percentage (between 0 and 1)
        if 0 <= default_value <= 1:
            value = st.slider(
                "Default Value",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state[value_key],
                step=0.01,
                format="%.2f",
                help="Set the default value for this decision",
                key=value_key
            )
            st.caption(f"Percentage: {value:.1%}")
        else:
            value = st.number_input(
                "Default Value",
                min_value=0.0,
                value=float(st.session_state[value_key]),
                step=0.1,
                help="Set the default value for this decision",
                key=value_key
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


