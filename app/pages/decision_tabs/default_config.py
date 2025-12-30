# app/pages/decision_tabs/default_config.py
"""
UI components for configuring default decision parameters before simulation.
These parameters apply to decisions that are NOT selected for custom configuration.
"""
import streamlit as st
from app.pages.decision_execution import DEFAULT_DECISION_VALUES


def save_to_persistent_storage(key):
    """Save a session state value to persistent storage dictionary.
    
    This creates a 'shadow state' that persists even if Streamlit widget keys 
    are cleared or reset during navigation.
    """
    if '_persistent_defaults' not in st.session_state:
        st.session_state._persistent_defaults = {}
    
    if key in st.session_state:
        st.session_state._persistent_defaults[key] = st.session_state[key]


def restore_from_persistent_storage(key, default_value):
    """Restore a value from persistent storage to session state if available.
    
    Returns the restored value or the default value.
    """
    if '_persistent_defaults' not in st.session_state:
        st.session_state._persistent_defaults = {}
    
    # Priority 1: Use value from persistent storage
    if key in st.session_state._persistent_defaults:
        val = st.session_state._persistent_defaults[key]
        # Restore to session state so widget finds it
        st.session_state[key] = val
        return val
    
    # Priority 2: Use existing session state value if present
    if key in st.session_state:
        return st.session_state[key]
        
    # Priority 3: Use default value
    st.session_state[key] = default_value
    return default_value


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
    from app.models import ALL_DECISIONS
    
    # Get decision number
    decision_number = ALL_DECISIONS.index(decision_name) + 1 if decision_name in ALL_DECISIONS else None
    
    # Special handling for purchase_vs_bid to show "Purchase Now" instead of "Purchase"
    if decision_name == "purchase_vs_bid":
        decision_title = "Purchase Now Vs Bid"
    elif decision_name == "purchasing_quantity":
        decision_title = "Purchase Request Quantity"
    elif decision_name == "purchasing_frequency":
        decision_title = "Purchase Request Frequency"
    else:
        decision_title = decision_name.replace('_', ' ').title()
    
    # Add number prefix
    if decision_number is not None:
        decision_title = f"{decision_number}. {decision_title}"
    
    default_value = DEFAULT_DECISION_VALUES.get(decision_name)
    
    st.markdown(f"**🎯 {decision_title}**")
    
    # Handle different types of default decisions
    if isinstance(default_value, dict) and default_value.get("type") == "random_probability":
        render_probability_default_config(decision_name, default_value)
    
    elif isinstance(default_value, dict) and default_value.get("type") == "prioritized_selection":
        render_prioritized_default_config(decision_name, default_value)
    
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
    
    # SHADOW STATE: Restore from persistent storage before rendering
    current_value = restore_from_persistent_storage(prob_key, default_probability)
    
    # Special handling for purchase_vs_bid - show it only applies to regular customers
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
        
        # Widget uses key parameter - value is read from session state
        # SHADOW STATE: Use on_change to save to persistent storage
        probability = st.slider(
            slider_label,
            min_value=0.0,
            max_value=1.0,
            value=current_value,
            step=0.01,
            help=slider_help,
            key=prob_key,
            on_change=save_to_persistent_storage,
            args=(prob_key,)
        )
    
    with col2:
        st.metric("Ratio", f"{probability:.1%} : {1-probability:.1%}")
        st.caption(f"{options[0]} : {options[1]}")
    
    with col3:
        st.metric("Default", f"{default_probability:.0%}")
        if probability != default_probability:
            st.caption("⚙️ Modified")
        else:
            st.caption("✓ Default")


def render_prioritized_default_config(decision_name, default_value):
    """Render UI for prioritized selection default decisions"""
    
    options = default_value.get("options", [])
    default_template = default_value.get("priority_template", ["forgo_transaction"])
    description = default_value.get("description", "")
    
    # Session state key for this decision's priority template
    template_key = f"{decision_name}_priority_template"
    
    # SHADOW STATE: Restore from persistent storage
    current_template = restore_from_persistent_storage(template_key, default_template)
    
    # Create option names mapping
    option_names = dict(options) if options else {}
    option_codes = [opt[0] for opt in options]
    
    st.info(f"ℹ️ {description}")
    
    # UI for configuring priority list
    st.markdown("**Configure Priority Template:**")
    st.caption("Select options in priority order. Agents will try options in this sequence when transactions are rejected.")
    
    # Available options (not yet in template)
    available_options = [opt for opt in option_codes if opt not in current_template]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display current priority list
        st.markdown("**Current Priority List:**")
        
        if current_template:
            for i, opt in enumerate(current_template, 1):
                col_item, col_remove = st.columns([4, 1])
                with col_item:
                    st.text(f"{i}. {option_names.get(opt, opt)}")
                with col_remove:
                    if st.button("❌", key=f"{decision_name}_remove_{i}", help=f"Remove {opt}"):
                        new_template = current_template.copy()
                        new_template.remove(opt)
                        st.session_state[template_key] = new_template
                        save_to_persistent_storage(template_key)  # SHADOW STATE: Save
                        st.rerun()
        else:
            st.caption("No options selected")
        
        # Add option selector
        st.markdown("---")
        st.markdown("**Add Option to List:**")
        
        if available_options:
            selected_to_add = st.selectbox(
                "Choose an option to add:",
                options=[""] + available_options,
                format_func=lambda x: "-- Select option --" if x == "" else option_names.get(x, x),
                key=f"{decision_name}_add_selector"
            )
            
            if selected_to_add and st.button("➕ Add to List", key=f"{decision_name}_add_btn"):
                new_template = current_template.copy() if current_template else []
                
                # If adding forgo_transaction, ensure it goes to the end
                if selected_to_add == "forgo_transaction":
                    new_template.append(selected_to_add)
                else:
                    # Add before forgo_transaction if it exists, otherwise at the end
                    if "forgo_transaction" in new_template:
                        insert_pos = new_template.index("forgo_transaction")
                        new_template.insert(insert_pos, selected_to_add)
                    else:
                        new_template.append(selected_to_add)
                
                st.session_state[template_key] = new_template
                save_to_persistent_storage(template_key)  # SHADOW STATE: Save
                st.rerun()
        else:
            st.success("✅ All options are in the priority list")
        
        # Reset to default button
        if st.button("🔄 Reset to Default", key=f"{decision_name}_reset"):
            st.session_state[template_key] = default_template
            save_to_persistent_storage(template_key)  # SHADOW STATE: Save
            st.rerun()
    
    with col2:
        st.metric("Options in List", len(current_template))
        st.metric("Available to Add", len(available_options))
        
        if current_template == default_template:
            st.caption("✓ Default")
        else:
            st.caption("⚙️ Modified")
        
        # Validation
        st.markdown("---")
        st.markdown("**Validation:**")
        
        if "forgo_transaction" in current_template:
            if current_template[-1] == "forgo_transaction":
                st.success("✅ Option 5 is last")
            else:
                st.error("⚠️ Option 5 must be last!")
        else:
            st.info("ℹ️ Option 5 not in list")


def render_radio_default_config(decision_name, default_value):
    """Render UI for radio selection default decisions"""
    
    options = default_value.get("options", [])
    default_option = default_value.get("default_option", "")
    
    # Session state key for this decision's selection
    selection_key = f"{decision_name}_default_selection"
    
    # SHADOW STATE: Restore from persistent storage
    current_selection = restore_from_persistent_storage(selection_key, default_option)
    
    # Create option names mapping
    option_names = dict(options) if options else {}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Compute default index based on current session state
        # CRITICAL: Don't use .get() with fallback - key must exist before widget renders
        option_codes = [opt[0] for opt in options]
        
        try:
            default_index = option_codes.index(current_selection) if current_selection in option_codes else 0
        except Exception:
            default_index = 0
        
        # Widget uses explicit index derived from session state
        # SHADOW STATE: Use on_change to save to persistent storage
        selected = st.radio(
            "Default Option",
            options=option_codes,
            format_func=lambda x: option_names.get(x, x),
            index=default_index,
            help="Choose the default option for this decision",
            key=selection_key,  # Streamlit manages value automatically via session state
            on_change=save_to_persistent_storage,
            args=(selection_key,)
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
    
    # SHADOW STATE: Restore from persistent storage
    # We don't use the returned value directly for checkboxes, but we ensure it's in session state
    restore_from_persistent_storage(selection_key, default_selection)
    
    st.caption("Select which parameters should be included (equal weight distribution)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_params = []
        
        for param_key, param_info in parameters.items():
            checkbox_key = f"{decision_name}_default_param_{param_key}"
            
            # SHADOW STATE: Restore individual checkbox state
            default_checked = param_key in default_selection
            restore_from_persistent_storage(checkbox_key, default_checked)
            
            # Create checkbox - reads from session state
            # CRITICAL: Don't use .get() with fallback - key must exist before widget renders
            is_selected = st.checkbox(
                f"{param_info['name']} - {param_info['description']}",
                value=st.session_state[checkbox_key],  # Read directly from key (no fallback)
                key=checkbox_key,
                on_change=save_to_persistent_storage,
                args=(checkbox_key,)
            )
            
            if is_selected:
                selected_params.append(param_key)
        
        # Update main selection state to reflect current checkbox states
        st.session_state[selection_key] = selected_params
        save_to_persistent_storage(selection_key)  # SHADOW STATE: Save list
    
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
    
    # Special handling for final_donation_rate when donation config is selected
    # Check that key exists AND value is not None
    if decision_name == "final_donation_rate" and st.session_state.get('selected_donation_config') is not None:
        render_final_donation_rate_with_config(default_value)
        return
    
    # Session state key for this decision's value
    value_key = f"{decision_name}_default_value"
    
    # SHADOW STATE: Restore from persistent storage
    current_value = restore_from_persistent_storage(value_key, default_value)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Determine if this is a percentage (between 0 and 1)
        if 0 <= default_value <= 1:
            # Use finer step for final_donation_rate to show precise values like 10.19%
            if decision_name == "final_donation_rate":
                step_size = 0.01  # 1% increments, but display 4 decimal places
                format_str = "%.4f"
                caption_format = f"Percentage: {st.session_state[value_key]:.2%}"
            else:
                step_size = 0.01
                format_str = "%.2f"
                caption_format = f"Percentage: {st.session_state[value_key]:.1%}"
            
            # Widget reads from session state
            # CRITICAL: Don't use .get() with fallback - key must exist before widget renders
            value = st.slider(
                "Default Value",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state[value_key],  # Read directly from key (no fallback)
                step=step_size,
                format=format_str,
                help="Set the default value for this decision",
                key=value_key,  # Streamlit manages value automatically via session state
                on_change=save_to_persistent_storage,
                args=(value_key,)
            )
            st.caption(f"Percentage: {value:.2%}" if decision_name == "final_donation_rate" else f"Percentage: {value:.1%}")
        else:
            # Widget reads from session state
            # CRITICAL: Don't use .get() with fallback - key must exist before widget renders
            value = st.number_input(
                "Default Value",
                min_value=0.0,
                value=float(st.session_state[value_key]),  # Read directly from key (no fallback)
                step=0.1,
                help="Set the default value for this decision",
                key=value_key,  # Streamlit manages value automatically via session state
                on_change=save_to_persistent_storage,
                args=(value_key,)
            )
    
    with col2:
        st.metric("Current", f"{value:.2f}")
        if value != default_value:
            st.caption("⚙️ Modified")
        else:
            st.caption("✓ Default")


def render_final_donation_rate_with_config(default_value):
    """Render final_donation_rate UI when a donation configuration is selected.
    
    Shows the slider synced to the selected donation configuration's mean rate.
    """
    config = st.session_state.selected_donation_config
    mean_donation = config['metrics']['mean_donation']
    
    # Session state key for this decision's value
    value_key = "final_donation_rate_default_value"
    
    # CRITICAL FIX: Do NOT force overwrite on every render!
    # The value is synced ONCE when config is selected (in save_selected_configuration or auto_populate).
    # After that, we must respect the user's manual adjustments in session state.
    
    # Initialize if missing
    if value_key not in st.session_state:
        st.session_state[value_key] = mean_donation
    
    # Ensure persistent storage is initialized
    if '_persistent_defaults' not in st.session_state:
        st.session_state._persistent_defaults = {}
    
    # Show that this is linked to the selected donation configuration
    st.success(f"✅ **Linked to Selected Donation Configuration**")
    st.caption(f"📊 Value synced from: {config['population_mode']} + {config['income_spec_mode']}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show slider with the config's mean donation as initial value, but allow changes
        # Step of 0.01 allows 1% increments, display shows 4 decimal places
        value = st.slider(
            "Default Value",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state[value_key],  # Read from session state (user modified)
            step=0.01,
            format="%.4f",
            help="This value is synced from the selected donation configuration. You can adjust it if needed.",
            key=value_key,
            on_change=save_to_persistent_storage,
            args=(value_key,)
        )
        st.caption(f"Percentage: {value:.2%}")
    
    with col2:
        st.metric("From Config", f"{mean_donation:.2%}")
        if abs(value - mean_donation) < 0.001:
            st.caption("🔗 Synced")
        else:
            st.caption("⚙️ Adjusted")


def render_placeholder_default_config(decision_name, default_value):
    """Render info for placeholder/computed default decisions"""
    
    descriptions = {
        "RANDOM_WITHIN_LIMIT": "Random value within purchasing limit (computed per agent based on income category)",
        "CALCULATED": "Calculated by spreading Purchasing quantity over simulation term",
        "RANDOM_WITHIN_RANGE": "Random bid amount within bidding price range (computed based on market parameters)",
        "deterministic": "Deterministic selection based on highest weighted vendor-product score",
        "NA": "Not applicable - this decision is not relevant given other parameter choices"
    }
    
    description = descriptions.get(default_value, f"Default behavior: {default_value}")
    
    st.info(f"ℹ️ **Computed During Simulation**")
    st.caption(description)
    
    # Decision-specific captions
    if decision_name in ["purchasing_quantity", "purchasing_frequency"]:
        st.caption("💡 No pre-configuration needed - this value is automatically calculated for all agents")
    elif decision_name == "bid_amount":
        st.caption("💡 No pre-configuration needed - this value is automatically calculated only for agents who ended up placing a bid")
    else:
        st.caption("💡 No pre-configuration needed - this value is automatically calculated during simulation")


def reset_all_default_parameters(unselected_decisions):
    """Reset all default parameters to system defaults"""
    
    for decision_name in unselected_decisions:
        # Clear all session state keys for this decision
        keys_to_clear = [
            f"{decision_name}_default_probability_y",
            f"{decision_name}_default_selection",
            f"{decision_name}_default_params",
            f"{decision_name}_default_value",
            f"{decision_name}_priority_template"  # For prioritized selection
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
            # SHADOW STATE: Clear from persistent storage too
            if '_persistent_defaults' in st.session_state and key in st.session_state._persistent_defaults:
                del st.session_state._persistent_defaults[key]
        
        # Clear checkbox states for vendor_choice_weights
        default_value = DEFAULT_DECISION_VALUES.get(decision_name)
        if isinstance(default_value, dict) and default_value.get("type") == "checkbox_selection":
            parameters = default_value.get("parameters", {})
            for param_key in parameters.keys():
                checkbox_key = f"{decision_name}_default_param_{param_key}"
                if checkbox_key in st.session_state:
                    del st.session_state[checkbox_key]
                # SHADOW STATE: Clear from persistent storage too
                if '_persistent_defaults' in st.session_state and checkbox_key in st.session_state._persistent_defaults:
                    del st.session_state._persistent_defaults[checkbox_key]


