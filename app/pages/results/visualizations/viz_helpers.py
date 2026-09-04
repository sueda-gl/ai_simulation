# app/pages/results/visualizations/viz_helpers.py
"""
Helper functions for decision visualizations.
"""
import streamlit as st


def get_dynamic_description(decision_name):
    """Get dynamic description for decisions showing current probability settings"""
    
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES, DEFAULT_DECISION_DESCRIPTIONS
    
    default_value = DEFAULT_DECISION_VALUES.get(decision_name)
    
    # For parametric random decisions, show current probability
    if isinstance(default_value, dict) and default_value.get("type") == "random_probability":
        options = default_value.get("options", ["Y", "N"])
        # Use _default_ key (consistent with Page 2 Overview tab)
        prob_key = f"{decision_name}_default_probability_y"
        current_prob = st.session_state.get(prob_key, default_value.get("probability_y", 0.5))
        
        return f"{current_prob:.0%} chance of {options[0]}, {1-current_prob:.0%} chance of {options[1]}"
    
    # For other decisions, use static description
    return DEFAULT_DECISION_DESCRIPTIONS.get(decision_name, "Standard default behavior")


