# app/pages/results/visualizations/viz_helpers.py
"""
Helper functions for decision visualizations, and the chart helpers shared by
several result views.
"""
import streamlit as st
import plotly.express as px


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


def yes_no_pie(value_counts):
    """The pie of a Y/N decision's value counts, as the results overview and the
    Disclose Income view draw it (formerly four identical copies of this call).

    px.pie applies ``color_discrete_map`` only together with ``color=``, so the map
    below (meant as green Y, red N) has no effect and the slices take the default
    colours. It is kept as the copies had it: adding ``color=value_counts.index``
    would recolour the pies on screen."""
    return px.pie(
        values=value_counts.values,
        names=value_counts.index,
        color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'}
    )
