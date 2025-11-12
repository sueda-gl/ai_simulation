# app/pages/results/decision_visualizations.py
"""
Decision visualization router for the Enhanced AI Agent Simulation.
This module acts as the main entry point and routes to category-specific visualizations.

The actual visualization functions are organized in the visualizations/ subdirectory by category:
- donation_viz.py: Donation-related decisions
- disclosure_viz.py: Disclosure-related decisions
- transaction_viz.py: Transaction and purchase decisions
- vendor_viz.py: Vendor choice decisions
- purchasing_viz.py: Purchasing decisions
- bidding_viz.py: Bidding decisions
- viz_helpers.py: Helper functions and utilities
"""
import streamlit as st

# Import the registry and all render functions from the visualizations module
from .visualizations import (
    DECISION_VISUALIZATIONS,
    render_probability_controls,
    get_dynamic_description,
)


def render_decision_results(df, decision_name, decision_title):
    """
    Main entry point for rendering decision-specific visualizations.
    
    This function routes to the appropriate visualization based on decision_name.
    All specific render functions are imported from the visualizations submodule.
    
    Args:
        df: DataFrame containing simulation results
        decision_name: Name of the decision (e.g., 'donation_default')
        decision_title: Display title for the decision
    """
    if decision_name not in df.columns:
        st.warning(f"No results available for {decision_title}")
        return
    
    decision_data = df[decision_name]
    
    # Get the appropriate visualization function for this decision
    viz_function = DECISION_VISUALIZATIONS.get(decision_name)
    
    if viz_function is None:
        # Fallback for decisions without specific visualizations
        st.info(f"No specialized visualization available for {decision_title}")
        st.write(f"**Data preview:**")
        st.write(decision_data.describe())
        return
    
    # Call the specific visualization function
    viz_function(df, decision_name, decision_title, decision_data)


# Re-export commonly used functions for backwards compatibility
__all__ = [
    'render_decision_results',
    'render_probability_controls',
    'get_dynamic_description',
    'DECISION_VISUALIZATIONS',
]
