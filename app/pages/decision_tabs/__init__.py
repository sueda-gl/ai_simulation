# app/pages/decision_tabs/__init__.py
"""
Decision tabs module for the Enhanced AI Agent Simulation.
"""
import streamlit as st
from app.pages.decision_tabs.donation_default import render_donation_default_tab
from app.pages.decision_tabs.generic_decision import render_generic_decision_tab
# Rejected transaction functionality moved to results page
from app.pages.decision_tabs.global_parameters import (
    render_global_parameters_readonly,
    render_global_parameters_tab
)


def render_decision_tab(decision_name):
    """Render configuration for a specific decision"""
    if decision_name == "donation_default":
        render_donation_default_tab()
    elif decision_name in ["rejected_transaction_defaults", "rejected_transaction_option"]:
        # These decisions are configured directly in the results page
        st.info(f"Configuration for {decision_name.replace('_', ' ').title()} is available in the Results page.")
        st.caption("💡 This decision uses radio buttons in the default values display for easy configuration.")
    else:
        render_generic_decision_tab(decision_name)


__all__ = [
    'render_decision_tab',
    'render_donation_default_tab',
    'render_generic_decision_tab',
    'render_global_parameters_readonly',
    'render_global_parameters_tab'
]
