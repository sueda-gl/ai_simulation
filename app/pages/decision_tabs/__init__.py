# app/pages/decision_tabs/__init__.py
"""
Decision tabs module for the Enhanced AI Agent Simulation.
"""
import streamlit as st
from app.pages.decision_tabs.donation_default import render_donation_default_tab
from app.pages.decision_tabs.disclose_income import render_disclose_income_tab
from app.pages.decision_tabs.disclose_documents import render_disclose_documents_tab
from app.pages.decision_tabs.generic_decision import render_generic_decision_tab
from app.pages.decision_tabs.rejected_transaction import render_rejected_transaction_defaults_tab
from app.pages.decision_tabs.global_parameters import (
    render_global_parameters_readonly,
    render_global_parameters_tab
)
from app.pages.decision_execution import render_simulation_buttons


def render_decision_tab(decision_name):
    """Render configuration for a specific decision"""
    if decision_name == "donation_default":
        render_donation_default_tab()
    elif decision_name == "disclose_income":
        render_disclose_income_tab()
    elif decision_name == "disclose_documents":
        render_disclose_documents_tab()
    elif decision_name == "bid_value":
        # Use dedicated bid_value tab with formula visualization
        from app.pages.decision_tabs.bid_value_tab import render_bid_value_tab
        render_bid_value_tab()
    elif decision_name == "rejected_transaction_defaults":
        # Decision 4: four trait-based sub-decision mechanisms (TTP length + three rankings)
        render_rejected_transaction_defaults_tab()
    elif decision_name == "rejected_transaction_option":
        # This decision is configured directly in the results page
        st.info(f"Configuration for {decision_name.replace('_', ' ').title()} is available in the Results page.")
        st.caption("💡 This decision uses radio buttons in the default values display for easy configuration.")

        # Still provide simulation buttons for this decision
        render_simulation_buttons(
            decision_name=decision_name,
            selected_decisions=st.session_state.decision_params.selected_decisions
        )
    else:
        render_generic_decision_tab(decision_name)


__all__ = [
    'render_decision_tab',
    'render_donation_default_tab',
    'render_disclose_documents_tab',
    'render_rejected_transaction_defaults_tab',
    'render_generic_decision_tab',
    'render_global_parameters_readonly',
    'render_global_parameters_tab',
    'render_simulation_buttons'
]
