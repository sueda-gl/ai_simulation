# app/pages/decision_tabs/generic_decision.py
"""
Generic decision tab configuration for decisions without specific implementation.
"""
import streamlit as st
from app.pages.decision_execution import render_simulation_buttons


def render_generic_decision_tab(decision_name):
    """Render configuration for a generic decision"""
    st.markdown(f'<h3 class="section-header">🎯 {decision_name.replace("_", " ").title()} Configuration</h3>', unsafe_allow_html=True)
    
    st.info(f"Configuration for {decision_name} will be implemented here.")
    st.caption("This decision currently uses default values.")
    
    # Render both individual and complete simulation buttons
    render_simulation_buttons(
        decision_name=decision_name,
        selected_decisions=st.session_state.decision_params.selected_decisions
    )
