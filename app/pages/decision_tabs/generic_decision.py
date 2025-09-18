# app/pages/decision_tabs/generic_decision.py
"""
Generic decision tab configuration for decisions without specific implementation.
"""
import streamlit as st
from app.pages.decision_execution import run_individual_decision


def render_generic_decision_tab(decision_name):
    """Render configuration for a generic decision"""
    st.markdown(f'<h3 class="section-header">🎯 {decision_name.replace("_", " ").title()} Configuration</h3>', unsafe_allow_html=True)
    
    st.info(f"Configuration for {decision_name} will be implemented here.")
    st.caption("This decision currently uses default values.")
    
    # Individual run button
    st.markdown("---")
    if st.button(f"🚀 Run {decision_name.replace('_', ' ').title()} Only", type="secondary", width="stretch", key=f"run_{decision_name}"):
        run_individual_decision(decision_name)
