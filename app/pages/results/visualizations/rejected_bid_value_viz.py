# app/pages/results/visualizations/rejected_bid_value_viz.py
"""
Decision 12 (rejected_bid_value) visualization: a single note, the decision is
not relevant given the choice of Option 5.
"""
import streamlit as st


def render_rejected_bid_value(df, decision_name, decision_title, decision_data):
    """Visualization for rejected_bid_value"""
    st.markdown("**Not relevant given choice of Option 5**")
