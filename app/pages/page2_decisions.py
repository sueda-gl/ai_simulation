# app/pages/page2_decisions.py
"""
Page 2: Decision-Specific Parameters for the Enhanced AI Agent Simulation.
"""
import streamlit as st
from app.models import ALL_DECISIONS
from app.pages.navigation import render_navigation
from app.pages.decision_tabs import render_decision_tab
from app.pages.decision_tabs.global_parameters import render_global_parameters_readonly
from app.pages.decision_execution import run_combined_simulation, DEFAULT_DECISION_VALUES


def render_overview_tab(selected_decisions):
    """Render the overview tab with combined execution option"""
    
    # Display Global Parameters
    render_global_parameters_readonly()
    
    # Add combined run button
    st.markdown('<h3 class="section-header">🚀 Complete Simulation</h3>', unsafe_allow_html=True)
    
    # Import ALL_DECISIONS for calculation
    from app.models import ALL_DECISIONS
    unselected_decisions = [d for d in ALL_DECISIONS if d not in selected_decisions]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"🎯 **Complete end-to-end simulation**: {len(selected_decisions)} decisions with custom parameters + {len(unselected_decisions)} decisions with default values")
        if len(unselected_decisions) > 0:
            st.caption(f"💡 Unselected decisions will use default values to provide complete simulation experience")
    with col2:
        if st.button("🚀 Run Complete Simulation", type="primary", width="stretch"):
            run_combined_simulation(selected_decisions)
    


def render_page2():
    """Render Page 2: Decision-Specific Parameters"""
    st.markdown('<h2 class="page-header">Page 2: Decision-Specific Parameters</h2>', unsafe_allow_html=True)
    
    # Decision selection
    st.markdown('<h3 class="section-header">🎯 Decision Selection</h3>', unsafe_allow_html=True)
    
    # Multi-select with "Select All" functionality
    select_all = st.checkbox("Select All Decisions", value=False)
    
    if select_all:
        selected_decisions = st.multiselect(
            "Selected Decisions",
            ALL_DECISIONS,
            default=ALL_DECISIONS,
            help="All decisions are selected",
            disabled=True
        )
    else:
        # Use session state to preserve selections when navigating between pages
        # But default to empty list if nothing was previously selected
        default_selections = st.session_state.decision_params.selected_decisions if hasattr(st.session_state.decision_params, 'selected_decisions') and st.session_state.decision_params.selected_decisions else []
        
        selected_decisions = st.multiselect(
            "Select Decisions to Run",
            ALL_DECISIONS,
            default=default_selections,
            help="Select one or more decisions to run",
            placeholder="Choose decisions..."
        )
    
    # Store selected decisions
    st.session_state.decision_params.selected_decisions = selected_decisions
    
    if not selected_decisions:
        st.warning("Please select at least one decision to configure parameters")
        # Navigation
        render_navigation('page2')
        return
    
    # Create tabs
    tab_names = ["📊 Overview"] + [f"🎯 {d.replace('_', ' ').title()}" for d in selected_decisions]
    tabs = st.tabs(tab_names)
    
    # Overview Tab
    with tabs[0]:
        render_overview_tab(selected_decisions)
    
    # Decision-specific tabs
    for i, decision in enumerate(selected_decisions):
        with tabs[i + 1]:
            render_decision_tab(decision)
    
    # Navigation
    render_navigation('page2')
