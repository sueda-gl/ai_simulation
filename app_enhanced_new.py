# Enhanced AI Agent Simulation Dashboard with Two-Page Interface
"""
Main entry point for the Enhanced AI Agent Simulation.
This file orchestrates the application flow and imports from modularized components.
"""
import streamlit as st
import sys
from pathlib import Path

# Add  thoject root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

# Import from our modularized app
from app.models import initialize_session_state
from app.components import get_css_styles
from app.pages import render_page1, render_page2, render_results_page
from app.state.widgets import begin_script_run, end_script_run

# Page configuration
st.set_page_config(
    page_title="COOPECON AI Agent Simulation",
    page_icon="🤖",
    layout="wide"
)

# Apply custom CSS styling
st.markdown(get_css_styles(), unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

# Rotate the per-run widget render log (app/state/widgets.py): a keyed widget the
# browser did not draw in the previous run gets its session-state value pushed.
begin_script_run()

# Main title
st.markdown('<h1 class="main-header">COOPECON AI Agent Simulation</h1>', unsafe_allow_html=True)

# Page routing
try:
    if st.session_state.page == 'page1':
        render_page1()
    elif st.session_state.page == 'page2':
        render_page2()
    elif st.session_state.page == 'results':
        render_results_page()
except Exception:
    # An error ends the run like a completed one (the browser drops what was not
    # redrawn); a rerun / stop request is a BaseException and leaves the run open.
    end_script_run()
    raise

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.9rem;'>
    Enhanced AI Agent Simulation Framework | Two-Page Interface
</div>
""", unsafe_allow_html=True)

# The run completed (a run stopped early by a newer interaction never gets here):
# the next run may take this run's widget render log as what the browser holds.
end_script_run()
