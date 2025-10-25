# app/pages.py
"""
Page logic for the Enhanced AI Agent Simulation.
This file now imports from the modular structure for better maintainability.
"""

# Re-export all functions from the modular structure for backward compatibility
from app.pages.navigation import go_to_page1, go_to_page2, go_to_results, render_navigation
from app.pages.page1_common_params import render_page1
from app.pages.page2_decisions import render_page2
from app.pages.results import render_results_page

# Also expose some internal functions that might be used elsewhere
from app.pages.decision_tabs import (
    render_decision_tab,
    render_donation_default_tab,
    render_global_parameters_readonly,
    render_global_parameters_tab
)

from app.pages.decision_execution import (
    run_individual_decision,
    run_combined_simulation
)

from app.pages.results import (
    render_single_run_results,
    render_all_modes_comparison,
    render_population_comparison,
    render_dependent_variable_results,
    render_income_comparison,
    render_parameter_applicability_summary,
    render_individual_agent_details,
    render_export_section
)
