# app/pages/results/__init__.py
"""
Results module for the Enhanced AI Agent Simulation.
"""
from app.pages.results.main_results import render_results_page, render_single_run_results
from app.pages.results.comparisons import (
    render_population_comparison,
    render_all_modes_comparison,
    render_income_comparison
)
from app.pages.results.details import (
    render_parameter_applicability_summary,
    render_individual_agent_details,
    render_export_section
)

__all__ = [
    'render_results_page',
    'render_single_run_results',
    'render_population_comparison',
    'render_all_modes_comparison',
    'render_income_comparison',
    'render_parameter_applicability_summary',
    'render_individual_agent_details',
    'render_export_section'
]
