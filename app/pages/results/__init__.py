# app/pages/results/__init__.py
"""
Results module for the Enhanced AI Agent Simulation.
"""
from app.pages.results.main_results import render_results_page, render_single_run_results
from app.pages.results.comparisons import (
    render_all_modes_comparison,
    render_income_comparison
)
from app.pages.results.components.agent_details import render_individual_agent_details
from app.pages.results.components.export_section import render_export_section
from app.pages.results.run_context import RunContext

__all__ = [
    'render_results_page',
    'render_single_run_results',
    'render_all_modes_comparison',
    'render_income_comparison',
    'render_individual_agent_details',
    'render_export_section',
    'RunContext'
]
