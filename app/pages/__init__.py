# app/pages/__init__.py
"""
Pages module for the Enhanced AI Agent Simulation.

This module organizes the application pages:
- Page 1: Common Parameters
- Page 2: Decision-Specific Parameters
- Results Page
"""

from app.pages.navigation import go_to_page1, go_to_page2, go_to_results, render_navigation
from app.pages.page1_common_params import render_page1
from app.pages.page2_decisions import render_page2
from app.pages.results import render_results_page

__all__ = [
    # Navigation functions
    'go_to_page1',
    'go_to_page2', 
    'go_to_results',
    'render_navigation',
    
    # Page rendering functions
    'render_page1',
    'render_page2',
    'render_results_page'
]
