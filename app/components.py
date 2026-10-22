# app/components.py
"""
Compatibility shim. What this module used to hold now lives with the page that
uses it:

    app/pages/css.py                      get_css_styles - the global CSS
    app/pages/page1_income_histogram.py   show_income_distribution_histogram
    app/pages/results/overview.py         show_overview, rtd_page_element,
                                          rtd_overview_metric and the disclose
                                          income / documents rate analyses
    app/pages/results/config_pin.py       the "Use This Config" buttons and the
                                          seed-mismatch error
    app/pages/results/mc_display.py       show_monte_carlo_results

Import from those modules. The re-exports below remain only for the files that
still import from here and are not yet free to change:

    tests/test_run_context_run_values.py  rtd_page_element, show_overview

Delete this module once that file imports from app.pages.results.overview.
"""
from app.pages.results.overview import rtd_page_element, show_overview  # noqa: F401
