"""Pure (Streamlit-free) numerics that back what the UI reports on screen.

`app.reports.preview` holds the Page-1 income-distribution preview maths that
used to live as methods on `SimulationParameters`.

`app.reports.xlsx`, `app.reports.agent_level` and `app.reports.transaction_level`
hold the results-page export builders: the workbook plumbing plus the two
two-level export DataFrames, moved out of
`app/pages/results/components/export_section.py`. Every function here takes its
inputs explicitly -- the results DataFrame, the vendors list, the simulation
parameter values, the timestamp converter -- so the page keeps the session-state
reads and the `st.*` calls, and the export numbers can be computed (and tested)
outside a Streamlit script run.
"""
from app.reports.agent_level import build_agent_level_dataframe
from app.reports.preview import (
    DEFAULT_PREVIEW_SEED,
    discount_qualification_rate,
    distribution_kwargs,
    sample_income_distribution,
)
from app.reports.transaction_level import build_transaction_level_dataframe
from app.reports.xlsx import (
    apply_bid_price_formatting,
    apply_disclosure_price_formatting,
    apply_donation_price_formatting,
    apply_export_price_formatting,
    apply_transaction_price_formatting,
    apply_vendor_price_formatting,
    to_xlsx_bytes,
)

__all__ = [
    "DEFAULT_PREVIEW_SEED",
    "discount_qualification_rate",
    "distribution_kwargs",
    "sample_income_distribution",
    # export builders
    "build_agent_level_dataframe",
    "build_transaction_level_dataframe",
    # workbook plumbing
    "to_xlsx_bytes",
    "apply_bid_price_formatting",
    "apply_disclosure_price_formatting",
    "apply_donation_price_formatting",
    "apply_export_price_formatting",
    "apply_transaction_price_formatting",
    "apply_vendor_price_formatting",
]
