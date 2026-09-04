"""Pure (Streamlit-free) numerics that back what the UI reports on screen.

`app.reports.preview` holds the Page-1 income-distribution preview maths that
used to live as methods on `SimulationParameters`.
"""
from app.reports.preview import (
    DEFAULT_PREVIEW_SEED,
    discount_qualification_rate,
    distribution_kwargs,
    sample_income_distribution,
)

__all__ = [
    "DEFAULT_PREVIEW_SEED",
    "discount_qualification_rate",
    "distribution_kwargs",
    "sample_income_distribution",
]
