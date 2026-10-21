# src/contract/columns.py
"""
User-facing column names of the Decision 4 rank aggregation, shared by the exports,
the Individual Agent Details panel and the CLI.
"""

# User-facing outputs say "integrated", never "consensus".
# The model keeps its internal rtd_consensus_* column names (saved-configuration
# hashes, tests, the
# Stata-parity checks read them); every export / on-screen table renames them with
# integrated_column_name / rename_consensus_columns.
INTERNAL_AGGREGATION_PREFIX = "rtd_consensus_"
EXPORT_AGGREGATION_PREFIX = "rtd_integrated_"


def integrated_column_name(column):
    """User-facing name of a model column: rtd_consensus_<x> -> rtd_integrated_<x>."""
    if isinstance(column, str) and column.startswith(INTERNAL_AGGREGATION_PREFIX):
        return EXPORT_AGGREGATION_PREFIX + column[len(INTERNAL_AGGREGATION_PREFIX):]
    return column


def rename_consensus_columns(df):
    """A copy-free rename of every rtd_consensus_* column of `df` to rtd_integrated_*."""
    renames = {c: integrated_column_name(c) for c in df.columns
               if integrated_column_name(c) != c}
    return df.rename(columns=renames) if renames else df
