"""Pure numerics and CSV building behind the Monte-Carlo results screen.

Moved verbatim (B5) out of `app.components.show_monte_carlo_results`, which now
only reads `st.session_state['mc_results']`, calls the functions here and hands
the results to `st.metric` / `st.plotly_chart` / `st.dataframe` /
`st.download_button`. Nothing in this module imports Streamlit.

`mc_results` is the dict written by `app/pages/decision_execution.py` from
`run_monte_carlo_study()`: `{'summary': DataFrame, 'detailed': DataFrame,
'log': str}`. The summary frame carries one row per decision with columns
`decision, mean, std, p2.5, p97.5, runs`; the detailed frame carries one row per
Monte-Carlo run with `run` and `<decision>_mean` columns.

Two behaviours are deliberately preserved rather than fixed:

* `add_running_mean` writes the `running_mean` column *into the frame it is
  given* — the very DataFrame held in session state — exactly as the screen did.
  That is why the "Download Detailed" CSV contains a `running_mean` column that
  `run_monte_carlo_study` never produced: the chart put it there first.
* the convergence band is `mean ± 1.96 * std / sqrt(n)` computed from the
  *per-run* standard deviation, i.e. a normal-approximation interval on the
  final running mean.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# The decision whose per-run mean the overview metrics and the convergence chart
# are built from. Both the metric block and the chart are skipped when the
# Monte-Carlo study did not include it.
DONATION_DECISION = 'donation_default'

# `<decision>_mean` column the detailed frame carries for that decision.
DONATION_MEAN_COLUMN = 'donation_default_mean'

# Summary columns rendered as percentages in the on-screen table.
PERCENT_SUMMARY_COLUMNS = ['mean', 'p2.5', 'p97.5']


def donation_summary_row(summary_df: pd.DataFrame) -> Optional[pd.Series]:
    """The summary row for `donation_default`, or None when it was not run.

    Args:
        summary_df: `mc_results['summary']`

    Returns:
        The first matching row as a Series, or None.
    """
    if DONATION_DECISION in summary_df['decision'].values:
        return summary_df[summary_df['decision'] == DONATION_DECISION].iloc[0]
    return None


def overview_metrics(summary_df: pd.DataFrame) -> Optional[Dict[str, str]]:
    """The four headline metrics, already formatted as the screen shows them.

    Args:
        summary_df: `mc_results['summary']`

    Returns:
        An ordered {label: value} mapping for the four `st.metric` calls, or
        None when the study did not include `donation_default` (in which case
        the screen shows the four empty columns and no metrics).
    """
    donation_row = donation_summary_row(summary_df)
    if donation_row is None:
        return None

    return {
        "Mean Donation Rate": f"{donation_row['mean']:.2%}",
        "Standard Deviation": f"{donation_row['std']:.2%}",
        "95% CI Lower": f"{donation_row['p2.5']:.2%}",
        "95% CI Upper": f"{donation_row['p97.5']:.2%}",
    }


def has_convergence_series(detailed_df: pd.DataFrame) -> bool:
    """Whether the detailed frame can drive the convergence chart."""
    return DONATION_MEAN_COLUMN in detailed_df.columns


def add_running_mean(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """Add the expanding mean of the per-run donation rate, IN PLACE.

    The column is written onto the frame that was passed in — the object living
    in `st.session_state['mc_results']['detailed']` — which is how the
    `running_mean` column ends up in the "Download Detailed" CSV. Kept as it was.

    Args:
        detailed_df: `mc_results['detailed']`, carrying `donation_default_mean`

    Returns:
        The same frame, now with a `running_mean` column.
    """
    detailed_df['running_mean'] = detailed_df[DONATION_MEAN_COLUMN].expanding().mean()
    return detailed_df


def convergence_interval(detailed_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """The 95% band drawn across the running-average panel.

    Must be called after `add_running_mean`, which supplies `running_mean`.

    Args:
        detailed_df: `mc_results['detailed']` with `running_mean` present

    Returns:
        {'final_mean', 'final_std', 'ci_upper', 'ci_lower'}, or None for a
        single-run study (the screen draws no band then).
    """
    if len(detailed_df) > 1:
        final_mean = detailed_df['running_mean'].iloc[-1]
        final_std = detailed_df[DONATION_MEAN_COLUMN].std()
        ci_upper = final_mean + 1.96 * final_std / np.sqrt(len(detailed_df))
        ci_lower = final_mean - 1.96 * final_std / np.sqrt(len(detailed_df))

        return {
            'final_mean': final_mean,
            'final_std': final_std,
            'ci_upper': ci_upper,
            'ci_lower': ci_lower,
        }
    return None


def format_summary_for_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    """The summary table as the screen renders it: rates as percentages.

    `std` is deliberately left unformatted, as on screen; only `mean`, `p2.5`
    and `p97.5` become `.2%` strings. Works on a copy, so the frame in session
    state (and therefore the downloaded CSV) keeps its numeric values.

    Args:
        summary_df: `mc_results['summary']`

    Returns:
        A formatted copy for `st.dataframe`.
    """
    display_summary = summary_df.copy()
    for col in PERCENT_SUMMARY_COLUMNS:
        if col in display_summary.columns:
            display_summary[col] = display_summary[col].apply(lambda x: f"{x:.2%}")
    return display_summary


def summary_csv(summary_df: pd.DataFrame) -> str:
    """Bytes behind "📥 Download Summary" (`monte_carlo_summary_<ts>.csv`)."""
    return summary_df.to_csv(index=False)


def detailed_csv(detailed_df: pd.DataFrame) -> str:
    """Bytes behind "📥 Download Detailed" (`monte_carlo_detailed_<ts>.csv`).

    Includes the `running_mean` column when the convergence chart has already
    added it to this frame (see `add_running_mean`).
    """
    return detailed_df.to_csv(index=False)
