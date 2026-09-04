"""Pure (Streamlit-free) Decision 4 sheet builders.

The workbook/frame builders behind the "rejected transaction defaults" MODEL run
(Decision 4): the four sub-decision elements (Options List Length / Tendency to
Plan, Loyalty, Willingness-to-Pay, Risk-Taking). Moved verbatim from
`app/pages/results/visualizations/transaction_viz.py`, which now imports them
back and keeps only the Streamlit rendering (charts, captions, download
buttons).

Nothing here imports Streamlit or reads session state: every input is passed in
explicitly. The two `prepare_*` builders RAISE on a malformed frame; the page
turns that back into the same inline `st.error(...)` it always showed.
"""
import numpy as np
import pandas as pd

from app.reports.xlsx import to_xlsx_bytes

# Per-element sheet / section names for the Decision 4 exports.
RTD_ELEMENT_SHEETS = {
    'ttp': 'Options List Length',
    'loyalty': 'Loyalty',
    'wtp': 'Willingness-to-Pay',
    'risk_taking': 'Risk-Taking',
}

# Independent variables per element (Stata-aligned trait column names, in each
# element's equation order). WTP and Risk-Taking additionally get 'Assigned
# Allowance Level' when the frame was computed with categorical income.
RTD_ELEMENT_INPUTS = {
    'ttp': ['ExtraversionBig5', 'Agreeable', 'NeuroticismBig5',
            'ConscientiousnessBig5', 'Education'],
    'loyalty': ['ExtraversionBig5', 'OpennessBig5', 'Agreeable'],
    'wtp': ['ExtraversionBig5', 'Agreeable', 'income'],
    'risk_taking': ['ExtraversionBig5', 'OpennessBig5', 'Agreeable',
                    'ConscientiousnessBig5', 'NeuroticismBig5', 'income'],
}


def rtd_frame_income_mode(df):
    """Income specification the frame was computed with ('categorical'/'continuous')."""
    if 'rtd_income_mode' in df.columns and len(df) > 0:
        first = df['rtd_income_mode'].dropna()
        if len(first) > 0 and str(first.iloc[0]) == 'categorical':
            return 'categorical'
    return 'continuous'


def rtd_agent_id_series(df):
    if 'agent_id' in df.columns:
        return df['agent_id']
    return pd.Series(range(1, len(df) + 1), index=df.index)


def rtd_element_inputs_frame(df, mech):
    """Agent ID + the element's OWN independent variables (Stata-aligned names).
    Categorical income adds 'Assigned Allowance Level' for wtp / risk_taking."""
    out = pd.DataFrame(index=df.index)
    out['Agent ID'] = rtd_agent_id_series(df)
    inputs = list(RTD_ELEMENT_INPUTS[mech])
    if mech in ('wtp', 'risk_taking') and rtd_frame_income_mode(df) == 'categorical':
        inputs.append('Assigned Allowance Level')
    for col in inputs:
        if col in df.columns:
            out[col] = df[col]
    return out


def rtd_choice_columns(out, rankings):
    """choice1..choice5 columns (option numbers; blank beyond the list length)."""
    for pos in range(1, 6):
        out[f'choice{pos}'] = rankings.apply(
            lambda lst, p=pos: lst[p - 1] if isinstance(lst, list) and len(lst) >= p else np.nan)


RTD_STATA_NAMES = {'loyalty': ('loyalty', 'loyalty'), 'wtp': ('wtp', 'WTP'),
                   'risk_taking': ('rt', 'RT')}


def prepare_rtd_element_export(df, mech):
    """Per-element Excel frame: Agent ID, ONLY this element's independent variables,
    its score, the segment (or list length for TTP) and the resulting option
    sequence per customer as choice1..choice5. Stata-aligned column names."""
    out = rtd_element_inputs_frame(df, mech)
    if mech == 'ttp':
        out['weighted_ttp'] = df['rtd_weighted_ttp']
        out['choice_length'] = df['rtd_choice_length']
        return out
    col_key, stata = RTD_STATA_NAMES[mech]
    out[f'{stata}_score'] = df[f'rtd_{col_key}_score']
    if f'rtd_{col_key}_z' in df.columns:
        out[f'z_{stata}'] = df[f'rtd_{col_key}_z']
    out[f'{stata}_segment'] = df[f'rtd_{col_key}_segment']
    rtd_choice_columns(out, df[f'rtd_{col_key}_ranking'])
    return out


def prepare_rtd_model_export(df):
    """Whole-decision Decision 4 workbook, organized as one self-contained sheet per
    element ('Options List Length', 'Loyalty', 'Willingness-to-Pay', 'Risk-Taking').

    Each sheet mirrors the per-element file (Agent ID + the element's own
    independent variables + choice1..choice5) and additionally carries the
    intermediate distributions: score, z (where present), deterministic and final
    segment / list length, and the sigma used. Stata-aligned column names.

    Returns an ordered {sheet_name: DataFrame} dict; raises on a malformed frame.
    """
    sheets = {}

    # ---- Options List Length (TTP) ----
    ttp = rtd_element_inputs_frame(df, 'ttp')
    for src, dst in [('rtd_weighted_ttp', 'weighted_ttp'),
                     ('rtd_weighted_ttp06', 'weighted_ttp06'),
                     ('rtd_choice_length_deterministic', 'choice_length_deterministic'),
                     ('rtd_choice_length', 'choice_length'),
                     ('rtd_sigma_used_ttp', 'sigma_used_ttp')]:
        if src in df.columns:
            ttp[dst] = df[src]
    sheets[RTD_ELEMENT_SHEETS['ttp']] = ttp

    # ---- Rankings: Loyalty / Willingness-to-Pay / Risk-Taking ----
    for mech, (col_key, stata) in RTD_STATA_NAMES.items():
        if f'rtd_{col_key}_score' not in df.columns:
            continue
        sheet = rtd_element_inputs_frame(df, mech)
        sheet[f'{stata}_score'] = df[f'rtd_{col_key}_score']
        if f'rtd_{col_key}_z' in df.columns:
            sheet[f'z_{stata}'] = df[f'rtd_{col_key}_z']
        if f'rtd_{col_key}_segment_deterministic' in df.columns:
            sheet[f'{stata}_segment_deterministic'] = df[f'rtd_{col_key}_segment_deterministic']
        sheet[f'{stata}_segment'] = df[f'rtd_{col_key}_segment']
        rtd_choice_columns(sheet, df[f'rtd_{col_key}_ranking'])
        if f'rtd_sigma_used_{col_key}' in df.columns:
            sheet[f'sigma_used_{stata}'] = df[f'rtd_sigma_used_{col_key}']
        sheets[RTD_ELEMENT_SHEETS[mech]] = sheet
    return sheets


# Human-readable filename slugs ('ttp' reads too much like 'wtp')
RTD_ELEMENT_FILE_SLUGS = {'ttp': 'options_list_length', 'loyalty': 'loyalty',
                          'wtp': 'willingness_to_pay', 'risk_taking': 'risk_taking'}


def rtd_element_xlsx_bytes(export_df, mech):
    """Per-element workbook bytes: the element frame on its own sheet, unformatted."""
    return to_xlsx_bytes({RTD_ELEMENT_SHEETS[mech]: export_df})


def rtd_model_xlsx_bytes(sheets):
    """Whole-decision workbook bytes: one sheet per element, in dict order, unformatted."""
    return to_xlsx_bytes(sheets)


# ---------------------------------------------------------------------------
# The Decision 4-only export of the results page's export section
# (`app/pages/results/components/export_section.py`): sheet naming for a
# multi-configuration run and the per-element subset.  Pure name/dict work -
# the page still calls its own `_prepare_rtd_model_export` wrapper so a
# malformed frame keeps showing the inline `st.error`.
# ---------------------------------------------------------------------------

# Result key -> sheet-name prefix used when one workbook holds several configurations.
RTD_CONFIG_SHEET_PREFIXES = {
    'copula_categorical': 'Copula_Cat', 'copula_continuous': 'Copula_Cont',
    'research_spec_categorical': 'ResSpec_Cat', 'research_spec_continuous': 'ResSpec_Cont',
    'research_baseline_categorical': 'ResBase_Cat', 'research_baseline_continuous': 'ResBase_Cont',
    'categorical': 'Cat', 'continuous': 'Cont',
}


def rtd_config_sheet_prefix(config_key):
    """Sheet-name prefix for one result key (unknown keys: first 12 characters)."""
    return RTD_CONFIG_SHEET_PREFIXES.get(config_key, str(config_key)[:12])


def rtd_prefixed_sheet_name(prefix, sheet_name):
    """'<prefix> <sheet>' truncated to Excel's 31-character sheet-name limit."""
    return f"{prefix} {sheet_name}"[:31]


def rtd_element_subset(sheets, active_element):
    """`sheets` restricted to `active_element`'s sheet (all of them when None)."""
    if not active_element:
        return sheets
    name = RTD_ELEMENT_SHEETS[active_element]
    return {name: sheets[name]} if name in sheets else {}


def rtd_score_stats_caption(series):
    """Summary line matching Stata's `summarize` output for the score variable."""
    s = pd.Series(series).astype(float)
    return (f"Mean {s.mean():.4f} · SD {s.std(ddof=1):.4f} · "
            f"Min {s.min():.4f} · Max {s.max():.4f} · N {s.notna().sum():,}")
