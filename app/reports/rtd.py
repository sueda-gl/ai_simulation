"""Pure (Streamlit-free) Decision 4 sheet builders.

The workbook/frame builders behind the "rejected transaction defaults" MODEL run
(Decision 4): the five sub-decision elements (Options List Length / Tendency to
Plan, Loyalty, Willingness-to-Pay, Risk-Taking, Flexibility) and the Section-6
rank aggregation's integrated default list.  Ported from the owner's September
2026 `app/pages/results/visualizations/transaction_viz.py` (original repo,
e973a99), which kept them next to the charts; the page now imports them back
and keeps only the Streamlit rendering (charts, captions, download buttons).

Nothing here imports Streamlit or reads session state: every input is passed in
explicitly.  The `prepare_*` builders RAISE on a malformed frame; the page
turns that back into the same inline `st.error(...)` the original showed.
"""
import math

import numpy as np
import pandas as pd

from app.reports.xlsx import to_xlsx_bytes

# Per-element sheet / section names for the Decision 4 exports.
RTD_ELEMENT_SHEETS = {
    'ttp': 'Options List Length',
    'loyalty': 'Loyalty',
    'wtp': 'Willingness-to-Pay',
    'risk_taking': 'Risk-Taking',
    'flexibility': 'Flexibility',
}
# Section-6 rank aggregation (integrated default list) sheet / section name.
RTD_AGG_SHEET = 'Integrated Default List'
# The value of rtd_run_element for "Run Integrated Default List Only".
RTD_AGGREGATION_ELEMENT = 'aggregation'
RTD_ALL_ELEMENTS = ('ttp', 'loyalty', 'wtp', 'risk_taking', 'flexibility')
# Elements whose equations contain NO income term: identical under the categorical and
# the continuous income specification, so a comparison layout renders them only once.
RTD_INCOME_FREE_ELEMENTS = ('ttp', 'loyalty', 'flexibility')

# Results-page subtitle of a single-element Decision 4 run (Lavie 2026-10: "below the
# '4. Rejected Transaction Defaults' title, add a subtitle for the decision element to
# which the results correspond, e.g. '4.1 Options List Length (Tendency to Plan)'").
# Numbers and names are the Decision 4 tab's sub-tabs (MECH_TITLES / AGGREGATION_TITLE
# in app/pages/decision_tabs/rejected_transaction.py), i.e. the document's order.
RTD_ELEMENT_SUBTITLES = {
    'ttp': (1, 'Options List Length (Tendency to Plan)'),
    'loyalty': (2, 'Loyalty Ranking'),
    'wtp': (3, 'Willingness-to-Pay Ranking'),
    'risk_taking': (4, 'Risk-Taking Ranking'),
    'flexibility': (5, 'Flexibility Ranking'),
    RTD_AGGREGATION_ELEMENT: (6, 'Integrated Default List (Rank Aggregation)'),
}


def rtd_element_subtitle(decision_number, element):
    """'4.1 Options List Length (Tendency to Plan)' for a single-element run of
    Decision `decision_number`; None for a whole-decision / complete run (element
    None) or an unknown element."""
    if element not in RTD_ELEMENT_SUBTITLES:
        return None
    index, name = RTD_ELEMENT_SUBTITLES[element]
    prefix = f"{decision_number}.{index}" if decision_number is not None else f"{index}."
    return f"{prefix} {name}"


# The two list-length averages a saved Decision 4 configuration summarises.
RTD_SUMMARY_METRICS = ('mean_choice_length', 'mean_default_list_length')


def rtd_summary_metric_keys(element=None):
    """Which list-length averages belong on a Decision 4 configuration summary shown for
    `element` (Lavie #8, professor 2026-10: drop the summary line unrelated to the page).

    None (the whole decision was run)       -> both averages;
    'ttp' (Options List Length element)     -> the options list length only;
    'aggregation' (integrated default list) -> the integrated default list length only;
    'loyalty' / 'wtp' / 'risk_taking' / 'flexibility' -> neither.
    """
    if element is None:
        return RTD_SUMMARY_METRICS
    if element == 'ttp':
        return ('mean_choice_length',)
    if element == RTD_AGGREGATION_ELEMENT:
        return ('mean_default_list_length',)
    return ()

RTD_STAGE_LABELS = {
    'kemeny': 'Kemeny alone (unique full ranking)',
    'schulze': 'Schulze',
    'copeland': 'Copeland',
    'footrule': 'Footrule',
    'random': 'Last resort (random)',
}
RTD_TRUNCATION_LABELS = {
    'none': 'Not truncated (full ranking kept)',
    'length': 'Options list length',
    'option5': 'Option 5 stop rule',
    'both': 'Both rules bind at the same position',
}
RTD_KEMENY_STATUS_LABELS = {
    'unique': 'Unique full ranking (no ties)',
    'unique_with_ties': 'Unique ordering with ties',
    'multiple': 'Several equally good orderings',
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
    # Flexibility: Big 5 (IVW equation order) + the observed anchor stdactions
    'flexibility': ['ExtraversionBig5', 'OpennessBig5', 'NeuroticismBig5', 'Agreeable',
                    'ConscientiousnessBig5', 'stdactions'],
}

# Raw input column -> (model column holding the standardized value, Stata name).
# These are the values the equations actually multiply, so every Decision 4 Excel
# carries them next to their raw inputs (professor 2026-09: "we are missing all the
# variables that are used in the calculation for all decision elements").
RTD_INPUT_Z = {
    'ExtraversionBig5': ('rtd_z_extraversion', 'z_extraversionbig5'),
    'Agreeable': ('rtd_z_agreeable', 'z_agreeable'),
    'NeuroticismBig5': ('rtd_z_neuroticism', 'z_neuroticismbig5'),
    'ConscientiousnessBig5': ('rtd_z_conscientiousness', 'z_conscientiousnessbig5'),
    'OpennessBig5': ('rtd_z_openness', 'z_opennessbig5'),
    'Education': ('rtd_reducation', 'reducation'),
    'income': ('rtd_z_income', 'z_net_income'),
    'stdactions': ('rtd_z_stdactions', 'z_stdactions'),
}

# mech -> (model column key, Stata name)
RTD_STATA_NAMES = {'loyalty': ('loyalty', 'loyalty'), 'wtp': ('wtp', 'WTP'),
                   'risk_taking': ('rt', 'RT'), 'flexibility': ('flex', 'Flexibility')}


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
    """Agent ID + the element's OWN independent variables: each raw input followed
    immediately by the standardized value its equation uses (Stata names
    z_extraversionbig5, reducation, z_net_income, z_stdactions, ...).

    With CATEGORICAL income the WTP / Risk-Taking equations replace the z_net_income
    term with the budget-level dummies, so those files carry 'Assigned Allowance Level'
    instead of z_net_income."""
    out = pd.DataFrame(index=df.index)
    out['Agent ID'] = rtd_agent_id_series(df)
    categorical = (mech in ('wtp', 'risk_taking')
                   and rtd_frame_income_mode(df) == 'categorical')
    for col in RTD_ELEMENT_INPUTS[mech]:
        if col in df.columns:
            out[col] = df[col]
        if col == 'income' and categorical:
            if 'Assigned Allowance Level' in df.columns:
                out['Assigned Allowance Level'] = df['Assigned Allowance Level']
            continue
        src, dst = RTD_INPUT_Z.get(col, (None, None))
        if src and src in df.columns:
            out[dst] = df[src]
    return out


def rtd_choice_columns(out, rankings):
    """choice1..choice5 columns (option numbers; blank beyond the list length)."""
    for pos in range(1, 6):
        out[f'choice{pos}'] = rankings.apply(
            lambda lst, p=pos: lst[p - 1] if isinstance(lst, list) and len(lst) >= p else np.nan)


def rtd_list_str(lst):
    """'a > b > c' rendering of an option-number list ('(empty)' for no options)."""
    if isinstance(lst, (list, tuple)):
        return ' > '.join(str(o) for o in lst) if len(lst) else '(empty)'
    return 'N/A'


def _rtd_flex_intermediates(out, df):
    """Flexibility intermediates in Stata naming: the calculated IVW score and its
    population z (z_stdactions travels with the inputs; the anchored score and its z
    follow as Flexibility_score / z_Flexibility)."""
    for src, dst in (('rtd_flex_ivw', 'Flexibility_calculated_ivw'),
                     ('rtd_flex_z_ivw', 'z_Flexibility_calculated_ivw')):
        if src in df.columns:
            out[dst] = df[src]


def prepare_rtd_element_export(df, mech):
    """Per-element Excel frame: Agent ID, ONLY this element's independent variables and
    the z-scores its equation uses, its score and intermediates, the deterministic and
    final segment (or the deterministic and final list length for the Options List
    Length), the sigma used, and the resulting option sequence as choice1..choice5.
    Stata-aligned column names. Used for both the per-element file and the element's
    sheet in the whole-decision workbook."""
    out = rtd_element_inputs_frame(df, mech)
    if mech == 'ttp':
        for src, dst in (('rtd_weighted_ttp', 'weighted_ttp'),
                         ('rtd_weighted_ttp06', 'weighted_ttp06'),
                         ('rtd_choice_length_deterministic', 'choice_length_deterministic'),
                         ('rtd_choice_length', 'choice_length'),
                         ('rtd_sigma_used_ttp', 'sigma_used_ttp')):
            if src in df.columns:
                out[dst] = df[src]
        return out
    col_key, stata = RTD_STATA_NAMES[mech]
    if mech == 'flexibility':
        _rtd_flex_intermediates(out, df)
    out[f'{stata}_score'] = df[f'rtd_{col_key}_score']
    if f'rtd_{col_key}_z' in df.columns:
        out[f'z_{stata}'] = df[f'rtd_{col_key}_z']
    if f'rtd_{col_key}_segment_deterministic' in df.columns:
        out[f'{stata}_segment_deterministic'] = df[f'rtd_{col_key}_segment_deterministic']
    out[f'{stata}_segment'] = df[f'rtd_{col_key}_segment']
    if f'rtd_sigma_used_{col_key}' in df.columns:
        out[f'sigma_used_{stata}'] = df[f'rtd_sigma_used_{col_key}']
    rtd_choice_columns(out, df[f'rtd_{col_key}_ranking'])
    return out


def prepare_rtd_integrated_export(df):
    """The 'Integrated Default List' sheet: ONE row per agent holding EVERY variable the
    decision uses, grouped left to right (professor 2026-09: "In the integrated Excel
    output we are missing all the variables that are used in the calculation for all
    decision elements"):

      Agent ID -> raw inputs (traits, Education, income / allowance level, stdactions)
      -> z-scores -> 1 Options List Length -> 2 Loyalty -> 3 Willingness-to-Pay
      -> 4 Risk-Taking -> 5 Flexibility -> 6 integration (integrated ranking, Kemeny
      diagnostics, tie-break stage, truncation, final_choice1..5).

    This sheet replaces the former separate 'All Elements' sheet (the two were
    near-identical).
    """
    out = pd.DataFrame(index=df.index)
    out['Agent ID'] = rtd_agent_id_series(df)
    categorical = rtd_frame_income_mode(df) == 'categorical'

    # ---- raw inputs ----
    raw_inputs = ['ExtraversionBig5', 'Agreeable', 'NeuroticismBig5',
                  'ConscientiousnessBig5', 'OpennessBig5', 'Education', 'income']
    if categorical:
        raw_inputs.append('Assigned Allowance Level')
    raw_inputs.append('stdactions')
    for col in raw_inputs:
        if col in df.columns:
            out[col] = df[col]

    # ---- z-scores actually used by the equations ----
    z_inputs = ['ExtraversionBig5', 'Agreeable', 'NeuroticismBig5',
                'ConscientiousnessBig5', 'OpennessBig5', 'Education']
    if not categorical:
        # categorical WTP / RT use the budget-level dummies instead of z_net_income
        z_inputs.append('income')
    for col in z_inputs:
        src, dst = RTD_INPUT_Z[col]
        if src in df.columns:
            out[dst] = df[src]

    # ---- 1. Options List Length ----
    for src, dst in (('rtd_weighted_ttp', 'weighted_ttp'),
                     ('rtd_weighted_ttp06', 'weighted_ttp06'),
                     ('rtd_choice_length_deterministic', 'choice_length_deterministic'),
                     ('rtd_choice_length', 'choice_length')):
        if src in df.columns:
            out[dst] = df[src]

    # ---- 2-5. Ranking elements ----
    for mech in ('loyalty', 'wtp', 'risk_taking', 'flexibility'):
        col_key, stata = RTD_STATA_NAMES[mech]
        if f'rtd_{col_key}_score' not in df.columns:
            continue
        if mech == 'flexibility':
            _rtd_flex_intermediates(out, df)
            if 'rtd_z_stdactions' in df.columns:
                out['z_stdactions'] = df['rtd_z_stdactions']
        out[f'{stata}_score'] = df[f'rtd_{col_key}_score']
        if f'rtd_{col_key}_z' in df.columns:
            out[f'z_{stata}'] = df[f'rtd_{col_key}_z']
        if f'rtd_{col_key}_segment_deterministic' in df.columns:
            out[f'{stata}_segment_deterministic'] = df[f'rtd_{col_key}_segment_deterministic']
        out[f'{stata}_segment'] = df[f'rtd_{col_key}_segment']
        out[f'{stata}_list'] = df[f'rtd_{col_key}_ranking'].apply(rtd_list_str)

    # ---- 6. Integration ----
    if 'rtd_default_list' in df.columns:
        out['integrated_ranking'] = df['rtd_consensus_ranking'].apply(rtd_list_str)
        for src, dst in (('rtd_consensus_kemeny_status', 'kemeny_status'),
                         ('rtd_consensus_n_kemeny_optimal', 'n_kemeny_optimal'),
                         ('rtd_consensus_is_kemeny_optimal', 'is_kemeny_optimal'),
                         ('rtd_consensus_settled_by', 'settled_by'),
                         ('rtd_consensus_truncated_by', 'truncated_by'),
                         ('rtd_default_list_length', 'default_list_length')):
            if src in df.columns:
                out[dst] = df[src]
        for pos in range(1, 6):
            out[f'final_choice{pos}'] = df['rtd_default_list'].apply(
                lambda lst, p=pos: lst[p - 1] if isinstance(lst, list) and len(lst) >= p else np.nan)
    return out


def prepare_rtd_model_export(df):
    """Whole-decision Decision 4 workbook: the 'Integrated Default List' sheet first
    (one row per agent with EVERY input, z-score, element score, segment, list and the
    integration diagnostics - see prepare_rtd_integrated_export), then one
    self-contained sheet per element ('Options List Length', 'Loyalty',
    'Willingness-to-Pay', 'Risk-Taking', 'Flexibility'), each identical to that
    element's own per-element file.

    Returns an ordered {sheet_name: DataFrame} dict; raises on a malformed frame.
    """
    sheets = {}
    if 'rtd_default_list' in df.columns:
        sheets[RTD_AGG_SHEET] = prepare_rtd_integrated_export(df)
    for mech in RTD_ALL_ELEMENTS:
        if mech == 'ttp':
            if 'rtd_weighted_ttp' not in df.columns:
                continue
        elif f'rtd_{RTD_STATA_NAMES[mech][0]}_score' not in df.columns:
            continue
        sheet = prepare_rtd_element_export(df, mech)
        if sheet is not None and not sheet.empty:
            sheets[RTD_ELEMENT_SHEETS[mech]] = sheet
    return sheets


# Human-readable filename slugs ('ttp' reads too much like 'wtp')
RTD_ELEMENT_FILE_SLUGS = {'ttp': 'options_list_length', 'loyalty': 'loyalty',
                          'wtp': 'willingness_to_pay', 'risk_taking': 'risk_taking',
                          'flexibility': 'flexibility'}


def rtd_element_xlsx_bytes(export_df, mech):
    """Per-element workbook bytes: the element frame on its own sheet, unformatted."""
    return to_xlsx_bytes({RTD_ELEMENT_SHEETS[mech]: export_df})


def rtd_model_xlsx_bytes(sheets):
    """Decision 4 workbook bytes: one sheet per entry, in dict order, unformatted."""
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
    """`sheets` restricted to what `active_element`'s run exports: all of them for a
    whole-decision run (None), the 'Integrated Default List' sheet for "Run
    Integrated Default List Only" ('aggregation'), else that element's sheet."""
    if not active_element:
        return sheets
    name = RTD_AGG_SHEET if active_element == RTD_AGGREGATION_ELEMENT else RTD_ELEMENT_SHEETS[active_element]
    return {name: sheets[name]} if name in sheets else {}


# ---------------------------------------------------------------------------
# Chart data: score histograms (Stata's bin rule), allocation shares
# ---------------------------------------------------------------------------

# Upper bound on the histogram bin count (professor 2026-09-17: at the default 1,000
# agents Stata's rule gives 29 bins, too fine to compare with the document's figures).
# 16 is exactly what Stata's default rule gives for the 280 participants, so the
# document's histograms and the app's charts share the same bins at every N >= 280.
RTD_MAX_BINS = 16


def rtd_stata_bin_count(n):
    """Stata's DEFAULT number of histogram bins for n non-missing observations:

        k = int( min( sqrt(n), 10 * ln(n) / ln(10) ) )

    (`help histogram`: "bins = min(sqrt(N), 10*ln(N)/ln(10))"). Stata TRUNCATES the
    value to an integer - it does not round: n = 280 -> min(16.733, 24.472) = 16.733
    -> 16 bins, which is what the Decision 4 figures in the design document use (the
    document's weighted_loyalty figure is bar-for-bar np.histogram(.dta, bins=16));
    `sysuse auto` / `histogram mpg` (n = 74, 8.60) reports bin=8; n = 50 -> 7,
    n = 100 -> 10.

    The expression is evaluated exactly as Stata writes it, 10*ln(n)/ln(10) in double
    precision, not via log10: at n = 1000 that is 29.999999999999996, so the rule gives
    29 bins (sqrt(1000) = 31.6 is the larger term). Never fewer than one bin."""
    n = int(n)
    if n < 1:
        return 1
    raw = min(math.sqrt(n), 10.0 * math.log(n) / math.log(10.0))
    return max(1, int(raw))


def rtd_bin_count(n):
    """Number of bins the Decision 4 score histograms draw for n observations: Stata's
    default rule (rtd_stata_bin_count) capped at RTD_MAX_BINS = 16 (the rule's own
    value for the 280 participants), so charts at the app's default 1,000 agents
    (rule: 29 bins) use the document's 16 bins."""
    return min(rtd_stata_bin_count(n), RTD_MAX_BINS)


def rtd_stata_bins(series):
    """(edges, counts) for a score histogram: k equal-width bins spanning min..max,
    k = rtd_bin_count(number of non-missing values) - Stata's default rule capped at 16.

    Same bins as Stata's `histogram` (start = min, width = (max - min) / k) and as
    np.histogram(x, bins=k): every bin is closed on the left and open on the right
    except the LAST, which also includes the maximum, so the counts always sum to N and
    the plotted proportions sum to 1. Bins that no agent falls into simply have a count
    of 0 - Stata does not draw them at all, which is why a 16-bin figure in the
    document can show only 14 or 15 bars."""
    s = pd.Series(series).dropna().astype(float)
    n = len(s)
    k = rtd_bin_count(n)
    vmin, vmax = (float(s.min()), float(s.max())) if n else (0.0, 0.0)
    if not n or vmax <= vmin:
        # Degenerate (constant or empty) score: one unit-wide bin holding everything.
        return np.array([vmin, vmin + 1.0]), np.array([n])
    # Edges built as vmin + i * size (NOT np.linspace) so they are bit-for-bit the
    # boundaries plotly derives from xbins(start=vmin, size=size) in the page's chart.
    size = (vmax - vmin) / k
    edges = vmin + size * np.arange(k + 1, dtype=float)
    edges[-1] = max(edges[-1], vmax)
    counts = np.histogram(s.to_numpy(), bins=edges)[0]
    return edges, counts


# Score each element's distribution chart plots, per element:
#   (deterministic score column, chart title,
#    stochastic draw column, the draw's anchor column, x-axis title of the draw)
# Deterministic charts plot the STANDARDIZED score of the ranking elements (professor
# 2026-08: "present the standardized loyalty graph rather than the one before
# standardization") and weighted_ttp for the Options List Length (the document
# defines no standardized TTP) - exactly the Stata figures (Research Baseline).
# With the Normal(anchor, sigma) draw on (Research Specification / Copula), the
# allocation charts and statistics describe the DRAWN scores - the document's
# draw_k ~ Normal(anchor, sigma), re-rescaled over the population and re-binned
# (choice_length_stochastic, sweighted_loyalty15, sWTP_calculated15,
# sRT_calculated15, sFlexibility_combined15) - so the chart plots the draw too.
RTD_SCORE_CHARTS = {
    'ttp': ('rtd_weighted_ttp', "Tendency to Plan score",
            'rtd_ttp_draw', 'rtd_weighted_ttp06',
            "Drawn score ~ Normal(0-6 rescaled Tendency to Plan score, σ)"),
    'loyalty': ('rtd_loyalty_z', "Loyalty score",
                'rtd_loyalty_draw', 'rtd_loyalty_z',
                "Drawn score ~ Normal(standardized Loyalty score, σ)"),
    'wtp': ('rtd_wtp_z', "Willingness-to-Pay score",
            'rtd_wtp_draw', 'rtd_wtp_score',
            "Drawn score ~ Normal(Willingness-to-Pay score, σ)"),
    'risk_taking': ('rtd_rt_z', "Risk-Taking score",
                    'rtd_rt_draw', 'rtd_rt_z',
                    "Drawn score ~ Normal(standardized Risk-Taking score, σ)"),
    'flexibility': ('rtd_flex_z', "Flexibility score",
                    'rtd_flex_draw', 'rtd_flex_z',
                    "Drawn score ~ Normal(standardized anchored Flexibility score, σ)"),
}


def rtd_score_chart_data(df, mech):
    """(series, title, x-axis title, drawn) for an element's score distribution chart.

    Without a stochastic draw (Research Baseline, or the draw switched off, or this
    element's σ coefficient 0) the chart shows the deterministic score, as before.
    When the element's scores were drawn, it shows each agent's DRAWN score - the
    value whose population min..max is re-cut into the segments (0-6 lengths) that the
    allocation chart, its table and the overview statistics count - so graph and
    statistics describe the same values (Lavie 2026-10: the Research Specification
    graphs were identical to the Research Baseline ones while the statistics were
    not). An agent left undrawn inside a drawn run (Quintiles mode with a budget
    level's coefficient at 0) is plotted at its anchor, the value the population's
    draw range was computed with for it."""
    det_col, title, draw_col, anchor_col, draw_x_title = RTD_SCORE_CHARTS[mech]
    if draw_col in df.columns and df[draw_col].notna().any():
        series = pd.to_numeric(df[draw_col], errors='coerce')
        if anchor_col in df.columns:
            series = series.fillna(pd.to_numeric(df[anchor_col], errors='coerce'))
        return series, f"{title} (stochastic draw)", draw_x_title, True
    return df[det_col], title, title, False


def rtd_score_stats_caption(series):
    """Range line under each score chart - Min and Max only (professor 2026-09: mean,
    SD and N dropped from the caption)."""
    s = pd.Series(series).astype(float)
    return f"Min {s.min():.4f} · Max {s.max():.4f}"


def rtd_first_choice(df, col_key):
    """Each agent's FIRST-ranked option for a ranking element, read from the element's
    own ranking column (rtd_<col_key>_ranking[0]).

    Never derive this from the segment: that would hard-code the segment -> priority
    sequence mapping direction, which the model owns. The ranking column is whatever
    the model produced, so every chart built on this helper stays correct in either
    direction. Agents with an empty ranking map to 0."""
    col = f'rtd_{col_key}_ranking'
    if col not in df.columns:
        return pd.Series(0, index=df.index)
    return df[col].apply(lambda lst: lst[0] if isinstance(lst, (list, tuple)) and len(lst) else 0)


def rtd_reversed_sequence(seq):
    """Allocation-chart category order: the element's priority sequence REVERSED, so the
    least likely option sits on the left and the most likely on the right (professor
    2026-09). Applied in every context - per-element, whole-decision and comparison."""
    return list(reversed(list(seq)))


def rtd_integrated_first_choice(df):
    """First option of each agent's integrated default list (0 = empty list)."""
    return df['rtd_default_list'].apply(lambda l: l[0] if isinstance(l, list) and len(l) else 0)


def rtd_kemeny_status_frame(df):
    """'Kemeny outcome' table of the tie statistics (share of agents per status)."""
    n = len(df)
    counts = df['rtd_consensus_kemeny_status'].astype(str).value_counts()
    return pd.DataFrame({
        'Kemeny outcome': [RTD_KEMENY_STATUS_LABELS[s] for s in RTD_KEMENY_STATUS_LABELS],
        '% of agents': [f"{counts.get(s, 0) / n * 100:.1f}%" for s in RTD_KEMENY_STATUS_LABELS],
    })


def rtd_settled_by_frame(df):
    """'Stage that settled the ranking' table of the tie statistics."""
    n = len(df)
    settled = df['rtd_consensus_settled_by'].astype(str).value_counts() \
        if 'rtd_consensus_settled_by' in df.columns else pd.Series(dtype=int)
    return pd.DataFrame({
        'Settled by': [RTD_STAGE_LABELS[s] for s in RTD_STAGE_LABELS],
        '% of agents': [f"{settled.get(s, 0) / n * 100:.1f}%" for s in RTD_STAGE_LABELS],
    })
