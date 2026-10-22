# app/reports/disclosure.py
"""Pure builders behind the disclosure results exports (Decisions 1 and 2).

The sheet frames behind `app/pages/results/visualizations/disclosure_viz.py`:
`round(..., 2)` on stored values, z-scores with `ddof=1`, the eligibility-gate
re-derivation with its `12500.0` default, the `'General'` number format.

No Streamlit: pandas / numpy / openpyxl (and `src.engine.config_access`) only.

`prepare_*` build the sheet frames; the `*_sheets` / `build_*_xlsx` helpers at
the bottom assemble the results page's DI-only and DD-only export workbooks.

The workbook plumbing (`to_xlsx_bytes`) and the disclosure number formatter
live in `app/reports/xlsx.py`; this module imports them rather than
keeping its own copy.
"""
from typing import Dict

import pandas as pd

from app.reports._common import (
    CONFIG_SHEET_NAMES, agent_id_column, column_or_blank, rounded_column_or_blank,
)
from app.reports.xlsx import apply_disclosure_price_formatting, to_xlsx_bytes


# ---------------------------------------------------------------------------
# On-screen statistics for the raw DI / DD histograms
# ---------------------------------------------------------------------------
def raw_value_stats(raw_values: pd.Series) -> dict:
    """Mean / std / median / min / max of a raw decision series."""
    return {
        'mean': raw_values.mean(),
        'std': raw_values.std(),
        'median': raw_values.median(),
        'min': raw_values.min(),
        'max': raw_values.max(),
    }


def disclose_income_choice_counts(decision_data: pd.Series) -> dict:
    """The Y/N metrics of the Decision 1 results: `value_counts`, `total`,
    `yes_count` / `pct_yes`, `no_count` / `pct_no` (% of all agents) and
    `disclosure_rate` (= `pct_yes`)."""
    value_counts = decision_data.value_counts()
    total = len(decision_data)
    yes_count = value_counts.get('Y', 0)
    pct_yes = (yes_count/total)*100
    no_count = value_counts.get('N', 0)
    pct_no = (no_count/total)*100
    disclosure_rate = (yes_count/total)*100
    return {'value_counts': value_counts, 'total': total, 'yes_count': yes_count,
            'pct_yes': pct_yes, 'no_count': no_count, 'pct_no': pct_no,
            'disclosure_rate': disclosure_rate}


def disclose_documents_choice_counts(decision_data: pd.Series) -> dict:
    """The metrics of the Decision 2 results: `total_agents`, `na_count` (not
    qualified), `qualified_agents`, `yes_count` / `no_count`, their % of the
    qualified agents `pct_yes` / `pct_no` and `disclosure_rate` (None without
    qualified agents), and `qualified_counts`, the value counts without 'NA'
    (the pie chart)."""
    # Separate NA (not applicable) from Y/N choices
    value_counts = decision_data.value_counts()
    total_agents = len(decision_data)

    na_count = value_counts.get('NA', 0)
    qualified_agents = total_agents - na_count

    yes_count = value_counts.get('Y', 0)
    no_count = value_counts.get('N', 0)
    if qualified_agents > 0:
        pct_yes = (yes_count/qualified_agents)*100
        pct_no = (no_count/qualified_agents)*100
        disclosure_rate = (yes_count/qualified_agents)*100
    else:
        pct_yes = pct_no = disclosure_rate = None

    # Filter out NA for the pie chart
    qualified_counts = {k: v for k, v in value_counts.items() if k != 'NA'}
    return {'total_agents': total_agents, 'na_count': na_count,
            'qualified_agents': qualified_agents, 'yes_count': yes_count,
            'no_count': no_count, 'pct_yes': pct_yes, 'pct_no': pct_no,
            'disclosure_rate': disclosure_rate, 'qualified_counts': qualified_counts}


def raw_stats_frame(stats: dict, value_label: str) -> pd.DataFrame:
    """The '📈 Statistics' table shown under a raw DI / DD histogram."""
    return pd.DataFrame({
        'Statistic': ['Mean', 'Std Dev', 'Median', 'Min', 'Max'],
        value_label: [
            f"{stats['mean']:.4f}",
            f"{stats['std']:.4f}",
            f"{stats['median']:.4f}",
            f"{stats['min']:.4f}",
            f"{stats['max']:.4f}"
        ]
    })


def customer_type_summary_frame(customer_stats: dict) -> pd.DataFrame:
    """The '📊 Customer Type Summary' table (breakdown by pricing model)."""
    return pd.DataFrame({
        'Type': ['Regular', 'Fixed', 'Discount'],
        'Agents': [
            f"{customer_stats['regular']['count']:,}",
            f"{customer_stats['fixed']['count']:,}",
            f"{customer_stats['discount']['count']:,}"
        ],
        'Share': [
            f"{customer_stats['regular']['percentage']:.2f}%",
            f"{customer_stats['fixed']['percentage']:.2f}%",
            f"{customer_stats['discount']['percentage']:.2f}%"
        ]
    })


# ---------------------------------------------------------------------------
# Sheet builders
# ---------------------------------------------------------------------------
def _y1_n0_or_blank(value):
    """A Y/N choice as 1/0; '' for anything else."""
    return 1 if value == 'Y' else (0 if value == 'N' else '')


def _y1_n0_or_na(value):
    """A Y/N choice as 1/0; 'N/A' for anything else."""
    return 1 if value == 'Y' else (0 if value == 'N' else 'N/A')


def prepare_disclose_income_excel_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare disclose income data for Excel export.

    Includes all variables used in the disclose income calculation:
    - Raw trait values (non-standardized): Agreeable, Openness, Honesty_Humility,
      Extraversion, Neuroticism, ReligiousAffiliation, ReligiousService, Religious composite
    - Income information: Assigned Allowance Level, I-High indicator
    - Observed prosocial behavior: TWT+Sospeso
    - Configuration values: WOPB, WPB, Intercept
    - Calculated values: PB_i (anchored prosocial behavior), DI_i (continuous value)
    - Income (actual income value)
    - Final decision: disclose_income (1/0)

    IMPORTANT: Columns after disclose_income are intentionally excluded.

    Args:
        df: Results dataframe with agent data

    Returns:
        DataFrame formatted for Excel export with 19 columns, or None if required columns missing
    """
    # Check required column
    if 'disclose_income' not in df.columns:
        return None

    # Create export dataframe
    export_df = pd.DataFrame()

    # A column below is the first source column the frame has, else blank ('').

    # ========================================================================
    # 1. Agent ID ('agent_id', else 'index' + 1, else 1..n)
    # ========================================================================
    export_df['Agent ID'] = agent_id_column(df)

    # ========================================================================
    # 2-6. Raw Personality Trait Values (non-standardized)
    # ========================================================================
    export_df['Agreeable'] = column_or_blank(df, 'Agreeable')                # 2.
    export_df['Openness'] = column_or_blank(df, 'OpennessBig5')              # 3.
    export_df['Honesty_Humility'] = column_or_blank(df, 'Honesty_Humility')  # 4.
    export_df['Extraversion'] = column_or_blank(df, 'ExtraversionBig5')      # 5.
    export_df['Neuroticism'] = column_or_blank(df, 'NeuroticismBig5')        # 6.

    # ========================================================================
    # 7-9. Religious Components (raw values + computed composite)
    # ========================================================================

    # 7. ReligiousAffiliation (raw binary 0/1); 8. ReligiousService (raw ordinal)
    export_df['ReligiousAffiliation'] = column_or_blank(df, 'ReligiousAffiliation')
    export_df['ReligiousService'] = column_or_blank(df, 'ReligiousService')

    # 9. Religious composite (computed, non-standardized)
    # This comes from the decision function output
    if 'disclose_income_religious_composite' in df.columns:
        export_df['Religious'] = df['disclose_income_religious_composite']
    else:
        # Fallback: compute it here if not available
        # Religious = (ReligiousAffiliation + scaled_ReligiousService) / 2
        # where scaled_ReligiousService = ReligiousService / 4 (assuming max=4)
        if 'ReligiousAffiliation' in df.columns and 'ReligiousService' in df.columns:
            rs_scaled = df['ReligiousService'] / 4.0  # Scale to 0-1
            export_df['Religious'] = (df['ReligiousAffiliation'] + rs_scaled) / 2
        else:
            export_df['Religious'] = ''

    # ========================================================================
    # 10-12. Income Information
    # ========================================================================

    # 10. Assigned Allowance Level
    export_df['Assigned Allowance Level'] = column_or_blank(df, 'Assigned Allowance Level', 'actual_allowance')

    # 11. Income (right after Assigned Allowance Level)
    export_df['income'] = column_or_blank(df, 'income', 'actual_allowance')

    # 12. I-High (income_high indicator: 1 if level > 3, else 0)
    if 'disclose_income_income_high' in df.columns:
        export_df['I-High'] = df['disclose_income_income_high']
    else:
        # Fallback: compute from Assigned Allowance Level
        if 'Assigned Allowance Level' in df.columns:
            export_df['I-High'] = (df['Assigned Allowance Level'] > 3).astype(int)
        else:
            export_df['I-High'] = ''

    # ========================================================================
    # 13. Observed Prosocial Behavior
    # ========================================================================

    # TWT+Sospeso (observed prosocial behavior)
    export_df['TWT+Sospeso'] = column_or_blank(df, 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}')

    # ========================================================================
    # 14. Predicted Prosocial Behavior (trait-based)
    # ========================================================================

    # calc_PB (weighted_prosocial from traits, before anchoring)
    export_df['calc_PB'] = column_or_blank(df, 'disclose_income_weighted_prosocial')

    # ========================================================================
    # 15-17. Configuration Values (Weights and Intercept)
    # ========================================================================

    # 14. WOPB (Observed Prosocial Behavior Weight)
    if 'disclose_income_wopb' in df.columns:
        export_df['WOPB'] = df['disclose_income_wopb']
    else:
        # Default value from config
        export_df['WOPB'] = 0.25

    # 15. WPB (Prosocial Behavior Weight in final equation)
    if 'disclose_income_wpb' in df.columns:
        export_df['WPB'] = df['disclose_income_wpb']
    else:
        # Default value from config
        export_df['WPB'] = 0.50

    # 16. Intercept (β₀)
    if 'disclose_income_intercept' in df.columns:
        export_df['Intercept'] = df['disclose_income_intercept']
    else:
        # Default value from config
        export_df['Intercept'] = 0.75

    # ========================================================================
    # 17-18. Calculated Values
    # ========================================================================

    # 17. PB_i (Anchored Prosocial Behavior)
    export_df['PB_i'] = column_or_blank(df, 'disclose_income_anchored_pb')

    # 18. Disclosure Income (Continuous value before Y/N classification)
    export_df['Disclosure Income'] = column_or_blank(df, 'disclose_income_di', 'disclose_income_raw')

    # ========================================================================
    # 19. Final Decision (LAST COLUMN - nothing after this)
    # ========================================================================

    # disclose_income (Y/N to 1/0)
    export_df['Disclose Income (Y=1)'] = df['disclose_income'].apply(_y1_n0_or_blank)

    return export_df


def prepare_disclose_documents_excel_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare disclose DOCUMENTS data for Excel export (privacy-calculus model, item-22 layout).

    Column order (item-22 layout):
        Agent ID,
        Extraversion, Neuroticism, Agreeable          (raw traits),
        Assigned Allowance Level, Income,
        TWT+Sospeso                                    (right after Income),
        PersonalIncentive = max_income - income        (document Personal Incentive),
        Intercept (beta0)                              (BEFORE the DD columns),
        PrivacyConcern, Trust                          (standardized, Eq 2 & 3, AFTER Intercept),
        Disclosure Document                            (the FINAL DD value = DD raw, post-stochastic),
        Disclose Income (Y=1),
        Disclose Documents (Y=1)                       (N/A unless qualified: DI=1 AND income<threshold),
        customer_type                                  (Discount/Fixed/Regular).

    Standardization note: PrivacyConcern and Trust are the TRAIT parts of the document's
    Equation 1 (Privacy Concern) and Equation 2 (Trust). The model emits them WITHOUT the
    Eq1/Eq2 beta0 baseline intercepts, which are constants identical for every agent and
    therefore drop out of the z-scored (standardized) values exported here. We z-score over
    the population (ddof=1, matching Stata's egen std).

    NOTE on the DD outcome: 'Disclosure Document' is the FINAL value = DD raw (after the
    optional Normal draw). The deterministic pre-draw score (DD_score) is intentionally NOT
    exported. (Reminder of the distinction: DD_score = deterministic; DD_raw = after draw;
    they coincide when the stochastic component is off.)

    Returns a DataFrame for export, or None if disclose_documents is absent.
    """
    if 'disclose_documents' not in df.columns:
        return None

    e = pd.DataFrame()

    # --- Agent ID -----------------------------------------------------------
    e['Agent ID'] = agent_id_column(df)

    # --- Raw traits ---------------------------------------------------------
    e['Extraversion'] = column_or_blank(df, 'ExtraversionBig5')
    e['Neuroticism'] = column_or_blank(df, 'NeuroticismBig5')
    e['Agreeable'] = column_or_blank(df, 'Agreeable')

    # --- Income / allowance -------------------------------------------------
    e['Assigned Allowance Level'] = column_or_blank(df, 'Assigned Allowance Level')
    income_series = None
    if 'income' in df.columns:
        income_series = pd.to_numeric(df['income'], errors='coerce')
    elif 'disclose_documents_agent_income' in df.columns:
        income_series = pd.to_numeric(df['disclose_documents_agent_income'], errors='coerce')
    e['Income'] = income_series if income_series is not None else ''

    # --- TWT+Sospeso (right AFTER income) -----------------------------------
    e['TWT+Sospeso'] = column_or_blank(df, 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}')

    # --- PersonalIncentive = max_income - income (document Personal Incentive) -----
    # max_income is taken over the population (the maximum observed income in the sample).
    if income_series is not None and income_series.notna().any():
        max_income = income_series.max()
        e['PersonalIncentive'] = max_income - income_series
    else:
        e['PersonalIncentive'] = ''

    # --- Intercept (beta0), BEFORE the DD columns ---------------------------
    e['Intercept'] = df.get('disclose_documents_intercept', '')

    # --- Mediators: PrivacyConcern, Trust (standardized) AFTER Intercept ----
    def _standardize(col_name):
        if col_name not in df.columns:
            return None
        vals = pd.to_numeric(df[col_name], errors='coerce')
        valid = vals.dropna()
        if len(valid) < 2:
            return vals  # nothing to standardize against
        sd = valid.std(ddof=1)
        if sd == 0 or pd.isna(sd):
            return vals - valid.mean()
        return (vals - valid.mean()) / sd

    pc_std = _standardize('disclose_documents_privacy_concern')
    tr_std = _standardize('disclose_documents_trust')
    e['PrivacyConcern'] = pc_std if pc_std is not None else ''
    e['Trust'] = tr_std if tr_std is not None else ''

    # --- Single DD outcome: the FINAL value (DD raw, post-stochastic) -------
    e['Disclosure Document'] = df.get('disclose_documents_raw', '')

    # --- Disclose Income (Y=1) ----------------------------------------------
    if 'disclose_income' in df.columns:
        e['Disclose Income (Y=1)'] = df['disclose_income'].apply(_y1_n0_or_na)
    else:
        # Standalone DD run: disclose_income was not computed (ungated model).
        e['Disclose Income (Y=1)'] = 'N/A'

    # --- Disclose Documents (Y=1): N/A unless QUALIFIED ----------------------
    # Qualified gate = Disclose Income == 'Y' AND income < discount threshold (strict).
    # Matches the model gate (disclose_documents*.py: income >= threshold -> "NA").
    # For every agent failing that gate (DI != Y, or income at/above threshold), force N/A.
    from src.engine.config_access import get_simulation_param
    threshold = None
    try:
        sim_cfg = df.attrs.get('simulation_config') if hasattr(df, 'attrs') else None
        if sim_cfg is not None:
            threshold = get_simulation_param(sim_cfg, 'discount_income_threshold', 12500.0)
    except Exception:
        threshold = None
    if threshold is None:
        threshold = 12500.0

    def _dd_y(idx, raw):
        # Base mapping of the model's choice.
        base = _y1_n0_or_na(raw)
        if base == 'N/A':
            return 'N/A'
        # Apply the qualified gate explicitly.
        if 'disclose_income' in df.columns:
            di = df['disclose_income'].iloc[idx]
            if di != 'Y':
                return 'N/A'
        if income_series is not None:
            inc = income_series.iloc[idx]
            if pd.notna(inc) and inc >= threshold:
                return 'N/A'
        return base

    e['Disclose Documents (Y=1)'] = [
        _dd_y(i, raw) for i, raw in enumerate(df['disclose_documents'].tolist())
    ]

    # --- customer_type (Discount / Fixed / Regular) -------------------------
    if 'customer_type' in df.columns:
        e['customer_type'] = df['customer_type']

    return e


def prepare_disclosure_excel_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare disclosure and customer type data for Excel export.

    Converts Y/N/NA values to 1/0 format and creates customer type indicator columns.

    Args:
        df: Results dataframe with agent data

    Returns:
        DataFrame formatted for Excel export, or None if required columns missing
    """
    # Check required columns
    required_cols = ['disclose_income', 'disclose_documents', 'customer_type']
    if not all(col in df.columns for col in required_cols):
        return None

    # Create export dataframe
    export_df = pd.DataFrame()

    # Agent ID - try multiple possible column names
    export_df['Agent ID'] = agent_id_column(df)

    # ====================================================================
    # AGENT TRAITS: Full agent trait columns (consistent with Disclose Income);
    # each is the first source column the frame has, else blank ('')
    # ====================================================================
    export_df['Honesty_Humility'] = rounded_column_or_blank(df, 'Honesty_Humility')

    # Assigned Allowance Level - use the actual allowance column which exists for ALL agents
    # Priority: 'Assigned Allowance Level' > 'actual_allowance' > 'income' (not income_category which is only for Discount/Fixed)
    export_df['Assigned Allowance Level'] = column_or_blank(
        df, 'Assigned Allowance Level', 'actual_allowance', 'income')

    export_df['Study Program'] = column_or_blank(df, 'Study Program')
    export_df['Group_experiment'] = column_or_blank(df, 'Group_experiment', 'group', 'group_experiment')
    export_df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'] = rounded_column_or_blank(
        df, 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}')
    export_df['income'] = rounded_column_or_blank(df, 'income', 'actual_allowance')

    # ====================================================================

    # disclose_income (Y/N to 1/0)
    if 'disclose_income' in df.columns:
        export_df['disclose_income'] = df['disclose_income'].apply(_y1_n0_or_blank)
    else:
        export_df['disclose_income'] = ''

    # disclose_documents (Y/N/NA to 1/0/N/A)
    if 'disclose_documents' in df.columns:
        export_df['disclose_documents'] = df['disclose_documents'].apply(_y1_n0_or_na)
    else:
        export_df['disclose_documents'] = ''

    # Customer type indicator columns
    if 'customer_type' in df.columns:
        export_df['Regular'] = (df['customer_type'] == 'regular').astype(int)
        export_df['Fixed'] = (df['customer_type'] == 'fixed').astype(int)
        export_df['Discount'] = (df['customer_type'] == 'discount').astype(int)
    else:
        export_df['Regular'] = ''
        export_df['Fixed'] = ''
        export_df['Discount'] = ''

    return export_df


# ---------------------------------------------------------------------------
# Workbook builders (one sheet each)
# ---------------------------------------------------------------------------
def build_disclose_income_xlsx(excel_data: pd.DataFrame) -> bytes:
    """`agent_disclose_income_data.xlsx`: one 'Agent Disclose Income Data' sheet."""
    return to_xlsx_bytes({'Agent Disclose Income Data': excel_data},
                         formatter=apply_disclosure_price_formatting)


def build_disclose_documents_xlsx(excel_data: pd.DataFrame) -> bytes:
    """`agent_disclose_documents_data.xlsx`: one 'Agent Disclose Documents Data' sheet."""
    return to_xlsx_bytes({'Agent Disclose Documents Data': excel_data},
                         formatter=apply_disclosure_price_formatting)


def build_disclosure_xlsx(excel_data: pd.DataFrame) -> bytes:
    """`agent_disclosure_customer_types.xlsx`: one 'Agent Disclosure Data' sheet."""
    return to_xlsx_bytes({'Agent Disclosure Data': excel_data},
                         formatter=apply_disclosure_price_formatting)


# ---------------------------------------------------------------------------
# The DI-only / DD-only export workbooks of the results page's export section
# (`app/pages/results/components/export_section.py`).  Sheet names, sheet order,
# the skip rules and the formatter are the ones the page wrote inline; the page
# keeps the `st.*` copy, metrics and previews.
# ---------------------------------------------------------------------------

# "Compare all" result key -> sheet name, in the order the workbook writes them.
DISCLOSURE_COMPARE_ALL_SHEETS = CONFIG_SHEET_NAMES


def build_disclosure_sheets_xlsx(sheets: dict) -> bytes:
    """Workbook bytes for a disclosure export: every sheet, disclosure formatter."""
    return to_xlsx_bytes(sheets, formatter=apply_disclosure_price_formatting)


def disclose_income_compare_all_sheets(results_dict: dict) -> dict:
    """DI-only "Compare all" export: one sheet per population x income configuration.

    Sheets follow `DISCLOSURE_COMPARE_ALL_SHEETS` order; absent, empty or
    unpreparable configurations are skipped.
    """
    sheets_data = {}
    for config_key, sheet_name in DISCLOSURE_COMPARE_ALL_SHEETS:
        config_df = results_dict.get(config_key)

        if config_df is None or config_df.empty:
            continue

        # Prepare the export DataFrame using the standard function
        # This gives us all 19 columns with correct PB_i, DI_i, disclose_income for this config
        sheet_df = prepare_disclose_income_excel_data(config_df)

        if sheet_df is None:
            continue

        sheets_data[sheet_name] = sheet_df
    return sheets_data


def disclose_income_compare_both_sheets(results_dict: dict) -> dict:
    """DI-only "Compare both" export: one sheet per income mode, in result order."""
    sheets_data = {}
    for config_key, config_df in results_dict.items():
        if config_df is None or config_df.empty:
            continue

        # Create readable sheet name
        if 'categorical' in config_key.lower():
            sheet_name = 'Categorical'
        elif 'continuous' in config_key.lower():
            sheet_name = 'Continuous'
        else:
            sheet_name = config_key.replace('_', ' ').title()

        # Prepare the export DataFrame with all 19 columns for this config
        sheet_df = prepare_disclose_income_excel_data(config_df)

        if sheet_df is not None:
            sheets_data[sheet_name] = sheet_df
    return sheets_data


def build_disclose_income_results_xlsx(export_df) -> bytes:
    """DI-only single-configuration export: one 'Disclose Income Results' sheet."""
    return build_disclosure_sheets_xlsx({'Disclose Income Results': export_df})


def disclose_documents_export_sheets(results_dict: dict, is_compare_all: bool) -> dict:
    """DD-only multi-configuration export sheets (mirror of the DI builders).

    "Compare all" uses the `DISCLOSURE_COMPARE_ALL_SHEETS` names; otherwise the
    sheet is named after the income mode, falling back to the raw result key.
    """
    if is_compare_all:
        items = [(results_dict.get(k), name) for k, name in DISCLOSURE_COMPARE_ALL_SHEETS]
    else:
        items = [(cdf, ('Categorical' if 'categorical' in k.lower() else ('Continuous' if 'continuous' in k.lower() else k)))
                 for k, cdf in results_dict.items()]
    sheets_data = {}
    for cdf, name in items:
        if cdf is None or cdf.empty:
            continue
        sdf = prepare_disclose_documents_excel_data(cdf)
        if sdf is not None:
            sheets_data[name] = sdf
    return sheets_data


def build_disclose_documents_results_xlsx(export_df) -> bytes:
    """DD-only single-configuration export: one 'Disclose Documents' sheet."""
    return build_disclosure_sheets_xlsx({'Disclose Documents': export_df})


# ---------------------------------------------------------------------------
# Customer type distribution (the results page's Customer Type Distribution block)
# ---------------------------------------------------------------------------
def analyze_customer_types(results_df) -> Dict[str, any]:
    """
    Analyze customer type distribution from simulation results.
    
    Provides counts and percentages for each customer type.
    
    Args:
        results_df: DataFrame with simulation results (must have 'customer_type' column)
        
    Returns:
        dict: Statistics about customer type distribution
        
    Example:
        stats = analyze_customer_types(results_df)
        # Returns: {
        #   'discount': {'count': 45, 'percentage': 15.0},
        #   'fixed': {'count': 135, 'percentage': 45.0},
        #   'regular': {'count': 120, 'percentage': 40.0},
        #   'total': 300
        # }
    """
    
    if 'customer_type' not in results_df.columns:
        return {
            'error': 'customer_type column not found in results',
            'discount': {'count': 0, 'percentage': 0.0},
            'fixed': {'count': 0, 'percentage': 0.0},
            'regular': {'count': 0, 'percentage': 0.0},
            'total': 0
        }
    
    total = len(results_df)
    
    if total == 0:
        return {
            'discount': {'count': 0, 'percentage': 0.0},
            'fixed': {'count': 0, 'percentage': 0.0},
            'regular': {'count': 0, 'percentage': 0.0},
            'total': 0
        }
    
    # Count each type
    discount_count = len(results_df[results_df['customer_type'] == 'discount'])
    fixed_count = len(results_df[results_df['customer_type'] == 'fixed'])
    regular_count = len(results_df[results_df['customer_type'] == 'regular'])
    
    return {
        'discount': {
            'count': discount_count,
            'percentage': (discount_count / total) * 100
        },
        'fixed': {
            'count': fixed_count,
            'percentage': (fixed_count / total) * 100
        },
        'regular': {
            'count': regular_count,
            'percentage': (regular_count / total) * 100
        },
        'total': total
    }
