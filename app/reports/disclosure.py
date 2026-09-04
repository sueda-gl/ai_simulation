# app/reports/disclosure.py
"""Pure builders behind the disclosure results exports (Decisions 1 and 2).

Everything here was moved VERBATIM out of
`app/pages/results/visualizations/disclosure_viz.py`: the same column order and
fallbacks, the same `round(..., 2)` on stored values, the same z-scores
(`ddof=1`), the same eligibility-gate re-derivation with its `12500.0` default,
the same `'General'` number format. Only the `st.*` calls were replaced by
return values -- these functions never read session state and never imported it.

No Streamlit: pandas / numpy / openpyxl (and `src.decisions.income_utils`) only.

`prepare_*` are the public names; the original `_prepare_*` / `_apply_*` names
are kept as aliases at the bottom of the module because
`app/pages/results/components/export_section.py` imports them by those names.

The workbook plumbing (`to_xlsx_bytes`) and the disclosure number formatter
live in `app/reports/xlsx.py` (B1); this module imports them rather than
keeping its own copy.
"""
import pandas as pd

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

    # ========================================================================
    # 1. Agent ID
    # ========================================================================
    if 'agent_id' in df.columns:
        export_df['Agent ID'] = df['agent_id']
    elif 'index' in df.columns:
        export_df['Agent ID'] = df['index'] + 1  # Convert 0-based to 1-based
    else:
        export_df['Agent ID'] = range(1, len(df) + 1)

    # ========================================================================
    # 2-6. Raw Personality Trait Values (non-standardized)
    # ========================================================================

    # 2. Agreeable
    if 'Agreeable' in df.columns:
        export_df['Agreeable'] = df['Agreeable']
    else:
        export_df['Agreeable'] = ''

    # 3. Openness (from OpennessBig5)
    if 'OpennessBig5' in df.columns:
        export_df['Openness'] = df['OpennessBig5']
    else:
        export_df['Openness'] = ''

    # 4. Honesty_Humility
    if 'Honesty_Humility' in df.columns:
        export_df['Honesty_Humility'] = df['Honesty_Humility']
    else:
        export_df['Honesty_Humility'] = ''

    # 5. Extraversion (from ExtraversionBig5)
    if 'ExtraversionBig5' in df.columns:
        export_df['Extraversion'] = df['ExtraversionBig5']
    else:
        export_df['Extraversion'] = ''

    # 6. Neuroticism (from NeuroticismBig5)
    if 'NeuroticismBig5' in df.columns:
        export_df['Neuroticism'] = df['NeuroticismBig5']
    else:
        export_df['Neuroticism'] = ''

    # ========================================================================
    # 7-9. Religious Components (raw values + computed composite)
    # ========================================================================

    # 7. ReligiousAffiliation (raw binary 0/1)
    if 'ReligiousAffiliation' in df.columns:
        export_df['ReligiousAffiliation'] = df['ReligiousAffiliation']
    else:
        export_df['ReligiousAffiliation'] = ''

    # 8. ReligiousService (raw ordinal)
    if 'ReligiousService' in df.columns:
        export_df['ReligiousService'] = df['ReligiousService']
    else:
        export_df['ReligiousService'] = ''

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
    if 'Assigned Allowance Level' in df.columns:
        export_df['Assigned Allowance Level'] = df['Assigned Allowance Level']
    elif 'actual_allowance' in df.columns:
        export_df['Assigned Allowance Level'] = df['actual_allowance']
    else:
        export_df['Assigned Allowance Level'] = ''

    # 11. Income (right after Assigned Allowance Level)
    if 'income' in df.columns:
        export_df['income'] = df['income']
    elif 'actual_allowance' in df.columns:
        export_df['income'] = df['actual_allowance']
    else:
        export_df['income'] = ''

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
    if 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}' in df.columns:
        export_df['TWT+Sospeso'] = df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
    else:
        export_df['TWT+Sospeso'] = ''

    # ========================================================================
    # 14. Predicted Prosocial Behavior (trait-based)
    # ========================================================================

    # calc_PB (weighted_prosocial from traits, before anchoring)
    if 'disclose_income_weighted_prosocial' in df.columns:
        export_df['calc_PB'] = df['disclose_income_weighted_prosocial']
    else:
        export_df['calc_PB'] = ''

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
    if 'disclose_income_anchored_pb' in df.columns:
        export_df['PB_i'] = df['disclose_income_anchored_pb']
    else:
        export_df['PB_i'] = ''

    # 18. Disclosure Income (Continuous value before Y/N classification)
    if 'disclose_income_di' in df.columns:
        export_df['Disclosure Income'] = df['disclose_income_di']
    elif 'disclose_income_raw' in df.columns:
        export_df['Disclosure Income'] = df['disclose_income_raw']
    else:
        export_df['Disclosure Income'] = ''

    # ========================================================================
    # 19. Final Decision (LAST COLUMN - nothing after this)
    # ========================================================================

    # disclose_income (Y/N to 1/0)
    export_df['Disclose Income (Y=1)'] = df['disclose_income'].apply(
        lambda x: 1 if x == 'Y' else (0 if x == 'N' else '')
    )

    return export_df


def prepare_disclose_documents_excel_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare disclose DOCUMENTS data for Excel export (privacy-calculus model, item-22 layout).

    Column order (per professor feedback item 22):
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
    if 'agent_id' in df.columns:
        e['Agent ID'] = df['agent_id']
    elif 'index' in df.columns:
        e['Agent ID'] = df['index'] + 1
    else:
        e['Agent ID'] = range(1, len(df) + 1)

    # --- Raw traits ---------------------------------------------------------
    e['Extraversion'] = df['ExtraversionBig5'] if 'ExtraversionBig5' in df.columns else ''
    e['Neuroticism'] = df['NeuroticismBig5'] if 'NeuroticismBig5' in df.columns else ''
    e['Agreeable'] = df['Agreeable'] if 'Agreeable' in df.columns else ''

    # --- Income / allowance -------------------------------------------------
    e['Assigned Allowance Level'] = df['Assigned Allowance Level'] if 'Assigned Allowance Level' in df.columns else ''
    income_series = None
    if 'income' in df.columns:
        income_series = pd.to_numeric(df['income'], errors='coerce')
    elif 'disclose_documents_agent_income' in df.columns:
        income_series = pd.to_numeric(df['disclose_documents_agent_income'], errors='coerce')
    e['Income'] = income_series if income_series is not None else ''

    # --- TWT+Sospeso (right AFTER income) -----------------------------------
    twt_col = 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}'
    e['TWT+Sospeso'] = df[twt_col] if twt_col in df.columns else ''

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
        e['Disclose Income (Y=1)'] = df['disclose_income'].apply(
            lambda x: 1 if x == 'Y' else (0 if x == 'N' else 'N/A')
        )
    else:
        # Standalone DD run: disclose_income was not computed (ungated model).
        e['Disclose Income (Y=1)'] = 'N/A'

    # --- Disclose Documents (Y=1): N/A unless QUALIFIED ----------------------
    # Qualified gate = Disclose Income == 'Y' AND income < discount threshold (strict).
    # Matches the model gate (disclose_documents*.py: income >= threshold -> "NA").
    # For every agent failing that gate (DI != Y, or income at/above threshold), force N/A.
    from src.decisions.income_utils import get_simulation_param
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
        base = 1 if raw == 'Y' else (0 if raw == 'N' else 'N/A')
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
    if 'agent_id' in df.columns:
        export_df['Agent ID'] = df['agent_id']
    elif 'index' in df.columns:
        export_df['Agent ID'] = df['index'] + 1  # Convert 0-based to 1-based
    else:
        export_df['Agent ID'] = range(1, len(df) + 1)

    # ====================================================================
    # AGENT TRAITS: Full agent trait columns (consistent with Disclose Income)
    # ====================================================================

    # Honesty_Humility
    if 'Honesty_Humility' in df.columns:
        export_df['Honesty_Humility'] = df['Honesty_Humility'].round(2)
    else:
        export_df['Honesty_Humility'] = ''

    # Assigned Allowance Level - use the actual allowance column which exists for ALL agents
    # Priority: 'Assigned Allowance Level' > 'actual_allowance' > 'income' (not income_category which is only for Discount/Fixed)
    if 'Assigned Allowance Level' in df.columns:
        export_df['Assigned Allowance Level'] = df['Assigned Allowance Level']
    elif 'actual_allowance' in df.columns:
        export_df['Assigned Allowance Level'] = df['actual_allowance']
    elif 'income' in df.columns:
        export_df['Assigned Allowance Level'] = df['income']
    else:
        export_df['Assigned Allowance Level'] = ''

    # Study Program
    if 'Study Program' in df.columns:
        export_df['Study Program'] = df['Study Program']
    else:
        export_df['Study Program'] = ''

    # Group_experiment (check for various possible column names, case-insensitive)
    if 'Group_experiment' in df.columns:
        export_df['Group_experiment'] = df['Group_experiment']
    elif 'group' in df.columns:
        export_df['Group_experiment'] = df['group']
    elif 'group_experiment' in df.columns:
        export_df['Group_experiment'] = df['group_experiment']
    else:
        export_df['Group_experiment'] = ''

    # TWT+Sospeso
    if 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}' in df.columns:
        export_df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'] = df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'].round(2)
    else:
        export_df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'] = ''

    # Income
    if 'income' in df.columns:
        export_df['income'] = df['income'].round(2)
    elif 'actual_allowance' in df.columns:
        export_df['income'] = df['actual_allowance'].round(2)
    else:
        export_df['income'] = ''

    # ====================================================================

    # disclose_income (Y/N to 1/0)
    if 'disclose_income' in df.columns:
        export_df['disclose_income'] = df['disclose_income'].apply(
            lambda x: 1 if x == 'Y' else (0 if x == 'N' else '')
        )
    else:
        export_df['disclose_income'] = ''

    # disclose_documents (Y/N/NA to 1/0/N/A)
    if 'disclose_documents' in df.columns:
        export_df['disclose_documents'] = df['disclose_documents'].apply(
            lambda x: 1 if x == 'Y' else (0 if x == 'N' else 'N/A')
        )
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
# Backward-compatible aliases: `app/pages/results/components/export_section.py`
# still imports the sheet builders and the formatter by these names.
# ---------------------------------------------------------------------------
_prepare_disclose_income_excel_data = prepare_disclose_income_excel_data
_prepare_disclose_documents_excel_data = prepare_disclose_documents_excel_data
_prepare_disclosure_excel_data = prepare_disclosure_excel_data
_apply_price_formatting_disclosure = apply_disclosure_price_formatting
