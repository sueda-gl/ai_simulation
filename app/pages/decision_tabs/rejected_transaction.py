# app/pages/decision_tabs/rejected_transaction.py
"""
Decision 4: Rejected Transaction Defaults - tab configuration.

Five trait-based sub-decision mechanisms (per the "Decision 4 - Rejected Transaction
Defaults" design document rev 280826-2, verified against the professor's Stata file
Stata_File_Decision4_290826.dta):

  1. Options List Length (Tendency to Plan) - how many default options to pre-select (0-5)
  2. Loyalty ranking                        - priority sequence Option 3 > 1 > 4 > 5 > 2
  3. Willingness-to-Pay ranking             - priority sequence Option 3 > 2 > 1 > 4 > 5
  4. Risk-Taking ranking                    - priority sequence Option 4 > 2 > 1 > 3 > 5
  5. Flexibility ranking          - priority sequence Option 2 > 4 > 3 > 1 > 5
     (doc Section 5: IVW Big-5 score, standardized, anchored 25/75 with the observed
     SD in actions per cycle (stdactions), re-standardized and binned; verified vs
     Stata_File_Decision4_290826.dta)

Each mechanism yields its own per-agent output; the four ranking mechanisms' lists
are then integrated into ONE default list per agent by the document's Section-6 rank
aggregation (Kemeny-Young consensus with the tie-break hierarchy Schulze -> Copeland
-> footrule -> random, truncated to the options list length and at Option 5) -
sub-tab 6 explains the procedure and carries its own Run button. The aggregation is
always on (no user switch; ties no criterion can separate are always broken at
random, the document's rule).

The model coefficients and sigma constants are fixed (dta-verified); the tab exposes
the income specification (categorical / continuous / compare both; WTP and
Risk-Taking are the only income-using elements) and the stochastic settings per
mechanism (sigma strategy and x0-2 coefficient).
Persistence follows the disclose_documents triple-layer pattern: canonical rtd_*
read keys + rtd_tab_* widget keys + a tab-persistence dict.
"""
import streamlit as st
import yaml
import pandas as pd
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "decisions.yaml"

MECHANISMS = ('ttp', 'loyalty', 'wtp', 'risk_taking', 'flexibility')

MECH_TITLES = {
    'ttp': "1. Options List Length (Tendency to Plan)",
    'loyalty': "2. Loyalty Ranking",
    'wtp': "3. Willingness-to-Pay Ranking",
    'risk_taking': "4. Risk-Taking Ranking",
    'flexibility': "5. Flexibility Ranking",
}

OPTION_LABELS = {
    1: "Option 1: higher price category, same vendor",
    2: "Option 2: other vendor at lower PN price",
    3: "Option 3: current vendor at PN price",
    4: "Option 4: place a bid",
    5: "Option 5: forgo the transaction",
}

LEVEL_LABELS = {
    '1': 'Level 1 (€12)', '2': 'Level 2 (€32)', '3': 'Level 3 (€72)',
    '4': 'Level 4 (€128)', '5': 'Level 5 (€200)',
}

# Fallback sigma constants (config/decisions.yaml is the source of truth).
# Values from the professor's revised Decision 4 document (2026-09): loyalty, WTP,
# risk-taking and flexibility sigmas were all restated; TTP is unchanged.
FALLBACK_SIGMA_OVERALL = {
    'ttp': 0.395446, 'loyalty': 0.4665145336,
    'wtp': 0.4526575455, 'risk_taking': 0.39522204,
    'flexibility': 0.4346228732,
}

# Fallback categorical-income effects (config/decisions.yaml is the source of truth).
# Level 1 (EUR 12) is the base level: intercept only; level_2..5 = EUR 32/72/128/200.
FALLBACK_CATEGORICAL_EFFECTS = {
    'wtp': {'intercept': -0.691843, 'level_2': 0.2671588, 'level_3': 0.5057716,
            'level_4': 0.9411747, 'level_5': 1.822471},
    'risk_taking': {'intercept': -0.0068307, 'level_2': 0.0026128, 'level_3': 0.0050555,
                    'level_4': 0.0092738, 'level_5': 0.0179812},
}


def load_rtd_config():
    """Load rejected_transaction_defaults configuration from YAML."""
    # Tolerates a concurrent non-atomic rewrite of decisions.yaml (the other tabs'
    # "Reset Config to Defaults" buttons); a half-written file used to crash the
    # whole app and force a new session (page 1, selections lost).
    from app.models import read_yaml_config
    config = read_yaml_config(CONFIG_PATH)
    return config.get('rejected_transaction_defaults', {})


def _mech_stoch_config(config, mech):
    return (config.get('stochastic', {}).get('mechanisms', {}) or {}).get(mech, {}) or {}


def initialize_rtd_session_state():
    """Initialize session state for the Decision 4 tab (canonical rtd_* read keys)."""
    config = load_rtd_config()

    if 'rejected_transaction_tab_persistence' not in st.session_state:
        st.session_state.rejected_transaction_tab_persistence = {}

    # Sigma is a DECISION-WIDE setting (one strategy + coefficient applied to all four
    # elements; each element keeps its own base sigma from config). Anchors and
    # intercepts remain per element.
    defaults = {
        'rtd_income_mode': 'Continuous only',
        'rtd_sigma_enabled': True,
        'rtd_sigma_in_copula': False,
        'rtd_sigma_strategy': 'overall',
        'rtd_scale_factor': 1.0,
        # Element selected via a per-element Run button (None = whole decision).
        # Controls display + export only; the model always computes all four elements.
        'rtd_run_element': None,
    }
    intercepts_cfg = config.get('intercepts') or {}
    for mech in MECHANISMS:
        defaults[f'rtd_intercept_{mech}'] = float(intercepts_cfg.get(mech, 0.0) or 0.0)
    # Flexibility Anchor Mix (W_OFlex; W_CFlex = 1 - W_OFlex), config default 0.25
    defaults['rtd_flex_observed_weight'] = float(
        (config.get('flexibility_anchor') or {}).get('observed_weight', 0.25))
    # Section-6 rank aggregation: always on. The flag is kept in session state because
    # app/simulation.py and app/pages/decision_execution.py read it, but no widget
    # renders it any more (it was a diagnostic switch with no basis in the document).
    defaults['rtd_aggregation_enabled'] = True

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    if 'rtd_quintile_scale_factors' not in st.session_state:
        st.session_state.rtd_quintile_scale_factors = {
            '1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0, '5': 1.0,
        }


def restore_widget_from_storage(widget_key, storage_dict, storage_key, default_value):
    """Restore a widget key from the storage dictionary before the widget renders."""
    if storage_dict and storage_key in storage_dict:
        st.session_state[widget_key] = storage_dict[storage_key]
        return storage_dict[storage_key]
    if widget_key in st.session_state:
        return st.session_state[widget_key]
    st.session_state[widget_key] = default_value
    return default_value


def save_to_rtd_storage(widget_key, storage_key):
    """Save a widget value to the Decision 4 tab persistence dict."""
    if 'rejected_transaction_tab_persistence' not in st.session_state:
        st.session_state.rejected_transaction_tab_persistence = {}
    if widget_key in st.session_state:
        st.session_state.rejected_transaction_tab_persistence[storage_key] = st.session_state[widget_key]


def _render_black_table(df, full_width=False):
    """Render a DataFrame as a compact HTML table whose column TITLES are black.

    st.dataframe renders its column headers in gray; the professor asked for black
    header ink on every table of this tab, so the tables are emitted as plain HTML
    with explicit header styling. All values come from the config / fixed labels
    (no user-entered text), so no HTML escaping is required and the ">" separators
    of the priority lists render literally.
    """
    head = "".join(
        '<th style="text-align:left;color:#000000;font-weight:600;'
        'border-bottom:1px solid #9aa0a6;padding:5px 14px 5px 0;">'
        f'{col}</th>' for col in df.columns)
    body = "".join(
        "<tr>" + "".join(
            '<td style="color:#262730;border-bottom:1px solid #ececec;'
            f'padding:5px 14px 5px 0;">{val}</td>' for val in row) + "</tr>"
        for row in df.astype(str).itertuples(index=False, name=None))
    st.markdown(
        f'<table style="border-collapse:collapse;font-size:0.88rem;'
        f'width:{"100%" if full_width else "auto"};margin-bottom:0.6rem;">'
        f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>',
        unsafe_allow_html=True)


def _segment_mapping_df(sequence, element_name):
    """Segment -> priority list table for a ranking mechanism.

    Segment s (1..5) receives the LAST s options of the priority sequence
    (seq[5 - s:]): segment 5 ("Highest 20%") gets the full sequence starting with the
    top option, segment 1 ("Lowest 20%") gets only the last option (Option 5).
    Mirrors src.decisions.rejected_transaction_defaults._ranking_for_segment.
    """
    rows = []
    for seg in range(1, 6):
        tail = sequence[len(sequence) - seg:] if sequence else []
        label = {1: f'1 (Lowest 20% of {element_name} score segment)',
                 5: f'5 (Highest 20% of {element_name} score segment)'}.get(seg, str(seg))
        rows.append({
            f'{element_name} score segment': label,
            'Priority list for rejected transaction options': ' > '.join(str(o) for o in tail),
            'Options list length': len(tail),
        })
    return pd.DataFrame(rows)


def render_option_labels():
    """The five rejected-transaction options, one line per option, in regular black
    markdown text (they used to be st.caption, which renders gray)."""
    st.markdown("  \n".join(OPTION_LABELS[num] for num in range(1, 6)))


def render_formula_section(config, mech):
    """Render the mechanism's equation, coefficients, and segment mapping."""
    coeffs = (config.get('coefficients', {}) or {}).get(mech, {}) or {}
    sequences = config.get('priority_sequences', {}) or {}

    if mech == 'ttp':
        st.markdown(
            "Estimates each agent's **Tendency to Plan** based on Big-5 "
            "personality traits and education, and then converts it into the "
            "number of pre-selected default options (0-5)."
        )
        st.latex(
            rf"TendencyToPlan_i\ (TTP_i) = \beta_0"
            rf" {coeffs.get('extraversion', -0.0152556564):.9f} \times z_{{Extroversion_i}}"
            rf" + {coeffs.get('agreeable', 0.0177638642):.8f} \times z_{{Agreeableness_i}}"
            rf" + {coeffs.get('neuroticism', 0.01959):.5f} \times z_{{Neuroticism_i}}"
            rf" + {coeffs.get('conscientiousness', 0.00901465):.8f} \times z_{{Conscientiousness_i}}"
            rf" + {coeffs.get('education', 0.0297):.4f} \times Education_i,"
            r"\quad Education_i \in \{0,1\}"
        )
        st.latex(r"OptionsListLength05_i = \left\lfloor (6 - 0.0001) \times"
                 r" \frac{TTP_i - \min(TTP)}{\max(TTP) - \min(TTP)} \right\rfloor \in \{0,\dots,5\}")
        return

    seq = sequences.get(mech, [])
    construct = {'loyalty': 'Loyalty to the vendor', 'wtp': 'Willingness to Pay',
                 'risk_taking': 'Risk-Taking propensity',
                 'flexibility': 'Flexibility'}[mech]
    if mech == 'loyalty':
        st.markdown(f"Estimates **{construct}** based on Big-5 personality traits.")
        st.latex(
            rf"Loyalty_i = \beta_1"
            rf" {coeffs.get('extraversion', -0.009828468):.9f} \times z_{{Extroversion_i}}"
            rf" + {coeffs.get('openness', 0.01096706):.8f} \times z_{{Openness_i}}"
            rf" + {coeffs.get('agreeable', 0.0123046):.7f} \times z_{{Agreeableness_i}}"
        )
    elif mech == 'wtp':
        st.markdown(f"Estimates **{construct}** based on personality traits and income.")
        _render_income_element_formula(config, coeffs, 'wtp')
    elif mech == 'flexibility':
        _render_flexibility_formula(config, coeffs)
    else:
        st.markdown(f"Estimates **{construct}** based on Big-5 personality traits and income.")
        _render_income_element_formula(config, coeffs, 'risk_taking')

    seg_name = {'loyalty': 'Loyalty', 'wtp': 'WTP', 'risk_taking': 'RiskTaking',
                'flexibility': 'Flexibility'}[mech]
    if mech == 'flexibility':
        # Stata bins the re-standardized anchored score; min-max rescaling is
        # affine-invariant, so binning AnchoredFlexibility itself is identical.
        sym, sym_i = r"AnchoredFlexibility", r"AnchoredFlexibility_i"
    else:
        sym, sym_i = seg_name, rf"{seg_name}_i"
    st.latex(
        rf"{seg_name}15_i = \left\lfloor 1 + (5 - 0.0001) \times"
        rf" \frac{{{sym_i} - \min({sym})}}{{\max({sym}) - \min({sym})}}"
        rf" \right\rfloor \in \{{1,\dots,5\}}"
    )
    st.markdown(f"**{ELEMENT_SHORT[mech]} rejected transaction options sequence:** "
                f"{' > '.join('Option ' + str(o) for o in seq)}")
    map_col, _ = st.columns(2)
    with map_col:
        _render_black_table(_segment_mapping_df(seq, ELEMENT_SHORT[mech]), full_width=True)
    render_option_labels()


def _flex_anchor_weights(config):
    """(W_OFlex, W_CFlex): the Anchor Mix slider's current value (session state) over the
    config default; the calculated weight is always 1 - W_OFlex."""
    default_w = float((config.get('flexibility_anchor') or {}).get('observed_weight', 0.25))
    w_obs = float(st.session_state.get('rtd_flex_observed_weight', default_w))
    return w_obs, 1.0 - w_obs


def _render_flexibility_formula(config, coeffs):
    """Flexibility (doc Section 5): the IVW Big-5 equation with its intercept beta4, and
    the anchoring of the calculated score on the observed flexibility (stdactions, the
    SD in the number of actions per cycle) with the Anchor Mix weights W_OFlex / W_CFlex."""
    w_obs, w_calc = _flex_anchor_weights(config)
    st.markdown(
        "Estimates **Flexibility** based on Big-5 personality traits and observed "
        "flexibility in the experiment (captured by standard deviation in the number of "
        "actions across the eight cycle-weeks)."
    )
    st.latex(
        rf"Flexibility_i = \beta_4"
        rf" + {coeffs.get('extraversion', 0.0206):.4f} \times z_{{Extroversion_i}}"
        rf" + {coeffs.get('openness', 0.0294118):.7f} \times z_{{Openness_i}}"
        rf" {coeffs.get('neuroticism', -0.04921357):.8f} \times z_{{Neuroticism_i}}"
        rf" + {coeffs.get('agreeable', 0.04339814):.8f} \times z_{{Agreeableness_i}}"
        rf" + {coeffs.get('conscientiousness', 0.04811179):.8f} \times z_{{Conscientiousness_i}}"
    )
    st.latex(
        rf"AnchoredFlexibility_i = W_{{OFlex}} \times ObservedFlexibility_i"
        rf" + W_{{CFlex}} \times Flexibility_i"
        rf" = {w_obs:.2f} \times ObservedFlexibility_i + {w_calc:.2f} \times Flexibility_i"
    )
    st.markdown(
        "ObservedFlexibility is the observed flexibility variable stdactions (the "
        "standard deviation in the number of actions across the eight cycle-weeks): a "
        "copula trait for synthetic populations and the participant's own value in the "
        "research baseline and research specification modes."
    )


def render_flex_anchor_mix(config):
    """Anchor Mix (Flexibility sub-tab): the W_OFlex slider, mirroring Decision 1's
    W_OPB control. W_CFlex is always 1 - W_OFlex."""
    st.markdown("**Anchor Mix**")
    default_w = float((config.get('flexibility_anchor') or {}).get('observed_weight', 0.25))
    widget_key, storage_key = 'rtd_tab_flex_observed_weight', 'rtd_flex_observed_weight'
    current = restore_widget_from_storage(
        widget_key, st.session_state.rejected_transaction_tab_persistence,
        storage_key, st.session_state.get(storage_key, default_w))

    def on_change():
        st.session_state['rtd_flex_observed_weight'] = float(
            st.session_state['rtd_tab_flex_observed_weight'])
        save_to_rtd_storage('rtd_tab_flex_observed_weight', 'rtd_flex_observed_weight')

    w_obs = st.slider(
        "W_OFlex: Observed vs Calculated flexibility weight",
        min_value=0.0, max_value=1.0, value=float(current), step=0.01,
        help="AnchoredFlexibility = W_OFlex × observed flexibility (stdactions) + "
             "(1 - W_OFlex) × calculated Flexibility (Big-5 equation); "
             f"Default: {default_w:.2f}",
        key=widget_key, on_change=on_change,
    )
    st.session_state['rtd_flex_observed_weight'] = float(w_obs)
    st.markdown(f"Observed flexibility weight (W_OFlex): {w_obs:.2f} · "
                f"Calculated flexibility weight (W_CFlex): {1.0 - w_obs:.2f}")


def _categorical_effects(config, mech):
    """Per-quintile income effects for wtp / risk_taking (YAML with fallbacks)."""
    cfg = (config.get('categorical_income_effects', {}) or {}).get(mech, {}) or {}
    return {k: float(cfg.get(k, v)) for k, v in FALLBACK_CATEGORICAL_EFFECTS[mech].items()}


# Quintile labels for the categorical income effects table - EXACTLY the
# disclose_income tab's wording (professor: same names/titles across decisions).
QUINTILE_LABELS = ['Q1 (€12)', 'Q2 (€32)', 'Q3 (€72)', 'Q4 (€128)', 'Q5 (€200)']


def render_categorical_effects_table(config, mech):
    """Per-quintile effects table for the categorical income specification (base
    intercept + quintile dummies), using the disclose_income tab's terminology."""
    eff = _categorical_effects(config, mech)
    base = eff['intercept']
    st.markdown("**Income Quintile Effects (β_income_q):**")
    table = pd.DataFrame({
        'Quintile': QUINTILE_LABELS,
        'β_income_q': [
            f"{base:.7f}",
            f"{base + eff['level_2']:.7f}",
            f"{base + eff['level_3']:.7f}",
            f"{base + eff['level_4']:.7f}",
            f"{base + eff['level_5']:.7f}",
        ],
    })
    _render_black_table(table)
    st.markdown("β_income_q: Income quintile effects based on agent's income category (Quintiles 1-5)")
    st.markdown(
        f"Each value = base intercept ({base:.7f}) + the quintile's differential "
        "income effect (Quintile 1 is the base level)."
    )


def _render_continuous_equation(coeffs, mech):
    """Continuous-income equation for wtp / risk_taking."""
    if mech == 'wtp':
        st.latex(
            rf"WTP_i = \beta_2"
            rf" + {coeffs.get('extraversion', 0.078863062):.9f} \times z_{{Extroversion_i}}"
            rf" {coeffs.get('agreeable', -0.012326128):.9f} \times z_{{Agreeableness_i}}"
            rf" + {coeffs.get('income', 0.698):.3f} \times z_{{Income_i}}"
        )
    else:
        st.latex(
            rf"RiskTaking_i = \beta_3"
            rf" + {coeffs.get('extraversion', 0.025942386297):.10f} \times z_{{Extroversion_i}}"
            rf" + {coeffs.get('openness', 0.023699214948):.10f} \times z_{{Openness_i}}"
            rf" {coeffs.get('agreeable', -0.038734315188):.10f} \times z_{{Agreeableness_i}}"
            rf" {coeffs.get('conscientiousness', -0.037739440732):.10f} \times z_{{Conscientiousness_i}}"
            rf" {coeffs.get('neuroticism', -0.025388697852):.10f} \times z_{{Neuroticism_i}}"
            rf" + {coeffs.get('income', 0.006874197106):.10f} \times z_{{Income_i}}"
        )


def _render_categorical_equation(config, coeffs, mech):
    """Categorical-income equation for wtp / risk_taking: the income term is replaced
    by a per-quintile income effect (base intercept + quintile dummy)."""
    if mech == 'wtp':
        st.latex(
            rf"WTP_i = \beta_2"
            rf" + {coeffs.get('extraversion', 0.078863062):.9f} \times z_{{Extroversion_i}}"
            rf" {coeffs.get('agreeable', -0.012326128):.9f} \times z_{{Agreeableness_i}}"
            rf" + \beta_{{income\_q}}[quintile_i]"
        )
    else:
        st.latex(
            rf"RiskTaking_i = \beta_3"
            rf" + {coeffs.get('extraversion', 0.025942386297):.10f} \times z_{{Extroversion_i}}"
            rf" + {coeffs.get('openness', 0.023699214948):.10f} \times z_{{Openness_i}}"
            rf" {coeffs.get('agreeable', -0.038734315188):.10f} \times z_{{Agreeableness_i}}"
            rf" {coeffs.get('conscientiousness', -0.037739440732):.10f} \times z_{{Conscientiousness_i}}"
            rf" {coeffs.get('neuroticism', -0.025388697852):.10f} \times z_{{Neuroticism_i}}"
            rf" + \beta_{{income\_q}}[quintile_i]"
        )
    render_categorical_effects_table(config, mech)


def _render_income_element_formula(config, coeffs, mech):
    """Render the wtp / risk_taking equation(s) following the tab's Income
    Specification selection (categorical / continuous / both)."""
    income_mode = st.session_state.get('rtd_income_mode', 'Continuous only')
    if income_mode == "Categorical only":
        _render_categorical_equation(config, coeffs, mech)
    elif income_mode == "Compare both":
        st.markdown("**Categorical Income Specification**")
        _render_categorical_equation(config, coeffs, mech)
        st.markdown("**Continuous Income Specification**")
        _render_continuous_equation(coeffs, mech)
    else:
        _render_continuous_equation(coeffs, mech)


ELEMENT_SHORT = {'ttp': 'Options List Length', 'loyalty': 'Loyalty',
                 'wtp': 'Willingness-to-Pay', 'risk_taking': 'Risk-Taking',
                 'flexibility': 'Flexibility'}


def render_decision_sigma_controls(config):
    """Decision-wide sigma controls: one strategy + coefficient applied to all four
    elements (each element keeps its own base sigma from config).

    Returns the per-budget-level sigma table (quintile mode) as a DataFrame so the
    caller can render it at full page width, or None in uniform mode."""
    bases = {m: float(_mech_stoch_config(config, m).get('sigma_overall', FALLBACK_SIGMA_OVERALL[m]))
             for m in MECHANISMS}
    base_quintiles = {m: {str(k): float(v) for k, v in
                          (_mech_stoch_config(config, m).get('sigma_quintile', {}) or {}).items()}
                      for m in MECHANISMS}

    st.markdown("**σ mode**")
    strategy_widget_key = 'rtd_tab_sigma_strategy'
    strategy_storage_key = 'rtd_sigma_strategy'
    current_strategy = st.session_state.get(strategy_storage_key, 'overall')

    strategy_val = restore_widget_from_storage(
        strategy_widget_key, st.session_state.rejected_transaction_tab_persistence,
        strategy_storage_key, current_strategy)
    strategy_val = 'quintile' if 'quintile' in str(strategy_val).lower() else 'overall'

    def on_strategy_change():
        st.session_state.rtd_sigma_strategy = st.session_state.rtd_tab_sigma_strategy
        save_to_rtd_storage('rtd_tab_sigma_strategy', 'rtd_sigma_strategy')

    sigma_strategy = st.radio(
        "Apply σ uniformly or per budget level?",
        options=['overall', 'quintile'],
        format_func=lambda x: 'Uniformly (single σ for all)' if x == 'overall' else 'Quintiles (σ per budget level)',
        index=0 if strategy_val == 'overall' else 1,
        key=strategy_widget_key, on_change=on_strategy_change, horizontal=True,
    )
    st.session_state.rtd_sigma_strategy = sigma_strategy

    st.markdown("---")

    if sigma_strategy == 'overall':
        coeff_widget_key = 'rtd_tab_sigma_coefficient'
        coeff_storage_key = 'rtd_sigma_coefficient'
        scale_fallback = st.session_state.get('rtd_scale_factor', 1.0)

        coeff_val = restore_widget_from_storage(
            coeff_widget_key, st.session_state.rejected_transaction_tab_persistence,
            coeff_storage_key, scale_fallback)
        coeff_val = max(0.0, min(float(coeff_val), 2.0))

        sigma_coefficient = st.slider(
            "σ Coefficient (multiplier)", min_value=0.0, max_value=2.0, value=coeff_val, step=0.01,
            help="Coefficient to multiply each element's base σ. Applies to all elements "
                 "of the decision. Final σ per element = base σ × coefficient.",
            key=coeff_widget_key,
            on_change=lambda: save_to_rtd_storage('rtd_tab_sigma_coefficient', 'rtd_sigma_coefficient'),
        )
        st.session_state.rtd_scale_factor = sigma_coefficient
        for m in MECHANISMS:
            st.markdown(f"{ELEMENT_SHORT[m]}: Effective σ = Base σ (empirical from 280 "
                        f"participants) × multiplier = {bases[m]:.6g} × "
                        f"{sigma_coefficient:.2f} = {bases[m] * sigma_coefficient:.6g}")
        return None
    else:
        st.markdown("**Per-Quintile σ Coefficients**")
        st.markdown("Each level has its own base σ from empirical data:")

        quintile_coefficients = {}
        default_scale = st.session_state.get('rtd_scale_factor', 1.0)
        current_scales = st.session_state.get('rtd_quintile_scale_factors', {
            '1': default_scale, '2': default_scale, '3': default_scale,
            '4': default_scale, '5': default_scale})
        for level in ['1', '2', '3', '4', '5']:
            level_scale = max(0.0, min(float(current_scales.get(level, default_scale)), 2.0))
            storage_key = f'rtd_sigma_quintile_{level}'
            widget_key = f'rtd_tab_sigma_q{level}'
            q_val = restore_widget_from_storage(
                widget_key, st.session_state.rejected_transaction_tab_persistence,
                storage_key, level_scale)
            q_val = max(0.0, min(float(q_val), 2.0))

            q_coeff = st.slider(
                f"{LEVEL_LABELS[level]}", min_value=0.0, max_value=2.0, value=q_val, step=0.01,
                key=widget_key,
                on_change=lambda l=level: save_to_rtd_storage(
                    f'rtd_tab_sigma_q{l}', f'rtd_sigma_quintile_{l}'),
            )
            quintile_coefficients[level] = q_coeff
        st.session_state.rtd_quintile_scale_factors = quintile_coefficients

        eff_rows = []
        for level in ['1', '2', '3', '4', '5']:
            row = {'Budget Level': LEVEL_LABELS[level]}
            for m in MECHANISMS:
                base = base_quintiles[m].get(level, bases[m])
                row[f'{ELEMENT_SHORT[m]} base σ'] = round(base, 6)
                row[f'{ELEMENT_SHORT[m]} effective σ'] = round(base * quintile_coefficients[level], 6)
            eff_rows.append(row)
        return pd.DataFrame(eff_rows)


# Intercept symbols per the Decision 4 document's notation: beta0 (TTP, doc line
# "β0 = Intercept that sets a baseline tendency to plan"), beta1 (Loyalty),
# beta2 (WTP), beta3 (Risk-Taking).
INTERCEPT_SYMBOLS = {'ttp': 'β₀', 'loyalty': 'β₁', 'wtp': 'β₂', 'risk_taking': 'β₃',
                     'flexibility': 'β₄'}


def render_intercept_control(config, mech):
    """Per-element intercept override (β0/β1/β2/β3 per the doc), mirroring the
    Research Default / Override Value / Impact Preview layout of the other decisions."""
    symbol = INTERCEPT_SYMBOLS[mech]
    research_default = float((config.get('intercepts') or {}).get(mech, 0.0) or 0.0)

    st.markdown("**Intercept Override**")
    widget_key = f'rtd_tab_intercept_{mech}'
    storage_key = f'rtd_intercept_{mech}'
    current = restore_widget_from_storage(
        widget_key, st.session_state.rejected_transaction_tab_persistence,
        storage_key, st.session_state.get(storage_key, research_default))

    def on_change(m=mech):
        st.session_state[f'rtd_intercept_{m}'] = st.session_state[f'rtd_tab_intercept_{m}']
        save_to_rtd_storage(f'rtd_tab_intercept_{m}', f'rtd_intercept_{m}')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Research Default: {research_default:.4f}**")
        st.markdown(f"Intercept ({symbol})")
        st.markdown("Baseline value")
    with col2:
        st.markdown("**Override Value**")
        value = st.number_input(
            f"Baseline {ELEMENT_SHORT[mech]} tendency", min_value=-5.0, max_value=5.0,
            value=float(current), step=0.01, format="%.4f",
            key=widget_key, on_change=on_change,
            help=f"{symbol} baseline for this element (research default "
                 f"{research_default:.4f}). The intercept shifts the element's "
                 "standardized score by β and thereby the allocation across the segment "
                 "boundaries: the boundaries are fixed from the intercept-free population "
                 "scores, so a nonzero intercept moves agents across them (a negative "
                 "value shifts agents toward the lower segments, a positive value toward "
                 "the higher ones, capped at the end bins).",
        )
        st.session_state[storage_key] = float(value)
    with col3:
        st.markdown("**Impact Preview**")
        change = float(value) - research_default
        if abs(change) > 0.00001:
            impact = "Higher baseline" if change > 0 else "Lower baseline"
            st.metric("Change", f"{change:+.4f}", delta=impact)
        else:
            st.metric("Change", "No change")


def render_stochastic_explanation(mech):
    """Short stochastic-component explanation, phrased consistently with the other
    decisions' Final Decision text (same structure for all four elements)."""
    score = {'ttp': 'TTP_i', 'loyalty': 'Loyalty_i', 'wtp': 'WTP_i',
             'risk_taking': 'RiskTaking_i',
             'flexibility': 'AnchoredFlexibility_i'}[mech]
    bins = {'ttp': 'the 0-5 options list length',
            'loyalty': 'the 1-5 Loyalty segment',
            'wtp': 'the 1-5 WTP segment',
            'risk_taking': 'the 1-5 Risk-Taking segment',
            'flexibility': 'the 1-5 Flexibility segment'}[mech]
    st.markdown("**Stochastic Component:**")
    st.markdown(
        f"If stochastic enabled: {score} ~ Normal(μ = anchor, σ) where the anchor is "
        f"the continuous {score} score and σ = base σ × coefficient (overall or per "
        f"budget level); the drawn values are re-rescaled over the population and "
        f"re-binned into {bins}."
    )


def render_element_reset_button(mech):
    """Per-element reset: restores only this element's settings (intercept; for
    Flexibility also the Anchor Mix weight).
    Decision-wide σ settings are untouched."""
    if st.button(f"Reset {ELEMENT_SHORT[mech]} to Defaults", type="secondary",
                 help="Reset this element's settings to research defaults "
                      "(decision-wide σ settings are not affected)",
                 key=f"rtd_reset_{mech}_btn"):
        if reset_rtd_element_to_defaults(mech):
            st.toast(f"{ELEMENT_SHORT[mech]} settings reset to defaults", icon="🔄")
            st.rerun()


ELEMENT_RUN_TITLES = {
    'ttp': 'Options List Length (Tendency to Plan)',
    'loyalty': 'Loyalty Ranking',
    'wtp': 'Willingness-to-Pay Ranking',
    'risk_taking': 'Risk-Taking Ranking',
    'flexibility': 'Flexibility Ranking',
}


def render_element_run_button(mech):
    """Per-element Run button: runs the SAME individual Decision 4 simulation as the
    whole-decision button (the model always computes all four elements) but flags
    st.session_state.rtd_run_element so the results page shows and exports only
    this element's results. The whole-decision / complete-simulation buttons clear
    the flag again (see render_rejected_transaction_defaults_tab)."""
    title = ELEMENT_RUN_TITLES[mech]
    if st.button(f"🔬 Run {title} Only", type="primary",
                 key=f"rtd_run_{mech}_btn",
                 help=f"Run the Decision 4 simulation with the current settings and "
                      f"present only the {title} results and Excel"):
        st.session_state.rtd_run_element = mech
        from app.pages.decision_execution import run_individual_decision
        run_individual_decision('rejected_transaction_defaults')


def render_mechanism_subtab(config, mech):
    """Render one mechanism's sub-tab: formula + stochastic explanation
    (+ Anchor Mix for Flexibility) + intercept + per-element reset + per-element run."""
    render_formula_section(config, mech)
    render_stochastic_explanation(mech)
    if mech == 'flexibility':
        st.markdown("---")
        render_flex_anchor_mix(config)
    st.markdown("---")
    render_intercept_control(config, mech)
    st.markdown("---")
    render_element_reset_button(mech)
    st.markdown("---")
    render_element_run_button(mech)


AGGREGATION_TITLE = "6. Integrated Default List (Rank Aggregation)"


def render_aggregation_run_button():
    """Run button of the aggregation sub-tab: the SAME individual Decision 4 run as
    the other Run buttons, flagged with rtd_run_element = 'aggregation' so the
    results page presents the integrated default list plus the tie statistics."""
    if st.button("🔬 Run Integrated Default List Only", type="primary",
                 key='rtd_run_aggregation_btn',
                 help="Run the Decision 4 simulation with the current settings and "
                      "present the integrated default list together with the "
                      "tie-resolution statistics"):
        st.session_state.rtd_run_element = 'aggregation'
        from app.pages.decision_execution import run_individual_decision
        run_individual_decision('rejected_transaction_defaults')


def render_aggregation_subtab(config):
    """Sub-tab 6: the Section-6 rank aggregation that integrates the ranking
    mechanisms' lists into one default list per agent - method explanation, the two
    output rules and the sub-tab's own Run button. The aggregation is always on
    (ties that no criterion can separate are always broken at random, the
    document's rule)."""
    sequences = config.get('priority_sequences', {}) or {}
    st.markdown(
        "The Loyalty, Willingness-to-Pay, Risk-Taking and Flexibility "
        "mechanisms each produce a priority list of the five options per agent. "
        "These lists do not necessarily concur, so they are reconciled into one "
        "integrated ranking (all sub-decision mechanisms receive equal weight), "
        "following two rules:"
    )
    st.markdown(
        "1. the integrated list is truncated to the Options List Length (Tendency to "
        "Plan, element 1);  \n"
        "2. every option listed after Option 5 (forgo the transaction) is dropped."
    )
    st.markdown(
        "Both rules apply to the integrated ranking only. The mechanisms' own priority "
        "lists are left as they are - the Loyalty sequence, for example, still lists "
        "Option 2 after Option 5."
    )
    st.markdown("##### Inputs (priority lists per mechanism)")
    inputs_df = pd.DataFrame([
        {'Mechanism': 'Loyalty', 'Priority sequence': ' > '.join(f"Option {o}" for o in sequences.get('loyalty', [3, 1, 4, 5, 2]))},
        {'Mechanism': 'Willingness-to-Pay', 'Priority sequence': ' > '.join(f"Option {o}" for o in sequences.get('wtp', [3, 2, 1, 4, 5]))},
        {'Mechanism': 'Risk-Taking', 'Priority sequence': ' > '.join(f"Option {o}" for o in sequences.get('risk_taking', [4, 2, 1, 3, 5]))},
        {'Mechanism': 'Flexibility', 'Priority sequence': ' > '.join(f"Option {o}" for o in sequences.get('flexibility', [2, 4, 3, 1, 5]))},
    ])
    _render_black_table(inputs_df)

    st.markdown("##### Aggregation method: Kemeny-Young with a tie-breaking hierarchy")
    st.markdown(
        "The distance between two rankings is measured by the Kendall-tau distance: the "
        "number of option pairs that the two rankings order differently. With five "
        "options there are 10 pairs; a pair (x, y) counts 1 when one ranking places x "
        "above y and the other places y above x, and 0 otherwise. The integrated ranking "
        "is the ordering of the five options whose total Kendall-tau distance to the four "
        "mechanism rankings is smallest (Kemeny-Young); it is found by checking all 120 "
        "possible orderings."
    )
    st.latex(r"\pi^{*} = \arg\min_{\pi \in S_5} \sum_{j} d_{K}(\pi, r_j)")
    st.markdown(
        "After applying the Kemeny-Young method, remaining ranking ties are resolved in "
        "two phases:  \n"
        "Phase 1 - If Kemeny returns several equally good orderings, the Schulze (2011) "
        "strongest-paths ordering is used to produce an initial ranking.  \n"
        "Phase 2 - leftover ties. After applying Kemeny and Schulze, remaining ties are "
        "resolved using Copeland (pairwise wins minus losses) and then by Spearman "
        "footrule (smallest total positional displacement). Any remaining ties are "
        "resolved by randomization which avoids any systematic bias."
    )
    st.markdown(
        "The random draw uses the agent's simulation seed, so runs are reproducible."
    )

    st.markdown("---")
    st.markdown(
        "This run presents the integrated default list results together with the "
        "tie-resolution statistics: the share of agents with initial ties after Kemeny "
        "and the stage at which the ties were settled."
    )
    render_aggregation_run_button()


def reset_rtd_to_defaults():
    """Reset all Decision 4 tab settings to research defaults (session state only -
    the coefficients and sigma constants live in config/decisions.yaml and are not
    user-editable, so no YAML write is needed)."""
    for key in [k for k in st.session_state.keys() if k.startswith('rtd_')]:
        del st.session_state[key]
    if 'rejected_transaction_tab_persistence' in st.session_state:
        del st.session_state['rejected_transaction_tab_persistence']
    initialize_rtd_session_state()
    return True


def reset_rtd_element_to_defaults(mech):
    """Reset ONLY one element's settings (intercept; for Flexibility also the
    Anchor Mix weight) to research defaults.

    Surgical version of reset_rtd_to_defaults: deletes only this element's canonical
    rtd_* keys, its rtd_tab_* widget keys, and its persistence-dict entries, then
    re-initializes so the widgets revert on rerun. Sigma is decision-wide and is
    deliberately NOT touched here (the whole-page reset covers it)."""
    keys = [f'rtd_intercept_{mech}', f'rtd_tab_intercept_{mech}']
    storage_keys = [f'rtd_intercept_{mech}']
    if mech == 'flexibility':
        keys += ['rtd_flex_observed_weight', 'rtd_tab_flex_observed_weight']
        storage_keys.append('rtd_flex_observed_weight')
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
    persistence = st.session_state.get('rejected_transaction_tab_persistence', {})
    for storage_key in storage_keys:
        persistence.pop(storage_key, None)
    initialize_rtd_session_state()
    return True


def render_rejected_transaction_defaults_tab():
    """Render the Decision 4 (Rejected Transaction Defaults) configuration tab."""
    initialize_rtd_session_state()
    config = load_rtd_config()

    st.markdown('<h3 class="section-header">Rejected Transaction Defaults Configuration</h3>',
                unsafe_allow_html=True)

    quintile_sigma_table = None
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h4 class="subsection-header">Income Specification</h4>', unsafe_allow_html=True)

        mode_options = ["Categorical only", "Continuous only", "Compare both"]
        if "rtd_tab_income_mode" not in st.session_state:
            current_mode = st.session_state.get('rtd_income_mode', 'Continuous only')
            st.session_state.rtd_tab_income_mode = current_mode if current_mode in mode_options else "Continuous only"

        def on_rtd_income_mode_change():
            st.session_state.rtd_income_mode = st.session_state.rtd_tab_income_mode
            save_to_rtd_storage('rtd_tab_income_mode', 'rtd_income_mode')

        income_val = restore_widget_from_storage(
            'rtd_tab_income_mode', st.session_state.rejected_transaction_tab_persistence,
            'rtd_income_mode', 'Continuous only')
        if income_val not in mode_options:
            income_val = next((o for o in mode_options if str(income_val).lower() == o.lower()),
                              "Continuous only")
            st.session_state.rtd_tab_income_mode = income_val

        income_mode = st.radio(
            "Income Specification for Rejected Transaction Model",
            mode_options,
            help="""
            **Categorical only**: Willingness-to-Pay and Risk-Taking use fitted
            income quintile effects (Quintiles 1-5)
            **Continuous only**: Willingness-to-Pay and Risk-Taking use the generated
            monetary income (z-scored)
            **Compare both**: Run both specifications for comparison

            Options List Length (Tendency to Plan) and Loyalty do not use income.
            """,
            key="rtd_tab_income_mode", on_change=on_rtd_income_mode_change,
        )
        st.session_state.rtd_income_mode = income_mode

    with col2:
        st.markdown('<h4 class="subsection-header">Stochastic Component</h4>', unsafe_allow_html=True)

        population_mode = st.session_state.get('population_mode', 'Copula (synthetic)')
        if population_mode == "Research Baseline":
            st.info("📊 Research Baseline always uses deterministic scores. Configure "
                    "stochastic settings for Copula / Research Specification runs.")

        st.markdown("**Copula Mode:**")
        copula_val = restore_widget_from_storage(
            'rtd_tab_sigma_in_copula', st.session_state.rejected_transaction_tab_persistence,
            'rtd_sigma_in_copula', False)
        sigma_in_copula = st.checkbox(
            "Add Normal(anchor, σ) draw to Copula runs", value=copula_val,
            help="When enabled, Copula mode will also use the stochastic component",
            key="rtd_tab_sigma_in_copula",
            on_change=lambda: save_to_rtd_storage('rtd_tab_sigma_in_copula', 'rtd_sigma_in_copula'))
        st.session_state.rtd_sigma_in_copula = sigma_in_copula

        st.markdown("**Research Specification Mode:**")
        res_val = restore_widget_from_storage(
            'rtd_tab_sigma_enabled', st.session_state.rejected_transaction_tab_persistence,
            'rtd_sigma_enabled', True)
        sigma_enabled = st.checkbox(
            "Use Normal(anchor, σ) draw in Research Specification mode", value=res_val,
            help="When enabled, adds stochastic variation via Normal(anchor, σ) draws.",
            key="rtd_tab_sigma_enabled",
            on_change=lambda: save_to_rtd_storage('rtd_tab_sigma_enabled', 'rtd_sigma_enabled'))
        st.session_state.rtd_sigma_enabled = sigma_enabled

        st.markdown("Research Baseline always uses anchor values only (deterministic).")

        # Decision-wide sigma settings: shown only when the Research Specification
        # stochastic checkbox is on; the settings apply to all elements of the decision.
        if sigma_enabled:
            quintile_sigma_table = render_decision_sigma_controls(config)
        elif sigma_in_copula:
            st.caption("Copula draws use each element's base σ (coefficient 1.0). "
                       "Check the Research Specification box to configure σ settings.")
        else:
            st.info("Stochastic component disabled - deterministic scores are used.")

    # Quintile-mode sigma table rendered outside the column so it spans the full
    # page width (base σ and effective σ per element, per budget level).
    if quintile_sigma_table is not None:
        st.markdown("**Base σ and effective σ per budget level:**")
        _render_black_table(quintile_sigma_table, full_width=True)

    # ---- Four mechanism sub-tabs + the rank-aggregation sub-tab ----
    st.markdown('<h4 class="subsection-header">Sub-Decision Mechanisms</h4>', unsafe_allow_html=True)
    sub_tabs = st.tabs([MECH_TITLES[m] for m in MECHANISMS] + [AGGREGATION_TITLE])
    for tab, mech in zip(sub_tabs[:len(MECHANISMS)], MECHANISMS):
        with tab:
            render_mechanism_subtab(config, mech)
    with sub_tabs[-1]:
        render_aggregation_subtab(config)

    # ---- Reset ----
    if st.button("Reset Decision 4 Settings to Defaults", type="secondary",
                 help="Reset all Decision 4 stochastic settings to research defaults",
                 key="rtd_reset_btn"):
        if reset_rtd_to_defaults():
            st.toast("Decision 4 settings reset to defaults", icon="🔄")
            st.rerun()

    # ---- What the whole-decision run produces (explained above its Run button) ----
    st.markdown("---")
    st.markdown("##### Running the whole decision")
    st.markdown(
        "Run Rejected Transaction Defaults Only presents every element's results - the "
        "score distribution and the option allocation of each of the five mechanisms - "
        "followed by the percentage share of each first integrated default option. The "
        "integrated list per agent is the Kemeny-Young consensus of the Loyalty, "
        "Willingness-to-Pay, Risk-Taking and Flexibility rankings, truncated to the "
        "agent's Options List Length and cut after Option 5 (forgo the transaction). "
        "The tie-resolution statistics are not part of this run; they are shown by the "
        "Run Integrated Default List Only button on the Integrated Default List "
        "sub-tab. The complete simulation presents only the default list length and the "
        "first integrated option per agent. The Excel files contain all element "
        "variables: each element's inputs, intermediate scores, segments and rankings "
        "together with the integrated default list."
    )

    # ---- Simulation buttons ----
    # The whole-decision and complete-simulation buttons must clear the per-element
    # run flag so the results page presents all four elements again. A clicked
    # button's widget state is already True at the start of the rerun its click
    # triggers - i.e. BEFORE render_simulation_buttons() executes the run below -
    # so clearing here keeps the generic decision_execution logic untouched.
    if (st.session_state.get('run_rejected_transaction_defaults_only_btn')
            or st.session_state.get('run_complete_from_rejected_transaction_defaults_btn')):
        st.session_state.rtd_run_element = None
    try:
        from app.pages.decision_execution import render_simulation_buttons
        selected_decs = getattr(st.session_state.decision_params, 'selected_decisions', [])
        render_simulation_buttons(decision_name="rejected_transaction_defaults",
                                  selected_decisions=selected_decs)
    except Exception as e:
        st.error(f"Error rendering simulation buttons: {e}")
        import traceback
        st.code(traceback.format_exc())
