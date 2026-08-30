# app/pages/decision_tabs/rejected_transaction.py
"""
Decision 4: Rejected Transaction Defaults - tab configuration.

Four trait-based sub-decision mechanisms (per the "Decision 4 - Rejected Transaction
Defaults" design document, verified against the professor's Stata file
Stata_File_Decision4_050626.dta):

  1. Options List Length (Tendency to Plan) - how many default options to pre-select (0-5)
  2. Loyalty ranking                        - priority sequence Option 3 > 1 > 4 > 5 > 2
  3. Willingness-to-Pay ranking             - priority sequence Option 3 > 2 > 1 > 4 > 5
  4. Risk-Taking ranking                    - priority sequence Option 4 > 2 > 1 > 3 > 5

Each mechanism yields its own per-agent output; rank aggregation across the mechanisms
is a later, separate step (explicitly out of scope in the source document), so no
combined default list is produced yet.

The model coefficients and sigma constants are fixed (dta-verified); the tab exposes
the income specification (categorical / continuous / compare both; WTP and
Risk-Taking are the only income-using elements) and the stochastic settings per
mechanism (sigma strategy, x0-2 coefficient, and the stochastic anchor).
Persistence follows the disclose_documents triple-layer pattern: canonical rtd_*
read keys + rtd_tab_* widget keys + a tab-persistence dict.
"""
import streamlit as st
import yaml
import pandas as pd
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "decisions.yaml"

MECHANISMS = ('ttp', 'loyalty', 'wtp', 'risk_taking')

MECH_TITLES = {
    'ttp': "1. Options List Length (Tendency to Plan)",
    'loyalty': "2. Loyalty Ranking",
    'wtp': "3. Willingness-to-Pay Ranking",
    'risk_taking': "4. Risk-Taking Ranking",
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
FALLBACK_SIGMA_OVERALL = {
    'ttp': 0.395446, 'loyalty': 0.0318293523,
    'wtp': 0.45265807275, 'risk_taking': 0.332208167,
}

# Fallback categorical-income effects (config/decisions.yaml is the source of truth).
# Level 1 (EUR 12) is the base level: intercept only; level_2..5 = EUR 32/72/128/200.
FALLBACK_CATEGORICAL_EFFECTS = {
    'wtp': {'intercept': -0.6919842, 'level_2': 0.2672136, 'level_3': 0.5058749,
            'level_4': 0.9413666, 'level_5': 1.822843},
    'risk_taking': {'intercept': -0.0068307, 'level_2': 0.0026128, 'level_3': 0.0050555,
                    'level_4': 0.0092738, 'level_5': 0.0179812},
}


def load_rtd_config():
    """Load rejected_transaction_defaults configuration from YAML."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
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
        mech_cfg = _mech_stoch_config(config, mech)
        defaults[f'rtd_anchor_{mech}'] = mech_cfg.get('anchor', 'continuous')
        defaults[f'rtd_intercept_{mech}'] = float(intercepts_cfg.get(mech, 0.0) or 0.0)

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


def _segment_mapping_df(sequence, element_name):
    """Segment -> priority list table for a ranking mechanism.

    Mirrored mapping (professor's ruling): segment s receives the tail of the priority
    sequence from position 6-s, so the HIGHEST segment gets the full list starting
    with the top option and the lowest segment gets only the last option.
    """
    rows = []
    for seg in range(1, 6):
        tail = sequence[5 - seg:]
        label = {1: f'1 (Lowest 20% of {element_name} score segment)',
                 5: f'5 (Highest 20% of {element_name} score segment)'}.get(seg, str(seg))
        rows.append({
            f'{element_name} score segment': label,
            'Priority list for rejected transaction options': ' > '.join(str(o) for o in tail),
            'Options list length': len(tail),
        })
    return pd.DataFrame(rows)


def render_formula_section(config, mech):
    """Render the mechanism's equation, coefficients, and segment mapping."""
    coeffs = (config.get('coefficients', {}) or {}).get(mech, {}) or {}
    sequences = config.get('priority_sequences', {}) or {}

    if mech == 'ttp':
        st.markdown(
            "Estimates each customer's **Tendency to Plan** based on Big-5 "
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
                 'risk_taking': 'Risk-Taking propensity'}[mech]
    if mech == 'loyalty':
        st.markdown(f"Estimates **{construct}** based on Big-5 personality traits.")
        st.latex(
            rf"Loyalty_i = \beta_1"
            rf" + {coeffs.get('extraversion', 0.09):.4f} \times z_{{Extroversion_i}}"
            rf" + {coeffs.get('openness', 0.0273):.4f} \times z_{{Openness_i}}"
            rf" + {coeffs.get('agreeable', 0.0045):.4f} \times z_{{Agreeableness_i}}"
        )
    elif mech == 'wtp':
        st.markdown(f"Estimates **{construct}** based on personality traits and income.")
        _render_income_element_formula(config, coeffs, 'wtp')
    else:
        st.markdown(f"Estimates **{construct}** based on Big-5 personality traits and income.")
        _render_income_element_formula(config, coeffs, 'risk_taking')

    seg_name = {'loyalty': 'Loyalty', 'wtp': 'WTP', 'risk_taking': 'RiskTaking'}[mech]
    st.latex(
        rf"{seg_name}15_i = \left\lfloor 1 + (5 - 0.0001) \times"
        rf" \frac{{{seg_name}_i - \min({seg_name})}}{{\max({seg_name}) - \min({seg_name})}}"
        rf" \right\rfloor \in \{{1,\dots,5\}}"
    )
    st.markdown(f"**{ELEMENT_SHORT[mech]} rejected transaction options sequence:** "
                f"{' > '.join('Option ' + str(o) for o in seq)}")
    map_col, _ = st.columns(2)
    with map_col:
        st.dataframe(_segment_mapping_df(seq, ELEMENT_SHORT[mech]), hide_index=True,
                     use_container_width=True)
    for num in range(1, 6):
        st.caption(OPTION_LABELS[num])


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
    st.dataframe(table, hide_index=True, use_container_width=False)
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
            rf" + {coeffs.get('extraversion', 0.0788796127824):.10f} \times z_{{Extroversion_i}}"
            rf" {coeffs.get('agreeable', -0.012328716):.9f} \times z_{{Agreeableness_i}}"
            rf" + {coeffs.get('income', 0.69814232):.8f} \times z_{{Income_i}}"
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
            rf" + {coeffs.get('extraversion', 0.0788796127824):.10f} \times z_{{Extroversion_i}}"
            rf" {coeffs.get('agreeable', -0.012328716):.9f} \times z_{{Agreeableness_i}}"
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
                 'wtp': 'Willingness-to-Pay', 'risk_taking': 'Risk-Taking'}


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


def render_anchor_control(mech):
    """Advanced stochastic-anchor option (loyalty / risk_taking only)."""
    with st.expander("Advanced: stochastic anchor", expanded=False):
        anchor_widget_key = f'rtd_tab_anchor_{mech}'
        anchor_storage_key = f'rtd_anchor_{mech}'
        anchor_val = restore_widget_from_storage(
            anchor_widget_key, st.session_state.rejected_transaction_tab_persistence,
            anchor_storage_key, st.session_state.get(anchor_storage_key, 'continuous'))
        anchor_val = 'binned' if str(anchor_val) == 'binned' else 'continuous'

        def on_anchor_change(m=mech):
            st.session_state[f'rtd_anchor_{m}'] = st.session_state[f'rtd_tab_anchor_{m}']
            save_to_rtd_storage(f'rtd_tab_anchor_{m}', f'rtd_anchor_{m}')

        anchor = st.radio(
            "Anchor of the Normal(anchor, σ) draw",
            options=['continuous', 'binned'],
            format_func=lambda x: ('Continuous score (default)'
                                   if x == 'continuous' else 'Binned 1-5 segment'),
            index=0 if anchor_val == 'continuous' else 1,
            key=anchor_widget_key, on_change=on_anchor_change,
        )
        st.session_state[f'rtd_anchor_{mech}'] = anchor
        st.caption(
            "'Continuous' anchors the draw on the mechanism's continuous score; "
            "'binned' anchors it on the deterministic 1-5 segment."
        )


# Intercept symbols per the Decision 4 document's notation: beta0 (TTP, doc line
# "β0 = Intercept that sets a baseline tendency to plan"), beta1 (Loyalty),
# beta2 (WTP), beta3 (Risk-Taking).
INTERCEPT_SYMBOLS = {'ttp': 'β₀', 'loyalty': 'β₁', 'wtp': 'β₂', 'risk_taking': 'β₃'}


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
                 f"{research_default:.4f}). Shifts the element's "
                 "score distribution and thereby the allocation: the segment boundaries are "
                 "fixed from the intercept-free population scores, so a nonzero intercept "
                 "moves agents across them (a negative value shifts agents toward the lower "
                 "segments, a positive value toward the higher ones, capped at the end bins).",
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
             'risk_taking': 'RiskTaking_i'}[mech]
    bins = {'ttp': 'the 0-5 options list length',
            'loyalty': 'the 1-5 Loyalty segment',
            'wtp': 'the 1-5 WTP segment',
            'risk_taking': 'the 1-5 Risk-Taking segment'}[mech]
    anchor = (f"the continuous {score} score (or the binned segment, see Advanced below)"
              if mech in ('loyalty', 'risk_taking') else f"the continuous {score} score")
    st.markdown("**Stochastic Component:**")
    st.markdown(
        f"- If stochastic enabled: `{score} ~ Normal(μ = anchor, σ)` where the anchor is "
        f"{anchor} and σ = base σ × coefficient (overall or per budget level); the drawn "
        f"values are re-rescaled over the population and re-binned into {bins}."
    )


def render_element_reset_button(mech):
    """Per-element reset: restores only this element's settings (intercept and,
    where applicable, stochastic anchor). Decision-wide σ settings are untouched."""
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
    """Render one mechanism's sub-tab: formula + stochastic explanation + intercept
    (+ anchor option) + per-element reset + per-element run."""
    render_formula_section(config, mech)
    render_stochastic_explanation(mech)
    st.markdown("---")
    render_intercept_control(config, mech)
    if mech in ('loyalty', 'risk_taking'):
        st.markdown("---")
        render_anchor_control(mech)
    st.markdown("---")
    render_element_reset_button(mech)
    st.markdown("---")
    render_element_run_button(mech)


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
    """Reset ONLY one element's settings (intercept + anchor) to research defaults.

    Surgical version of reset_rtd_to_defaults: deletes only this element's canonical
    rtd_* keys, its rtd_tab_* widget keys, and its persistence-dict entries, then
    re-initializes so the widgets revert on rerun. Sigma is decision-wide and is
    deliberately NOT touched here (the whole-page reset covers it)."""
    for key in (f'rtd_intercept_{mech}', f'rtd_anchor_{mech}',
                f'rtd_tab_intercept_{mech}', f'rtd_tab_anchor_{mech}'):
        if key in st.session_state:
            del st.session_state[key]
    persistence = st.session_state.get('rejected_transaction_tab_persistence', {})
    for storage_key in (f'rtd_intercept_{mech}', f'rtd_anchor_{mech}'):
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
        st.dataframe(quintile_sigma_table, hide_index=True, use_container_width=True)

    # ---- Four mechanism sub-tabs ----
    st.markdown('<h4 class="subsection-header">Sub-Decision Mechanisms</h4>', unsafe_allow_html=True)
    sub_tabs = st.tabs([MECH_TITLES[m] for m in MECHANISMS])
    for tab, mech in zip(sub_tabs, MECHANISMS):
        with tab:
            render_mechanism_subtab(config, mech)

    # ---- Reset ----
    if st.button("Reset Decision 4 Settings to Defaults", type="secondary",
                 help="Reset all Decision 4 stochastic settings to research defaults",
                 key="rtd_reset_btn"):
        if reset_rtd_to_defaults():
            st.toast("Decision 4 settings reset to defaults", icon="🔄")
            st.rerun()

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
