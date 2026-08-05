# app/pages/decision_tabs/rejected_transaction.py
"""
Decision 4: Rejected Transaction Defaults - tab configuration.

Four trait-based sub-decision mechanisms (per the "Decision 4 - Rejected Transaction
Defaults" design document, verified against the professor's Stata file
Stata_File_Decision4_050626.dta):

  1. List Length (Tendency to Plan)   - how many default options to pre-select (0-5)
  2. Loyalty ranking                  - priority sequence Option 3 > 1 > 4 > 5 > 2
  3. Willingness-to-Pay ranking       - priority sequence Option 3 > 2 > 1 > 4 > 5
  4. Risk-Taking ranking              - priority sequence Option 4 > 2 > 1 > 3 > 5

Each mechanism yields its own per-agent output; rank aggregation across the mechanisms
is a later, separate step (explicitly out of scope in the source document), so no
combined default list is produced yet.

The model coefficients and sigma constants are fixed (dta-verified); the tab exposes
the stochastic settings per mechanism (sigma strategy, x0-2 coefficient, and the
stochastic anchor). Persistence follows the disclose_documents triple-layer pattern:
canonical rtd_* read keys + rtd_tab_* widget keys + a tab-persistence dict.
"""
import streamlit as st
import yaml
import pandas as pd
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "decisions.yaml"

MECHANISMS = ('ttp', 'loyalty', 'wtp', 'risk_taking')

MECH_TITLES = {
    'ttp': "1. List Length (Tendency to Plan)",
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

    defaults = {
        'rtd_sigma_enabled': True,
        'rtd_sigma_in_copula': False,
    }
    for mech in MECHANISMS:
        mech_cfg = _mech_stoch_config(config, mech)
        defaults[f'rtd_sigma_strategy_{mech}'] = mech_cfg.get('sigma_strategy', 'overall')
        defaults[f'rtd_scale_factor_{mech}'] = mech_cfg.get('scale_factor', 1.0)
        defaults[f'rtd_anchor_{mech}'] = mech_cfg.get('anchor', 'continuous')

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    for mech in MECHANISMS:
        q_key = f'rtd_quintile_scale_factors_{mech}'
        if q_key not in st.session_state:
            mech_cfg = _mech_stoch_config(config, mech)
            default_scale = mech_cfg.get('scale_factor', 1.0)
            st.session_state[q_key] = mech_cfg.get('quintile_scale_factors', {
                '1': default_scale, '2': default_scale, '3': default_scale,
                '4': default_scale, '5': default_scale,
            })


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


def _segment_mapping_df(sequence):
    """Segment -> priority list table for a ranking mechanism (dta-verified tails)."""
    rows = []
    for seg in range(1, 6):
        tail = sequence[seg - 1:]
        label = {1: '1 (lowest fifth of score range)', 5: '5 (highest fifth of score range)'}.get(
            seg, str(seg))
        rows.append({
            'Score segment': label,
            'Priority list (option numbers)': ' > '.join(str(o) for o in tail),
            'List length': len(tail),
        })
    return pd.DataFrame(rows)


def render_formula_section(config, mech):
    """Render the mechanism's equation, coefficients, and segment mapping."""
    coeffs = (config.get('coefficients', {}) or {}).get(mech, {}) or {}
    sequences = config.get('priority_sequences', {}) or {}

    if mech == 'ttp':
        st.markdown(
            "Estimates each customer's **Tendency to Plan** from Big-5 traits and "
            "education, then converts it into the number of pre-selected default "
            "options (0-5)."
        )
        st.latex(
            rf"TTP_i = {coeffs.get('extraversion', -0.0152556564):.10f}\,z_{{E_i}}"
            rf" + {coeffs.get('agreeable', 0.0177638642):.10f}\,z_{{A_i}}"
            rf" + {coeffs.get('neuroticism', 0.01959):.5f}\,z_{{N_i}}"
            rf" + {coeffs.get('conscientiousness', 0.00901465):.8f}\,z_{{C_i}}"
            rf" + {coeffs.get('education', 0.0297):.4f}\,Ed_i"
        )
        st.latex(r"ttp06_i = (6 - 0.0001)\cdot\frac{TTP_i - \min(TTP)}{\max(TTP) - \min(TTP)}"
                 r"\quad\Rightarrow\quad length_i = \lfloor ttp06_i \rfloor \in \{0,\dots,5\}")
        st.caption(
            "Ed = 1 for graduate students, 0 for undergraduates. "
            "min/max are computed over the simulated population."
        )
        return

    seq = sequences.get(mech, [])
    construct = {'loyalty': 'Loyalty to the vendor', 'wtp': 'Willingness to Pay',
                 'risk_taking': 'Risk-Taking propensity'}[mech]
    if mech == 'loyalty':
        st.markdown(f"Estimates **{construct}** from Big-5 traits.")
        st.latex(
            rf"Loyalty_i = {coeffs.get('extraversion', 0.09):.4f}\,z_{{E_i}}"
            rf" + {coeffs.get('openness', 0.0273):.4f}\,z_{{O_i}}"
            rf" + {coeffs.get('agreeable', 0.0045):.4f}\,z_{{A_i}}"
        )
    elif mech == 'wtp':
        st.markdown(f"Estimates **{construct}** from traits and income.")
        st.latex(
            rf"WTP_i = {coeffs.get('extraversion', 0.0788796127824):.10f}\,z_{{E_i}}"
            rf" {coeffs.get('agreeable', -0.012328716):.9f}\,z_{{A_i}}"
            rf" + {coeffs.get('income', 0.69814232):.8f}\,z_{{I_i}}"
        )
        st.caption("z_I = standardized continuous income (population-level stats, "
                   "mirroring the disclose-decisions' continuous-income handling).")
    else:
        st.markdown(f"Estimates **{construct}** from all Big-5 traits and income.")
        st.latex(
            rf"RT_i = {coeffs.get('extraversion', 0.025942386297):.12f}\,z_{{E_i}}"
            rf" + {coeffs.get('openness', 0.023699214948):.12f}\,z_{{O_i}}"
            rf" {coeffs.get('agreeable', -0.038734315188):.12f}\,z_{{A_i}}"
            rf" {coeffs.get('conscientiousness', -0.037739440732):.12f}\,z_{{C_i}}"
            rf" {coeffs.get('neuroticism', -0.025388697852):.12f}\,z_{{N_i}}"
            rf" + {coeffs.get('income', 0.006874197106):.12f}\,z_{{I_i}}"
        )

    st.latex(r"segment_i = \left\lfloor 1 + (5 - 0.0001)\cdot"
             r"\frac{score_i - \min(score)}{\max(score) - \min(score)} \right\rfloor \in \{1,\dots,5\}")
    st.markdown(f"**Priority sequence:** {' > '.join('Option ' + str(o) for o in seq)}")
    st.dataframe(_segment_mapping_df(seq), hide_index=True, use_container_width=True)
    st.caption(
        "Segment s receives the tail of the priority sequence starting at position s: "
        "segment 1 gets all five options, segment 5 gets only the last option."
    )
    with st.expander("Option definitions", expanded=False):
        for num in range(1, 6):
            st.markdown(f"- **{OPTION_LABELS[num]}**")


def render_sigma_controls(config, mech):
    """Per-mechanism sigma controls: strategy radio + x0-2 coefficient slider(s)."""
    mech_cfg = _mech_stoch_config(config, mech)
    base_overall = float(mech_cfg.get('sigma_overall', FALLBACK_SIGMA_OVERALL[mech]))
    base_quintiles = {str(k): float(v) for k, v in (mech_cfg.get('sigma_quintile', {}) or {}).items()}

    st.markdown("**σ mode**")
    strategy_widget_key = f'rtd_tab_sigma_strategy_{mech}'
    strategy_storage_key = f'rtd_sigma_strategy_{mech}'
    current_strategy = st.session_state.get(strategy_storage_key, 'overall')

    strategy_val = restore_widget_from_storage(
        strategy_widget_key, st.session_state.rejected_transaction_tab_persistence,
        strategy_storage_key, current_strategy)
    strategy_val = 'quintile' if 'quintile' in str(strategy_val).lower() else 'overall'

    def on_strategy_change(m=mech):
        st.session_state[f'rtd_sigma_strategy_{m}'] = st.session_state[f'rtd_tab_sigma_strategy_{m}']
        save_to_rtd_storage(f'rtd_tab_sigma_strategy_{m}', f'rtd_sigma_strategy_{m}')

    sigma_strategy = st.radio(
        "Apply σ uniformly or per budget level?",
        options=['overall', 'quintile'],
        format_func=lambda x: 'Uniformly (single σ for all)' if x == 'overall' else 'Per budget level (σ per allowance group)',
        index=0 if strategy_val == 'overall' else 1,
        key=strategy_widget_key, on_change=on_strategy_change, horizontal=True,
    )
    st.session_state[f'rtd_sigma_strategy_{mech}'] = sigma_strategy

    if sigma_strategy == 'overall':
        st.markdown(f"Base σ = {base_overall:.6g}")
        coeff_widget_key = f'rtd_tab_sigma_coefficient_{mech}'
        coeff_storage_key = f'rtd_sigma_coefficient_{mech}'
        scale_fallback = st.session_state.get(f'rtd_scale_factor_{mech}', 1.0)

        coeff_val = restore_widget_from_storage(
            coeff_widget_key, st.session_state.rejected_transaction_tab_persistence,
            coeff_storage_key, scale_fallback)
        coeff_val = max(0.0, min(float(coeff_val), 2.0))

        sigma_coefficient = st.slider(
            "σ Coefficient (multiplier)", min_value=0.0, max_value=2.0, value=coeff_val, step=0.01,
            help=f"Final σ = {base_overall:.6g} × coefficient. Set to 0 to make this mechanism deterministic.",
            key=coeff_widget_key,
            on_change=lambda m=mech: save_to_rtd_storage(
                f'rtd_tab_sigma_coefficient_{m}', f'rtd_sigma_coefficient_{m}'),
        )
        st.session_state[f'rtd_scale_factor_{mech}'] = sigma_coefficient
        st.markdown(f"Effective σ = {base_overall:.6g} × {sigma_coefficient:.2f} = {base_overall * sigma_coefficient:.6g}")
    else:
        st.markdown("**Per-budget-level σ coefficients:**")
        quintile_coefficients = {}
        default_scale = st.session_state.get(f'rtd_scale_factor_{mech}', 1.0)
        current_scales = st.session_state.get(f'rtd_quintile_scale_factors_{mech}', {
            '1': default_scale, '2': default_scale, '3': default_scale,
            '4': default_scale, '5': default_scale})
        for level in ['1', '2', '3', '4', '5']:
            level_scale = max(0.0, min(float(current_scales.get(level, default_scale)), 2.0))
            storage_key = f'rtd_sigma_quintile_{level}_{mech}'
            widget_key = f'rtd_tab_sigma_q{level}_{mech}'
            q_val = restore_widget_from_storage(
                widget_key, st.session_state.rejected_transaction_tab_persistence,
                storage_key, level_scale)
            q_val = max(0.0, min(float(q_val), 2.0))
            base_sigma = base_quintiles.get(level, base_overall)

            col_slider, col_result = st.columns([3, 1])
            with col_slider:
                q_coeff = st.slider(
                    f"{LEVEL_LABELS[level]} (base σ={base_sigma:.6g})",
                    min_value=0.0, max_value=2.0, value=q_val, step=0.01,
                    key=widget_key,
                    on_change=lambda l=level, m=mech: save_to_rtd_storage(
                        f'rtd_tab_sigma_q{l}_{m}', f'rtd_sigma_quintile_{l}_{m}'),
                )
            with col_result:
                st.metric("Effective σ", f"{base_sigma * q_coeff:.6g}")
            quintile_coefficients[level] = q_coeff
        st.session_state[f'rtd_quintile_scale_factors_{mech}'] = quintile_coefficients

    # Stochastic anchor (loyalty / risk_taking only: the doc's literal text anchors the
    # draw on the binned 1-5 segment, but the only mechanism with stochastic ground
    # truth in the .dta (WTP) provably anchors on the continuous score).
    if mech in ('loyalty', 'risk_taking'):
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


def render_mechanism_subtab(config, mech, stochastic_active):
    """Render one mechanism's sub-tab: formula + sigma controls."""
    render_formula_section(config, mech)
    st.markdown("---")
    if stochastic_active:
        render_sigma_controls(config, mech)
    else:
        st.info("Stochastic component disabled for all modes - this mechanism uses the "
                "deterministic score directly.")


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


def render_rejected_transaction_defaults_tab():
    """Render the Decision 4 (Rejected Transaction Defaults) configuration tab."""
    initialize_rtd_session_state()
    config = load_rtd_config()

    st.markdown('<h3 class="section-header">Rejected Transaction Defaults Configuration</h3>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h4 class="subsection-header">Income Specification</h4>', unsafe_allow_html=True)
        st.radio(
            "Income Specification for Rejected Transaction Model",
            ["Continuous only"],
            index=0,
            key="rtd_tab_income_mode",
            help="""
            **Continuous only**: The model uses the generated monetary income (z-scored)
            in the Willingness-to-Pay and Risk-Taking equations.
            List Length (Tendency to Plan) and Loyalty do not use income.
            """,
        )
        st.session_state.rtd_income_mode = "Continuous only"

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
            "Add Normal(anchor, σ) draws to Copula runs", value=copula_val,
            key="rtd_tab_sigma_in_copula",
            on_change=lambda: save_to_rtd_storage('rtd_tab_sigma_in_copula', 'rtd_sigma_in_copula'))
        st.session_state.rtd_sigma_in_copula = sigma_in_copula

        st.markdown("**Research Specification Mode:**")
        res_val = restore_widget_from_storage(
            'rtd_tab_sigma_enabled', st.session_state.rejected_transaction_tab_persistence,
            'rtd_sigma_enabled', True)
        sigma_enabled = st.checkbox(
            "Use Normal(anchor, σ) draws in Research Specification mode", value=res_val,
            key="rtd_tab_sigma_enabled",
            on_change=lambda: save_to_rtd_storage('rtd_tab_sigma_enabled', 'rtd_sigma_enabled'))
        st.session_state.rtd_sigma_enabled = sigma_enabled

        st.markdown("Research Baseline always uses anchor values only (deterministic).")
        st.caption("Each mechanism has its own σ — configure it inside each sub-decision "
                   "tab below. Set a mechanism's σ coefficient to 0 to keep that mechanism "
                   "deterministic while others draw.")

    stochastic_active = sigma_in_copula or sigma_enabled

    # ---- Four mechanism sub-tabs ----
    st.markdown('<h4 class="subsection-header">Sub-Decision Mechanisms</h4>', unsafe_allow_html=True)
    sub_tabs = st.tabs([MECH_TITLES[m] for m in MECHANISMS])
    for tab, mech in zip(sub_tabs, MECHANISMS):
        with tab:
            render_mechanism_subtab(config, mech, stochastic_active)

    # ---- Actions ----
    st.markdown('<h4 class="subsection-header">Actions & Management</h4>', unsafe_allow_html=True)
    if st.button("Reset Decision 4 Settings to Defaults", type="secondary",
                 help="Reset all Decision 4 stochastic settings to research defaults",
                 key="rtd_reset_btn"):
        if reset_rtd_to_defaults():
            st.toast("Decision 4 settings reset to defaults", icon="🔄")
            st.rerun()

    # ---- Simulation buttons ----
    try:
        from app.pages.decision_execution import render_simulation_buttons
        selected_decs = getattr(st.session_state.decision_params, 'selected_decisions', [])
        render_simulation_buttons(decision_name="rejected_transaction_defaults",
                                  selected_decisions=selected_decs)
    except Exception as e:
        st.error(f"Error rendering simulation buttons: {e}")
        import traceback
        st.code(traceback.format_exc())
