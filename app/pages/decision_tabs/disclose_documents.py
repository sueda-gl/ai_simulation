# app/pages/decision_tabs/disclose_documents.py
"""
Disclose Documents decision tab configuration.

Decision 2: Disclose supporting documents to qualify for Discount Customer status.
Privacy-Calculus model (Bansal et al. 2016 + Dinev & Hart 2006). The model
coefficients/intercepts are fixed (verified bit-for-bit against the reference Stata
Decision 2 file); the tab exposes the income mode, optional intercept override, and the
stochastic settings. UI mirrors the Disclose Income tab for consistency.
"""
import traceback

import streamlit as st
from app.pages.decision_execution import render_simulation_buttons
from app.pages.decision_tabs._tab_state import seed_if_absent
from app.pages.decision_tabs._controls import (
    LEVEL_LABELS, MODE_OPTIONS, TabControls,
    render_income_mode, render_sigma_controls, render_stochastic_tick_boxes,
)
from app.pages.decision_tabs._disclosure import (
    DisclosureTab, apply_pending_reset, render_actions_and_management_section,
    render_intercept_override_section,
)
from app.state.registry import CONTROL_KEYS, DISCLOSURE_KEYS
from app.seam.config_repo import DEFAULT_DECISIONS_PATH, read_yaml_file
import pandas as pd


CONFIG_PATH = DEFAULT_DECISIONS_PATH  # the shared constant (src/config.py)

# Base σ values (empirical, from the reference Stata file's constranssospeso2periods01 column)
BASE_SIGMA_OVERALL = 0.16066
BASE_SIGMAS = {
    '1': 0.1690045319,   # Level 1 (€12)
    '2': 0.1909536245,   # Level 2 (€32)
    '3': 0.1077552751,   # Level 3 (€72)
    '4': 0.1436159112,   # Level 4 (€128)
    '5': 0.1562794728,   # Level 5 (€200)
}
RESEARCH_DEFAULT_INTERCEPT = -0.75


def load_disclose_documents_config():
    """Load disclose_documents configuration from YAML."""
    config = read_yaml_file(CONFIG_PATH)
    return config.get('disclose_documents', {})


def research_default_config():
    """The tab's research defaults: the ``disclose_documents`` block of
    config/decisions.yaml, read fresh (the file is read-only) - the single source
    of truth for both the first draw (``initialize_disclose_documents_session_state``)
    and "Reset Config to Defaults" (which used to apply its own hard-coded dict)."""
    return load_disclose_documents_config()


def initialize_disclose_documents_session_state():
    """Initialize session state for disclose_documents tab."""
    config = research_default_config()
    stochastic = config.get('stochastic', {})

    if 'disclose_documents_tab_persistence' not in st.session_state:
        st.session_state.disclose_documents_tab_persistence = {}

    defaults = {
        'dd_intercept': config.get('intercept', RESEARCH_DEFAULT_INTERCEPT),
        'dd_sigma_enabled': True,
        'dd_scale_factor': stochastic.get('scale_factor', 1.0),
        'dd_sigma_strategy': stochastic.get('sigma_strategy', 'overall'),
        'dd_income_mode': config.get('income_mode', 'Categorical only'),
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    if 'dd_quintile_scale_factors' not in st.session_state:
        default_scale = stochastic.get('scale_factor', 1.0)
        st.session_state.dd_quintile_scale_factors = stochastic.get('quintile_scale_factors', {
            '1': default_scale, '2': default_scale, '3': default_scale,
            '4': default_scale, '5': default_scale
        })

    if 'dd_sigma_in_copula' not in st.session_state:
        st.session_state.dd_sigma_in_copula = False
    seed_if_absent('dd_tab_sigma_in_copula', st.session_state.get('dd_sigma_in_copula', False))


#: The tab's shared controls (decision_tabs._controls): keys from the registry,
#: σ constants and the tab's own wording (the base σ is printed unformatted, the
#: effective σ with four decimals).
DD_CONTROLS = TabControls(
    keys=CONTROL_KEYS['disclose_documents'],
    research_storage='sigma_enabled',
    income_storage='income_mode',
    income_label="Income Specification for Disclosure Model",
    income_help="""
            **Categorical only**: Uses per-allowance-level intercepts (5 levels)
            **Continuous only**: Uses the generated monetary income (inverse-income term; fresh draw each run)
            **Compare both**: Run both specifications for comparison
            """,
    income_options=MODE_OPTIONS,
    income_default='Categorical only',
    income_seed='exact',
    draw_noun='score',
    base_sigma=BASE_SIGMA_OVERALL,
    effective_base=BASE_SIGMA_OVERALL,
    base_fmt='',
    effective_fmt='.4f',
    quintile_fmt='.4f',
    overall_summary="{coeff:.2f} × {base} = {eff:.4f}",
    base_sigmas=BASE_SIGMAS,
    level_labels=LEVEL_LABELS,
)


def render_disclose_documents_tab():
    """Render disclose_documents specific configuration (mirrors disclose_income)."""
    # Must run BEFORE any dd_* widget is instantiated in this script run.
    apply_pending_reset(DD_TAB)
    initialize_disclose_documents_session_state()
    config = load_disclose_documents_config()

    # Overlay session-state intercept override so the formula display reflects it.
    if 'dd_intercept' in st.session_state:
        config['intercept'] = st.session_state.dd_intercept

    st.markdown('<h3 class="section-header">Disclose Documents Configuration</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h4 class="subsection-header">Income Specification</h4>', unsafe_allow_html=True)

        render_income_mode(DD_CONTROLS)

    with col2:
        st.markdown('<h4 class="subsection-header">Stochastic Component</h4>', unsafe_allow_html=True)

        population_mode = st.session_state.get('population_mode', 'Copula (synthetic)')
        if population_mode == "Research Baseline":
            st.info("📊 Research Baseline always uses anchor values only (deterministic). "
                    "Configure stochastic settings below for Copula / Research Specification runs.")

        sigma_in_copula, sigma_enabled = render_stochastic_tick_boxes(DD_CONTROLS)

        st.markdown("Research Baseline always uses anchor values only (deterministic).")

        if sigma_in_copula or sigma_enabled:
            render_sigma_controls(DD_CONTROLS)
        else:
            st.info("Stochastic component disabled for all modes - using deterministic score directly")

    # Mathematical Model Formula Section
    st.markdown('<h4 class="subsection-header">Mathematical Model Formula</h4>', unsafe_allow_html=True)
    render_formula_display(config)

    # Intercept Override Section
    st.markdown('<h4 class="subsection-header">Intercept Override</h4>', unsafe_allow_html=True)
    render_intercept_override_section(DD_TAB)

    # Actions & Management Section
    st.markdown('<h4 class="subsection-header">Actions & Management</h4>', unsafe_allow_html=True)
    render_actions_and_management_section(DD_TAB)

    # Simulation buttons
    try:
        selected_decs = getattr(st.session_state.decision_params, 'selected_decisions', [])
        render_simulation_buttons(decision_name="disclose_documents", selected_decisions=selected_decs)
    except Exception as e:
        st.error(f"Error rendering simulation buttons: {e}")
        st.code(traceback.format_exc())


def render_formula_display(config):
    """Render the mathematical model formula based on selected income mode."""
    income_mode = st.session_state.get('dd_income_mode', 'Categorical only')

    # Dynamically read the discount threshold from the Main Parameters page so the
    # eligibility-gate text tracks whatever the user set there (config/simulation.yaml
    # discount_income_threshold -> st.session_state.sim_params.discount_income_threshold).
    sim_params = st.session_state.get('sim_params', None)
    discount_threshold = getattr(sim_params, 'discount_income_threshold', 12500.0) if sim_params else 12500.0

    with st.expander("Current Model Equation", expanded=True):
        st.markdown("""
        **Decision 2: Disclose Documents** applies only to qualified agents (income below the
        discount threshold) who decided to disclose income. It uses a two-stage mediation model
        with three equations:
        - **Equation 1: Privacy Concern (Mediating Variable)** based on intercept, Neuroticism and
          Agreeableness – same for both modes
        - **Equation 2: Trust (Mediating Variable)** based on intercept, Neuroticism, Extraversion,
          and Agreeableness – same for both modes
        - **Equation 3: Disclose Document (Dependent Variable)** based on β₀ intercept, Privacy
          Concern, Trust, and Personal Incentive (inverse of income = maximum income − personal
          income) – different for categorical vs continuous modes

        Output: "Y" if DD > 0 after stochastic draw, "N" otherwise.
        """)
        render_mediator_equations()

        if income_mode == "Categorical only":
            render_categorical_dd_formula(config)
        elif income_mode == "Continuous only":
            render_continuous_dd_formula(config)
        else:  # Compare both
            st.markdown("---"); st.markdown("### Categorical Income Specification")
            render_categorical_dd_formula(config)
            st.markdown("---"); st.markdown("### Continuous Income Specification")
            render_continuous_dd_formula(config)

        st.markdown("### Final Decision")
        st.markdown(f"""
        - If stochastic enabled: `dd_i ~ Normal(μ = dd_i, σ)` where σ = sd(consumed transfers+sospeso over two periods, 0–1 scaled) × coefficient (overall or per quintile)
        - **disclose_documents = "Y"** if dd_i > 0, else **"N"**
        - **Eligibility gate:** only agents who disclosed income (`disclose_income = Y`) AND have income below the discount threshold of {discount_threshold:,.0f} are considered; everyone else is **"NA"**.
        """)

    with st.expander("Variable Definitions", expanded=False):
        st.markdown("""
        | Variable | Definition |
        |----------|------------|
        | z_E | Z-scored Extraversion |
        | z_N | Z-scored Neuroticism |
        | z_A | Z-scored Agreeableness |
        | z_picont | Standardized Personal Incentive (picont = maximum income − personal income) |
        | β_PIcat_q | Income quintile effect / intercept (categorical mode, Quintiles 1-5) |
        | β₀ | Baseline disclosure tendency (default −0.75) |
        """)


def render_mediator_equations():
    """Render Equation 1 (Privacy Concern) and Equation 2 (Trust) - same for both modes."""
    st.markdown("### Equation 1: Privacy Concern (PC)")
    st.latex(
        rf"PC_i = \beta + 0.12\,z_{{N_i}} + 0.14\,z_{{A_i}}"
    )
    st.markdown("### Equation 2: Trust (T)")
    st.latex(
        rf"T_i = \beta - 0.0204\,z_{{N_i}} + 0.13\,z_{{E_i}} + 0.0762\,z_{{A_i}}"
    )


def render_categorical_dd_formula(config):
    """Render the categorical-income DD formula."""
    coeffs = config.get('equation_coefficients', {})
    bE = coeffs.get('extraversion', 0.015584630336545)
    bN = coeffs.get('neuroticism', -0.024781455105683)
    bA = coeffs.get('agreeable', -0.016923520441338)
    st.markdown("### Equation 3: Disclosure Documents (Categorical)")
    st.latex(
        rf"DiscloseDocuments_i = \beta_0 + {bE:.6f}\,z_{{E_i}} {bN:.6f}\,z_{{N_i}} {bA:.6f}\,z_{{A_i}} + \beta_{{PIcat\_q}}[quintile_i]"
    )
    icpts = config.get('categorical_intercepts', {})
    st.markdown("**PIcat = 200 if income-level = 12; 128 if = 32; 72 if = 72; 32 if = 128; 12 if = 200**")
    st.markdown("**Income Quintile Effects (β_PIcat_q):**")
    df = pd.DataFrame({
        # €-values are the INVERSE of income (PIcat), per the Decision 2 document: "we use the inverse of
        # income throughout" — so income-12 agents show as €200, income-200 as €12, etc.
        'Quintile': ['Q1 (€200)', 'Q2 (€128)', 'Q3 (€72)', 'Q4 (€32)', 'Q5 (€12)'],
        'β_PIcat_q': [
            f"{icpts.get('level_1', 0.1464773):.7f}",
            f"{icpts.get('level_2', 0.0902694):.7f}",
            f"{icpts.get('level_3', 0.0384204):.7f}",
            f"{icpts.get('level_4', -0.0522756):.7f}",
            f"{icpts.get('level_5', -0.2393718):.7f}",
        ]
    })
    st.dataframe(df, hide_index=True, use_container_width=False)
    st.markdown(
        "Each value = regression base intercept = −0.2393718 + differential income-quintile dummy."
    )
    st.markdown("β_PIcat_q: Income quintile effects based on agent's income category (Quintiles 1-5)")


def render_continuous_dd_formula(config):
    """Render the continuous-income DD formula."""
    coeffs = config.get('equation_coefficients', {})
    bE = coeffs.get('extraversion', 0.015584630336545)
    bN = coeffs.get('neuroticism', -0.024781455105683)
    bA = coeffs.get('agreeable', -0.016923520441338)
    bPI = coeffs.get('personal_incentive', 0.14735467793568)
    beta0 = config.get('intercept', RESEARCH_DEFAULT_INTERCEPT)
    st.markdown("### Equation 3: Disclosure Documents (Continuous)")
    st.latex(
        rf"DiscloseDocuments_i = [\beta_0 = {beta0:.2f}] + [{bE:.6f}\times z_{{E_i}} {bN:.6f}\times z_{{N_i}} {bA:.6f}\times z_{{A_i}} + {bPI:.6f}\times z_{{PIcont_i}}]"
    )
    st.markdown("z_PIcont = standardized Personal Incentive (picont), where picont = maximum income − personal income.")


def _apply_extra_widget_keys(config):
    """The σ coefficient and σ-mode widget keys, Disclose Documents' own part of a
    whole-tab reset (_disclosure.apply_config_to_widget_keys); both widgets are
    drawn in the run that applies the reset."""
    stochastic = config.get('stochastic', {})
    st.session_state.dd_tab_sigma_coefficient_stochastic = stochastic.get('scale_factor', 1.0)
    st.session_state.dd_tab_sigma_strategy_stochastic = stochastic.get('sigma_strategy', 'overall')


#: The intercept override, the resets and the action buttons (decision_tabs._disclosure).
DD_TAB = DisclosureTab(
    keys=DISCLOSURE_KEYS['disclose_documents'],
    decision_label='disclose documents',
    load_config=research_default_config,
    default_intercept=RESEARCH_DEFAULT_INTERCEPT,
    research_default_label='−0.75',
    intercept_min=-5.0,
    intercept_max=0.0,
    intercept_help="β₀ = −0.75 in the disclose documents equation. Higher values increase baseline probability of disclosure.",
    apply_extra_widget_keys=_apply_extra_widget_keys,
)
