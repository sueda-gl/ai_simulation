# app/pages/decision_tabs/disclose_income.py
"""
Disclose Income decision tab configuration.

Decision 1: Disclose income for Fixed status at time of registration/review.
Uses a two-stage mediation model when specified (research spec mode).
"""
import traceback

import streamlit as st
from app.state.widgets import stateful
from app.state.saved_configs import clear_decision_config
from app.pages.decision_execution import render_simulation_buttons
from app.pages.decision_tabs._tab_state import restore, seed_if_absent
from app.pages.decision_tabs._controls import (
    EMPIRICAL_BASE_SIGMAS, LEVEL_LABELS, MODE_OPTIONS, TabControls,
    render_income_mode, render_sigma_controls, render_stochastic_tick_boxes,
)
from app.pages.decision_tabs._disclosure import (
    DisclosureTab, apply_pending_reset, render_actions_and_management_section,
    render_intercept_override_section,
)
from app.state.registry import CONTROL_KEYS, DISCLOSURE_KEYS
from app.seam.config_repo import DEFAULT_DECISIONS_PATH, read_yaml_file, get_config_repo
import pandas as pd


CONFIG_PATH = DEFAULT_DECISIONS_PATH  # the shared constant (src/config.py)


def load_disclose_income_config():
    """Load disclose_income configuration from YAML."""
    config = read_yaml_file(CONFIG_PATH)
    return config.get('disclose_income', {})


def _load_base_sigma_overall():
    """Read the base sigma from the configuration file (read-only)."""
    return float(get_config_repo().disclose_income_sigma_overall())


# Single sigma constant: disclose_income.stochastic.sigma_overall from
# config/decisions.yaml, loaded once at import. The file is never written.
BASE_SIGMA_OVERALL = _load_base_sigma_overall()


def research_default_config():
    """The tab's research defaults: the ``disclose_income`` block of
    config/decisions.yaml, read fresh (the file is read-only).

    The single source of truth for both the first draw
    (``initialize_disclose_income_session_state``) and "Reset Config to Defaults"
    - e.g. the σ coefficient ``stochastic.scale_factor: 0.1`` (the intended default).
    The reset used to apply a hard-coded dict on top of the file, which set the σ
    coefficient to 1.0 while the first draw showed 0.1.
    """
    return load_disclose_income_config()


def initialize_disclose_income_session_state():
    """Initialize session state for disclose_income tab."""
    config = research_default_config()

    # Initialize model parameters from config
    anchor_weights = config['anchor_weights']
    stochastic = config.get('stochastic', {})

    defaults = {
        'di_intercept': config['intercept'],
        'di_wopb': anchor_weights['observed_prosocial'],
        'di_wpb': anchor_weights['prosocial_weight'],
        'di_sigma_enabled': True,
        'di_scale_factor': stochastic.get('scale_factor', 1.0),
        'di_sigma_strategy': stochastic.get('sigma_strategy', 'overall'),
        'di_income_mode': config.get('income_mode', 'categorical'),
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Initialize quintile scale factors
    if 'di_quintile_scale_factors' not in st.session_state:
        default_scale = stochastic.get('scale_factor', 1.0)
        quintile_scales = stochastic.get('quintile_scale_factors', {
            '1': default_scale, '2': default_scale, '3': default_scale,
            '4': default_scale, '5': default_scale
        })
        st.session_state.di_quintile_scale_factors = quintile_scales

    # Initialize copula stochastic key
    if 'di_sigma_in_copula' not in st.session_state:
        st.session_state.di_sigma_in_copula = False

    # Initialize canonical widget key for copula stochastic checkbox
    seed_if_absent('di_tab_sigma_in_copula', st.session_state.get('di_sigma_in_copula', False))


#: The tab's shared controls (decision_tabs._controls): keys from the registry,
#: σ constants and the tab's own wording.
DI_CONTROLS = TabControls(
    keys=CONTROL_KEYS['disclose_income'],
    income_label="Income Specification for Disclosure Model",
    income_help="""
            **Categorical only**: Uses level-specific intercepts based on income categories (5 levels)
            **Continuous only**: Uses single β₀ with income coefficient based on actual income
            **Compare both**: Run both specifications for comparison
            """,
    income_options=MODE_OPTIONS,
    income_default='Categorical only',
    income_seed='match',
    after_income_change=lambda: clear_decision_config('disclose_income'),
    base_sigma=BASE_SIGMA_OVERALL,
    base_sigmas=EMPIRICAL_BASE_SIGMAS,
    level_labels=LEVEL_LABELS,
)


def render_disclose_income_tab():
    """Render disclose_income specific configuration."""
    # Must run BEFORE any di_* widget is instantiated in this script run.
    apply_pending_reset(DI_TAB)
    initialize_disclose_income_session_state()
    config = load_disclose_income_config()

    # Overlay the session-state intercept so the formula display shows the value
    # in effect (the configuration file is read-only).
    if 'di_intercept' in st.session_state:
        config['intercept'] = st.session_state.di_intercept

    st.markdown('<h3 class="section-header">Disclose Income Configuration</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Income Specification
        st.markdown('<h4 class="subsection-header">Income Specification</h4>', unsafe_allow_html=True)

        render_income_mode(DI_CONTROLS)

    with col2:
        # Stochastic Component
        st.markdown('<h4 class="subsection-header">Stochastic Component</h4>', unsafe_allow_html=True)

        population_mode = st.session_state.get('population_mode', 'Copula (synthetic)')

        if population_mode == "Research Baseline":
            st.info("📊 Research Baseline always uses anchor values only (deterministic). "
                    "Configure stochastic settings below for Copula / Research Specification runs.")

        sigma_in_copula, sigma_enabled = render_stochastic_tick_boxes(DI_CONTROLS)

        st.markdown("Research Baseline always uses anchor values only (deterministic).")

        # --- Shared sigma controls (strategy, coefficient, quintiles) ---
        if sigma_in_copula or sigma_enabled:
            render_sigma_controls(DI_CONTROLS)
        else:
            st.info("Stochastic component disabled for all modes - using anchor values directly")

        # Anchor Mix
        st.markdown('<h4 class="subsection-header">Anchor Mix</h4>', unsafe_allow_html=True)

        anchor_weights = config['anchor_weights']
        current_wopb = anchor_weights['observed_prosocial']
        current_wpb = anchor_weights['prosocial_weight']

        # Seed WOPB from its canonical key (di_wopb) when the slider comes back
        restore('di_wopb_widget', current_wopb)

        # WOPB - Weight for observed vs calculated prosocial behavior
        new_wopb = stateful(st.slider,
            "W_OPB: Observed vs Calculated prosocial behavior weight",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="anchored_PB = WOPB×Observed_PB + (1-WOPB×Calculated_PB based on Honesty-Humility, Agreeableness, Openness, and Religiosity); Default: 0.25",
            key="di_wopb_widget",
        )
        st.markdown(f"Observed prosocial behavior weight: {new_wopb:.2f}")
        st.session_state.di_wopb = new_wopb

        # Seed WPB from its canonical key (di_wpb) when the slider comes back
        restore('di_wpb_widget', current_wpb)

        # WPB - Weight for prosocial effect in disclosure equation
        new_wpb = stateful(st.slider,
            "W_PB: Prosocial behavior (Equation 1) effect weight",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="DI = β0+(1-WPB)×(Income, Extroversion, Neuroticism, Honesty-Humility, and Agreeableness) + WPB×(Prosocial Behavior×Income_High); Default: 0.5",
            key="di_wpb_widget",
        )
        st.markdown(f"Prosocial behavior weight: {new_wpb:.2f}")
        st.session_state.di_wpb = new_wpb

    # Mathematical Model Formula Section
    st.markdown('<h4 class="subsection-header">Mathematical Model Formula</h4>', unsafe_allow_html=True)
    render_formula_display(config)

    # Intercept Override Section
    st.markdown('<h4 class="subsection-header">Intercept Override</h4>', unsafe_allow_html=True)
    render_intercept_override_section(DI_TAB)

    # Actions & Management Section
    st.markdown('<h4 class="subsection-header">Actions & Management</h4>', unsafe_allow_html=True)
    render_actions_and_management_section(DI_TAB)

    # Simulation buttons - same as donation_default
    try:
        # Safety: Get selected_decisions with a default
        selected_decs = getattr(st.session_state.decision_params, 'selected_decisions', [])
        
        render_simulation_buttons(
            decision_name="disclose_income",
            selected_decisions=selected_decs
        )
    except Exception as e:
        st.error(f"Error rendering simulation buttons: {e}")
        st.code(traceback.format_exc())


def render_formula_display(config):
    """Render the mathematical model formula based on selected income mode."""
    # Get current income mode
    income_mode = st.session_state.get('di_income_mode', 'Categorical only')
    
    with st.expander("Current Model Equation", expanded=True):
        # Show mode-specific description
        if income_mode == "Categorical only":
            st.markdown("""
            **Decision 1: Disclose Income** uses a two-stage mediation model:
            - **Equation 1**: Calculated Prosocial Behavior (Calculated_PB) based on Honesty-Humility, Agreeableness, Openness, and Religiosity – same for both modes
            - **Equation 2**: Disclose Income (DI) based on β₀ intercept, PB, Income, Extroversion, Neuroticism, Honesty-Humility, and Agreeableness - different for categorical vs continuous modes
            
            Output: "Y" if DI > 0 after stochastic draw, "N" otherwise.
            """)
            render_equation1_and_combining(config)
            render_categorical_di_formula(config)
            
        elif income_mode == "Continuous only":
            st.markdown("""
            **Decision 1: Disclose Income** uses a two-stage mediation model:
            - **Equation 1**: Calculated Prosocial Behavior (Calculated_PB) based on Honesty-Humility, Agreeableness, Openness, and Religiosity – same for both modes
            - **Equation 2**: Disclose Income (DI) based on β₀ intercept, PB, Income, Extroversion, Neuroticism, Honesty-Humility, and Agreeableness - different for categorical vs continuous modes
            
            Output: "Y" if DI > 0 after stochastic draw, "N" otherwise.
            """)
            render_equation1_and_combining(config)
            render_continuous_di_formula(config)
            
        else:  # Compare both
            st.markdown("""
            **Decision 1: Disclose Income** uses a two-stage mediation model:
            - **Equation 1**: Calculated Prosocial Behavior (Calculated_PB) based on Honesty-Humility, Agreeableness, Openness, and Religiosity – same for both modes
            - **Equation 2**: Disclose Income (DI) based on β₀ intercept, PB, Income, Extroversion, Neuroticism, Honesty-Humility, and Agreeableness - different for categorical vs continuous modes
            
            Output: "Y" if DI > 0 after stochastic draw, "N" otherwise.
            """)
            render_equation1_and_combining(config)
            
            st.markdown("---")
            st.markdown("### Categorical Income Specification")
            render_categorical_di_formula(config)
            
            st.markdown("---")
            st.markdown("### Continuous Income Specification")
            render_continuous_di_formula(config)

        # Final decision (same for all modes)
        st.markdown("### Final Decision")
        st.markdown("""
        - If stochastic enabled: `PB_i ~ Normal(μ = Anchor_i, σ)` where σ = sd(TWT+Sospeso) × coefficient (overall or per quintile)
        - The stochastic PB_i is then z-scored and used in the DI equation above
        - **disclose_income = "Y"** if DI_i > 0, else **"N"**
        """)

    with st.expander("Variable Definitions", expanded=False):
        render_variable_definitions(income_mode)


def render_equation1_and_combining(config):
    """Render Equation 1 (Prosocial Behavior) and Combining section - same for all modes."""
    st.markdown("### Equation 1: Calculated Prosocial Behavior (calc_PB_i)")
    st.latex(r"""
    calc\_PB_i = 0.023776 \times z_{Agreeable_i} + 0.016537 \times z_{Openness_i} + 0.0295482 \times z_{HH_i} + 0.0677157 \times z_{Religious_i}
    """)

    st.markdown("### Combining with Observed Behavior")
    wopb = config['anchor_weights']['observed_prosocial']
    st.latex(f"""
    PB_i = {wopb:.2f} \\times z_{{obs\\_PB_i}} + {1-wopb:.2f} \\times z_{{calc\\_PB_i}}
    """)
    st.markdown("Note: Both observed and calculated PB are standardized (z-scored) before combining.")


def render_categorical_di_formula(config):
    """Render Equation 2 for CATEGORICAL income mode."""
    st.markdown("### Equation 2: Disclose Income (Categorical)")
    
    wpb = config['anchor_weights']['prosocial_weight']
    
    # Show full expanded formula (per the Decision 1 document - no separate "direct effects" line)
    st.latex(f"""
    DiscloseIncome_i = \\beta_{{income\\_q}}[quintile_i] + [1 - W_{{PB}} = {1-wpb:.2f}] \\times [0.00680238 \\times z_{{E_i}} + 0.0173732 \\times z_{{N_i}} + 0.0163905 \\times z_{{HH_i}}] + [W_{{PB}} = {wpb:.2f}] \\times (PB_i \\times I_{{high}})
    """)
    
    # Show level-specific intercepts table
    st.markdown("**Income Quintile Effects (β_income_q):**")
    intercept_data = {
        'Quintile': ['Q1 (€12)', 'Q2 (€32)', 'Q3 (€72)', 'Q4 (€128)', 'Q5 (€200)'],
        'β_income_q': [
            '0.0089007',
            '0.0055352',
            '0.0023109',
            '-0.0032216',
            '-0.0145324'
        ]
    }
    intercept_df = pd.DataFrame(intercept_data)
    st.dataframe(intercept_df, hide_index=True, use_container_width=True)
    
    st.markdown("β_income_q: Income quintile effects based on agent's income category (Quintiles 1-5)")


def render_continuous_di_formula(config):
    """Render Equation 2 for CONTINUOUS income mode."""
    st.markdown("### Equation 2: Disclosure Intention (Continuous)")
    
    wpb = config['anchor_weights']['prosocial_weight']
    beta0 = config['intercept']
    
    # Show full expanded formula only (per the Decision 1 document)
    st.latex(f"""
    DiscloseIncome_i = [\\beta_0 = {beta0}] + [1 - W_{{PB}} = {1-wpb:.2f}] \\times [0.00680238 \\times z_{{E_i}} + 0.0173732 \\times z_{{N_i}} + 0.0163905 \\times z_{{HH_i}} - 0.008988 \\times z_{{I_i}}] + [W_{{PB}} = {wpb:.2f}] \\times (PB_i \\times I_{{high}})
    """)
    st.markdown("z_I: Z-scored actual income of the agent (continuous)")


def render_variable_definitions(income_mode):
    """Render variable definitions based on income mode."""
    st.markdown("""
    | Variable | Definition |
    |----------|------------|
    | z_Agreeable | Z-scored Agreeableness |
    | z_Openness | Z-scored Openness to Experience |
    | z_HH | Z-scored Honesty-Humility |
    | z_Religious | Z-scored religiosity composite |
    | z_E | Z-scored Extraversion |
    | z_N | Z-scored Neuroticism |
    | obs_PB | Observed prosocial behavior (TWT+Sospeso) |
    """)
    
    # Mode-specific definitions
    if income_mode == "Continuous only":
        st.markdown("""
        | z_I | Z-scored actual income of the agent (continuous) |
        """)
    elif income_mode == "Categorical only":
        st.markdown("""
        | β_income_q | Quintile-specific intercept based on income category (5 quintiles: €12, €32, €72, €128, €200) |
        """)
    else:  # Compare both
        st.markdown("""
        **Categorical mode:**
        | β_income_q | Quintile-specific intercept based on income category (5 quintiles: €12, €32, €72, €128, €200) |
        
        **Continuous mode:**
        | z_I | Z-scored actual income of the agent (continuous) |
        """)


def _apply_extra_widget_keys(config):
    """The anchor-mix sliders and their canonical keys, Disclose Income's own
    part of a whole-tab reset (_disclosure.apply_config_to_widget_keys).

    NOTE: the sigma coefficient / strategy / quintile sliders are NOT set here.
    Their live keys carry the `_stochastic` suffix
    (`di_tab_sigma_coefficient_stochastic`, `di_tab_sigma_strategy_stochastic`,
    `di_tab_sigma_q{1..5}_stochastic`); the reset clears them with its bulk delete
    of every `di_` key instead, and the σ controls seed them again from the
    canonical keys. The unsuffixed writes that used to stand here set keys no
    widget and no reader ever used, so they are gone.
    """
    anchor = config['anchor_weights']
    st.session_state.di_wopb_widget = anchor['observed_prosocial']
    st.session_state.di_wpb_widget = anchor['prosocial_weight']
    st.session_state.di_wopb = anchor['observed_prosocial']
    st.session_state.di_wpb = anchor['prosocial_weight']


#: The intercept override, the resets and the action buttons (decision_tabs._disclosure).
DI_TAB = DisclosureTab(
    keys=DISCLOSURE_KEYS['disclose_income'],
    decision_label='disclose income',
    load_config=research_default_config,
    default_intercept=float(research_default_config()['intercept']),
    research_default_label='0.75',
    intercept_min=0.0,
    intercept_max=5.0,
    intercept_help="β₀ = 0.75 in the disclose income equation. Override value, with higher values increasing baseline probability of disclosure.",
    apply_extra_widget_keys=_apply_extra_widget_keys,
)
