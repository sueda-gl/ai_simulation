# app/pages/decision_tabs/donation_default.py
"""
Donation Default decision tab configuration.
"""
import traceback

import streamlit as st
from app.state.widgets import drawn_in_previous_run, stateful
from app.state.saved_configs import clear_decision_config
from app.pages.decision_tabs._tab_state import restore, seed_if_absent
from app.pages.decision_tabs._controls import (
    LEVEL_LABELS, LEVELS, TabControls,
    render_income_mode, render_sigma_controls, render_stochastic_tick_boxes,
)
from app.state.registry import CONTROL_KEYS
import pandas as pd
from app.seam.config_repo import DEFAULT_DECISIONS_PATH, read_yaml_file
from app.seam.sentinels import donation_sigma_overall
from app.pages.decision_execution import render_simulation_buttons
from app.models import load_donation_coefficients_from_yaml


CONFIG_PATH = DEFAULT_DECISIONS_PATH  # the shared constant (src/config.py)


def _load_base_sigma_overall():
    """Read the donation base sigma from the configuration file (read-only)."""
    return float(donation_sigma_overall())


# Single sigma constant: donation_default.stochastic.sigma_overall from
# config/decisions.yaml, loaded once at import. The file is never written.
BASE_SIGMA_OVERALL = _load_base_sigma_overall()


def _load_base_sigmas():
    """{level: σ}: ``stochastic.sigma_quintile`` of the ``donation_default`` block of
    config/decisions.yaml (read-only), the way the Disclose Documents tab reads its own."""
    stochastic = read_yaml_file(CONFIG_PATH)['donation_default']['stochastic']
    return {str(level): float(sigma) for level, sigma in stochastic['sigma_quintile'].items()}


# Per-quintile base σ shown by the slider labels and the metrics, loaded once at
# import; display only (the model reads the same keys from its params).
BASE_SIGMAS = _load_base_sigmas()


def research_defaults():
    """The tab's research defaults, read from the ``donation_default`` block of
    config/decisions.yaml (read-only): the single source of truth for "Reset Config
    to Defaults" - it used to write its own hard-coded copies of these numbers.

    Not read here, so the reset sets them itself: the income mode and the two σ
    tick boxes (Copula off, Research Specification on). The σ controls (mode
    radio, coefficient and per-quintile sliders) come back at the file's
    ``stochastic`` values: the reset writes their canonical keys from here and
    deletes their widget keys, which the shared controls seed again from the
    canonical keys when they are next drawn (``_tab_state.restore``)."""
    block = read_yaml_file(CONFIG_PATH)['donation_default']
    stochastic = block.get('stochastic', {})
    default_scale = float(stochastic.get('scale_factor', 1.0))
    quintiles = stochastic.get('quintile_scale_factors') or {}
    return {
        'intercepts': {
            'categorical': float(block['regression_coefficients']['categorical']['intercept']),
            'continuous': float(block['regression_coefficients']['continuous']['intercept']),
        },
        'shift_value': float((block.get('adjustment') or {}).get('shift_value', 0.0)),
        'anchor_observed_weight': float(block['anchor_weights']['observed']),
        'sigma_strategy': stochastic.get('sigma_strategy', 'overall'),
        'scale_factor': default_scale,
        'quintile_scale_factors': {str(level): float(quintiles.get(level, quintiles.get(int(level), default_scale)))
                                   for level in ('1', '2', '3', '4', '5')},
    }


def ensure_coefficients_loaded():
    """Ensure donation coefficients are loaded from configuration file before any access"""
    if 'donation_coeff_intercept' not in st.session_state:
        load_donation_coefficients_from_yaml()


def get_coefficient(name, mode_suffix=None):
    """Get a coefficient value from session state. Configuration file is the single source of truth."""
    ensure_coefficients_loaded()
    if mode_suffix:
        key = f'donation_coeff_{name}_{mode_suffix}'
    else:
        key = f'donation_coeff_{name}'
    
    if key not in st.session_state:
        # Try to reload from configuration file
        load_donation_coefficients_from_yaml()
        if key not in st.session_state:
            st.error(f"Coefficient '{name}' not found in default configuration!")
            return 0.0
    
    return st.session_state[key]


def initialize_donation_widget_keys():
    """Initialize widget keys for donation_default tab to preserve values across navigation"""

    # Initialize canonical checkbox widget keys (one set, mode-independent)
    seed_if_absent("tab_sigma_in_copula", st.session_state.get('sigma_in_copula', False))
    seed_if_absent("tab_sigma_in_research", st.session_state.get('sigma_in_research', True))

    # (tab_sigma_coefficient_stochastic is seeded by _controls.render_sigma_controls,
    # right before the slider and only in a run that draws it: seeded here it would
    # carry 1.0 - the quintile branch's coefficient - into a key no widget draws,
    # which then wins over the stored coefficient when the slider comes back.)
    seed_if_absent("tab_anchor_weight", st.session_state.get(
        'anchor_observed_weight', research_defaults()['anchor_observed_weight']))

    # Initialize sigma strategy (overall vs quintile)
    if "donation_sigma_strategy" not in st.session_state:
        st.session_state.donation_sigma_strategy = st.session_state.get('donation_sigma_strategy', 'overall')

    # Initialize quintile scale factors
    if "donation_quintile_scale_factors" not in st.session_state:
        st.session_state.donation_quintile_scale_factors = dict(
            research_defaults()['quintile_scale_factors'])


def _after_income_mode_change():
    """After the income-mode radio changes: reload the coefficient set of the new
    mode and drop a pinned donation configuration (it was run in the old mode)."""
    reload_coefficients_for_income_mode()
    clear_decision_config('donation_default')


#: The tab's shared controls (decision_tabs._controls): keys from the registry,
#: σ constants and the tab's own wording.
DONATION_CONTROLS = TabControls(
    keys=CONTROL_KEYS['donation_default'],
    income_label="Income Mode for Donation Model",
    income_help="Choose income treatment: categorical (5 categories), continuous (linear), or Compare both",
    income_options=["categorical only", "continuous only", "Compare both"],
    income_default='categorical only',
    income_seed=None,
    after_income_change=_after_income_mode_change,
    base_sigma=BASE_SIGMA_OVERALL,
    overall_summary="{base:.2f} × coefficient",
    coefficient_fallback=1.0,
    quintile_level_fallback=1.0,
    base_sigmas=BASE_SIGMAS,
    level_labels=LEVEL_LABELS,
)


def render_donation_default_tab():
    """Render donation_default specific configuration"""
    # Ensure coefficients are loaded from configuration file
    ensure_coefficients_loaded()
    
    # Initialize widget keys to preserve values across navigation
    initialize_donation_widget_keys()
    
    # Handle full config reset flag BEFORE any widgets render
    if st.session_state.get('_reset_config_to_defaults_flag', False):
        st.session_state._reset_config_to_defaults_flag = False

        # Research defaults from config/decisions.yaml (read-only) - the same values
        # the first draw shows; nothing below is a hard-coded number.
        defaults = research_defaults()

        # Reset intercept widget keys (both, whichever one this income mode draws;
        # see _seed_intercept_input for the one it does not)
        st.session_state.override_categorical_intercept = defaults['intercepts']['categorical']
        st.session_state.override_continuous_intercept = defaults['intercepts']['continuous']
        st.session_state.intercept_override_values = {}
        
        # Reset adjustment widget key
        st.session_state.override_adjustment_shift = defaults['shift_value']
        st.session_state.adjustment_override_values = {}
        
        # Reset income mode (the app's default, a Page-2 setting the file does not carry)
        st.session_state.page2_tab_income_spec_mode = "categorical only"
        st.session_state.income_spec_mode = "categorical only"
        
        # Reset the σ controls - the mode radio, the coefficient slider and the five
        # quintile sliders - to the file's values: write their canonical keys and
        # delete their widget keys, which the shared controls seed again from the
        # canonical keys right before each control is next drawn (the Disclose
        # Income / Documents resets leave their σ widget keys to the same seeding).
        # The widget keys are not written here: the quintile sliders are hidden once
        # the σ mode is back at 'overall', and a widget key written in a run that
        # does not draw its widget resurfaces, stale, once the widget is culled.
        st.session_state.donation_sigma_strategy = defaults['sigma_strategy']
        st.session_state.donation_sigma_coefficient_setting = defaults['scale_factor']
        st.session_state.donation_quintile_scale_factors = dict(defaults['quintile_scale_factors'])
        for key in ('donation_tab_sigma_strategy_stochastic', 'tab_sigma_coefficient_stochastic'):
            if key in st.session_state:
                del st.session_state[key]
        for level in LEVELS:
            if f'donation_tab_sigma_q{level}_stochastic' in st.session_state:
                del st.session_state[f'donation_tab_sigma_q{level}_stochastic']
        
        # Reset anchor weight
        st.session_state.tab_anchor_weight = defaults['anchor_observed_weight']
        
        # Reset sigma checkboxes
        st.session_state.tab_sigma_in_copula = False
        st.session_state.tab_sigma_in_research = True
    
    st.markdown('<h3 class="section-header"> Donation Default Configuration</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        population_mode = st.session_state.population_mode  # Use value from Page 1
        
        # Income specification selector
        st.markdown('<h4 class="subsection-header">Income Specification</h4>', unsafe_allow_html=True)
        render_income_mode(DONATION_CONTROLS)
    
    with col2:
        # Stochastic component option
        st.markdown('<h4 class="subsection-header">Stochastic Component</h4>', unsafe_allow_html=True)
        
        if population_mode == "Research Baseline":
            st.info("📊 Research Baseline always uses anchor values only (deterministic). "
                    "Configure stochastic settings below for Copula / Research Specification runs.")

        sigma_in_copula, sigma_in_research = render_stochastic_tick_boxes(DONATION_CONTROLS)

        st.markdown("Research Baseline always uses anchor values only (deterministic).")

        # --- Shared sigma controls (strategy, coefficient, quintiles) ---
        if sigma_in_copula or sigma_in_research:
            sigma_strategy, sigma_coefficient = render_sigma_controls(DONATION_CONTROLS)
            if sigma_strategy == 'overall':
                # The coefficient in effect (the seam's engine input) is the slider;
                # the slider's own canonical key, donation_sigma_coefficient_setting,
                # was written by the shared controls right after it.
                st.session_state.sigma_coefficient = sigma_coefficient
                st.session_state.sigma_value_ui = BASE_SIGMA_OVERALL * sigma_coefficient
            else:
                # Set sigma_value_ui to indicate stochastic is enabled (using overall sigma as fallback display)
                st.session_state.sigma_value_ui = BASE_SIGMA_OVERALL
                st.session_state.sigma_coefficient = 1.0  # Default coefficient for overall fallback
        else:
            st.info("Stochastic component disabled for all modes - using anchor values directly")
        
        # Anchor weights
        st.markdown('<h4 class="subsection-header">Anchor Mix</h4>', unsafe_allow_html=True)

        # Seed the anchor weight from its canonical key (anchor_observed_weight)
        restore('tab_anchor_weight', research_defaults()['anchor_observed_weight'])

        anchor_observed_weight = stateful(st.slider,
            "Weight for observed vs modeled prosocial behavior",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="Anchor = w × Observed + (1-w) × Predicted",
            key="tab_anchor_weight",
        )
        st.session_state.anchor_observed_weight = anchor_observed_weight
        st.markdown(f"Predicted weight: {1 - anchor_observed_weight:.2f}")
    
    
    # Mathematical Formula Display Section
    st.markdown('<h4 class="subsection-header">📐 Mathematical Model Formula</h4>', unsafe_allow_html=True)
    render_formula_display()
    
    # Intercept Override Section
    st.markdown('<h4 class="subsection-header">🔢 Intercept Override</h4>', unsafe_allow_html=True)
    render_intercept_override_section()
    
    # Distribution Adjustment Section
    st.markdown('<h4 class="subsection-header">📊 Distribution Adjustment</h4>', unsafe_allow_html=True)
    render_adjustment_override_section()
    
    # Actions and Default Coefficient Management section - combined
    st.markdown("---")
    st.markdown('<h4 class="subsection-header">⚙️ Actions & Management</h4>', unsafe_allow_html=True)
    render_actions_and_management_section()
    
    # Render both individual and complete simulation buttons
    try:
        # Safety: Get selected_decisions with a default
        selected_decs = getattr(st.session_state.decision_params, 'selected_decisions', [])
        
        render_simulation_buttons(
            decision_name="donation_default",
            selected_decisions=selected_decs
        )
    except Exception as e:
        st.error(f"Error rendering simulation buttons: {e}")
        st.code(traceback.format_exc())


def render_formula_display():
    """Render mathematical formulas with current coefficient values"""
    
    # Get current income specification mode
    income_mode = st.session_state.get('income_spec_mode', 'categorical')
    
    with st.expander("📊 Current Model Equation", expanded=True):
        if income_mode == "categorical only":
            render_categorical_formula()
        elif income_mode == "continuous only":
            render_continuous_formula()
        else:  # Compare both
            st.markdown("**📊 Both specifications will be used for comparison:**")
            st.markdown("**Categorical Income Specification:**")
            render_categorical_formula_specific()
            st.markdown("---")
            st.markdown("**Continuous Income Specification:**")
            render_continuous_formula_specific()
    
    with st.expander("📚 Variable Definitions", expanded=False):
        render_variable_definitions()


def render_categorical_formula():
    """Render categorical income specification formula"""
    
    # The categorical coefficient set the model runs (session state, loaded from the file)
    _render_categorical_formula(
        intercept=get_coefficient('intercept', 'cat'),
        hh_coeff=get_coefficient('hh', 'cat'),
        group=[
            get_coefficient('midsub', 'cat'),
            get_coefficient('nosub', 'cat'),
            get_coefficient('fullsub', 'cat')
        ],
        study=[
            get_coefficient('incoming', 'cat'),
            get_coefficient('law', 'cat'),
            get_coefficient('ug', 'cat'),
            get_coefficient('grad', 'cat')
        ],
        income_labels=['Q1 (Level 12, ref)', 'Q2 (Level 32)', 'Q3 (Level 72)', 'Q4 (Level 128)', 'Q5 (Level 200)'],
        income=[
            get_coefficient('q1', 'cat'),
            get_coefficient('q2', 'cat'),
            get_coefficient('q3', 'cat'),
            get_coefficient('q4', 'cat'),
            get_coefficient('q5', 'cat')
        ],
    )


def render_categorical_formula_specific():
    """Render categorical income specification formula with categorical-specific coefficients"""
    
    # The categorical coefficient set the model runs (session state, loaded from the file)
    _render_categorical_formula(
        intercept=get_coefficient('intercept', 'cat'),
        hh_coeff=get_coefficient('hh', 'cat'),
        group=[
            get_coefficient('midsub', 'cat'),
            get_coefficient('nosub', 'cat'),
            get_coefficient('fullsub', 'cat')
        ],
        study=[
            get_coefficient('incoming', 'cat'),
            get_coefficient('law', 'cat'),
            get_coefficient('ug', 'cat'),
            get_coefficient('grad', 'cat')
        ],
        income_labels=['Q1 (Level 12, ref)', 'Q2 (Level 32)', 'Q3 (Level 72)', 'Q4 (Level 128)', 'Q5 (Level 200)'],
        income=[
            get_coefficient('q1', 'cat'),
            get_coefficient('q2', 'cat'),
            get_coefficient('q3', 'cat'),
            get_coefficient('q4', 'cat'),
            get_coefficient('q5', 'cat')
        ],
    )


def _render_categorical_formula(intercept, hh_coeff, group, study, income_labels, income):
    """The categorical formula display, from the coefficient values the caller
    looked up."""
    
    # Symbolic formula
    st.markdown("**📐 Symbolic Formula:**")
    st.latex(r"""
    \hat{y}_i = \beta_0 + \beta_{group}[group_i] + \beta_{income\_q}[quintile_i] + \beta_{study}[study_i] + \beta_{hh} \times HH\_raw_i
    """)
    
    # Numerical formula with current values
    st.markdown("**🔢 With Current Coefficient Values:**")
    st.latex(f"""
    \\hat{{y}}_i = {intercept:.6f} + \\beta_{{group}}[group_i] + \\beta_{{income\\_q}}[quintile_i] + \\beta_{{study}}[study_i] + {hh_coeff:.6f} \\times HH\\_raw_i
    """)
    
    # Show coefficient lookup tables
    col1, col2 = st.columns(2)
    
    with col1:
        _render_group_and_study_tables(group, study)
    
    with col2:
        st.markdown("**💰 Income Quintile Effects (β_income_q):**")
        income_data = {
            'Quintile': income_labels,
            'Intercept': income
        }
        income_df = pd.DataFrame(income_data)
        income_df['Intercept'] = income_df['Intercept'].map('{:.6f}'.format)
        st.dataframe(income_df, hide_index=True, use_container_width=True)


def _render_group_and_study_tables(group, study):
    """The β_group and β_study lookup tables (the left column of both formulas)."""
    st.markdown("**👥 Group Effects (β_group):**")
    group_data = {
        'Group': ['MidSub', 'NoSub', 'FullSub (ref)'],
        'Intercept': group
    }
    group_df = pd.DataFrame(group_data)
    group_df['Intercept'] = group_df['Intercept'].map('{:.6f}'.format)
    st.dataframe(group_df, hide_index=True, use_container_width=True)
    
    st.markdown("**🎓 Study Programme Effects (β_study):**")
    study_data = {
        'Programme': ['Incoming', 'Law5yr', 'UG3yr', 'Grad2yr (ref)'],
        'Intercept': study
    }
    study_df = pd.DataFrame(study_data)
    study_df['Intercept'] = study_df['Intercept'].map('{:.6f}'.format)
    st.dataframe(study_df, hide_index=True, use_container_width=True)


def render_continuous_formula():
    """Render continuous income specification formula"""
    
    # The continuous coefficient set the model runs (session state, loaded from the file)
    _render_continuous_formula(
        intercept=get_coefficient('intercept', 'cont'),
        hh_coeff=get_coefficient('hh', 'cont'),
        linear_coeff=get_coefficient('linear', 'cont'),
        group=[
            get_coefficient('midsub', 'cont'),
            get_coefficient('nosub', 'cont'),
            get_coefficient('fullsub', 'cont')
        ],
        study=[
            get_coefficient('incoming', 'cont'),
            get_coefficient('law', 'cont'),
            get_coefficient('ug', 'cont'),
            get_coefficient('grad', 'cont')
        ],
    )


def render_continuous_formula_specific():
    """Render continuous income specification formula with continuous-specific coefficients"""
    
    # The continuous coefficient set the model runs (session state, loaded from the file)
    _render_continuous_formula(
        intercept=get_coefficient('intercept', 'cont'),
        hh_coeff=get_coefficient('hh', 'cont'),
        linear_coeff=get_coefficient('linear', 'cont'),
        group=[
            get_coefficient('midsub', 'cont'),
            get_coefficient('nosub', 'cont'),
            get_coefficient('fullsub', 'cont')
        ],
        study=[
            get_coefficient('incoming', 'cont'),
            get_coefficient('law', 'cont'),
            get_coefficient('ug', 'cont'),
            get_coefficient('grad', 'cont')
        ],
    )


def _render_continuous_formula(intercept, hh_coeff, linear_coeff, group, study):
    """The continuous formula display, from the coefficient values the caller
    looked up."""
    
    # Symbolic formula
    st.markdown("**📐 Symbolic Formula:**")
    st.latex(r"""
    \hat{y}_i = \beta_0 + \beta_{group}[group_i] + \beta_{linear} \times income\_level_i + \beta_{study}[study_i] + \beta_{hh} \times HH\_raw_i
    """)
    
    # Numerical formula with current values
    st.markdown("**🔢 With Current Coefficient Values:**")
    st.latex(f"""
    \\hat{{y}}_i = {intercept:.6f} + \\beta_{{group}}[group_i] + {linear_coeff:.6f} \\times income\\_level_i + \\beta_{{study}}[study_i] + {hh_coeff:.6f} \\times HH\\_raw_i
    """)
    
    # Show coefficient lookup tables and linear effect
    col1, col2 = st.columns(2)
    
    with col1:
        _render_group_and_study_tables(group, study)
    
    with col2:
        st.markdown("**📈 Linear Income Effect (β_linear):**")
        st.markdown("Effect = β_linear × actual_allowance_amount")
        
        # Show linear effect for actual allowance amounts (not level numbers)
        allowance_mapping = {1: 12, 2: 32, 3: 72, 4: 128, 5: 200}
        linear_data = {
            'Income Level': ['1 (€12)', '2 (€32)', '3 (€72)', '4 (€128)', '5 (€200)'],
            'Effect': [linear_coeff * allowance_mapping[i] for i in range(1, 6)]
        }
        linear_df = pd.DataFrame(linear_data)
        linear_df['Effect'] = linear_df['Effect'].map('{:.6f}'.format)
        st.dataframe(linear_df, hide_index=True, use_container_width=True)
        
        # Show total range effect using actual allowances
        total_range = linear_coeff * (200 - 12)  # from €12 to €200
        st.metric("Total Range Effect (€12→€200)", f"{total_range:.6f}")


def render_variable_definitions():
    """Render variable definitions table"""
    
    st.markdown("**📚 Variable Definitions and Notation:**")
    
    definitions_data = {
        'Variable': [
            'ŷᵢ',
            'β₀',
            'β_group[group_i]',
            'β_income_q[quintile_i]',
            'β_linear × income_level_i',
            'β_study[study_i]',
            'β_hh',
            'HH_raw_i'
        ],
        'Definition': [
            'Predicted prosocial behavior for agent i',
            'Model intercept (baseline prediction when all other terms = 0)',
            'Experimental group effect (MidSub, NoSub vs FullSub reference)',
            'Income quintile effect (Q1, Q2, Q3 vs Q4_Q5 reference) - Categorical mode',
            'Linear income effect (coefficient × income level 1-5) - Continuous mode',
            'Study programme effect (Incoming, Law5yr, UG3yr vs Grad2yr reference)',
            'Honesty-Humility coefficient (effect per point of the raw score)',
            'Raw Honesty-Humility score (not standardized), as in the methodology document'
        ]
    }
    
    definitions_df = pd.DataFrame(definitions_data)
    st.dataframe(definitions_df, hide_index=True, use_container_width=True)
    
    st.markdown("**🔍 Key Notes:**")
    st.markdown("""
    - **Reference categories** have coefficient = 0.0 and serve as baseline for comparison
    - **Income Level Mapping (Categorical)**: 1→Q1, 2→Q2, 3→Q3, 4&5→Q4_Q5
    - **Honesty-Humility** enters as its raw score (the regression coefficient is per raw point)
    - **Final prediction** is sum of all terms: ŷᵢ = β₀ + Σ(effects)
    """)
    
    # Show processing pipeline
    with st.expander("🔄 Complete Processing Pipeline", expanded=False):
        st.markdown("""
        **Step-by-step agent processing:**
        
        1. **Trait Extraction**: Extract agent's traits from synthetic population
        2. **Regression Prediction**: Calculate ŷᵢ using formulas above  
        3. **Scaling to 0-100**: Transform both observed and predicted to 0-100 scale
        4. **Anchor Computation**: anchor = 0.75 × observed + 0.25 × predicted
        5. **Stochastic Component**: (Optional) draw ~ Normal(anchor, σ)
        6. **Truncation**: Floor negative values at 0
        7. **Final Scaling**: Divide by the population maximum of the floored draws → [0,1] donation rate
        """)
        
        st.markdown("**📊 Scaling Constants:**")
        scaling_data = {
            'Variable': [
                'Observed Prosocial (TWT+Sospeso)',
                'Predicted Prosocial (Regression)',
                'Anchor Weights'
            ],
            'Range': [
                '[0, 112] → [0, 100]',
                '280-sample [min, max] → [0, 100] (categorical [-0.5084, 8.5422]; continuous [-1.2784, 8.7549])',
                'observed: 0.75, predicted: 0.25'
            ]
        }
        scaling_df = pd.DataFrame(scaling_data)
        st.dataframe(scaling_df, hide_index=True, use_container_width=True)


def reload_coefficients_for_income_mode():
    """Reload coefficients from configuration file when income mode changes"""
    # The configuration file is read-only, so the intercept / adjustment overrides
    # live in session state; carry them across the reload.
    overrides = {
        key: st.session_state[key]
        for key in ('donation_coeff_intercept_cat',
                    'donation_coeff_intercept_cont',
                    'donation_adjustment_shift')
        if key in st.session_state
    }
    load_donation_coefficients_from_yaml()
    for key, value in overrides.items():
        st.session_state[key] = value
    # Point the unsuffixed intercept at the newly selected specification.
    active_key = f'donation_coeff_intercept_{_active_intercept_suffix()}'
    if active_key in st.session_state:
        st.session_state.donation_coeff_intercept = st.session_state[active_key]


def _seed_intercept_input(widget_key, spec_type, current):
    """Seed an intercept input's key from the typed override (else ``current``, the
    intercept in effect) when the key is absent, and when the input was not drawn
    in the previous run - it comes back after an income-mode switch, navigation or
    a reset. On a steady rerun the key is NOT written: a key written through the
    Session State API is re-sent to the browser with that run's response, and
    writing the stored override over it on every rerun (the old restore) made the
    +/- clicks that arrived while a rerun was in flight disappear (2-4 of six
    fast clicks landed).

    The second condition is what lets the resets keep writing BOTH keys, including
    the one whose input the income mode does not draw: Streamlit keeps a key written
    in a run that does not draw its widget under the user key, never refreshes that
    entry once the widget exists, and lets it resurface once the widget is culled -
    in front of a later typed override. Seeding on every return of the input (the
    run in which stateful() pushes the value to the browser anyway) makes the typed
    override win over such a leftover."""
    if widget_key not in st.session_state or not drawn_in_previous_run(widget_key):
        st.session_state[widget_key] = float(
            st.session_state.intercept_override_values.get(spec_type, current))


def render_intercept_override_section():
    """Render the intercept override section with ability to modify default coefficient values"""
    
    # Research default values (config/decisions.yaml, read-only)
    research_intercepts = research_defaults()['intercepts']
    CATEGORICAL_DEFAULT = research_intercepts['categorical']
    CONTINUOUS_DEFAULT = research_intercepts['continuous']
    
    # Handle reset flag BEFORE widgets render
    if st.session_state.get('_reset_intercept_flag', False):
        st.session_state._reset_intercept_flag = False
        st.session_state.override_categorical_intercept = CATEGORICAL_DEFAULT
        st.session_state.override_continuous_intercept = CONTINUOUS_DEFAULT
        st.session_state.intercept_override_values = {}
    
    # Initialize override values if not present
    _intercept_overrides()
    
    # Get current income mode to determine which intercepts to show
    income_mode = st.session_state.get('income_spec_mode', 'categorical only')
    
    # Show current configuration values for reference
    try:
        current_yaml_values = get_current_intercepts()

        # Use 3 columns: Categorical Current, Continuous Current, Override Values
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📋 Categorical**")
            st.metric("Current Value", f"{current_yaml_values['categorical']:.6f}")
        
        with col2:
            st.markdown("**📋 Continuous**")
            st.metric("Current Value", f"{current_yaml_values['continuous']:.6f}")
        
        with col3:
            st.markdown("**✏️ Override Values**")
            
            # Override input fields based on income mode (the keys are seeded by
            # _seed_intercept_input; the live widget value wins on a steady rerun)
            if income_mode == "Compare both":
                # Show both categorical and continuous
                
                _seed_intercept_input('override_categorical_intercept', 'categorical',
                                      current_yaml_values['categorical'])
                
                new_cat_intercept = stateful(st.number_input,
                    "Categorical",
                    step=0.001,
                    format="%.6f",
                    help="Override value for categorical income specification",
                    key="override_categorical_intercept",
                    on_change=lambda: INTERCEPT_OVERRIDE.auto_save('categorical', st.session_state.override_categorical_intercept)
                )
                st.session_state.intercept_override_values['categorical'] = new_cat_intercept
                
                _seed_intercept_input('override_continuous_intercept', 'continuous',
                                      current_yaml_values['continuous'])
                
                new_cont_intercept = stateful(st.number_input,
                    "Continuous", 
                    step=0.001,
                    format="%.6f",
                    help="Override value for continuous income specification",
                    key="override_continuous_intercept",
                    on_change=lambda: INTERCEPT_OVERRIDE.auto_save('continuous', st.session_state.override_continuous_intercept)
                )
                st.session_state.intercept_override_values['continuous'] = new_cont_intercept
                    
            elif "continuous" in income_mode.lower():
                # Show only continuous
                
                _seed_intercept_input('override_continuous_intercept', 'continuous',
                                      current_yaml_values['continuous'])
                
                new_cont_intercept = stateful(st.number_input,
                    "Continuous",
                    step=0.001,
                    format="%.6f", 
                    help="Override value for continuous income specification",
                    key="override_continuous_intercept",
                    on_change=lambda: INTERCEPT_OVERRIDE.auto_save('continuous', st.session_state.override_continuous_intercept)
                )
                st.session_state.intercept_override_values['continuous'] = new_cont_intercept
                
            else:
                # Show only categorical (default)
                
                _seed_intercept_input('override_categorical_intercept', 'categorical',
                                      current_yaml_values['categorical'])
                
                new_cat_intercept = stateful(st.number_input,
                    "Categorical",
                    step=0.001,
                    format="%.6f",
                    help="Override value for categorical income specification", 
                    key="override_categorical_intercept",
                    on_change=lambda: INTERCEPT_OVERRIDE.auto_save('categorical', st.session_state.override_categorical_intercept)
                )
                st.session_state.intercept_override_values['categorical'] = new_cat_intercept
        
        # Show impact preview
        st.markdown("---")
        if st.session_state.intercept_override_values:
            st.markdown("**📊 Impact Preview:**")
            
            impact_data = []
            for spec_type, new_value in st.session_state.intercept_override_values.items():
                current_value = current_yaml_values[spec_type]
                change = new_value - current_value
                impact_data.append({
                    'Specification': spec_type.title(),
                    'Current': f"{current_value:.6f}",
                    'New': f"{new_value:.6f}", 
                    'Change': f"{change:+.6f}",
                    'Impact': "Higher baseline" if change > 0 else "Lower baseline" if change < 0 else "No change"
                })
            
            if impact_data:
                impact_df = pd.DataFrame(impact_data)
                st.dataframe(impact_df, hide_index=True, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading configuration values: {e}")


def get_current_yaml_intercepts():
    """Get current intercept values from configuration file"""
    config = read_yaml_file(CONFIG_PATH)
    
    regression_coeffs = config['donation_default']['regression_coefficients']
    
    return {
        'categorical': regression_coeffs['categorical']['intercept'],
        'continuous': regression_coeffs['continuous']['intercept']
    }


def get_current_intercepts():
    """Get the intercept values currently in effect.

    The configuration file is read-only, so an override lives in the
    donation_coeff_intercept_cat / _cont session keys (the coefficient set the
    model runs); the file only supplies the value when no session value exists.
    """
    file_values = get_current_yaml_intercepts()
    return {
        'categorical': st.session_state.get('donation_coeff_intercept_cat', file_values['categorical']),
        'continuous': st.session_state.get('donation_coeff_intercept_cont', file_values['continuous'])
    }


def _active_intercept_suffix():
    """Coefficient suffix ('cat'/'cont') for the active income specification."""
    income_mode = str(st.session_state.get('income_spec_mode', 'categorical only'))
    if 'continuous' in income_mode.lower() and 'compare' not in income_mode.lower():
        return 'cont'
    return 'cat'


def apply_intercept_overrides(override_values):
    """Apply intercept values to session state (the configuration file is read-only)"""
    try:
        ensure_coefficients_loaded()
        
        suffixes = {'categorical': 'cat', 'continuous': 'cont'}
        active_suffix = _active_intercept_suffix()
        
        for spec_type, value in override_values.items():
            suffix = suffixes[spec_type]
            st.session_state[f'donation_coeff_intercept_{suffix}'] = float(value)
            # Mirror load_donation_coefficients_from_yaml: the unsuffixed key holds
            # the coefficient set of the active income specification.
            if suffix == active_suffix:
                st.session_state.donation_coeff_intercept = float(value)
        
        return True
        
    except Exception as e:
        st.error(f"Error applying the override: {e}")
        return False


class OverrideField:
    """One override section's session-state plumbing - the intercept and the
    adjustment sections share it: the dict of typed override values, the
    configuration file's defaults, the apply step and the auto-save toasts."""

    def __init__(self, overrides, read_defaults, apply, saved_message, failed_message, icon):
        self.overrides = overrides            # () -> the override-values dict (created if absent)
        self.read_defaults = read_defaults    # () -> {field: file value}
        self.apply = apply                    # ({field: value}) -> bool
        self.saved_message = saved_message    # (field, value) -> toast text
        self.failed_message = failed_message  # (field) -> toast text
        self.icon = icon

    def reset_to_defaults(self):
        """Restore the configuration file's values into session state"""
        try:
            research_defaults = self.read_defaults()
        except Exception as e:
            st.error(f"Error applying the reset: {e}")
            return False
        
        return self.apply(research_defaults)

    def auto_save(self, field, new_value):
        """Auto-save a typed override to session state"""
        try:
            # Update the override values
            self.overrides()[field] = new_value
            
            # Save to session state immediately (the configuration file is read-only)
            success = self.apply({field: new_value})
            
            if success:
                # Show a brief success message
                st.toast(self.saved_message(field, new_value), icon=self.icon)
            else:
                st.toast(self.failed_message(field), icon="⚠️")
                
        except Exception as e:
            st.toast(f"❌ Auto-save error: {str(e)}", icon="⚠️")


def _intercept_overrides():
    if 'intercept_override_values' not in st.session_state:
        st.session_state.intercept_override_values = {}
    return st.session_state.intercept_override_values


def _adjustment_overrides():
    if 'adjustment_override_values' not in st.session_state:
        st.session_state.adjustment_override_values = {}
    return st.session_state.adjustment_override_values


def render_adjustment_override_section():
    """Render the distribution adjustment override section"""
    
    # Research default (config/decisions.yaml, read-only)
    research_default = research_defaults()['shift_value']
    
    # Handle reset flag BEFORE widget renders
    if st.session_state.get('_reset_adjustment_flag', False):
        st.session_state._reset_adjustment_flag = False
        st.session_state.override_adjustment_shift = 0.0
        st.session_state.adjustment_override_values = {}
    
    # Initialize adjustment values if not present
    _adjustment_overrides()
    
    try:
        # Use 3 columns: Research Default, Override Value, Impact Preview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Research Default: {research_default:.3f}**")
            st.markdown("Adjustment Shift")
            st.markdown("Fixed reference value from original research")
        
        with col2:
            st.markdown("**✏️ Override Value**")
            
            # Seed the widget key only when absent (after navigation); the live widget
            # value otherwise wins, and no rerun-dependent `value=` is passed, so the
            # widget id stays stable (the +/- buttons used to jump back one step under
            # fast clicks).
            seed_if_absent('override_adjustment_shift', float(
                st.session_state.adjustment_override_values.get('shift_value', research_default)))

            # Adjustment input field
            new_adjustment = stateful(st.number_input,
                "Distribution Shift Value",
                step=0.1,
                format="%.3f",
                help="Shift the distribution up (positive) or down (negative) on 0-100 scale before stochastic component",
                key="override_adjustment_shift",
                on_change=lambda: ADJUSTMENT_OVERRIDE.auto_save('shift_value', st.session_state.override_adjustment_shift)
            )
            st.session_state.adjustment_override_values['shift_value'] = new_adjustment
        
        with col3:
            st.markdown("**📊 Impact Preview**")
            change = new_adjustment - research_default
            if abs(change) > 0.001:
                impact = "Higher donations" if change > 0 else "Lower donations"
                st.metric("Change", f"{change:+.3f}", delta=impact)
            else:
                st.metric("Change", "No change")
        
        st.markdown("💡 How it works: Positive values shift the distribution up (greater donation), negative values shift it down (smaller donation)")
        
    except Exception as e:
        st.error(f"Error loading adjustment values: {e}")


def render_actions_and_management_section():
    """Render the combined actions and management section with three columns"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔄 Reset Intercept**")
        if st.button("Reset Intercept to Default Value", type="secondary", use_container_width=True, help="Reset the intercept values to the research defaults", key="reset_intercept_btn"):
            # Reset to the research default values held in the configuration file
            success = INTERCEPT_OVERRIDE.reset_to_defaults()
            if success:
                # Set flag to reset widgets on next rerun (before widgets render)
                st.session_state._reset_intercept_flag = True
                st.toast("✅ Intercepts reset to research defaults", icon="🔄")
                st.rerun()
            else:
                st.toast("❌ Failed to reset intercepts", icon="⚠️")
    
    with col2:
        st.markdown("**🔄 Reset Adjustment**")
        if st.button("Reset Adjustment to 0", type="secondary", use_container_width=True, help="Reset the adjustment value to 0", key="reset_adjustment_btn"):
            # Reset to default value of 0.0
            success = apply_adjustment_overrides({'shift_value': 0.0})
            if success:
                # Set flag to reset widget on next rerun (before widget renders)
                st.session_state._reset_adjustment_flag = True
                st.toast("✅ Adjustment reset to 0", icon="🔄")
                st.rerun()
            else:
                st.toast("❌ Failed to reset adjustment", icon="⚠️")
    
    with col3:
        st.markdown("**🔄 Reset All**")
        if st.button("Reset Config to Defaults", type="secondary", use_container_width=True, help="Reset entire page to research defaults (intercepts, adjustment, sliders, radio buttons)", key="reset_config_btn"):
            # Reset intercepts to research defaults
            success_intercept = INTERCEPT_OVERRIDE.reset_to_defaults()
            # Reset adjustment to research default (-4.0)
            success_adjustment = ADJUSTMENT_OVERRIDE.reset_to_defaults()
            
            if success_intercept and success_adjustment:
                # Set flag to reset ALL widgets on next rerun (before widgets render)
                st.session_state._reset_config_to_defaults_flag = True
                st.toast("✅ All values reset to research defaults", icon="🔄")
                st.rerun()
            else:
                st.toast("❌ Failed to reset configuration", icon="⚠️")


def get_current_yaml_adjustment():
    """Get current adjustment values from configuration file"""
    config = read_yaml_file(CONFIG_PATH)
    
    adjustment_params = config['donation_default'].get('adjustment', {})
    
    return {
        'shift_value': float(adjustment_params.get('shift_value', 0.0))
    }


def apply_adjustment_overrides(override_values):
    """Apply adjustment values to session state (the configuration file is read-only)"""
    try:
        if 'shift_value' in override_values:
            st.session_state.donation_adjustment_shift = float(override_values['shift_value'])
        
        return True
        
    except Exception as e:
        st.error(f"Error applying the override: {e}")
        return False


INTERCEPT_OVERRIDE = OverrideField(
    overrides=_intercept_overrides,
    read_defaults=get_current_yaml_intercepts,
    apply=apply_intercept_overrides,
    saved_message=lambda intercept_type, new_value: f"✅ {intercept_type.title()} intercept auto-saved: {new_value:.6f}",
    failed_message=lambda intercept_type: f"❌ Failed to save {intercept_type} intercept",
    icon="💾",
)

ADJUSTMENT_OVERRIDE = OverrideField(
    overrides=_adjustment_overrides,
    read_defaults=get_current_yaml_adjustment,
    apply=apply_adjustment_overrides,
    saved_message=lambda adjustment_type, new_value: f"✅ {adjustment_type.replace('_', ' ').title()} auto-saved: {new_value:.3f}",
    failed_message=lambda adjustment_type: f"❌ Failed to save {adjustment_type}",
    icon="📊",
)
