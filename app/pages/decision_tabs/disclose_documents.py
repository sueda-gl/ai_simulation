# app/pages/decision_tabs/disclose_documents.py
"""
Disclose Documents decision tab configuration.

Decision 2: Disclose supporting documents to qualify for Discount Customer status.
Privacy-Calculus model (Bansal et al. 2016 + Dinev & Hart 2006). The model
coefficients/intercepts are fixed (verified bit-for-bit against the professor's Stata
Decision 2 file); the tab exposes the income mode, optional intercept override, and the
stochastic settings. UI mirrors the Disclose Income tab for consistency.
"""
import streamlit as st
import yaml
import pandas as pd
from pathlib import Path


CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "decisions.yaml"

# Base σ values (empirical, from the professor's constranssospeso2periods01 column)
BASE_SIGMA_OVERALL = 0.16066
BASE_SIGMAS = {
    '1': 0.1690045319,   # Level 1 (€12)
    '2': 0.1909536245,   # Level 2 (€32)
    '3': 0.1077552751,   # Level 3 (€72)
    '4': 0.1436159112,   # Level 4 (€128)
    '5': 0.1562794728,   # Level 5 (€200)
}
LEVEL_LABELS = {
    '1': 'Level 1 (€12)', '2': 'Level 2 (€32)', '3': 'Level 3 (€72)',
    '4': 'Level 4 (€128)', '5': 'Level 5 (€200)',
}
RESEARCH_DEFAULT_INTERCEPT = -0.75


def load_disclose_documents_config():
    """Load disclose_documents configuration from YAML."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('disclose_documents', {})


def save_disclose_documents_config(updates: dict):
    """Save updates to disclose_documents configuration in YAML."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        if 'disclose_documents' not in config:
            config['disclose_documents'] = {}
        for key, value in updates.items():
            if '.' in key:
                parts = key.split('.')
                target = config['disclose_documents']
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = value
            else:
                config['disclose_documents'][key] = value
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        st.error(f"Error saving configuration: {e}")
        return False


def initialize_disclose_documents_session_state():
    """Initialize session state for disclose_documents tab."""
    config = load_disclose_documents_config()
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
    if 'dd_tab_sigma_in_copula' not in st.session_state:
        st.session_state.dd_tab_sigma_in_copula = st.session_state.get('dd_sigma_in_copula', False)


def restore_widget_from_storage(widget_key, storage_dict, storage_key, default_value):
    """Restore a widget key from storage dictionary before widget renders."""
    if storage_dict and storage_key in storage_dict:
        st.session_state[widget_key] = storage_dict[storage_key]
        return storage_dict[storage_key]
    if widget_key in st.session_state:
        return st.session_state[widget_key]
    st.session_state[widget_key] = default_value
    return default_value


def save_to_dd_storage(widget_key, storage_key):
    """Save a widget value to disclose documents storage."""
    if "disclose_documents_tab_persistence" not in st.session_state:
        st.session_state.disclose_documents_tab_persistence = {}
    if widget_key in st.session_state:
        st.session_state.disclose_documents_tab_persistence[storage_key] = st.session_state[widget_key]


def render_dd_sigma_controls(mode_suffix: str):
    """Render σ strategy controls (overall vs quintile) for disclose_documents."""
    st.markdown("**σ mode**")

    strategy_widget_key = f'dd_tab_sigma_strategy_{mode_suffix}'
    strategy_storage_key = f'dd_sigma_strategy_{mode_suffix}'
    current_strategy = st.session_state.get('dd_sigma_strategy', 'overall')

    strategy_val = restore_widget_from_storage(
        strategy_widget_key, st.session_state.disclose_documents_tab_persistence,
        strategy_storage_key, current_strategy
    )
    strategy_val = 'quintile' if 'quintile' in str(strategy_val).lower() else 'overall'

    def on_strategy_change():
        st.session_state.dd_sigma_strategy = st.session_state[strategy_widget_key]
        save_to_dd_storage(strategy_widget_key, strategy_storage_key)

    sigma_strategy = st.radio(
        "Apply σ uniformly or per budget level?",
        options=['overall', 'quintile'],
        format_func=lambda x: 'Uniformly (single σ for all)' if x == 'overall' else 'Quintiles (σ per budget level)',
        index=0 if strategy_val == 'overall' else 1,
        key=strategy_widget_key, on_change=on_strategy_change, horizontal=True
    )
    st.session_state.dd_sigma_strategy = sigma_strategy

    st.markdown("---")

    if sigma_strategy == 'overall':
        st.markdown(f"Base σ = {BASE_SIGMA_OVERALL} (empirical from 280 participants)")
        coeff_widget_key = f'dd_tab_sigma_coefficient_{mode_suffix}'
        coeff_storage_key = f'dd_sigma_coefficient_{mode_suffix}'
        scale_fallback = st.session_state.get('dd_scale_factor', 1.0) or 1.0

        coeff_val = restore_widget_from_storage(
            coeff_widget_key, st.session_state.disclose_documents_tab_persistence,
            coeff_storage_key, scale_fallback
        )
        coeff_val = max(0.0, min(float(coeff_val), 2.0)) or 1.0

        sigma_coefficient = st.slider(
            "σ Coefficient (multiplier)", min_value=0.0, max_value=2.0, value=coeff_val, step=0.01,
            help=f"Coefficient to multiply the base σ. Final σ = {BASE_SIGMA_OVERALL} × coefficient",
            key=coeff_widget_key,
            on_change=lambda: save_to_dd_storage(coeff_widget_key, coeff_storage_key)
        )
        st.session_state.dd_scale_factor = sigma_coefficient
        st.markdown(f"Effective σ = {BASE_SIGMA_OVERALL} × {sigma_coefficient:.2f} = {BASE_SIGMA_OVERALL * sigma_coefficient:.4f}")
    else:
        current_income_mode = st.session_state.get('dd_income_mode', 'Categorical only')
        overall_coeff = st.session_state.get('dd_scale_factor', 1.0)
        effective_sigma = BASE_SIGMA_OVERALL * overall_coeff
        if 'continuous' in str(current_income_mode).lower() and 'compare' not in str(current_income_mode).lower():
            st.warning(
                "**Continuous mode uses overall σ.** Per-quintile σ values are based on "
                "categorical budget levels and are not applicable to the continuous income "
                f"specification. The simulation will use the overall σ "
                f"({overall_coeff:.2f} × {BASE_SIGMA_OVERALL} = {effective_sigma:.4f}) for continuous runs."
            )
        elif 'compare' in str(current_income_mode).lower():
            st.info(
                "**Note:** Per-quintile σ values will only apply to the **categorical** run. "
                f"The continuous run will use the overall σ ({overall_coeff:.2f} × {BASE_SIGMA_OVERALL} = {effective_sigma:.4f})."
            )
        st.markdown("**Per-Quintile σ Coefficients**")
        st.markdown("Each level has its own base σ from empirical data:")

        quintile_coefficients = {}
        default_scale = st.session_state.get('dd_scale_factor', 1.0)
        current_quintile_scales = st.session_state.get('dd_quintile_scale_factors', {
            '1': default_scale, '2': default_scale, '3': default_scale,
            '4': default_scale, '5': default_scale
        })
        for level in ['1', '2', '3', '4', '5']:
            level_scale = max(0.0, min(float(current_quintile_scales.get(level, default_scale)), 2.0))
            storage_key = f'dd_sigma_quintile_{level}_{mode_suffix}'
            widget_key = f'dd_tab_sigma_q{level}_{mode_suffix}'
            q_val = restore_widget_from_storage(
                widget_key, st.session_state.disclose_documents_tab_persistence, storage_key, level_scale
            )
            q_val = max(0.0, min(float(q_val), 2.0))
            base_sigma = BASE_SIGMAS[level]

            col_slider, col_result = st.columns([3, 1])
            with col_slider:
                q_coeff = st.slider(
                    f"{LEVEL_LABELS[level]} (base σ={base_sigma:.4f})",
                    min_value=0.0, max_value=2.0, value=q_val if q_val != 0.0 else 1.0, step=0.01,
                    key=widget_key,
                    on_change=lambda l=level: save_to_dd_storage(
                        f'dd_tab_sigma_q{l}_{mode_suffix}', f'dd_sigma_quintile_{l}_{mode_suffix}')
                )
            with col_result:
                st.metric("Effective σ", f"{base_sigma * q_coeff:.4f}")
            quintile_coefficients[level] = q_coeff

        st.session_state.dd_quintile_scale_factors = quintile_coefficients


def render_disclose_documents_tab():
    """Render disclose_documents specific configuration (mirrors disclose_income)."""
    initialize_disclose_documents_session_state()
    config = load_disclose_documents_config()

    # Overlay session-state intercept override so the formula display reflects it.
    if 'dd_intercept' in st.session_state:
        config['intercept'] = st.session_state.dd_intercept

    st.markdown('<h3 class="section-header">Disclose Documents Configuration</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h4 class="subsection-header">Income Specification</h4>', unsafe_allow_html=True)

        mode_options = ["Categorical only", "Continuous only", "Compare both"]
        if "dd_tab_income_mode" not in st.session_state:
            current_mode = st.session_state.get('dd_income_mode', 'Categorical only')
            st.session_state.dd_tab_income_mode = current_mode if current_mode in mode_options else "Categorical only"

        def on_dd_income_mode_change():
            st.session_state.dd_income_mode = st.session_state.dd_tab_income_mode
            save_to_dd_storage('dd_tab_income_mode', 'income_mode')

        income_val = restore_widget_from_storage(
            'dd_tab_income_mode', st.session_state.disclose_documents_tab_persistence,
            'income_mode', 'Categorical only'
        )
        if income_val not in mode_options:
            income_val = next((o for o in mode_options if str(income_val).lower() == o.lower()), "Categorical only")
            st.session_state.dd_tab_income_mode = income_val

        income_mode = st.radio(
            "Income Specification for Disclosure Model",
            mode_options,
            help="""
            **Categorical only**: Uses per-allowance-level intercepts (5 levels)
            **Continuous only**: Uses the generated monetary income (inverse-income term; fresh draw each run)
            **Compare both**: Run both specifications for comparison
            """,
            key="dd_tab_income_mode", on_change=on_dd_income_mode_change
        )
        st.session_state.dd_income_mode = income_mode

    with col2:
        st.markdown('<h4 class="subsection-header">Stochastic Component</h4>', unsafe_allow_html=True)

        population_mode = st.session_state.get('population_mode', 'Copula (synthetic)')
        if population_mode == "Research Baseline":
            st.info("📊 Research Baseline always uses anchor values only (deterministic). "
                    "Configure stochastic settings below for Copula / Research Specification runs.")

        st.markdown("**Copula Mode:**")
        copula_val = restore_widget_from_storage(
            'dd_tab_sigma_in_copula', st.session_state.disclose_documents_tab_persistence,
            'dd_sigma_in_copula', False
        )
        sigma_in_copula = st.checkbox(
            "Add Normal(score, σ) draw to Copula runs", value=copula_val,
            help="When enabled, Copula mode will also use the stochastic component",
            key="dd_tab_sigma_in_copula",
            on_change=lambda: save_to_dd_storage('dd_tab_sigma_in_copula', 'dd_sigma_in_copula')
        )
        st.session_state.dd_sigma_in_copula = sigma_in_copula

        st.markdown("**Research Specification Mode:**")
        res_val = restore_widget_from_storage(
            'dd_tab_sigma_enabled', st.session_state.disclose_documents_tab_persistence,
            'sigma_enabled', True
        )
        sigma_enabled = st.checkbox(
            "Use Normal(score, σ) draw in Research Specification mode", value=res_val,
            help="When enabled, adds stochastic variation via Normal(score, σ) draws.",
            key="dd_tab_sigma_enabled",
            on_change=lambda: save_to_dd_storage('dd_tab_sigma_enabled', 'sigma_enabled')
        )
        st.session_state.dd_sigma_enabled = sigma_enabled

        st.markdown("Research Baseline always uses anchor values only (deterministic).")

        if sigma_in_copula or sigma_enabled:
            render_dd_sigma_controls('stochastic')
        else:
            st.info("Stochastic component disabled for all modes - using deterministic score directly")

    # Mathematical Model Formula Section
    st.markdown('<h4 class="subsection-header">Mathematical Model Formula</h4>', unsafe_allow_html=True)
    render_formula_display(config)

    # Intercept Override Section
    st.markdown('<h4 class="subsection-header">Intercept Override</h4>', unsafe_allow_html=True)
    render_intercept_override_section(config)

    # Actions & Management Section
    st.markdown('<h4 class="subsection-header">Actions & Management</h4>', unsafe_allow_html=True)
    render_actions_and_management_section(config)

    # Simulation buttons
    try:
        from app.pages.decision_execution import render_simulation_buttons
        selected_decs = getattr(st.session_state.decision_params, 'selected_decisions', [])
        render_simulation_buttons(decision_name="disclose_documents", selected_decisions=selected_decs)
    except Exception as e:
        st.error(f"Error rendering simulation buttons: {e}")
        import traceback
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
        render_mediator_equations(config)

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


def render_mediator_equations(config):
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
        # €-values are the INVERSE of income (PIcat), per professor: "we use the inverse of
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


def get_current_yaml_intercept():
    """Get current intercept value from YAML config."""
    return load_disclose_documents_config().get('intercept', RESEARCH_DEFAULT_INTERCEPT)


def render_intercept_override_section(config):
    """Render the intercept override section (mirrors disclose_income)."""
    if 'dd_intercept_override_values' not in st.session_state:
        st.session_state.dd_intercept_override_values = {}

    try:
        research_default = RESEARCH_DEFAULT_INTERCEPT
        current_config_value = get_current_yaml_intercept()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Research Default: {research_default:.4f}**")
            st.markdown("Intercept (β₀)")
            st.markdown("Fixed reference value from original research")
        with col2:
            st.markdown("**Override Value**")
            int_val = restore_widget_from_storage(
                'dd_override_intercept', st.session_state.dd_intercept_override_values,
                'intercept', current_config_value
            )
            new_intercept = st.number_input(
                "Baseline disclosure tendency", min_value=-5.0, max_value=0.0,
                value=float(int_val), step=0.01, format="%.4f",
                help="β₀ = −0.75 in the disclose documents equation. Higher values increase baseline probability of disclosure.",
                key="dd_override_intercept",
                on_change=lambda: auto_save_intercept(st.session_state.dd_override_intercept)
            )
            st.session_state.dd_intercept_override_values['intercept'] = new_intercept
        with col3:
            st.markdown("**Impact Preview**")
            change = new_intercept - research_default
            if abs(change) > 0.00001:
                impact = "Higher baseline" if change > 0 else "Lower baseline"
                st.metric("Change", f"{change:+.4f}", delta=impact)
            else:
                st.metric("Change", "No change")
    except Exception as e:
        st.error(f"Error loading configuration values: {e}")


def auto_save_intercept(new_value):
    """Update intercept override in session state (YAML only written by Reset)."""
    if 'dd_intercept_override_values' not in st.session_state:
        st.session_state.dd_intercept_override_values = {}
    st.session_state.dd_intercept_override_values['intercept'] = new_value
    st.session_state.dd_intercept = new_value


def _apply_config_to_widget_keys(config):
    """Explicitly set every widget key from a config dict (defeats Streamlit's stale cache)."""
    stochastic = config.get('stochastic', {})
    st.session_state.dd_override_intercept = config.get('intercept', RESEARCH_DEFAULT_INTERCEPT)
    st.session_state.dd_tab_sigma_enabled = stochastic.get('sigma_value', 0) > 0
    st.session_state.dd_tab_sigma_coefficient_stochastic = stochastic.get('scale_factor', 1.0)
    st.session_state.dd_tab_sigma_strategy_stochastic = stochastic.get('sigma_strategy', 'overall')
    st.session_state.dd_tab_income_mode = config.get('income_mode', 'Categorical only')
    st.session_state.dd_tab_sigma_in_copula = False
    st.session_state.dd_sigma_in_copula = False

    default_scale = stochastic.get('scale_factor', 1.0)
    quintile_scales = stochastic.get('quintile_scale_factors', {})
    for level in ['1', '2', '3', '4', '5']:
        st.session_state[f'dd_tab_sigma_q{level}_stochastic'] = quintile_scales.get(level, default_scale)

    # Also set the READ-keys the simulation consumes (so reset is correct regardless of rerun timing).
    st.session_state.dd_income_mode = config.get('income_mode', 'Categorical only')
    st.session_state.dd_intercept = config.get('intercept', RESEARCH_DEFAULT_INTERCEPT)
    st.session_state.dd_sigma_enabled = stochastic.get('sigma_value', 0) > 0
    st.session_state.dd_sigma_strategy = stochastic.get('sigma_strategy', 'overall')
    st.session_state.dd_scale_factor = default_scale
    st.session_state.dd_quintile_scale_factors = {
        level: quintile_scales.get(level, default_scale) for level in ['1', '2', '3', '4', '5']
    }


def reset_to_defaults():
    """Reset all configuration values to their research defaults."""
    default_config = {
        'intercept': RESEARCH_DEFAULT_INTERCEPT,
        'income_mode': 'Categorical only',
        'stochastic.sigma_value': 0,
        'stochastic.scale_factor': 1.0,
        'stochastic.sigma_strategy': 'overall',
        'stochastic.quintile_scale_factors.1': 1.0,
        'stochastic.quintile_scale_factors.2': 1.0,
        'stochastic.quintile_scale_factors.3': 1.0,
        'stochastic.quintile_scale_factors.4': 1.0,
        'stochastic.quintile_scale_factors.5': 1.0,
    }
    success = all(save_disclose_documents_config({k: v}) for k, v in default_config.items())
    if success:
        for key in [k for k in st.session_state.keys() if k.startswith('dd_')]:
            del st.session_state[key]
        if 'disclose_documents_tab_persistence' in st.session_state:
            del st.session_state['disclose_documents_tab_persistence']
        _apply_config_to_widget_keys(load_disclose_documents_config())
    return success


def render_actions_and_management_section(config):
    """Render the combined actions and management section (mirrors disclose_income)."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔄 Reset All**")
        if st.button("Reset Config to Defaults", type="secondary", use_container_width=True,
                     help="Reset all disclose documents values to research defaults", key="dd_reset_btn"):
            if reset_to_defaults():
                st.toast("Configuration reset to defaults", icon="🔄")
                st.rerun()
            else:
                st.toast("Failed to reset configuration", icon="⚠️")
    with col2:
        st.markdown("**🔄 Reset Intercept**")
        if st.button("Reset Intercept to Default Value", type="secondary", use_container_width=True,
                     help="Reset intercept value to research default (−0.75)", key="dd_reload_btn"):
            if save_disclose_documents_config({'intercept': RESEARCH_DEFAULT_INTERCEPT}):
                st.session_state.dd_intercept = RESEARCH_DEFAULT_INTERCEPT
                if 'dd_intercept_override_values' in st.session_state:
                    st.session_state.dd_intercept_override_values['intercept'] = RESEARCH_DEFAULT_INTERCEPT
                st.toast("✅ Intercept reset to research default (−0.75)", icon="🔄")
                st.rerun()
            else:
                st.toast("❌ Failed to reset intercept", icon="⚠️")

    with st.expander("Debug: Current Session State Values", expanded=False):
        st.write("**Current values in session state:**")
        st.write(f"- Intercept (dd_intercept): {st.session_state.get('dd_intercept', 'NOT SET')}")
        st.write(f"- Income Mode (dd_income_mode): {st.session_state.get('dd_income_mode', 'NOT SET')}")
        st.write(f"- Sigma Enabled (dd_sigma_enabled): {st.session_state.get('dd_sigma_enabled', 'NOT SET')}")
        st.write(f"- Sigma in Copula (dd_sigma_in_copula): {st.session_state.get('dd_sigma_in_copula', 'NOT SET')}")
        st.write(f"- σ mode (dd_sigma_strategy): {st.session_state.get('dd_sigma_strategy', 'NOT SET')}")
        st.write(f"- σ Coefficient (dd_scale_factor): {st.session_state.get('dd_scale_factor', 'NOT SET')}")
        st.write(f"- Quintile Scale Factors: {st.session_state.get('dd_quintile_scale_factors', 'NOT SET')}")
