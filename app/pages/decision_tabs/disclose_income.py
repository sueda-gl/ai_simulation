# app/pages/decision_tabs/disclose_income.py
"""
Disclose Income decision tab configuration.

Decision 1: Disclose income for Fixed status at time of registration/review.
Uses a two-stage mediation model when specified (research spec mode).
"""
import streamlit as st
import yaml
import pandas as pd
from pathlib import Path


CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "decisions.yaml"


def load_disclose_income_config():
    """Load disclose_income configuration from YAML."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('disclose_income', {})


def save_disclose_income_config(updates: dict):
    """Save updates to disclose_income configuration in YAML."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        # Update disclose_income section
        if 'disclose_income' not in config:
            config['disclose_income'] = {}

        for key, value in updates.items():
            if '.' in key:
                # Handle nested keys like 'anchor_weights.observed_prosocial'
                parts = key.split('.')
                target = config['disclose_income']
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = value
            else:
                config['disclose_income'][key] = value

        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return True
    except Exception as e:
        st.error(f"Error saving configuration: {e}")
        return False


def initialize_disclose_income_session_state():
    """Initialize session state for disclose_income tab."""
    config = load_disclose_income_config()

    # Initialize storage for persistence
    if 'disclose_income_tab_persistence' not in st.session_state:
        st.session_state.disclose_income_tab_persistence = {}

    # Initialize model parameters from config
    anchor_weights = config.get('anchor_weights', {})
    stochastic = config.get('stochastic', {})

    defaults = {
        'di_intercept': config.get('intercept', 0.75),
        'di_wopb': anchor_weights.get('observed_prosocial', 0.25),
        'di_wpb': anchor_weights.get('prosocial_weight', 0.50),
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
    if 'di_tab_sigma_in_copula' not in st.session_state:
        st.session_state.di_tab_sigma_in_copula = st.session_state.get('di_sigma_in_copula', False)


def restore_widget_from_storage(widget_key, storage_dict, storage_key, default_value):
    """Restore a widget key from storage dictionary before widget renders."""
    # Priority 1: Use value from storage dict
    if storage_dict and storage_key in storage_dict:
        val = storage_dict[storage_key]
        st.session_state[widget_key] = val
        return val

    # Priority 2: Use existing session state value if present
    if widget_key in st.session_state:
        return st.session_state[widget_key]

    # Priority 3: Use default value
    st.session_state[widget_key] = default_value
    return default_value


def save_to_disclose_income_storage(widget_key, storage_key):
    """Save a widget value to disclose income storage."""
    if "disclose_income_tab_persistence" not in st.session_state:
        st.session_state.disclose_income_tab_persistence = {}

    if widget_key in st.session_state:
        st.session_state.disclose_income_tab_persistence[storage_key] = st.session_state[widget_key]


def render_di_sigma_controls(mode_suffix: str):
    """
    Render sigma strategy controls (overall vs quintile) for disclose_income.

    Args:
        mode_suffix: One of 'copula', 'research', or 'compare' to distinguish widget keys
    """
    # Base sigma values per quintile (from empirical data)
    BASE_SIGMAS = {
        '1': 5.705052,   # Level 1 (€12)
        '2': 3.069326,   # Level 2 (€32)
        '3': 3.532226,   # Level 3 (€72)
        '4': 12.219622,  # Level 4 (€128)
        '5': 16.854622,  # Level 5 (€200)
    }

    LEVEL_LABELS = {
        '1': 'Level 1 (€12)',
        '2': 'Level 2 (€32)',
        '3': 'Level 3 (€72)',
        '4': 'Level 4 (€128)',
        '5': 'Level 5 (€200)',
    }

    # Strategy selection: Overall vs Quintiles
    st.markdown("**σ mode**")

    strategy_widget_key = f'di_tab_sigma_strategy_{mode_suffix}'
    strategy_storage_key = f'di_sigma_strategy_{mode_suffix}'
    current_strategy = st.session_state.get('di_sigma_strategy', 'overall')

    strategy_val = restore_widget_from_storage(
        strategy_widget_key,
        st.session_state.disclose_income_tab_persistence,
        strategy_storage_key,
        current_strategy
    )

    # Normalize strategy value
    if 'quintile' in str(strategy_val).lower():
        strategy_val = 'quintile'
    else:
        strategy_val = 'overall'

    def on_strategy_change():
        """Handle sigma strategy changes."""
        new_strategy = st.session_state[strategy_widget_key]
        save_to_disclose_income_storage(strategy_widget_key, strategy_storage_key)
        st.session_state.di_sigma_strategy = new_strategy

    sigma_strategy = st.radio(
        "Apply σ uniformly or per budget level?",
        options=['overall', 'quintile'],
        format_func=lambda x: 'Uniformly (single σ for all)' if x == 'overall' else 'Quintiles (σ per budget level)',
        index=0 if strategy_val == 'overall' else 1,
        key=strategy_widget_key,
        on_change=on_strategy_change,
        horizontal=True
    )
    st.session_state.di_sigma_strategy = sigma_strategy

    st.markdown("---")

    if sigma_strategy == 'overall':
        # OVERALL MODE: Single slider
        st.markdown("Base σ = 9.8995 (empirical from 280 participants)")

        coeff_widget_key = f'di_tab_sigma_coefficient_{mode_suffix}'
        coeff_storage_key = f'di_sigma_coefficient_{mode_suffix}'

        # Fallback to 1.0 (the true default coefficient), never 0.0.
        # di_scale_factor could have been corrupted in older versions; guard against it.
        scale_fallback = st.session_state.get('di_scale_factor', 1.0)
        if scale_fallback == 0.0:
            scale_fallback = 1.0

        coeff_val = restore_widget_from_storage(
            coeff_widget_key,
            st.session_state.disclose_income_tab_persistence,
            coeff_storage_key,
            scale_fallback
        )

        # Clamp value to valid range [0, 2]; treat 0.0 as "never intentionally set"
        coeff_val = max(0.0, min(float(coeff_val), 2.0))
        if coeff_val == 0.0:
            coeff_val = 1.0
            # Also fix the widget key so Streamlit picks up the corrected value
            st.session_state[coeff_widget_key] = coeff_val

        sigma_coefficient = st.slider(
            "σ Coefficient (multiplier)",
            min_value=0.0,
            max_value=2.0,
            value=coeff_val,
            step=0.01,
            help="Coefficient to multiply the base σ. Final σ = 9.8995 × coefficient",
            key=coeff_widget_key,
            on_change=lambda: save_to_disclose_income_storage(coeff_widget_key, coeff_storage_key)
        )
        st.session_state.di_scale_factor = sigma_coefficient

        effective_sigma = 9.8995 * sigma_coefficient
        st.markdown(f"Effective σ = 9.8995 × {sigma_coefficient:.2f} = {effective_sigma:.2f}")

    else:
        # QUINTILE MODE: 5 sliders (one per income level)
        current_income_mode = st.session_state.get('di_income_mode', 'Categorical only')
        overall_coeff = st.session_state.get('di_scale_factor', 1.0)
        effective_sigma = 9.8995 * overall_coeff
        if 'continuous' in str(current_income_mode).lower() and 'compare' not in str(current_income_mode).lower():
            st.warning(
                "**Continuous mode uses overall σ.** "
                "Per-quintile σ values are based on categorical budget levels and are "
                "not applicable to the continuous income specification. "
                f"The simulation will use the overall σ ({overall_coeff:.2f} × 9.8995 = {effective_sigma:.4f}) for continuous runs."
            )
        elif 'compare' in str(current_income_mode).lower():
            st.info(
                "**Note:** Per-quintile σ values will only apply to the **categorical** run. "
                f"The continuous run will use the overall σ ({overall_coeff:.2f} × 9.8995 = {effective_sigma:.4f})."
            )
        st.markdown("**Per-Quintile σ Coefficients**")
        st.markdown("Each level has its own base σ from empirical data:")

        quintile_coefficients = {}
        default_scale = st.session_state.get('di_scale_factor', 1.0)
        current_quintile_scales = st.session_state.get('di_quintile_scale_factors', {
            '1': default_scale, '2': default_scale, '3': default_scale,
            '4': default_scale, '5': default_scale
        })

        for level in ['1', '2', '3', '4', '5']:
            level_scale = current_quintile_scales.get(level, default_scale)
            level_scale = max(0.0, min(float(level_scale), 2.0))

            storage_key = f'di_sigma_quintile_{level}_{mode_suffix}'
            widget_key = f'di_tab_sigma_q{level}_{mode_suffix}'

            q_val = restore_widget_from_storage(
                widget_key,
                st.session_state.disclose_income_tab_persistence,
                storage_key,
                level_scale
            )
            q_val = max(0.0, min(float(q_val), 2.0))

            base_sigma = BASE_SIGMAS[level]

            col_slider, col_result = st.columns([3, 1])
            with col_slider:
                q_coeff = st.slider(
                    f"{LEVEL_LABELS[level]} (base σ={base_sigma:.2f})",
                    min_value=0.0,
                    max_value=2.0,
                    value=q_val if q_val != 0.0 else 1.0,
                    step=0.01,
                    key=widget_key,
                    on_change=lambda l=level: save_to_disclose_income_storage(
                        f'di_tab_sigma_q{l}_{mode_suffix}', f'di_sigma_quintile_{l}_{mode_suffix}'
                    )
                )
            with col_result:
                effective = base_sigma * q_coeff
                st.metric("Effective σ", f"{effective:.2f}")

            quintile_coefficients[level] = q_coeff

        # Save quintile scale factors to session state
        st.session_state.di_quintile_scale_factors = quintile_coefficients


def render_disclose_income_tab():
    """Render disclose_income specific configuration."""
    initialize_disclose_income_session_state()
    config = load_disclose_income_config()

    # Overlay session-state intercept override so the formula display
    # and other readers of `config` see the latest value even though
    # the YAML is only written on explicit Reset/Save.
    if 'di_intercept' in st.session_state:
        config['intercept'] = st.session_state.di_intercept

    st.markdown('<h3 class="section-header">Disclose Income Configuration</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Income Specification
        st.markdown('<h4 class="subsection-header">Income Specification</h4>', unsafe_allow_html=True)

        # Initialize the widget key if it doesn't exist
        if "di_tab_income_mode" not in st.session_state:
            current_mode = st.session_state.get('di_income_mode', 'Categorical only')
            valid_modes = ["Categorical only", "Continuous only", "Compare both"]
            
            # Try exact match first
            if current_mode in valid_modes:
                st.session_state.di_tab_income_mode = current_mode
            else:
                # Try case-insensitive match
                match_found = False
                for m in valid_modes:
                    if m.lower() == str(current_mode).lower():
                        st.session_state.di_tab_income_mode = m
                        match_found = True
                        break
                
                if not match_found:
                    st.session_state.di_tab_income_mode = "Categorical only"

        def on_di_income_mode_change():
            """Handle income spec mode changes for disclose income."""
            new_mode = st.session_state.di_tab_income_mode
            st.session_state.di_income_mode = new_mode
            save_to_disclose_income_storage('di_tab_income_mode', 'income_mode')
            from app.pages.decision_execution import clear_decision_config
            clear_decision_config('disclose_income')

        # Restore income mode from storage
        income_val = restore_widget_from_storage(
            'di_tab_income_mode',
            st.session_state.disclose_income_tab_persistence,
            'income_mode',
            'Categorical only'
        )

        # Ensure value is valid (handling case sensitivity)
        mode_options = ["Categorical only", "Continuous only", "Compare both"]
        
        # Try to find a matching option (case-insensitive)
        if income_val not in mode_options:
            match_found = False
            for opt in mode_options:
                if str(income_val).lower() == opt.lower():
                    income_val = opt
                    match_found = True
                    break
            
            if match_found:
                # Update session state to match normalized value (critical for st.radio)
                st.session_state.di_tab_income_mode = income_val
            else:
                income_val = "Categorical only"
                st.session_state.di_tab_income_mode = income_val

        income_mode = st.radio(
            "Income Specification for Disclosure Model",
            mode_options,
            help="""
            **Categorical only**: Uses level-specific intercepts based on income categories (5 levels)
            **Continuous only**: Uses single β₀ with income coefficient based on actual income
            **Compare both**: Run both specifications for comparison
            """,
            key="di_tab_income_mode",
            on_change=on_di_income_mode_change
        )
        # Sync session state
        st.session_state.di_income_mode = income_mode

    with col2:
        # Stochastic Component
        st.markdown('<h4 class="subsection-header">Stochastic Component</h4>', unsafe_allow_html=True)

        population_mode = st.session_state.get('population_mode', 'Copula (synthetic)')

        if population_mode == "Research Baseline":
            st.info("📊 Research Baseline always uses anchor values only (deterministic). "
                    "Configure stochastic settings below for Copula / Research Specification runs.")

        # --- Copula sigma checkbox (always visible) ---
        st.markdown("**Copula Mode:**")
        copula_val = restore_widget_from_storage(
            'di_tab_sigma_in_copula',
            st.session_state.disclose_income_tab_persistence,
            'di_sigma_in_copula',
            False
        )
        sigma_in_copula = st.checkbox(
            "Add Normal(anchor, σ) draw to Copula runs",
            value=copula_val,
            help="When enabled, Copula mode will also use the stochastic component",
            key="di_tab_sigma_in_copula",
            on_change=lambda: save_to_disclose_income_storage('di_tab_sigma_in_copula', 'di_sigma_in_copula')
        )
        st.session_state.di_sigma_in_copula = sigma_in_copula

        # --- Research Specification sigma checkbox (always visible) ---
        st.markdown("**Research Specification Mode:**")
        res_val = restore_widget_from_storage(
            'di_tab_sigma_enabled',
            st.session_state.disclose_income_tab_persistence,
            'sigma_enabled',
            True
        )
        sigma_enabled = st.checkbox(
            "Use Normal(anchor, σ) draw in Research Specification mode",
            value=res_val,
            help="When enabled, adds stochastic variation via Normal(anchor, σ) draws.",
            key="di_tab_sigma_enabled",
            on_change=lambda: save_to_disclose_income_storage('di_tab_sigma_enabled', 'sigma_enabled')
        )
        st.session_state.di_sigma_enabled = sigma_enabled

        st.markdown("Research Baseline always uses anchor values only (deterministic).")

        # --- Shared sigma controls (strategy, coefficient, quintiles) ---
        if sigma_in_copula or sigma_enabled:
            render_di_sigma_controls('stochastic')
        else:
            st.info("Stochastic component disabled for all modes - using anchor values directly")

        # Anchor Mix
        st.markdown('<h4 class="subsection-header">Anchor Mix</h4>', unsafe_allow_html=True)

        anchor_weights = config.get('anchor_weights', {})
        current_wopb = anchor_weights.get('observed_prosocial', 0.25)
        current_wpb = anchor_weights.get('prosocial_weight', 0.50)

        # Restore WOPB from storage
        wopb_val = restore_widget_from_storage(
            'di_wopb_widget',
            st.session_state.disclose_income_tab_persistence,
            'wopb',
            current_wopb
        )

        # WOPB - Weight for observed vs calculated prosocial behavior
        new_wopb = st.slider(
            "W_OPB: Observed vs Calculated prosocial behavior weight",
            min_value=0.0,
            max_value=1.0,
            value=float(wopb_val),
            step=0.01,
            help="anchored_PB = WOPB×Observed_PB + (1-WOPB×Calculated_PB based on Honesty-Humility, Agreeableness, Openness, and Religiosity); Default: 0.25",
            key="di_wopb_widget",
            on_change=lambda: save_to_disclose_income_storage('di_wopb_widget', 'wopb')
        )
        st.markdown(f"Observed prosocial behavior weight: {new_wopb:.2f}")
        st.session_state.di_wopb = new_wopb

        # Restore WPB from storage
        wpb_val = restore_widget_from_storage(
            'di_wpb_widget',
            st.session_state.disclose_income_tab_persistence,
            'wpb',
            current_wpb
        )

        # WPB - Weight for prosocial effect in disclosure equation
        new_wpb = st.slider(
            "W_PB: Prosocial behavior (Equation 1) effect weight",
            min_value=0.0,
            max_value=1.0,
            value=float(wpb_val),
            step=0.01,
            help="DI = β0+(1-WPB)×(Income, Extroversion, Neuroticism, Honesty-Humility, and Agreeableness) + WPB×(Prosocial Behavior×Income_High); Default: 0.5",
            key="di_wpb_widget",
            on_change=lambda: save_to_disclose_income_storage('di_wpb_widget', 'wpb')
        )
        st.markdown(f"Prosocial behavior weight: {new_wpb:.2f}")
        st.session_state.di_wpb = new_wpb

    # Mathematical Model Formula Section
    st.markdown('<h4 class="subsection-header">Mathematical Model Formula</h4>', unsafe_allow_html=True)
    render_formula_display(config)

    # Intercept Override Section
    st.markdown('<h4 class="subsection-header">Intercept Override</h4>', unsafe_allow_html=True)
    render_intercept_override_section(config)

    # Actions & Management Section
    st.markdown('<h4 class="subsection-header">Actions & Management</h4>', unsafe_allow_html=True)
    render_actions_and_management_section(config)

    # Simulation buttons - same as donation_default
    try:
        from app.pages.decision_execution import render_simulation_buttons
        
        # Safety: Get selected_decisions with a default
        selected_decs = getattr(st.session_state.decision_params, 'selected_decisions', [])
        
        render_simulation_buttons(
            decision_name="disclose_income",
            selected_decisions=selected_decs
        )
    except Exception as e:
        st.error(f"Error rendering simulation buttons: {e}")
        import traceback
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
    wopb = config.get('anchor_weights', {}).get('observed_prosocial', 0.25)
    st.latex(f"""
    PB_i = {wopb:.2f} \\times z_{{obs\_PB_i}} + {1-wopb:.2f} \\times z_{{calc\_PB_i}}
    """)
    st.markdown("Note: Both observed and calculated PB are standardized (z-scored) before combining.")


def render_categorical_di_formula(config):
    """Render Equation 2 for CATEGORICAL income mode."""
    st.markdown("### Equation 2: Disclose Income (Categorical)")
    
    wpb = config.get('anchor_weights', {}).get('prosocial_weight', 0.50)
    
    # Show full expanded formula (per professor's specification - no separate "direct effects" line)
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
    
    wpb = config.get('anchor_weights', {}).get('prosocial_weight', 0.50)
    beta0 = config.get('intercept', 0.75)
    
    # Show full expanded formula only (per professor's specification)
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


def get_current_yaml_intercept():
    """Get current intercept value from YAML config."""
    config = load_disclose_income_config()
    return config.get('intercept', 0.75)


def render_intercept_override_section(config):
    """Render the intercept override section with ability to modify the intercept value."""

    # Initialize override values if not present
    if 'di_intercept_override_values' not in st.session_state:
        st.session_state.di_intercept_override_values = {}

    # Show current configuration values for reference
    try:
        # Fixed research default — Impact Preview always compares against this,
        # regardless of what auto-save wrote to the YAML.
        research_default = 0.75

        # Current YAML value is used as the widget's initial value
        # so it reflects what's actually saved in the config.
        current_config_value = get_current_yaml_intercept()

        # Use 3 columns: Current Value, Override Value, Impact Preview
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**Research Default: {research_default:.4f}**")
            st.markdown("Intercept (β₀)")
            st.markdown("Fixed reference value from original research")

        with col2:
            st.markdown("**Override Value**")

            # Restore intercept widget key, defaulting to current config value
            int_val = restore_widget_from_storage(
                'di_override_intercept',
                st.session_state.di_intercept_override_values,
                'intercept',
                current_config_value
            )

            new_intercept = st.number_input(
                "Baseline disclosure tendency",
                min_value=0.0,
                max_value=5.0,
                value=float(int_val),
                step=0.01,
                format="%.4f",
                help="β₀ = 0.75 in the disclose income equation. Override value, with higher values increasing baseline probability of disclosure.",
                key="di_override_intercept",
                on_change=lambda: auto_save_intercept(st.session_state.di_override_intercept)
            )
            st.session_state.di_intercept_override_values['intercept'] = new_intercept

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
    """Update intercept override in session state (not YAML).

    The YAML is only written by the Reset button.  Reload Configuration
    reads from the unchanged YAML, so it can restore the last saved value.
    The simulation reads di_intercept from session state, so it always
    picks up the latest override without needing a YAML write.
    """
    if 'di_intercept_override_values' not in st.session_state:
        st.session_state.di_intercept_override_values = {}

    st.session_state.di_intercept_override_values['intercept'] = new_value
    st.session_state.di_intercept = new_value


def _apply_config_to_widget_keys(config):
    """Explicitly set every widget key from a config dict.

    Streamlit's internal widget cache can retain stale slider values even
    after their session-state key is deleted.  Explicitly *setting* the key
    is more reliable than deletion because it overwrites the cache.
    """
    anchor = config.get('anchor_weights', {})
    stochastic = config.get('stochastic', {})

    st.session_state.di_wopb_widget = anchor.get('observed_prosocial', 0.25)
    st.session_state.di_wpb_widget = anchor.get('prosocial_weight', 0.50)
    st.session_state.di_override_intercept = config.get('intercept', 0.75)
    st.session_state.di_tab_sigma_enabled = stochastic.get('sigma_value', 0) > 0
    st.session_state.di_tab_sigma_coefficient = stochastic.get('scale_factor', 1.0)
    st.session_state.di_tab_sigma_strategy = stochastic.get('sigma_strategy', 'overall')
    st.session_state.di_tab_income_mode = config.get('income_mode', 'Categorical only')
    st.session_state.di_tab_sigma_in_copula = False
    st.session_state.di_tab_sigma_in_copula_compare = False
    st.session_state.di_sigma_in_copula = False

    default_scale = stochastic.get('scale_factor', 1.0)
    quintile_scales = stochastic.get('quintile_scale_factors', {})
    for level in ['1', '2', '3', '4', '5']:
        st.session_state[f'di_tab_sigma_q{level}'] = quintile_scales.get(
            level, default_scale
        )


def reset_to_defaults():
    """Reset all configuration values to their defaults."""
    default_config = {
        'intercept': 0.75,
        'income_mode': 'Categorical only',
        'anchor_weights.observed_prosocial': 0.25,
        'anchor_weights.prosocial_weight': 0.50,
        'stochastic.sigma_value': 0,
        'stochastic.scale_factor': 1.0,
        'stochastic.sigma_strategy': 'overall',
        'stochastic.quintile_scale_factors.1': 1.0,
        'stochastic.quintile_scale_factors.2': 1.0,
        'stochastic.quintile_scale_factors.3': 1.0,
        'stochastic.quintile_scale_factors.4': 1.0,
        'stochastic.quintile_scale_factors.5': 1.0,
    }

    success = True
    for key, value in default_config.items():
        if not save_disclose_income_config({key: value}):
            success = False

    if success:
        # 1. Clear all di_ keys and persistence storage
        keys_to_clear = [k for k in st.session_state.keys() if k.startswith('di_')]
        for key in keys_to_clear:
            del st.session_state[key]
        if 'disclose_income_tab_persistence' in st.session_state:
            del st.session_state['disclose_income_tab_persistence']

        # 2. Re-read the now-reset YAML and SET every widget key to
        #    the default value.  This prevents Streamlit's internal
        #    widget cache from retaining stale slider values that would
        #    be auto-saved back to YAML on the next rerun.
        _apply_config_to_widget_keys(load_disclose_income_config())

    return success


def render_actions_and_management_section(config):
    """Render the combined actions and management section."""

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔄 Reset All**")
        if st.button("Reset Config to Defaults", type="secondary", use_container_width=True,
                     help="Reset all disclose income values to research defaults",
                     key="di_reset_btn"):
            success = reset_to_defaults()
            if success:
                st.toast("Configuration reset to defaults", icon="🔄")
                st.rerun()
            else:
                st.toast("Failed to reset configuration", icon="⚠️")

    with col2:
        st.markdown("**🔄 Reset Intercept**")
        if st.button("Reset Intercept to Default Value", type="secondary", use_container_width=True,
                     help="Reset intercept value to research default (0.75)",
                     key="di_reload_btn"):
            # Only reset the intercept value - NOT all configuration values
            default_intercept = 0.75
            success = save_disclose_income_config({'intercept': default_intercept})
            if success:
                # Update session state for intercept only
                st.session_state.di_intercept = default_intercept
                if 'di_intercept_override_values' in st.session_state:
                    st.session_state.di_intercept_override_values['intercept'] = default_intercept
                st.toast("✅ Intercept reset to research default (0.75)", icon="🔄")
                st.rerun()
            else:
                st.toast("❌ Failed to reset intercept", icon="⚠️")
