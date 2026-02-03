# app/pages/decision_tabs/donation_default.py
"""
Donation Default decision tab configuration.
"""
# Force rebuild timestamp: 2025-10-01
import streamlit as st
import pandas as pd
from app.pages.decision_execution import run_individual_decision
from app.models import load_donation_coefficients_from_yaml


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


def get_coefficient_for_input(name):
    """Get coefficient value for input field based on current income mode"""
    ensure_coefficients_loaded()
    income_mode = st.session_state.get('income_spec_mode', 'categorical only')
    
    # For input fields, show the appropriate coefficient set based on current mode
    if 'continuous' in income_mode.lower() and 'compare' not in income_mode.lower():
        return get_coefficient(name, 'cont')
    else:
        return get_coefficient(name, 'cat')  # Default to categorical for "compare both" and "categorical only"


def restore_widget_from_storage(widget_key, storage_dict, storage_key, default_value):
    """Restore a widget key from storage dictionary before widget renders.
    
    This ensures widget keys persist across navigation by:
    1. Checking storage dict for the value
    2. Restoring to session state so widget finds it
    """
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


def initialize_donation_widget_keys():
    """Initialize widget keys for donation_default tab to preserve values across navigation"""

    # Initialize persistence storage if missing
    if "donation_tab_persistence" not in st.session_state:
        st.session_state.donation_tab_persistence = {}

    # Initialize checkbox widget keys
    if "tab_sigma_in_copula" not in st.session_state:
        st.session_state.tab_sigma_in_copula = st.session_state.get('sigma_in_copula', False)

    if "tab_sigma_in_research" not in st.session_state:
        st.session_state.tab_sigma_in_research = st.session_state.get('sigma_in_research', True)

    if "tab_sigma_in_copula_compare" not in st.session_state:
        st.session_state.tab_sigma_in_copula_compare = st.session_state.get('sigma_in_copula', False)

    if "tab_sigma_in_research_compare" not in st.session_state:
        st.session_state.tab_sigma_in_research_compare = st.session_state.get('sigma_in_research', True)

    # Initialize slider widget keys
    if "tab_sigma_coefficient" not in st.session_state:
        st.session_state.tab_sigma_coefficient = st.session_state.get('sigma_coefficient', 1.0)

    if "tab_sigma_coefficient_research" not in st.session_state:
        st.session_state.tab_sigma_coefficient_research = st.session_state.get('sigma_coefficient', 1.0)

    if "tab_sigma_coefficient_compare" not in st.session_state:
        st.session_state.tab_sigma_coefficient_compare = st.session_state.get('sigma_coefficient', 1.0)

    if "tab_anchor_weight" not in st.session_state:
        st.session_state.tab_anchor_weight = st.session_state.get('anchor_observed_weight', 0.75)

    # Initialize sigma strategy (overall vs quintile)
    if "donation_sigma_strategy" not in st.session_state:
        st.session_state.donation_sigma_strategy = st.session_state.get('donation_sigma_strategy', 'overall')

    # Initialize quintile scale factors
    if "donation_quintile_scale_factors" not in st.session_state:
        st.session_state.donation_quintile_scale_factors = {
            '1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0, '5': 1.0
        }


def save_to_donation_storage(widget_key, storage_key):
    """Save a widget value to donation storage.

    Args:
        widget_key: The session state key of the widget
        storage_key: The key to use in the storage dictionary
    """
    if "donation_tab_persistence" not in st.session_state:
        st.session_state.donation_tab_persistence = {}

    if widget_key in st.session_state:
        st.session_state.donation_tab_persistence[storage_key] = st.session_state[widget_key]


def render_donation_sigma_controls(mode_suffix: str):
    """
    Render sigma strategy controls (overall vs quintile) for donation_default.

    Args:
        mode_suffix: One of 'copula', 'research', or 'compare' to distinguish widget keys
    """
    # Base sigma values per quintile (from empirical data)
    BASE_SIGMAS = {
        '1': 5.705052,   # Level 1 (€16)
        '2': 3.069326,   # Level 2 (€32)
        '3': 3.532226,   # Level 3 (€72)
        '4': 12.219622,  # Level 4 (€128)
        '5': 16.854622,  # Level 5 (€200)
    }

    LEVEL_LABELS = {
        '1': 'Level 1 (€16)',
        '2': 'Level 2 (€32)',
        '3': 'Level 3 (€72)',
        '4': 'Level 4 (€128)',
        '5': 'Level 5 (€200)',
    }

    # Restore sigma strategy from storage
    strategy_widget_key = f'donation_tab_sigma_strategy_{mode_suffix}'
    strategy_storage_key = f'sigma_strategy_{mode_suffix}'
    current_strategy = st.session_state.get('donation_sigma_strategy', 'overall')

    strategy_val = restore_widget_from_storage(
        strategy_widget_key,
        st.session_state.donation_tab_persistence,
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
        save_to_donation_storage(strategy_widget_key, strategy_storage_key)
        st.session_state.donation_sigma_strategy = new_strategy

    sigma_strategy = st.radio(
        "Apply σ uniformly or per income level?",
        options=['overall', 'quintile'],
        format_func=lambda x: 'Overall (single σ for all)' if x == 'overall' else 'Quintiles (σ per budget level)',
        index=0 if strategy_val == 'overall' else 1,
        key=strategy_widget_key,
        on_change=on_strategy_change,
        horizontal=True
    )
    st.session_state.donation_sigma_strategy = sigma_strategy

    st.markdown("---")

    if sigma_strategy == 'overall':
        # OVERALL MODE: Single slider
        st.caption("Base σ = 9.8995 (empirical from 280 participants)")

        # Restore coefficient widget value
        coeff_widget_key = f'tab_sigma_coefficient_{mode_suffix}'
        coeff_storage_key = f'sigma_coefficient_{mode_suffix}'

        coeff_val = restore_widget_from_storage(
            coeff_widget_key,
            st.session_state.donation_tab_persistence,
            coeff_storage_key,
            1.0
        )

        # Clamp value to valid range [0, 2]
        coeff_val = max(0.0, min(float(coeff_val), 2.0))

        sigma_coefficient = st.slider(
            "σ Coefficient (multiplier)",
            min_value=0.0,
            max_value=2.0,
            value=coeff_val if coeff_val != 0.0 else 1.0,
            step=0.01,
            help="Coefficient to multiply the base σ. Final σ = 9.8995 × coefficient",
            key=coeff_widget_key,
            on_change=lambda: save_to_donation_storage(coeff_widget_key, coeff_storage_key)
        )
        st.session_state.sigma_coefficient = sigma_coefficient

        effective_sigma = 9.8995 * sigma_coefficient
        st.caption(f"Effective σ = 9.8995 × {sigma_coefficient:.2f} = {effective_sigma:.2f}")
        st.session_state.sigma_value_ui = effective_sigma

    else:
        # QUINTILE MODE: 5 sliders (one per income level)
        st.markdown("**Per-Quintile σ Coefficients**")
        st.caption("Each level has its own base σ from empirical data:")

        quintile_coefficients = {}

        for level in ['1', '2', '3', '4', '5']:
            # Get current value for this level
            current_scale = st.session_state.donation_quintile_scale_factors.get(level, 1.0)
            current_scale = max(0.0, min(float(current_scale), 2.0))

            # Restore from storage
            storage_key = f'donation_sigma_quintile_{level}_{mode_suffix}'
            widget_key = f'donation_tab_sigma_q{level}_{mode_suffix}'

            q_val = restore_widget_from_storage(
                widget_key,
                st.session_state.donation_tab_persistence,
                storage_key,
                current_scale
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
                    on_change=lambda l=level, wk=widget_key, sk=storage_key: save_to_donation_storage(wk, sk)
                )
            with col_result:
                effective = base_sigma * q_coeff
                st.metric("Effective σ", f"{effective:.2f}")

            quintile_coefficients[level] = q_coeff

        # Save quintile scale factors to session state
        st.session_state.donation_quintile_scale_factors = quintile_coefficients

        # Set sigma_value_ui to indicate stochastic is enabled (using overall sigma as fallback display)
        st.session_state.sigma_value_ui = 9.8995
        st.session_state.sigma_coefficient = 1.0  # Default coefficient for overall fallback


def render_donation_default_tab():
    """Render donation_default specific configuration"""
    # Ensure coefficients are loaded from configuration file
    ensure_coefficients_loaded()
    
    # Initialize widget keys to preserve values across navigation
    initialize_donation_widget_keys()
    
    # Handle full config reset flag BEFORE any widgets render
    if st.session_state.get('_reset_config_to_defaults_flag', False):
        st.session_state._reset_config_to_defaults_flag = False
        
        # Clear persistence storage
        st.session_state.donation_tab_persistence = {}
        
        # Reset intercept widget keys
        st.session_state.override_categorical_intercept = 1.519818
        st.session_state.override_continuous_intercept = -0.139596
        st.session_state.intercept_override_values = {}
        
        # Reset adjustment widget key
        st.session_state.override_adjustment_shift = -4.0
        st.session_state.adjustment_override_values = {}
        
        # Reset income mode
        st.session_state.page2_tab_income_spec_mode = "categorical only"
        st.session_state.income_spec_mode = "categorical only"
        
        # Reset sigma strategy (main state variable)
        st.session_state.donation_sigma_strategy = "overall"
        
        # Reset sigma strategy widget keys for all mode suffixes
        for mode_suffix in ['copula', 'research', 'compare']:
            st.session_state[f'donation_tab_sigma_strategy_{mode_suffix}'] = "overall"
            st.session_state[f'tab_sigma_coefficient_{mode_suffix}'] = 1.0
            # Reset quintile slider widget keys
            for level in ['1', '2', '3', '4', '5']:
                st.session_state[f'donation_tab_sigma_q{level}_{mode_suffix}'] = 1.0
        
        # Reset sigma coefficient sliders (legacy keys)
        st.session_state.tab_sigma_coefficient = 1.0
        st.session_state.tab_sigma_coefficient_research = 1.0
        st.session_state.tab_sigma_coefficient_compare = 1.0
        
        # Reset quintile scale factors
        st.session_state.donation_quintile_scale_factors = {
            '1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0, '5': 1.0
        }
        
        # Reset anchor weight
        st.session_state.tab_anchor_weight = 0.75
        
        # Reset sigma checkboxes
        st.session_state.tab_sigma_in_copula = False
        st.session_state.tab_sigma_in_research = True
        st.session_state.tab_sigma_in_copula_compare = False
        st.session_state.tab_sigma_in_research_compare = True
    
    st.markdown('<h3 class="section-header"> Donation Default Configuration</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        population_mode = st.session_state.population_mode  # Use value from Page 1
        
        # Income specification selector
        if population_mode != "Dependent variable resampling":
            st.markdown('<h4 class="subsection-header">Income Specification</h4>', unsafe_allow_html=True)
            
            # Initialize the widget key if it doesn't exist
            if "page2_tab_income_spec_mode" not in st.session_state:
                # Map current income_spec_mode to radio button options
                if st.session_state.income_spec_mode in ["Categorical only", "Continuous only", "Compare both"]:
                    st.session_state.page2_tab_income_spec_mode = st.session_state.income_spec_mode
                elif st.session_state.income_spec_mode in ["compare both", "compare side-by-side"]:
                    st.session_state.page2_tab_income_spec_mode = "Compare both"
                else:
                    st.session_state.page2_tab_income_spec_mode = "Categorical only"
            
            def on_income_spec_mode_change():
                """Handle income spec mode changes"""
                setattr(st.session_state, 'income_spec_mode', st.session_state.page2_tab_income_spec_mode)
                save_to_donation_storage('page2_tab_income_spec_mode', 'income_spec_mode')
                reload_coefficients_for_income_mode()
                clear_input_field_cache()
                # FIX: Clear selected config from BOTH legacy AND unified storage
                # This prevents stale configs from persisting when user changes income mode
                # Legacy storage
                if hasattr(st.session_state, 'selected_donation_config'):
                    delattr(st.session_state, 'selected_donation_config')
                # Unified storage - must also clear to prevent migration from restoring stale config
                if 'selected_decision_configs' in st.session_state:
                    if 'donation_default' in st.session_state.selected_decision_configs:
                        del st.session_state.selected_decision_configs['donation_default']
            
            # Restore income spec mode
            income_val = restore_widget_from_storage(
                'page2_tab_income_spec_mode',
                st.session_state.donation_tab_persistence,
                'income_spec_mode',
                'categorical only'
            )
            
            # Ensure index matches logic if needed, but for radio string value is enough if in options
            # If value not in options, default to categorical
            if income_val not in ["categorical only", "continuous only", "Compare both"]:
                income_val = "categorical only"
            
            income_spec_mode = st.radio(
                "Income Mode for Donation Model",
                ["categorical only", "continuous only", "Compare both"],
                help="Choose income treatment: categorical (5 categories), continuous (linear), or Compare both",
                key="page2_tab_income_spec_mode",
                on_change=on_income_spec_mode_change
            )
            # Ensure session state is synced if we just restored a value
            st.session_state.income_spec_mode = income_spec_mode
        else:
            st.session_state.income_spec_mode = "categorical only"
    
    with col2:
        # Stochastic component option
        st.markdown('<h4 class="subsection-header">Stochastic Component</h4>', unsafe_allow_html=True)
        
        if population_mode == "Copula (synthetic)":
            # Show only Copula controls

            # Restore widget value
            copula_val = restore_widget_from_storage(
                'tab_sigma_in_copula',
                st.session_state.donation_tab_persistence,
                'sigma_in_copula',
                False
            )

            sigma_in_copula = st.checkbox(
                "Add Normal(anchor, σ) draw to Copula runs",
                value=copula_val,
                help="When enabled, Copula mode will also use the stochastic component",
                key="tab_sigma_in_copula",
                on_change=lambda: save_to_donation_storage('tab_sigma_in_copula', 'sigma_in_copula')
            )
            st.session_state.sigma_in_copula = sigma_in_copula
            st.session_state.sigma_in_research = True  # Default for research mode

            if sigma_in_copula:
                # Show sigma strategy selector
                render_donation_sigma_controls('copula')
            
        elif population_mode == "Research Specification":
            # Show only Research controls

            # Restore research checkbox
            res_val = restore_widget_from_storage(
                'tab_sigma_in_research',
                st.session_state.donation_tab_persistence,
                'sigma_in_research',
                True
            )

            sigma_in_research = st.checkbox(
                "Use Normal(anchor, σ) draw in Research mode",
                value=res_val,
                help="When enabled, Research mode will add stochastic variation via Normal(anchor, σ) draws. When disabled, only the anchor value is used.",
                key="tab_sigma_in_research",
                on_change=lambda: save_to_donation_storage('tab_sigma_in_research', 'sigma_in_research')
            )
            st.session_state.sigma_in_research = sigma_in_research
            st.session_state.sigma_in_copula = False  # Not applicable

            # Show sigma controls only if stochastic component is enabled
            if sigma_in_research:
                render_donation_sigma_controls('research')
            else:
                st.info("Stochastic component disabled - using anchor values directly")
                # Set sigma to 0 when disabled to ensure no variability
                st.session_state.sigma_coefficient = 0.0
                st.session_state.sigma_value_ui = 0.0
                
        elif population_mode == "Research Baseline":
            # Research Baseline mode - no stochastic component, anchor values only
            st.session_state.sigma_in_copula = False  # Not applicable
            st.session_state.sigma_in_research = False  # No stochastic component
            st.session_state.sigma_coefficient = 0.0
            st.session_state.sigma_value_ui = 0.0
            
            st.info("📊 Research Baseline Mode: Uses original 280 participants with anchor values only (no stochastic component)")
            st.caption("🎯 This mode returns the deterministic anchor = 0.75 × observed + 0.25 × predicted")
                
        else:  # Compare all
            # Show controls for all three modes
            st.markdown("**Copula Mode Controls:**")

            # Restore copula checkbox for compare
            copula_comp_val = restore_widget_from_storage(
                'tab_sigma_in_copula_compare',
                st.session_state.donation_tab_persistence,
                'sigma_in_copula_compare',
                False
            )

            sigma_in_copula = st.checkbox(
                "Add Normal(anchor, σ) draw to Copula runs",
                value=copula_comp_val,
                help="When enabled, Copula mode will also use the stochastic component",
                key="tab_sigma_in_copula_compare",
                on_change=lambda: save_to_donation_storage('tab_sigma_in_copula_compare', 'sigma_in_copula_compare')
            )
            st.session_state.sigma_in_copula = sigma_in_copula

            st.markdown("**Research Specification Controls:**")

            # Restore research checkbox for compare
            res_comp_val = restore_widget_from_storage(
                'tab_sigma_in_research_compare',
                st.session_state.donation_tab_persistence,
                'sigma_in_research_compare',
                True
            )

            sigma_in_research = st.checkbox(
                "Use Normal(anchor, σ) draw in Research Specification mode",
                value=res_comp_val,
                help="When enabled, Research Specification mode will add stochastic variation via Normal(anchor, σ) draws. When disabled, only the anchor value is used.",
                key="tab_sigma_in_research_compare",
                on_change=lambda: save_to_donation_storage('tab_sigma_in_research_compare', 'sigma_in_research_compare')
            )
            st.session_state.sigma_in_research = sigma_in_research

            st.markdown("**Research Baseline:** Always uses anchor values only (no stochastic component)")
            st.caption("Research Baseline = deterministic anchor = 0.75 × observed + 0.25 × predicted")

            # Show sigma controls if either mode has stochastic enabled
            if sigma_in_copula or sigma_in_research:
                render_donation_sigma_controls('compare')
            else:
                st.info("Stochastic component disabled for both modes - using anchor values directly")
                st.session_state.sigma_coefficient = 0.0
                st.session_state.sigma_value_ui = 0.0
        
        # Anchor weights
        if population_mode != "Dependent variable resampling":
            st.markdown('<h4 class="subsection-header">Anchor Mix</h4>', unsafe_allow_html=True)
            
            # Restore anchor weight
            anchor_val = restore_widget_from_storage(
                'tab_anchor_weight',
                st.session_state.donation_tab_persistence,
                'anchor_weight',
                0.75
            )
            
            anchor_observed_weight = st.slider(
                "Weight for observed vs modeled prosocial behavior",
                min_value=0.0,
                max_value=1.0,
                value=float(anchor_val),
                step=0.01,
                help="Anchor = w × Observed + (1-w) × Predicted",
                key="tab_anchor_weight",
                on_change=lambda: save_to_donation_storage('tab_anchor_weight', 'anchor_weight')
            )
            st.session_state.anchor_observed_weight = anchor_observed_weight
            st.caption(f"Predicted weight: {1 - anchor_observed_weight:.2f}")
        else:
            st.session_state.anchor_observed_weight = 0.75
    
    
    # NEW: Mathematical Formula Display Section
    st.markdown('<h4 class="subsection-header">📐 Mathematical Model Formula</h4>', unsafe_allow_html=True)
    render_formula_display()
    
    # NEW: Regression Coefficients Section
    st.markdown('<h4 class="subsection-header">🔢 Intercept Override</h4>', unsafe_allow_html=True)
    render_intercept_override_section()
    
    # NEW: Distribution Adjustment Section
    st.markdown('<h4 class="subsection-header">📊 Distribution Adjustment</h4>', unsafe_allow_html=True)
    render_adjustment_override_section()
    
    # Actions and Default Coefficient Management section - combined
    st.markdown("---")
    st.markdown('<h4 class="subsection-header">⚙️ Actions & Management</h4>', unsafe_allow_html=True)
    render_actions_and_management_section()
    
    # Render both individual and complete simulation buttons
    try:
        from app.pages.decision_execution import render_simulation_buttons
        
        # Safety: Get selected_decisions with a default
        selected_decs = getattr(st.session_state.decision_params, 'selected_decisions', [])
        
        render_simulation_buttons(
            decision_name="donation_default",
            selected_decisions=selected_decs
        )
    except Exception as e:
        st.error(f"Error rendering simulation buttons: {e}")
        import traceback
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
    
    # Get current coefficient values FROM YAML ONLY
    intercept = get_coefficient('intercept', 'cat')
    hh_coeff = get_coefficient('hh', 'cat')
    
    # Symbolic formula
    st.markdown("**📐 Symbolic Formula:**")
    st.latex(r"""
    \hat{y}_i = \beta_0 + \beta_{group}[group_i] + \beta_{income\_q}[quintile_i] + \beta_{study}[study_i] + \beta_{hh} \times HH\_zscore_i
    """)
    
    # Numerical formula with current values
    st.markdown("**🔢 With Current Coefficient Values:**")
    st.latex(f"""
    \\hat{{y}}_i = {intercept:.6f} + \\beta_{{group}}[group_i] + \\beta_{{income\\_q}}[quintile_i] + \\beta_{{study}}[study_i] + {hh_coeff:.6f} \\times HH\\_zscore_i
    """)
    
    # Show coefficient lookup tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**👥 Group Effects (β_group):**")
        group_data = {
            'Group': ['MidSub', 'NoSub', 'FullSub (ref)'],
            'Intercept': [
                st.session_state.get('donation_coeff_midsub', 0.856140306694656),
                st.session_state.get('donation_coeff_nosub', -0.926633374153906),
                st.session_state.get('donation_coeff_fullsub', 0.0)
            ]
        }
        group_df = pd.DataFrame(group_data)
        group_df['Intercept'] = group_df['Intercept'].map('{:.6f}'.format)
        st.dataframe(group_df, hide_index=True, use_container_width=True)
        
        st.markdown("**🎓 Study Programme Effects (β_study):**")
        study_data = {
            'Programme': ['Incoming', 'Law5yr', 'UG3yr', 'Grad2yr (ref)'],
            'Intercept': [
                st.session_state.get('donation_coeff_incoming', -6.920193024391676),
                st.session_state.get('donation_coeff_law', -2.081331674770856),
                st.session_state.get('donation_coeff_ug', -2.139093511519692),
                st.session_state.get('donation_coeff_grad', 0.0)
            ]
        }
        study_df = pd.DataFrame(study_data)
        study_df['Intercept'] = study_df['Intercept'].map('{:.6f}'.format)
        st.dataframe(study_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**💰 Income Quintile Effects (β_income_q):**")
        income_data = {
            'Quintile': ['Q1 (Level 1)', 'Q2 (Level 2)', 'Q3 (Level 3)', 'Q4_Q5 (Levels 4-5, ref)'],
            'Intercept': [
                st.session_state.get('donation_coeff_q1', -0.520290427509808),
                st.session_state.get('donation_coeff_q2', 3.754612744416796),
                st.session_state.get('donation_coeff_q3', 4.001714810873598),
                st.session_state.get('donation_coeff_q45', 0.0)
            ]
        }
        income_df = pd.DataFrame(income_data)
        income_df['Intercept'] = income_df['Intercept'].map('{:.6f}'.format)
        st.dataframe(income_df, hide_index=True, use_container_width=True)


def render_continuous_formula():
    """Render continuous income specification formula"""
    
    # Get current coefficient values FROM YAML ONLY
    intercept = get_coefficient('intercept', 'cont')
    hh_coeff = get_coefficient('hh', 'cont')
    linear_coeff = get_coefficient('linear', 'cont')
    
    # Symbolic formula
    st.markdown("**📐 Symbolic Formula:**")
    st.latex(r"""
    \hat{y}_i = \beta_0 + \beta_{group}[group_i] + \beta_{linear} \times income\_level_i + \beta_{study}[study_i] + \beta_{hh} \times HH\_zscore_i
    """)
    
    # Numerical formula with current values
    st.markdown("**🔢 With Current Coefficient Values:**")
    st.latex(f"""
    \\hat{{y}}_i = {intercept:.6f} + \\beta_{{group}}[group_i] + {linear_coeff:.6f} \\times income\\_level_i + \\beta_{{study}}[study_i] + {hh_coeff:.6f} \\times HH\\_zscore_i
    """)
    
    # Show coefficient lookup tables and linear effect
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**👥 Group Effects (β_group):**")
        group_data = {
            'Group': ['MidSub', 'NoSub', 'FullSub (ref)'],
            'Intercept': [
                st.session_state.get('donation_coeff_midsub', 0.856140306694656),
                st.session_state.get('donation_coeff_nosub', -0.926633374153906),
                st.session_state.get('donation_coeff_fullsub', 0.0)
            ]
        }
        group_df = pd.DataFrame(group_data)
        group_df['Intercept'] = group_df['Intercept'].map('{:.6f}'.format)
        st.dataframe(group_df, hide_index=True, use_container_width=True)
        
        st.markdown("**🎓 Study Programme Effects (β_study):**")
        study_data = {
            'Programme': ['Incoming', 'Law5yr', 'UG3yr', 'Grad2yr (ref)'],
            'Intercept': [
                st.session_state.get('donation_coeff_incoming', -6.920193024391676),
                st.session_state.get('donation_coeff_law', -2.081331674770856),
                st.session_state.get('donation_coeff_ug', -2.139093511519692),
                st.session_state.get('donation_coeff_grad', 0.0)
            ]
        }
        study_df = pd.DataFrame(study_data)
        study_df['Intercept'] = study_df['Intercept'].map('{:.6f}'.format)
        st.dataframe(study_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**📈 Linear Income Effect (β_linear):**")
        st.caption("Effect = β_linear × actual_allowance_amount")
        
        # Show linear effect for actual allowance amounts (not level numbers)
        allowance_mapping = {1: 16, 2: 32, 3: 72, 4: 128, 5: 200}
        linear_data = {
            'Income Level': ['1 (€16)', '2 (€32)', '3 (€72)', '4 (€128)', '5 (€200)'],
            'Effect': [linear_coeff * allowance_mapping[i] for i in range(1, 6)]
        }
        linear_df = pd.DataFrame(linear_data)
        linear_df['Effect'] = linear_df['Effect'].map('{:.6f}'.format)
        st.dataframe(linear_df, hide_index=True, use_container_width=True)
        
        # Show total range effect using actual allowances
        total_range = linear_coeff * (200 - 16)  # from €16 to €200
        st.metric("Total Range Effect (€16→€200)", f"{total_range:.6f}")


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
            'HH_zscore_i'
        ],
        'Definition': [
            'Predicted prosocial behavior for agent i',
            'Model intercept (baseline prediction when all other terms = 0)',
            'Experimental group effect (MidSub, NoSub vs FullSub reference)',
            'Income quintile effect (Q1, Q2, Q3 vs Q4_Q5 reference) - Categorical mode',
            'Linear income effect (coefficient × income level 1-5) - Continuous mode',
            'Study programme effect (Incoming, Law5yr, UG3yr vs Grad2yr reference)',
            'Honesty-Humility coefficient (effect per z-score unit)',
            'Standardized Honesty-Humility score: (HH_raw - 3.3922) / 0.5587'
        ]
    }
    
    definitions_df = pd.DataFrame(definitions_data)
    st.dataframe(definitions_df, hide_index=True, use_container_width=True)
    
    st.markdown("**🔍 Key Notes:**")
    st.markdown("""
    - **Reference categories** have coefficient = 0.0 and serve as baseline for comparison
    - **Income Level Mapping (Categorical)**: 1→Q1, 2→Q2, 3→Q3, 4&5→Q4_Q5
    - **Z-score standardization** ensures Honesty-Humility has mean=0, std=1 in original data
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
        7. **Final Scaling**: Convert to [0,1] proportion for donation rate
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
                '[-4.0778, 7.2030] → [0, 100]',
                'observed: 0.75, predicted: 0.25'
            ]
        }
        scaling_df = pd.DataFrame(scaling_data)
        st.dataframe(scaling_df, hide_index=True, use_container_width=True)


def reload_coefficients_for_income_mode():
    """Reload coefficients from configuration file when income mode changes"""
    load_donation_coefficients_from_yaml()


def clear_input_field_cache():
    """Clear Streamlit input field cache to force refresh with new values"""
    # Force refresh of input fields by clearing their keys from session state
    # This makes Streamlit re-evaluate the default values
    input_keys = [
        'donation_coeff_intercept_input',
        'donation_coeff_hh_input',
        'donation_coeff_midsub_input',
        'donation_coeff_nosub_input',
        'donation_coeff_fullsub_input',
        'donation_coeff_q1_input',
        'donation_coeff_q2_input',
        'donation_coeff_q3_input',
        'donation_coeff_q4_input',
        'donation_coeff_q5_input',
        'donation_coeff_q45_input',  # Legacy
        'donation_coeff_linear_input',
        'donation_coeff_incoming_input',
        'donation_coeff_law_input',
        'donation_coeff_ug_input',
        'donation_coeff_grad_input'
    ]
    
    for key in input_keys:
        if key in st.session_state:
            del st.session_state[key]


def render_categorical_formula_specific():
    """Render categorical income specification formula with categorical-specific coefficients"""
    
    # Get categorical-specific coefficient values FROM YAML ONLY
    intercept = get_coefficient('intercept', 'cat')
    hh_coeff = get_coefficient('hh', 'cat')
    
    # Symbolic formula
    st.markdown("**📐 Symbolic Formula:**")
    st.latex(r"""
    \hat{y}_i = \beta_0 + \beta_{group}[group_i] + \beta_{income\_q}[quintile_i] + \beta_{study}[study_i] + \beta_{hh} \times HH\_zscore_i
    """)
    
    # Numerical formula with current values
    st.markdown("**🔢 With Current Coefficient Values:**")
    st.latex(f"""
    \\hat{{y}}_i = {intercept:.6f} + \\beta_{{group}}[group_i] + \\beta_{{income\\_q}}[quintile_i] + \\beta_{{study}}[study_i] + {hh_coeff:.6f} \\times HH\\_zscore_i
    """)
    
    # Show coefficient lookup tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**👥 Group Effects (β_group):**")
        group_data = {
            'Group': ['MidSub', 'NoSub', 'FullSub (ref)'],
            'Intercept': [
                get_coefficient('midsub', 'cat'),
                get_coefficient('nosub', 'cat'),
                get_coefficient('fullsub', 'cat')
            ]
        }
        group_df = pd.DataFrame(group_data)
        group_df['Intercept'] = group_df['Intercept'].map('{:.6f}'.format)
        st.dataframe(group_df, hide_index=True, use_container_width=True)
        
        st.markdown("**🎓 Study Programme Effects (β_study):**")
        study_data = {
            'Programme': ['Incoming', 'Law5yr', 'UG3yr', 'Grad2yr (ref)'],
            'Intercept': [
                get_coefficient('incoming', 'cat'),
                get_coefficient('law', 'cat'),
                get_coefficient('ug', 'cat'),
                get_coefficient('grad', 'cat')
            ]
        }
        study_df = pd.DataFrame(study_data)
        study_df['Intercept'] = study_df['Intercept'].map('{:.6f}'.format)
        st.dataframe(study_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**💰 Income Quintile Effects (β_income_q):**")
        income_data = {
            'Quintile': ['Q1 (Level 16, ref)', 'Q2 (Level 32)', 'Q3 (Level 72)', 'Q4 (Level 128)', 'Q5 (Level 200)'],
            'Intercept': [
                get_coefficient('q1', 'cat'),
                get_coefficient('q2', 'cat'),
                get_coefficient('q3', 'cat'),
                get_coefficient('q4', 'cat'),
                get_coefficient('q5', 'cat')
            ]
        }
        income_df = pd.DataFrame(income_data)
        income_df['Intercept'] = income_df['Intercept'].map('{:.6f}'.format)
        st.dataframe(income_df, hide_index=True, use_container_width=True)


def render_continuous_formula_specific():
    """Render continuous income specification formula with continuous-specific coefficients"""
    
    # Get continuous-specific coefficient values FROM YAML ONLY
    intercept = get_coefficient('intercept', 'cont')
    hh_coeff = get_coefficient('hh', 'cont')
    linear_coeff = get_coefficient('linear', 'cont')
    
    # Symbolic formula
    st.markdown("**📐 Symbolic Formula:**")
    st.latex(r"""
    \hat{y}_i = \beta_0 + \beta_{group}[group_i] + \beta_{linear} \times income\_level_i + \beta_{study}[study_i] + \beta_{hh} \times HH\_zscore_i
    """)
    
    # Numerical formula with current values
    st.markdown("**🔢 With Current Coefficient Values:**")
    st.latex(f"""
    \\hat{{y}}_i = {intercept:.6f} + \\beta_{{group}}[group_i] + {linear_coeff:.6f} \\times income\\_level_i + \\beta_{{study}}[study_i] + {hh_coeff:.6f} \\times HH\\_zscore_i
    """)
    
    # Show coefficient lookup tables and linear effect
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**👥 Group Effects (β_group):**")
        group_data = {
            'Group': ['MidSub', 'NoSub', 'FullSub (ref)'],
            'Intercept': [
                get_coefficient('midsub', 'cont'),
                get_coefficient('nosub', 'cont'),
                get_coefficient('fullsub', 'cont')
            ]
        }
        group_df = pd.DataFrame(group_data)
        group_df['Intercept'] = group_df['Intercept'].map('{:.6f}'.format)
        st.dataframe(group_df, hide_index=True, use_container_width=True)
        
        st.markdown("**🎓 Study Programme Effects (β_study):**")
        study_data = {
            'Programme': ['Incoming', 'Law5yr', 'UG3yr', 'Grad2yr (ref)'],
            'Intercept': [
                get_coefficient('incoming', 'cont'),
                get_coefficient('law', 'cont'),
                get_coefficient('ug', 'cont'),
                get_coefficient('grad', 'cont')
            ]
        }
        study_df = pd.DataFrame(study_data)
        study_df['Intercept'] = study_df['Intercept'].map('{:.6f}'.format)
        st.dataframe(study_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**📈 Linear Income Effect (β_linear):**")
        st.caption("Effect = β_linear × actual_allowance_amount")
        
        # Show linear effect for actual allowance amounts (not level numbers)
        allowance_mapping = {1: 16, 2: 32, 3: 72, 4: 128, 5: 200}
        linear_data = {
            'Income Level': ['1 (€16)', '2 (€32)', '3 (€72)', '4 (€128)', '5 (€200)'],
            'Effect': [
                f"{linear_coeff * allowance_mapping[1]:.6f}",
                f"{linear_coeff * allowance_mapping[2]:.6f}",
                f"{linear_coeff * allowance_mapping[3]:.6f}",
                f"{linear_coeff * allowance_mapping[4]:.6f}",
                f"{linear_coeff * allowance_mapping[5]:.6f}"
            ]
        }
        linear_df = pd.DataFrame(linear_data)
        st.dataframe(linear_df, hide_index=True, use_container_width=True)
        
        # Show total range effect using actual allowances
        total_range = linear_coeff * (200 - 16)  # from €16 to €200
        st.metric("Total Range Effect (€16→€200)", f"{total_range:.6f}")


def render_intercept_override_section():
    """Render the intercept override section with ability to modify default coefficient values"""
    
    # Research default values
    CATEGORICAL_DEFAULT = 1.519818
    CONTINUOUS_DEFAULT = -0.139596
    
    # Handle reset flag BEFORE widgets render
    if st.session_state.get('_reset_intercept_flag', False):
        st.session_state._reset_intercept_flag = False
        st.session_state.override_categorical_intercept = CATEGORICAL_DEFAULT
        st.session_state.override_continuous_intercept = CONTINUOUS_DEFAULT
        st.session_state.intercept_override_values = {}
    
    # Initialize override values if not present
    if 'intercept_override_values' not in st.session_state:
        st.session_state.intercept_override_values = {}
    
    # Get current income mode to determine which intercepts to show
    income_mode = st.session_state.get('income_spec_mode', 'categorical only')
    
    # Show current configuration values for reference
    try:
        current_yaml_values = get_current_yaml_intercepts()
        
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
            
            # Override input fields based on income mode
            if income_mode == "Compare both":
                # Show both categorical and continuous
                
                # Restore categorical widget key
                cat_val = restore_widget_from_storage(
                    'override_categorical_intercept',
                    st.session_state.intercept_override_values,
                    'categorical',
                    current_yaml_values['categorical']
                )
                
                new_cat_intercept = st.number_input(
                    "Categorical",
                    value=cat_val,
                    step=0.001,
                    format="%.6f",
                    help="Override value for categorical income specification",
                    key="override_categorical_intercept",
                    on_change=lambda: auto_save_intercept('categorical', st.session_state.override_categorical_intercept)
                )
                st.session_state.intercept_override_values['categorical'] = new_cat_intercept
                
                # Restore continuous widget key
                cont_val = restore_widget_from_storage(
                    'override_continuous_intercept',
                    st.session_state.intercept_override_values,
                    'continuous',
                    current_yaml_values['continuous']
                )
                
                new_cont_intercept = st.number_input(
                    "Continuous", 
                    value=cont_val,
                    step=0.001,
                    format="%.6f",
                    help="Override value for continuous income specification",
                    key="override_continuous_intercept",
                    on_change=lambda: auto_save_intercept('continuous', st.session_state.override_continuous_intercept)
                )
                st.session_state.intercept_override_values['continuous'] = new_cont_intercept
                    
            elif "continuous" in income_mode.lower():
                # Show only continuous
                
                # Restore continuous widget key
                cont_val = restore_widget_from_storage(
                    'override_continuous_intercept',
                    st.session_state.intercept_override_values,
                    'continuous',
                    current_yaml_values['continuous']
                )
                
                new_cont_intercept = st.number_input(
                    "Continuous",
                    value=cont_val,
                    step=0.001,
                    format="%.6f", 
                    help="Override value for continuous income specification",
                    key="override_continuous_intercept",
                    on_change=lambda: auto_save_intercept('continuous', st.session_state.override_continuous_intercept)
                )
                st.session_state.intercept_override_values['continuous'] = new_cont_intercept
                
            else:
                # Show only categorical (default)
                
                # Restore categorical widget key
                cat_val = restore_widget_from_storage(
                    'override_categorical_intercept',
                    st.session_state.intercept_override_values,
                    'categorical',
                    current_yaml_values['categorical']
                )
                
                new_cat_intercept = st.number_input(
                    "Categorical",
                    value=cat_val,
                    step=0.001,
                    format="%.6f",
                    help="Override value for categorical income specification", 
                    key="override_categorical_intercept",
                    on_change=lambda: auto_save_intercept('categorical', st.session_state.override_categorical_intercept)
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
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "decisions.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    regression_coeffs = config['donation_default']['regression_coefficients']
    
    return {
        'categorical': regression_coeffs['categorical']['intercept'],
        'continuous': regression_coeffs['continuous']['intercept']
    }


def update_yaml_intercepts(override_values):
    """Update configuration file with new intercept values"""
    import yaml
    from pathlib import Path
    
    try:
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "decisions.yaml"
        
        # Load current configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update intercept values
        regression_coeffs = config['donation_default']['regression_coefficients']
        
        if 'categorical' in override_values:
            regression_coeffs['categorical']['intercept'] = float(override_values['categorical'])
            # Also update legacy regression block for backward compatibility
            config['donation_default']['regression']['intercept'] = float(override_values['categorical'])
        
        if 'continuous' in override_values:
            regression_coeffs['continuous']['intercept'] = float(override_values['continuous'])
        
        # Write back to configuration file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return True
        
    except Exception as e:
        st.error(f"Error updating configuration file: {e}")
        return False


def auto_save_intercept(intercept_type, new_value):
    """Auto-save intercept changes to configuration file"""
    try:
        # Update the override values
        if 'intercept_override_values' not in st.session_state:
            st.session_state.intercept_override_values = {}
        
        st.session_state.intercept_override_values[intercept_type] = new_value
        
        # Save to configuration file immediately
        success = update_yaml_intercepts({intercept_type: new_value})
        
        if success:
            # Reload coefficients to reflect changes
            load_donation_coefficients_from_yaml()
            # Show a brief success message
            st.toast(f"✅ {intercept_type.title()} intercept auto-saved: {new_value:.6f}", icon="💾")
        else:
            st.toast(f"❌ Failed to save {intercept_type} intercept", icon="⚠️")
            
    except Exception as e:
        st.toast(f"❌ Auto-save error: {str(e)}", icon="⚠️")


def render_adjustment_override_section():
    """Render the distribution adjustment override section"""
    
    # Fixed research default - this never changes
    research_default = -4.0
    
    # Handle reset flag BEFORE widget renders
    if st.session_state.get('_reset_adjustment_flag', False):
        st.session_state._reset_adjustment_flag = False
        st.session_state.override_adjustment_shift = 0.0
        st.session_state.adjustment_override_values = {}
    
    # Initialize adjustment values if not present
    if 'adjustment_override_values' not in st.session_state:
        st.session_state.adjustment_override_values = {}
    
    # Show current YAML values for reference
    try:
        current_yaml_values = get_current_yaml_adjustment()
        
        # Use 3 columns: Research Default, Override Value, Impact Preview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Research Default: {research_default:.3f}**")
            st.markdown("Adjustment Shift")
            st.caption("Fixed reference value from original research")
        
        with col2:
            st.markdown("**✏️ Override Value**")
            
            # Restore adjustment widget key (default to research default -4.0)
            adj_val = restore_widget_from_storage(
                'override_adjustment_shift',
                st.session_state.adjustment_override_values,
                'shift_value',
                research_default
            )
            
            # Adjustment input field
            new_adjustment = st.number_input(
                "Distribution Shift Value",
                value=float(adj_val),
                step=0.1,
                format="%.3f",
                help="Shift the distribution up (positive) or down (negative) on 0-100 scale before stochastic component",
                key="override_adjustment_shift",
                on_change=lambda: auto_save_adjustment('shift_value', st.session_state.override_adjustment_shift)
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
        
        st.caption("💡 How it works: Positive values shift the distribution up (greater donation), negative values shift it down (smaller donation)")
        
    except Exception as e:
        st.error(f"Error loading adjustment values: {e}")


def render_actions_and_management_section():
    """Render the combined actions and management section with three columns"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔄 Reset Intercept**")
        if st.button("Reset Intercept to Default Values", type="secondary", use_container_width=True, help="Reset intercept values to research defaults and update configuration", key="reset_intercept_btn"):
            # Reset to research default values
            default_values = {
                'categorical': 1.519818,  # Research default for categorical
                'continuous': -0.139596   # Research default for continuous
            }
            success = update_yaml_intercepts(default_values)
            if success:
                # Set flag to reset widgets on next rerun (before widgets render)
                st.session_state._reset_intercept_flag = True
                load_donation_coefficients_from_yaml()
                st.toast("✅ Intercepts reset to research defaults", icon="🔄")
                st.rerun()
            else:
                st.toast("❌ Failed to reset intercepts", icon="⚠️")
    
    with col2:
        st.markdown("**🔄 Reset Adjustment**")
        if st.button("Reset Adjustment to 0", type="secondary", use_container_width=True, help="Reset adjustment value to 0 and update configuration", key="reset_adjustment_btn"):
            # Reset to default value of 0.0
            success = update_yaml_adjustment({'shift_value': 0.0})
            if success:
                # Set flag to reset widget on next rerun (before widget renders)
                st.session_state._reset_adjustment_flag = True
                load_donation_coefficients_from_yaml()
                st.toast("✅ Adjustment reset to 0", icon="🔄")
                st.rerun()
            else:
                st.toast("❌ Failed to reset adjustment", icon="⚠️")
    
    with col3:
        st.markdown("**🔄 Reset All**")
        if st.button("Reset Config to Defaults", type="secondary", use_container_width=True, help="Reset entire page to research defaults (intercepts, adjustment, sliders, radio buttons)", key="reset_config_btn"):
            # Reset intercepts to research defaults
            success_intercept = update_yaml_intercepts({
                'categorical': 1.519818,
                'continuous': -0.139596
            })
            # Reset adjustment to research default (-4.0)
            success_adjustment = update_yaml_adjustment({'shift_value': -4.0})
            
            if success_intercept and success_adjustment:
                # Set flag to reset ALL widgets on next rerun (before widgets render)
                st.session_state._reset_config_to_defaults_flag = True
                load_donation_coefficients_from_yaml()
                st.toast("✅ All values reset to research defaults", icon="🔄")
                st.rerun()
            else:
                st.toast("❌ Failed to reset configuration", icon="⚠️")
    
    # Debug expander below the buttons
    with st.expander("🔍 Debug: Current Session State Values", expanded=False):
        st.write("**Current intercept values in session state:**")
        st.write(f"- Main intercept: {st.session_state.get('donation_coeff_intercept', 'NOT SET')}")
        st.write(f"- Categorical intercept: {st.session_state.get('donation_coeff_intercept_cat', 'NOT SET')}")
        st.write(f"- Continuous intercept: {st.session_state.get('donation_coeff_intercept_cont', 'NOT SET')}")
        
        # Load current configuration values for comparison
        try:
            import yaml
            from pathlib import Path
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "decisions.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            yaml_cat = config['donation_default']['regression_coefficients']['categorical']['intercept']
            yaml_cont = config['donation_default']['regression_coefficients']['continuous']['intercept']
            
            st.write("**Current configuration file values:**")
            st.write(f"- Configuration categorical: {yaml_cat}")
            st.write(f"- Configuration continuous: {yaml_cont}")
            
            # Check if they match
            if st.session_state.get('donation_coeff_intercept_cat') != yaml_cat:
                st.error("❌ Session state doesn't match configuration! Click 'Reload from Configuration' button.")
            else:
                st.success("✅ Session state matches configuration values")
                
        except Exception as e:
            st.error(f"Error reading configuration: {e}")


def get_current_yaml_adjustment():
    """Get current adjustment values from configuration file"""
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "decisions.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    adjustment_params = config['donation_default'].get('adjustment', {})
    
    return {
        'shift_value': float(adjustment_params.get('shift_value', 0.0))
    }


def update_yaml_adjustment(override_values):
    """Update configuration file with new adjustment values"""
    import yaml
    from pathlib import Path
    
    try:
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "decisions.yaml"
        
        # Load current configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update adjustment values
        if 'adjustment' not in config['donation_default']:
            config['donation_default']['adjustment'] = {}
        
        if 'shift_value' in override_values:
            config['donation_default']['adjustment']['shift_value'] = float(override_values['shift_value'])
        
        # Write back to configuration file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return True
        
    except Exception as e:
        st.error(f"Error updating configuration file: {e}")
        return False


def auto_save_adjustment(adjustment_type, new_value):
    """Auto-save adjustment changes to configuration file"""
    try:
        # Update the override values
        if 'adjustment_override_values' not in st.session_state:
            st.session_state.adjustment_override_values = {}
        
        st.session_state.adjustment_override_values[adjustment_type] = new_value
        
        # Save to configuration file immediately
        success = update_yaml_adjustment({adjustment_type: new_value})
        
        if success:
            # Reload coefficients to reflect changes
            load_donation_coefficients_from_yaml()
            # Show a brief success message
            st.toast(f"✅ {adjustment_type.replace('_', ' ').title()} auto-saved: {new_value:.3f}", icon="📊")
        else:
            st.toast(f"❌ Failed to save {adjustment_type}", icon="⚠️")
            
    except Exception as e:
        st.toast(f"❌ Auto-save error: {str(e)}", icon="⚠️")