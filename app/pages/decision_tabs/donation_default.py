# app/pages/decision_tabs/donation_default.py
"""
Donation Default decision tab configuration.
"""
import streamlit as st
import pandas as pd
from app.pages.decision_execution import run_individual_decision
from app.models import load_donation_coefficients_from_yaml


def ensure_coefficients_loaded():
    """Ensure donation coefficients are loaded from YAML before any access"""
    if 'donation_coeff_intercept' not in st.session_state:
        load_donation_coefficients_from_yaml()


def get_coefficient(name, mode_suffix=None):
    """Get a coefficient value from session state. YAML is the single source of truth."""
    ensure_coefficients_loaded()
    if mode_suffix:
        key = f'donation_coeff_{name}_{mode_suffix}'
    else:
        key = f'donation_coeff_{name}'
    
    if key not in st.session_state:
        # Try to reload from YAML
        load_donation_coefficients_from_yaml()
        if key not in st.session_state:
            st.error(f"Coefficient '{name}' not found in YAML configuration!")
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


def render_donation_default_tab():
    """Render donation_default specific configuration"""
    # Ensure coefficients are loaded from YAML
    ensure_coefficients_loaded()
    
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
                if st.session_state.income_spec_mode in ["categorical only", "continuous only", "Compare both"]:
                    st.session_state.page2_tab_income_spec_mode = st.session_state.income_spec_mode
                elif st.session_state.income_spec_mode in ["compare both", "compare side-by-side"]:
                    st.session_state.page2_tab_income_spec_mode = "Compare both"
                else:
                    st.session_state.page2_tab_income_spec_mode = "categorical only"
            
            income_spec_mode = st.radio(
                "Income Mode for Donation Model",
                ["categorical only", "continuous only", "Compare both"],
                help="Choose income treatment: categorical (5 categories), continuous (linear), or Compare both",
                key="page2_tab_income_spec_mode",
                on_change=lambda: [
                    setattr(st.session_state, 'income_spec_mode', st.session_state.page2_tab_income_spec_mode),
                    reload_coefficients_for_income_mode(),
                    clear_input_field_cache()
                ]
            )
        else:
            st.session_state.income_spec_mode = "categorical only"
    
    with col2:
        # Stochastic component option
        st.markdown('<h4 class="subsection-header">Stochastic Component</h4>', unsafe_allow_html=True)
        
        if population_mode == "Copula (synthetic)":
            # Show only Copula controls
            sigma_in_copula = st.checkbox(
                "Add Normal(anchor, σ) draw to Copula runs",
                value=st.session_state.sigma_in_copula,
                help="When enabled, Copula mode will also use the stochastic component",
                key="tab_sigma_in_copula"
            )
            st.session_state.sigma_in_copula = sigma_in_copula
            st.session_state.sigma_in_research = True  # Default for research mode
            
            # Show static sigma value and coefficient slider
            st.caption(f"📊 Base σ = 9.8995 (empirical from 280 participants)")
            sigma_coefficient = st.slider(
                "σ Coefficient (multiplier)",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.sigma_coefficient,
                step=0.1,
                help="Coefficient to multiply the base σ. Final σ = 9.8995 × coefficient",
                key="tab_sigma_coefficient"
            )
            st.session_state.sigma_coefficient = sigma_coefficient
            effective_sigma = 9.8995 * sigma_coefficient
            st.caption(f"🎯 Effective σ = 9.8995 × {sigma_coefficient:.1f} = {effective_sigma:.2f}")
            # Keep the static value in sigma_value_ui for backward compatibility
            st.session_state.sigma_value_ui = effective_sigma
            
        elif population_mode == "Research Specification":
            # Show only Research controls
            sigma_in_research = st.checkbox(
                "Use Normal(anchor, σ) draw in Research mode",
                value=st.session_state.sigma_in_research,
                help="When enabled, Research mode will add stochastic variation via Normal(anchor, σ) draws. When disabled, only the anchor value is used.",
                key="tab_sigma_in_research"
            )
            st.session_state.sigma_in_research = sigma_in_research
            st.session_state.sigma_in_copula = False  # Not applicable
            
            # Show sigma coefficient slider only if stochastic component is enabled
            if sigma_in_research:
                st.caption(f"📊 Base σ = 9.8995 (empirical from 280 participants)")
                sigma_coefficient = st.slider(
                    "σ Coefficient (multiplier)",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.sigma_coefficient,
                    step=0.1,
                    help="Coefficient to multiply the base σ. Final σ = 9.8995 × coefficient",
                    key="tab_sigma_coefficient_research"
                )
                st.session_state.sigma_coefficient = sigma_coefficient
                effective_sigma = 9.8995 * sigma_coefficient
                st.caption(f"🎯 Effective σ = 9.8995 × {sigma_coefficient:.1f} = {effective_sigma:.2f}")
                # Keep the static value in sigma_value_ui for backward compatibility
                st.session_state.sigma_value_ui = effective_sigma
            else:
                st.info("ℹ️ Stochastic component disabled - using anchor values directly")
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
            sigma_in_copula = st.checkbox(
                "Add Normal(anchor, σ) draw to Copula runs",
                value=st.session_state.sigma_in_copula,
                help="When enabled, Copula mode will also use the stochastic component",
                key="tab_sigma_in_copula_compare"
            )
            st.session_state.sigma_in_copula = sigma_in_copula
            
            st.markdown("**Research Specification Controls:**")
            sigma_in_research = st.checkbox(
                "Use Normal(anchor, σ) draw in Research Specification mode",
                value=st.session_state.sigma_in_research,
                help="When enabled, Research Specification mode will add stochastic variation via Normal(anchor, σ) draws. When disabled, only the anchor value is used.",
                key="tab_sigma_in_research_compare"
            )
            st.session_state.sigma_in_research = sigma_in_research
            
            st.markdown("**Research Baseline:** Always uses anchor values only (no stochastic component)")
            st.caption("🎯 Research Baseline = deterministic anchor = 0.75 × observed + 0.25 × predicted")
            
            # Show sigma coefficient slider if either mode has stochastic enabled
            if sigma_in_copula or sigma_in_research:
                st.caption(f"📊 Base σ = 9.8995 (empirical from 280 participants)")
                sigma_coefficient = st.slider(
                    "σ Coefficient (multiplier)",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.sigma_coefficient,
                    step=0.1,
                    help="Coefficient to multiply the base σ. Final σ = 9.8995 × coefficient",
                    key="tab_sigma_coefficient_compare"
                )
                st.session_state.sigma_coefficient = sigma_coefficient
                effective_sigma = 9.8995 * sigma_coefficient
                st.caption(f"🎯 Effective σ = 9.8995 × {sigma_coefficient:.1f} = {effective_sigma:.2f}")
                # Keep the static value in sigma_value_ui for backward compatibility
                st.session_state.sigma_value_ui = effective_sigma
            else:
                st.info("ℹ️ Stochastic component disabled for both modes - using anchor values directly")
                st.session_state.sigma_coefficient = 0.0
                st.session_state.sigma_value_ui = 0.0
        
        # Anchor weights
        if population_mode != "Dependent variable resampling":
            st.markdown('<h4 class="subsection-header">Anchor Mix</h4>', unsafe_allow_html=True)
            anchor_observed_weight = st.slider(
                "Weight for observed vs modeled prosocial behavior",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.anchor_observed_weight,
                step=0.05,
                help="Anchor = w × Observed + (1-w) × Predicted",
                key="tab_anchor_weight"
            )
            st.session_state.anchor_observed_weight = anchor_observed_weight
            st.caption(f"Predicted weight: {1 - anchor_observed_weight:.2f}")
        else:
            st.session_state.anchor_observed_weight = 0.75
    
    # Raw output option - only show if stochastic component is enabled
    stochastic_enabled = (
        (population_mode == "Copula (synthetic)" and st.session_state.sigma_in_copula) or
        (population_mode == "Research Specification" and st.session_state.sigma_in_research) or
        (population_mode == "Compare both" and (st.session_state.sigma_in_copula or st.session_state.sigma_in_research))
    )
    
    if stochastic_enabled:
        st.markdown('<h4 class="subsection-header">Output Options</h4>', unsafe_allow_html=True)
        raw_draw_mode = st.checkbox(
            "Show pre-truncation (raw) donation rate",
            value=st.session_state.raw_draw_mode,
            help="Display the raw Normal(anchor, σ) draw before processing",
            key="tab_raw_draw_mode"
        )
        st.session_state.raw_draw_mode = raw_draw_mode
    else:
        st.session_state.raw_draw_mode = False
    
    # NEW: Mathematical Formula Display Section
    st.markdown('<h4 class="subsection-header">📐 Mathematical Model Formula</h4>', unsafe_allow_html=True)
    render_formula_display()
    
    # NEW: Regression Coefficients Section
    st.markdown('<h4 class="subsection-header">🔢 Intercept Override</h4>', unsafe_allow_html=True)
    render_intercept_override_section()
    
    # NEW: Distribution Adjustment Section
    st.markdown('<h4 class="subsection-header">📊 Distribution Adjustment</h4>', unsafe_allow_html=True)
    render_adjustment_override_section()
    
    # Coefficient refresh and debug section
    st.markdown("---")
    st.markdown('<h4 class="subsection-header">🔧 Coefficient Management</h4>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Debug: Show current session state values
        with st.expander("🔍 Debug: Current Session State Values", expanded=False):
            st.write("**Current intercept values in session state:**")
            st.write(f"- Main intercept: {st.session_state.get('donation_coeff_intercept', 'NOT SET')}")
            st.write(f"- Categorical intercept: {st.session_state.get('donation_coeff_intercept_cat', 'NOT SET')}")
            st.write(f"- Continuous intercept: {st.session_state.get('donation_coeff_intercept_cont', 'NOT SET')}")
            
            # Load current YAML values for comparison
            try:
                import yaml
                from pathlib import Path
                config_path = Path(__file__).parent.parent.parent.parent / "config" / "decisions.yaml"
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                yaml_cat = config['donation_default']['regression_coefficients']['categorical']['intercept']
                yaml_cont = config['donation_default']['regression_coefficients']['continuous']['intercept']
                
                st.write("**Current YAML values:**")
                st.write(f"- YAML categorical: {yaml_cat}")
                st.write(f"- YAML continuous: {yaml_cont}")
                
                # Check if they match
                if st.session_state.get('donation_coeff_intercept_cat') != yaml_cat:
                    st.error("❌ Session state doesn't match YAML! Click 'Reload Coefficients' button.")
                else:
                    st.success("✅ Session state matches YAML values")
                    
            except Exception as e:
                st.error(f"Error reading YAML: {e}")
    
    with col2:
        if st.button("🔄 Reload Coefficients from YAML", type="secondary", use_container_width=True, help="Force reload all coefficients from the YAML file"):
            load_donation_coefficients_from_yaml()
            st.success("✅ Coefficients reloaded from YAML!")
            st.rerun()
    
    # Render both individual and complete simulation buttons
    from app.pages.decision_execution import render_simulation_buttons
    render_simulation_buttons(
        decision_name="donation_default",
        selected_decisions=st.session_state.decision_params.selected_decisions
    )


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
    
    with st.expander("🧮 Live Example Calculation", expanded=False):
        render_interactive_example()
    
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
            'Coefficient': [
                st.session_state.get('donation_coeff_midsub', 0.856140306694656),
                st.session_state.get('donation_coeff_nosub', -0.926633374153906),
                st.session_state.get('donation_coeff_fullsub', 0.0)
            ]
        }
        group_df = pd.DataFrame(group_data)
        group_df['Coefficient'] = group_df['Coefficient'].map('{:.6f}'.format)
        st.dataframe(group_df, hide_index=True, use_container_width=True)
        
        st.markdown("**🎓 Study Programme Effects (β_study):**")
        study_data = {
            'Programme': ['Incoming', 'Law5yr', 'UG3yr', 'Grad2yr (ref)'],
            'Coefficient': [
                st.session_state.get('donation_coeff_incoming', -6.920193024391676),
                st.session_state.get('donation_coeff_law', -2.081331674770856),
                st.session_state.get('donation_coeff_ug', -2.139093511519692),
                st.session_state.get('donation_coeff_grad', 0.0)
            ]
        }
        study_df = pd.DataFrame(study_data)
        study_df['Coefficient'] = study_df['Coefficient'].map('{:.6f}'.format)
        st.dataframe(study_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**💰 Income Quintile Effects (β_income_q):**")
        income_data = {
            'Quintile': ['Q1 (Level 1)', 'Q2 (Level 2)', 'Q3 (Level 3)', 'Q4_Q5 (Levels 4-5, ref)'],
            'Coefficient': [
                st.session_state.get('donation_coeff_q1', -0.520290427509808),
                st.session_state.get('donation_coeff_q2', 3.754612744416796),
                st.session_state.get('donation_coeff_q3', 4.001714810873598),
                st.session_state.get('donation_coeff_q45', 0.0)
            ]
        }
        income_df = pd.DataFrame(income_data)
        income_df['Coefficient'] = income_df['Coefficient'].map('{:.6f}'.format)
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
            'Coefficient': [
                st.session_state.get('donation_coeff_midsub', 0.856140306694656),
                st.session_state.get('donation_coeff_nosub', -0.926633374153906),
                st.session_state.get('donation_coeff_fullsub', 0.0)
            ]
        }
        group_df = pd.DataFrame(group_data)
        group_df['Coefficient'] = group_df['Coefficient'].map('{:.6f}'.format)
        st.dataframe(group_df, hide_index=True, use_container_width=True)
        
        st.markdown("**🎓 Study Programme Effects (β_study):**")
        study_data = {
            'Programme': ['Incoming', 'Law5yr', 'UG3yr', 'Grad2yr (ref)'],
            'Coefficient': [
                st.session_state.get('donation_coeff_incoming', -6.920193024391676),
                st.session_state.get('donation_coeff_law', -2.081331674770856),
                st.session_state.get('donation_coeff_ug', -2.139093511519692),
                st.session_state.get('donation_coeff_grad', 0.0)
            ]
        }
        study_df = pd.DataFrame(study_data)
        study_df['Coefficient'] = study_df['Coefficient'].map('{:.6f}'.format)
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


def render_interactive_example():
    """Render interactive example calculation"""
    
    st.markdown("**🎛️ Interactive Example Calculation**")
    st.caption("Select agent characteristics to see step-by-step prediction calculation")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        example_group = st.selectbox(
            "Group", 
            ["MidSub", "NoSub", "FullSub"],
            help="Experimental group assignment"
        )
    
    with col2:
        example_income = st.slider(
            "Income Level", 
            1, 5, 3,
            help="Income level from 1 (lowest) to 5 (highest)"
        )
    
    with col3:
        example_study = st.selectbox(
            "Study Programme", 
            ["Incoming", "Law5yr", "UG3yr", "Grad2yr"],
            help="Academic programme category"
        )
    
    with col4:
        example_hh = st.slider(
            "Honesty-Humility", 
            1.0, 5.0, 3.4, 0.1,
            help="Honesty-Humility score (1-5 scale)"
        )
    
    # Calculate step-by-step
    income_mode = st.session_state.get('income_spec_mode', 'categorical')
    
    if income_mode in ["categorical only", "Compare both"]:
        st.markdown("**📊 Categorical Mode Calculation:**")
        calculate_categorical_example(example_group, example_income, example_study, example_hh)
    
    if income_mode in ["continuous only", "Compare both"]:
        if income_mode == "Compare both":
            st.markdown("---")
        st.markdown("**📈 Continuous Mode Calculation:**")
        calculate_continuous_example(example_group, example_income, example_study, example_hh)


def calculate_categorical_example(group, income_level, study, hh_score):
    """Calculate and display categorical mode example"""
    
    # Get coefficients FROM YAML ONLY - no hardcoded fallbacks
    intercept = get_coefficient('intercept', 'cat')
    hh_coeff = get_coefficient('hh', 'cat')
    
    # Group coefficient
    group_coeffs = {
        'MidSub': get_coefficient('midsub', 'cat'),
        'NoSub': get_coefficient('nosub', 'cat'),
        'FullSub': get_coefficient('fullsub', 'cat')
    }
    group_coeff = group_coeffs[group]
    
    # Income quintile coefficient
    income_mapping = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'Q5'}
    income_quintile = income_mapping[income_level]
    income_coeffs = {
        'Q1': get_coefficient('q1', 'cat'),
        'Q2': get_coefficient('q2', 'cat'),
        'Q3': get_coefficient('q3', 'cat'),
        'Q4': get_coefficient('q4', 'cat'),
        'Q5': get_coefficient('q5', 'cat')
    }
    income_coeff = income_coeffs[income_quintile]
    
    # Study coefficient
    study_coeffs = {
        'Incoming': get_coefficient('incoming', 'cat'),
        'Law5yr': get_coefficient('law', 'cat'),
        'UG3yr': get_coefficient('ug', 'cat'),
        'Grad2yr': get_coefficient('grad', 'cat')
    }
    study_coeff = study_coeffs[study]
    
    # HH z-score calculation
    hh_mean = 3.3922
    hh_std = 0.5587
    hh_zscore = (hh_score - hh_mean) / hh_std
    hh_term = hh_coeff * hh_zscore
    
    # Step-by-step calculation
    steps_data = {
        'Component': [
            'Intercept (β₀)',
            f'Group Effect: {group}',
            f'Income Effect: {income_quintile} (Level {income_level})',
            f'Study Effect: {study}',
            f'HH Effect: {hh_score:.1f} → z-score = {hh_zscore:.3f}'
        ],
        'Coefficient': [
            f'{intercept:.6f}',
            f'{group_coeff:.6f}',
            f'{income_coeff:.6f}',
            f'{study_coeff:.6f}',
            f'{hh_coeff:.6f} × {hh_zscore:.3f} = {hh_term:.6f}'
        ],
        'Value': [
            intercept,
            group_coeff,
            income_coeff,
            study_coeff,
            hh_term
        ]
    }
    
    steps_df = pd.DataFrame(steps_data)
    st.dataframe(steps_df, hide_index=True, use_container_width=True)
    
    # Final prediction
    predicted = intercept + group_coeff + income_coeff + study_coeff + hh_term
    st.success(f"**🎯 Predicted Prosocial Behavior: {predicted:.6f}**")


def calculate_continuous_example(group, income_level, study, hh_score):
    """Calculate and display continuous mode example"""
    
    # Get coefficients FROM YAML ONLY - no hardcoded fallbacks
    intercept = get_coefficient('intercept', 'cont')
    hh_coeff = get_coefficient('hh', 'cont')
    linear_coeff = get_coefficient('linear', 'cont')
    
    # Group coefficient
    group_coeffs = {
        'MidSub': get_coefficient('midsub', 'cont'),
        'NoSub': get_coefficient('nosub', 'cont'),
        'FullSub': get_coefficient('fullsub', 'cont')
    }
    group_coeff = group_coeffs[group]
    
    # Linear income term - use actual allowance amount
    allowance_mapping = {1: 16, 2: 32, 3: 72, 4: 128, 5: 200}
    actual_allowance = allowance_mapping[income_level]
    income_term = linear_coeff * actual_allowance
    
    # Study coefficient
    study_coeffs = {
        'Incoming': get_coefficient('incoming', 'cont'),
        'Law5yr': get_coefficient('law', 'cont'),
        'UG3yr': get_coefficient('ug', 'cont'),
        'Grad2yr': get_coefficient('grad', 'cont')
    }
    study_coeff = study_coeffs[study]
    
    # HH z-score calculation
    hh_mean = 3.3922
    hh_std = 0.5587
    hh_zscore = (hh_score - hh_mean) / hh_std
    hh_term = hh_coeff * hh_zscore
    
    # Step-by-step calculation
    steps_data = {
        'Component': [
            'Intercept (β₀)',
            f'Group Effect: {group}',
            f'Income Effect: Linear (Level {income_level})',
            f'Study Effect: {study}',
            f'HH Effect: {hh_score:.1f} → z-score = {hh_zscore:.3f}'
        ],
        'Coefficient': [
            f'{intercept:.6f}',
            f'{group_coeff:.6f}',
            f'{linear_coeff:.6f} × {income_level} = {income_term:.6f}',
            f'{study_coeff:.6f}',
            f'{hh_coeff:.6f} × {hh_zscore:.3f} = {hh_term:.6f}'
        ],
        'Value': [
            intercept,
            group_coeff,
            income_term,
            study_coeff,
            hh_term
        ]
    }
    
    steps_df = pd.DataFrame(steps_data)
    st.dataframe(steps_df, hide_index=True, use_container_width=True)
    
    # Final prediction
    predicted = intercept + group_coeff + income_term + study_coeff + hh_term
    st.success(f"**🎯 Predicted Prosocial Behavior: {predicted:.6f}**")


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
    """Reload coefficients from YAML when income mode changes"""
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
            'Coefficient': [
                get_coefficient('midsub', 'cat'),
                get_coefficient('nosub', 'cat'),
                get_coefficient('fullsub', 'cat')
            ]
        }
        group_df = pd.DataFrame(group_data)
        group_df['Coefficient'] = group_df['Coefficient'].map('{:.6f}'.format)
        st.dataframe(group_df, hide_index=True, use_container_width=True)
        
        st.markdown("**🎓 Study Programme Effects (β_study):**")
        study_data = {
            'Programme': ['Incoming', 'Law5yr', 'UG3yr', 'Grad2yr (ref)'],
            'Coefficient': [
                get_coefficient('incoming', 'cat'),
                get_coefficient('law', 'cat'),
                get_coefficient('ug', 'cat'),
                get_coefficient('grad', 'cat')
            ]
        }
        study_df = pd.DataFrame(study_data)
        study_df['Coefficient'] = study_df['Coefficient'].map('{:.6f}'.format)
        st.dataframe(study_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**💰 Income Quintile Effects (β_income_q):**")
        income_data = {
            'Quintile': ['Q1 (Level 16, ref)', 'Q2 (Level 32)', 'Q3 (Level 72)', 'Q4 (Level 128)', 'Q5 (Level 200)'],
            'Coefficient': [
                get_coefficient('q1', 'cat'),
                get_coefficient('q2', 'cat'),
                get_coefficient('q3', 'cat'),
                get_coefficient('q4', 'cat'),
                get_coefficient('q5', 'cat')
            ]
        }
        income_df = pd.DataFrame(income_data)
        income_df['Coefficient'] = income_df['Coefficient'].map('{:.6f}'.format)
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
            'Coefficient': [
                get_coefficient('midsub', 'cont'),
                get_coefficient('nosub', 'cont'),
                get_coefficient('fullsub', 'cont')
            ]
        }
        group_df = pd.DataFrame(group_data)
        group_df['Coefficient'] = group_df['Coefficient'].map('{:.6f}'.format)
        st.dataframe(group_df, hide_index=True, use_container_width=True)
        
        st.markdown("**🎓 Study Programme Effects (β_study):**")
        study_data = {
            'Programme': ['Incoming', 'Law5yr', 'UG3yr', 'Grad2yr (ref)'],
            'Coefficient': [
                get_coefficient('incoming', 'cont'),
                get_coefficient('law', 'cont'),
                get_coefficient('ug', 'cont'),
                get_coefficient('grad', 'cont')
            ]
        }
        study_df = pd.DataFrame(study_data)
        study_df['Coefficient'] = study_df['Coefficient'].map('{:.6f}'.format)
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
    """Render the intercept override section with ability to modify YAML values"""
    
    # Initialize override values if not present
    if 'intercept_override_values' not in st.session_state:
        st.session_state.intercept_override_values = {}
    
    # Get current income mode to determine which intercepts to show
    income_mode = st.session_state.get('income_spec_mode', 'categorical only')
    
    # Show current YAML values for reference
    try:
        current_yaml_values = get_current_yaml_intercepts()
        
        st.markdown("**📋 Current YAML Values:**")
        yaml_col1, yaml_col2 = st.columns(2)
        
        with yaml_col1:
            st.metric("Categorical Intercept", f"{current_yaml_values['categorical']:.6f}")
        
        with yaml_col2:
            st.metric("Continuous Intercept", f"{current_yaml_values['continuous']:.6f}")
        
        st.markdown("---")
        st.markdown("**✏️ Override Values:**")
        
        # Override input fields based on income mode
        if income_mode == "Compare both":
            # Show both categorical and continuous
            col1, col2 = st.columns(2)
            
            with col1:
                new_cat_intercept = st.number_input(
                    "New Categorical Intercept",
                    value=st.session_state.intercept_override_values.get('categorical', current_yaml_values['categorical']),
                    step=0.001,
                    format="%.6f",
                    help="Override value for categorical income specification",
                    key="override_categorical_intercept",
                    on_change=lambda: auto_save_intercept('categorical', st.session_state.override_categorical_intercept)
                )
                st.session_state.intercept_override_values['categorical'] = new_cat_intercept
            
            with col2:
                new_cont_intercept = st.number_input(
                    "New Continuous Intercept", 
                    value=st.session_state.intercept_override_values.get('continuous', current_yaml_values['continuous']),
                    step=0.001,
                    format="%.6f",
                    help="Override value for continuous income specification",
                    key="override_continuous_intercept",
                    on_change=lambda: auto_save_intercept('continuous', st.session_state.override_continuous_intercept)
                )
                st.session_state.intercept_override_values['continuous'] = new_cont_intercept
                
        elif "continuous" in income_mode.lower():
            # Show only continuous
            new_cont_intercept = st.number_input(
                "New Continuous Intercept",
                value=st.session_state.intercept_override_values.get('continuous', current_yaml_values['continuous']),
                step=0.001,
                format="%.6f", 
                help="Override value for continuous income specification",
                key="override_continuous_intercept",
                on_change=lambda: auto_save_intercept('continuous', st.session_state.override_continuous_intercept)
            )
            st.session_state.intercept_override_values['continuous'] = new_cont_intercept
            
        else:
            # Show only categorical (default)
            new_cat_intercept = st.number_input(
                "New Categorical Intercept",
                value=st.session_state.intercept_override_values.get('categorical', current_yaml_values['categorical']),
                step=0.001,
                format="%.6f",
                help="Override value for categorical income specification", 
                key="override_categorical_intercept",
                on_change=lambda: auto_save_intercept('categorical', st.session_state.override_categorical_intercept)
            )
            st.session_state.intercept_override_values['categorical'] = new_cat_intercept
        
        # Action buttons
        st.markdown("---")
        if st.button("🔄 Reset to Default Values", help="Reset intercept values to research defaults and update YAML"):
            # Reset to research default values
            default_values = {
                'categorical': 1.519818,  # Research default for categorical
                'continuous': -0.139596   # Research default for continuous
            }
            success = update_yaml_intercepts(default_values)
            if success:
                st.session_state.intercept_override_values = {}
                load_donation_coefficients_from_yaml()
                st.toast("✅ Intercepts reset to research defaults", icon="🔄")
                st.rerun()
            else:
                st.toast("❌ Failed to reset intercepts", icon="⚠️")
        
        # Show impact preview
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
        st.error(f"Error loading YAML values: {e}")


def get_current_yaml_intercepts():
    """Get current intercept values from YAML file"""
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
    """Update YAML file with new intercept values"""
    import yaml
    from pathlib import Path
    
    try:
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "decisions.yaml"
        
        # Load current YAML
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
        
        # Write back to YAML
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return True
        
    except Exception as e:
        st.error(f"Error updating YAML: {e}")
        return False


def auto_save_intercept(intercept_type, new_value):
    """Auto-save intercept changes to YAML file"""
    try:
        # Update the override values
        if 'intercept_override_values' not in st.session_state:
            st.session_state.intercept_override_values = {}
        
        st.session_state.intercept_override_values[intercept_type] = new_value
        
        # Save to YAML immediately
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
    
    # Initialize adjustment values if not present
    if 'adjustment_override_values' not in st.session_state:
        st.session_state.adjustment_override_values = {}
    
    # Show current YAML values for reference
    try:
        current_yaml_values = get_current_yaml_adjustment()
        
        st.markdown("**📋 Current YAML Values:**")
        st.metric("Adjustment Shift", f"{current_yaml_values['shift_value']:.3f}")
        
        st.markdown("---")
        st.markdown("**✏️ Override Values:**")
        
        # Adjustment input field
        new_adjustment = st.number_input(
            "Distribution Shift Value",
            value=st.session_state.adjustment_override_values.get('shift_value', current_yaml_values['shift_value']),
            step=0.1,
            format="%.3f",
            help="Shift the distribution up (positive) or down (negative) on 0-100 scale before stochastic component",
            key="override_adjustment_shift",
            on_change=lambda: auto_save_adjustment('shift_value', st.session_state.override_adjustment_shift)
        )
        st.session_state.adjustment_override_values['shift_value'] = new_adjustment
        
        st.caption("💡 **How it works**: Positive values shift the distribution higher (more donation), negative values shift it lower (less donation)")
        
        # Action button
        st.markdown("---")
        if st.button("🔄 Reset Adjustment to Default (0.0)", help="Reset adjustment value to default 0.0 and update YAML"):
            # Reset to default value of 0.0
            success = update_yaml_adjustment({'shift_value': 0.0})
            if success:
                st.session_state.adjustment_override_values = {}
                load_donation_coefficients_from_yaml()
                st.toast("✅ Adjustment reset to default (0.0)", icon="🔄")
                st.rerun()
            else:
                st.toast("❌ Failed to reset adjustment", icon="⚠️")
        
        # Show impact preview
        if st.session_state.adjustment_override_values:
            current_value = current_yaml_values['shift_value']
            new_value = new_adjustment
            change = new_value - current_value
            
            if abs(change) > 0.001:  # Only show if there's a meaningful change
                st.markdown("**📊 Impact Preview:**")
                
                impact_data = [{
                    'Parameter': 'Distribution Shift',
                    'Current': f"{current_value:.3f}",
                    'New': f"{new_value:.3f}", 
                    'Change': f"{change:+.3f}",
                    'Impact': "Higher donations" if change > 0 else "Lower donations" if change < 0 else "No change"
                }]
                
                impact_df = pd.DataFrame(impact_data)
                st.dataframe(impact_df, hide_index=True, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading adjustment values: {e}")


def get_current_yaml_adjustment():
    """Get current adjustment values from YAML file"""
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "decisions.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    adjustment_params = config['donation_default'].get('adjustment', {})
    
    return {
        'shift_value': adjustment_params.get('shift_value', 0.0)
    }


def update_yaml_adjustment(override_values):
    """Update YAML file with new adjustment values"""
    import yaml
    from pathlib import Path
    
    try:
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "decisions.yaml"
        
        # Load current YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update adjustment values
        if 'adjustment' not in config['donation_default']:
            config['donation_default']['adjustment'] = {}
        
        if 'shift_value' in override_values:
            config['donation_default']['adjustment']['shift_value'] = float(override_values['shift_value'])
        
        # Write back to YAML
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return True
        
    except Exception as e:
        st.error(f"Error updating YAML: {e}")
        return False


def auto_save_adjustment(adjustment_type, new_value):
    """Auto-save adjustment changes to YAML file"""
    try:
        # Update the override values
        if 'adjustment_override_values' not in st.session_state:
            st.session_state.adjustment_override_values = {}
        
        st.session_state.adjustment_override_values[adjustment_type] = new_value
        
        # Save to YAML immediately
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