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
    st.markdown('<h4 class="subsection-header">🔬 Regression Model Coefficients</h4>', unsafe_allow_html=True)
    st.caption("Modify regression coefficients for sensitivity analysis and alternative model specifications")
    
    # Create expandable sections for different coefficient groups
    with st.expander("📊 Model Intercept & Base Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            intercept = st.number_input(
                "Model Intercept",
                value=get_coefficient_for_input('intercept'),
                step=0.001,
                format="%.6f",
                help="Baseline prediction level (loaded from YAML based on current income mode)",
                key="donation_coeff_intercept_input"
            )
            st.session_state.donation_coeff_intercept = intercept
            
        with col2:
            hh_coeff = st.number_input(
                "Honesty-Humility Coefficient",
                value=get_coefficient_for_input('hh'),
                step=0.001,
                format="%.6f",
                help="Effect of Honesty-Humility z-score on predicted prosocial behavior (loaded from YAML based on current income mode)",
                key="donation_coeff_hh_input"
            )
            st.session_state.donation_coeff_hh = hh_coeff
    
    with st.expander("👥 Group Effects (Reference: FullSub = 0)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            midsub_coeff = st.number_input(
                "MidSub Effect",
                value=get_coefficient_for_input('midsub'),
                step=0.001,
                format="%.6f",
                help="MidSub vs FullSub difference",
                key="donation_coeff_midsub_input"
            )
            st.session_state.donation_coeff_midsub = midsub_coeff
            
        with col2:
            nosub_coeff = st.number_input(
                "NoSub Effect", 
                value=get_coefficient_for_input('nosub'),
                step=0.001,
                format="%.6f",
                help="NoSub vs FullSub difference",
                key="donation_coeff_nosub_input"
            )
            st.session_state.donation_coeff_nosub = nosub_coeff
            
        with col3:
            fullsub_coeff = st.number_input(
                "FullSub Effect",
                value=get_coefficient_for_input('fullsub'),
                step=0.001,
                format="%.6f",
                help="Reference category effect (should remain 0)",
                key="donation_coeff_fullsub_input"
            )
            st.session_state.donation_coeff_fullsub = fullsub_coeff
    
    with st.expander("💰 Categorical Income Effects (Reference: Q1 = 0)", expanded=False):
        st.caption("Used when Income Mode = 'categorical only' or in categorical part of 'Compare both'")
        st.caption("Income levels: Q1(€16, ref), Q2(€32), Q3(€72), Q4(€128), Q5(€200)")
        
        # Split into two rows for 5 quintiles
        col1, col2, col3 = st.columns(3)
        with col1:
            q1_coeff = st.number_input(
                "Q1 Effect (€16)",
                value=get_coefficient_for_input('q1'),
                step=0.001,
                format="%.6f",
                help="Q1 coefficient - reference category (should be 0.0)",
                key="donation_coeff_q1_input"
            )
            st.session_state.donation_coeff_q1 = q1_coeff
            
        with col2:
            q2_coeff = st.number_input(
                "Q2 Effect (€32)",
                value=get_coefficient_for_input('q2'),
                step=0.001,
                format="%.6f",
                help="Q2 coefficient (loaded from YAML based on current income mode)",
                key="donation_coeff_q2_input"
            )
            st.session_state.donation_coeff_q2 = q2_coeff
            
        with col3:
            q3_coeff = st.number_input(
                "Q3 Effect (€72)",
                value=get_coefficient_for_input('q3'),
                step=0.001,
                format="%.6f",
                help="Q3 coefficient (loaded from YAML based on current income mode)",
                key="donation_coeff_q3_input"
            )
            st.session_state.donation_coeff_q3 = q3_coeff
        
        # Second row for Q4 and Q5
        col4, col5, col_empty = st.columns(3)
        with col4:
            q4_coeff = st.number_input(
                "Q4 Effect (€128)",
                value=get_coefficient_for_input('q4'),
                step=0.001,
                format="%.6f",
                help="Q4 coefficient (loaded from YAML based on current income mode)",
                key="donation_coeff_q4_input"
            )
            st.session_state.donation_coeff_q4 = q4_coeff
            
        with col5:
            q5_coeff = st.number_input(
                "Q5 Effect (€200)",
                value=get_coefficient_for_input('q5'),
                step=0.001,
                format="%.6f",
                help="Q5 coefficient (loaded from YAML based on current income mode)",
                key="donation_coeff_q5_input"
            )
            st.session_state.donation_coeff_q5 = q5_coeff
            # Also update legacy Q4_Q5 for backward compatibility
            st.session_state.donation_coeff_q45 = q5_coeff
            
    with st.expander("📈 Continuous Income Effect", expanded=False):
        st.caption("Used when Income Mode = 'continuous only' or in continuous part of 'Compare both'")
        col1, col2 = st.columns([3, 1])
        with col1:
            linear_coeff = st.number_input(
                "Linear Income Coefficient",
                value=get_coefficient_for_input('linear'),
                step=0.0001,
                format="%.6f",
                help="Linear effect of actual allowance amount on predicted prosocial behavior (loaded from YAML)",
                key="donation_coeff_linear_input"
            )
            st.session_state.donation_coeff_linear = linear_coeff
        with col2:
            st.metric("Range Effect", f"{linear_coeff * 4:.4f}", 
                     help="Total effect from income level 1 to 5")
    
    with st.expander("🎓 Study Programme Effects (Reference: Grad2yr = 0)", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            incoming_coeff = st.number_input(
                "Incoming Effect",
                value=get_coefficient_for_input('incoming'),
                step=0.001,
                format="%.6f",
                help="Incoming/Exchange coefficient (loaded from YAML based on current income mode)",
                key="donation_coeff_incoming_input"
            )
            st.session_state.donation_coeff_incoming = incoming_coeff
            
        with col2:
            law_coeff = st.number_input(
                "Law5yr Effect",
                value=get_coefficient_for_input('law'),
                step=0.001,
                format="%.6f",
                help="Law5yr coefficient (loaded from YAML based on current income mode)",
                key="donation_coeff_law_input"
            )
            st.session_state.donation_coeff_law = law_coeff
            
        with col3:
            ug_coeff = st.number_input(
                "UG3yr Effect",
                value=get_coefficient_for_input('ug'),
                step=0.001,
                format="%.6f",
                help="UG3yr coefficient (loaded from YAML based on current income mode)",
                key="donation_coeff_ug_input"
            )
            st.session_state.donation_coeff_ug = ug_coeff
            
        with col4:
            grad_coeff = st.number_input(
                "Grad2yr Effect",
                value=get_coefficient_for_input('grad'),
                step=0.001,
                format="%.6f",
                help="Reference category (should remain 0)",
                key="donation_coeff_grad_input"
            )
            st.session_state.donation_coeff_grad = grad_coeff
    
    # Reset to defaults button
    if st.button("🔄 Reset All Coefficients to Defaults", key="reset_donation_coefficients"):
        # Reset all coefficients to their default values
        st.session_state.donation_coeff_intercept = 1.22985660120368
        st.session_state.donation_coeff_hh = 0.634001208840808
        st.session_state.donation_coeff_midsub = 0.856140306694656
        st.session_state.donation_coeff_nosub = -0.926633374153906
        st.session_state.donation_coeff_fullsub = 0.0
        st.session_state.donation_coeff_q1 = -0.520290427509808
        st.session_state.donation_coeff_q2 = 3.754612744416796
        st.session_state.donation_coeff_q3 = 4.001714810873598
        st.session_state.donation_coeff_q45 = 0.0
        st.session_state.donation_coeff_linear = 0.0256
        st.session_state.donation_coeff_incoming = -6.920193024391676
        st.session_state.donation_coeff_law = -2.081331674770856
        st.session_state.donation_coeff_ug = -2.139093511519692
        st.session_state.donation_coeff_grad = 0.0
        st.success("✅ All coefficients reset to default values")
        st.rerun()
    
    # Coefficient refresh button
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reload Coefficients from YAML", type="secondary", help="Force reload all coefficients from the YAML file"):
            load_donation_coefficients_from_yaml()
            st.success("✅ Coefficients reloaded from YAML!")
            st.rerun()
    
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
            config_path = Path(__file__).parent.parent.parent / "config" / "decisions.yaml"
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
        if st.button("🚀 Run Donation Default Only", type="primary", key="run_donation_default"):
            run_individual_decision("donation_default")


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
