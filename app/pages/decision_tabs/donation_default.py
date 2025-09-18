# app/pages/decision_tabs/donation_default.py
"""
Donation Default decision tab configuration.
"""
import streamlit as st
from app.pages.decision_execution import run_individual_decision


def render_donation_default_tab():
    """Render donation_default specific configuration"""
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
                on_change=lambda: setattr(st.session_state, 'income_spec_mode', st.session_state.page2_tab_income_spec_mode)
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
    
    # Individual run button
    st.markdown("---")
    if st.button("🚀 Run Donation Default Only", type="secondary", width="stretch"):
        run_individual_decision("donation_default")
