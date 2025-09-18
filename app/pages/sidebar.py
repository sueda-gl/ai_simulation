# app/pages/sidebar.py
"""
Sidebar configuration for the Enhanced AI Agent Simulation.
"""
import streamlit as st
from app.models import get_decision_global_parameters, get_all_global_parameters
from app.simulation import run_simulation_from_sidebar, run_monte_carlo_study


def configure_sidebar(selected_decisions):
    """Configure the sidebar based on selected decisions"""
    st.sidebar.title("⚙️ Decision Parameters")
    
    # Only show sidebar content if decisions are selected
    if not selected_decisions:
        st.sidebar.info("👈 Select decisions on the main page to see applicable parameters")
        # Store minimal settings in session state for no decisions selected
        st.session_state.n_agents = 1000
        st.session_state.simulation_mode = "Single Run"
        st.session_state.seed = 42
        st.session_state.show_individual_agents = False
        st.session_state.save_results = True
    else:
        # Show decision-specific parameters only for selected decisions
        
        # Only show donation-specific parameters if donation_default is selected
        if "donation_default" in selected_decisions:
            # Show current population mode (configured on Page 1)
            st.sidebar.subheader("Population Generation")
            population_mode = st.session_state.population_mode  # Use value from Page 1
            st.sidebar.info(f"🔧 **Mode**: {population_mode}")
            st.sidebar.caption("💡 Configured on Page 1")
            
            # Income specification selector
            if population_mode != "Dependent variable resampling":
                st.sidebar.subheader("Income Specification")
                income_spec_mode = st.sidebar.radio(
                    "Income Mode for Donation Model",
                    ["categorical only", "continuous only", "Compare both"],
                    index=0,
                    help="Choose income treatment: categorical (5 categories), continuous (linear), or Compare both"
                )
            else:
                income_spec_mode = "categorical only"
            
            # Stochastic component option
            st.sidebar.subheader("Stochastic Component")
            
            if population_mode == "Copula (synthetic)":
                # Show only Copula controls
                sigma_in_copula = st.sidebar.checkbox(
                    "Add Normal(anchor, σ) draw to Copula runs",
                    value=st.session_state.sigma_in_copula,
                    help="When enabled, Copula mode will also use the stochastic component"
                )
                sigma_in_research = True  # Default for research mode
                
                # Show static sigma value and coefficient slider
                st.sidebar.caption("📊 Base σ = 9.8995 (empirical)")
                sigma_coefficient = st.sidebar.slider(
                    "σ Coefficient (multiplier)",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.sigma_coefficient,
                    step=0.1,
                    help="Coefficient to multiply the base σ"
                )
                effective_sigma = 9.8995 * sigma_coefficient
                st.sidebar.caption(f"🎯 Effective σ = {effective_sigma:.2f}")
                # Keep the static value in sigma_value_ui for backward compatibility
                sigma_value_ui = effective_sigma
                
            elif population_mode == "Research Specification":
                # Show only Research controls
                sigma_in_research = st.sidebar.checkbox(
                    "Use Normal(anchor, σ) draw in Research mode",
                    value=st.session_state.sigma_in_research,
                    help="When enabled, Research mode will add stochastic variation via Normal(anchor, σ) draws. When disabled, only the anchor value is used."
                )
                sigma_in_copula = False  # Not applicable
                
                # Show sigma coefficient slider only if stochastic component is enabled
                if sigma_in_research:
                    st.sidebar.caption("📊 Base σ = 9.8995 (empirical)")
                    sigma_coefficient = st.sidebar.slider(
                        "σ Coefficient (multiplier)",
                        min_value=0.0,
                        max_value=2.0,
                        value=st.session_state.sigma_coefficient,
                        step=0.1,
                        help="Coefficient to multiply the base σ"
                    )
                    effective_sigma = 9.8995 * sigma_coefficient
                    st.sidebar.caption(f"🎯 Effective σ = {effective_sigma:.2f}")
                    sigma_value_ui = effective_sigma
                else:
                    st.sidebar.info("ℹ️ Stochastic component disabled")
                    sigma_coefficient = 0.0
                    sigma_value_ui = 0.0
                    
            elif population_mode == "Research Baseline":
                # Research Baseline mode - no stochastic component
                sigma_in_copula = False  # Not applicable
                sigma_in_research = False  # No stochastic component
                sigma_coefficient = 0.0
                sigma_value_ui = 0.0
                
                st.sidebar.info("📊 Research Baseline: Anchor values only (no stochastic component)")
                st.sidebar.caption("🎯 Returns deterministic anchor = 0.75 × observed + 0.25 × predicted")
                    
            else:  # Compare all
                # Show controls for all three modes
                st.sidebar.markdown("**Copula Mode:**")
                sigma_in_copula = st.sidebar.checkbox(
                    "Add Normal(anchor, σ) draw to Copula runs",
                    value=st.session_state.sigma_in_copula,
                    help="When enabled, Copula mode will also use the stochastic component"
                )
                
                st.sidebar.markdown("**Research Specification:**")
                sigma_in_research = st.sidebar.checkbox(
                    "Use Normal(anchor, σ) draw in Research Specification mode",
                    value=st.session_state.sigma_in_research,
                    help="When enabled, Research Specification mode will add stochastic variation"
                )
                
                st.sidebar.markdown("**Research Baseline:** Always uses anchor values only")
                st.sidebar.caption("🎯 Baseline = deterministic anchor")
                
                # Show sigma coefficient slider if either mode has stochastic enabled
                if sigma_in_copula or sigma_in_research:
                    st.sidebar.caption("📊 Base σ = 9.8995 (empirical)")
                    sigma_coefficient = st.sidebar.slider(
                        "σ Coefficient (multiplier)",
                        min_value=0.0,
                        max_value=2.0,
                        value=st.session_state.sigma_coefficient,
                        step=0.1,
                        help="Coefficient to multiply the base σ"
                    )
                    effective_sigma = 9.8995 * sigma_coefficient
                    st.sidebar.caption(f"🎯 Effective σ = {effective_sigma:.2f}")
                    sigma_value_ui = effective_sigma
                else:
                    st.sidebar.info("ℹ️ Stochastic disabled for both modes")
                    sigma_coefficient = 0.0
                    sigma_value_ui = 0.0
            
            # Anchor weights slider
            if population_mode != "Dependent variable resampling":
                st.sidebar.subheader("Anchor Mix")
                anchor_observed_weight = st.sidebar.slider(
                    "Weight for observed versus modeled prosocial behavior",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.75,
                    step=0.05,
                    help="Anchor = w × Observed + (1-w) × Predicted. Default is 0.75 observed + 0.25 predicted."
                )
                st.sidebar.caption(f"Predicted weight: {1 - anchor_observed_weight:.2f}")
            else:
                anchor_observed_weight = 0.75
            
            # Raw output option - only show if stochastic component is enabled
            stochastic_enabled = (
                (population_mode == "Copula (synthetic)" and sigma_in_copula) or
                (population_mode == "Research Specification" and sigma_in_research) or
                (population_mode == "Compare both" and (sigma_in_copula or sigma_in_research))
            )
            
            if stochastic_enabled:
                st.sidebar.subheader("Output Options")
                raw_draw_mode = st.sidebar.checkbox(
                    "Show pre-truncation (raw) donation rate",
                    value=st.session_state.raw_draw_mode,
                    help="Display the raw Normal(anchor, σ) draw before any processing. This shows negative values and the full range of the stochastic draw before flooring at 0 and rescaling by personal maximum."
                )
            else:
                raw_draw_mode = False
        else:
            # If donation_default is not selected, set default values
            population_mode = "Copula (synthetic)"
            income_spec_mode = "categorical only"
            sigma_in_copula = False
            sigma_value_ui = 9.0
            anchor_observed_weight = 0.75
            raw_draw_mode = False

        # Show applicable global parameters dynamically
        all_applicable = get_decision_global_parameters(selected_decisions)
        
        # Only show income parameters if they are applicable
        if any(param in all_applicable for param in ['income_distribution', 'income_min', 'income_max', 'income_avg', 'discount_income_threshold']):
            st.sidebar.subheader("💵 Income Distribution")
            st.sidebar.caption("✅ Applicable for selected decisions")
            
            if 'income_distribution' in all_applicable:
                income_dist_type = st.sidebar.selectbox(
                    "Distribution Type",
                    ["lognormal", "pareto", "weibull"],
                    index=["lognormal", "pareto", "weibull"].index(st.session_state.sim_params.income_distribution),
                    help="✅ Applicable for selected decisions"
                )
                st.session_state.sim_params.income_distribution = income_dist_type
            
            if 'income_min' in all_applicable:
                income_min = st.sidebar.number_input(
                    "Minimum Income",
                    min_value=0.0,
                    value=st.session_state.sim_params.income_min,
                    help="✅ Applicable for selected decisions"
                )
                st.session_state.sim_params.income_min = income_min
            
            if 'income_avg' in all_applicable:
                # Show current type and value
                current_type = st.session_state.sim_params.income_avg_type.title()
                income_avg = st.sidebar.number_input(
                    f"{current_type} Income",
                    min_value=st.session_state.sim_params.income_min,
                    value=st.session_state.sim_params.income_avg,
                    help=f"✅ Applicable for selected decisions (configured as {current_type.lower()} on Page 1)"
                )
                st.session_state.sim_params.income_avg = income_avg
            
            if 'income_max' in all_applicable:
                income_max = st.sidebar.number_input(
                    "Maximum Income",
                    min_value=st.session_state.sim_params.income_avg,
                    value=st.session_state.sim_params.income_max,
                    help="✅ Applicable for selected decisions"
                )
                st.session_state.sim_params.income_max = income_max
            
            if 'discount_income_threshold' in all_applicable:
                discount_threshold = st.sidebar.number_input(
                    "Discount Threshold",
                    min_value=st.session_state.sim_params.income_min,
                    max_value=st.session_state.sim_params.income_max,
                    value=st.session_state.sim_params.discount_income_threshold,
                    help="✅ Income threshold for discount qualification"
                )
                st.session_state.sim_params.discount_income_threshold = discount_threshold

        # Only show market parameters if they are applicable
        market_params = ['num_vendors', 'market_price', 'vendor_price_min', 'vendor_price_max']
        if any(param in all_applicable for param in market_params):
            st.sidebar.subheader("🏪 Market Parameters")
            st.sidebar.caption("✅ Applicable for selected decisions")
            
            if 'num_vendors' in all_applicable:
                num_vendors = st.sidebar.number_input(
                    "Number of Vendors",
                    min_value=1,
                    max_value=50,
                    value=st.session_state.sim_params.num_vendors,
                    help="✅ Applicable for selected decisions"
                )
                st.session_state.sim_params.num_vendors = num_vendors
            
            if 'market_price' in all_applicable:
                market_price = st.sidebar.number_input(
                    "Average Market Price ($)",
                    min_value=1,
                    max_value=100000,
                    value=int(st.session_state.sim_params.market_price * 100),
                    help="✅ Applicable for selected decisions (in cents, converted to dollars)"
                ) / 100.0
                st.session_state.sim_params.market_price = market_price

        # Only show pricing parameters if they are applicable
        pricing_params = ['platform_markup', 'price_range', 'price_grid', 'bidding_percentage']
        if any(param in all_applicable for param in pricing_params):
            st.sidebar.subheader("💰 Pricing Parameters")
            st.sidebar.caption("✅ Applicable for selected decisions")
            
            if 'platform_markup' in all_applicable:
                platform_markup = st.sidebar.slider(
                    "Platform Markup (m)",
                    min_value=0.0,
                    max_value=0.5,
                    value=st.session_state.sim_params.platform_markup,
                    step=0.01,
                    help="✅ Applicable for selected decisions"
                )
                st.session_state.sim_params.platform_markup = platform_markup
            
            if 'bidding_percentage' in all_applicable:
                bidding_percentage = st.sidebar.slider(
                    "Bidding Percentage (bp)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.sim_params.bidding_percentage,
                    step=0.05,
                    help="✅ Applicable for selected decisions"
                )
                st.session_state.sim_params.bidding_percentage = bidding_percentage
        
        # Simulation parameters (always show if decisions are selected)
        st.sidebar.subheader("Simulation Parameters")
        
        # Show current simulation mode (set on Page 1)
        st.sidebar.info(f"📊 Mode: {st.session_state.sim_params.simulation_mode} (configured on Page 1)")
        
        n_agents = st.sidebar.number_input(
            "Number of Agents",
            min_value=10,
            max_value=50000,
            value=1000,
            step=100,
            help="Number of synthetic agents to generate"
        )
        
        # Use simulation mode from Page 1 (stored in session state)
        simulation_mode = st.session_state.sim_params.simulation_mode
        
        if simulation_mode == "Single Run":
            seed = st.sidebar.number_input(
                "Random Seed",
                min_value=1,
                max_value=2147483647,
                value=42,
                help="Seed for reproducible results"
            )
        else:
            n_runs = st.sidebar.number_input(
                "Number of Runs",
                min_value=2,
                max_value=1000,
                value=10,
                step=10,
                help="Number of Monte-Carlo repetitions (Note: 100+ runs can take several minutes)"
            )
            base_seed = st.sidebar.number_input(
                "Base Seed",
                min_value=1,
                max_value=2147483647,
                value=42,
                help="Starting seed (subsequent runs use base_seed + i)"
            )
        
        # Advanced options
        st.sidebar.subheader("🔧 Advanced Options")

        show_individual_agents = st.sidebar.checkbox(
            "Show Individual Agent Details",
            value=False,
            help="Display detailed breakdown of individual agents"
        )
        
        save_results = st.sidebar.checkbox(
            "Save Results to File",
            value=True,
            help="Save simulation outputs to outputs/ directory"
        )
        
        # Store settings in session state
        st.session_state.population_mode = population_mode
        st.session_state.income_spec_mode = income_spec_mode
        st.session_state.sigma_in_copula = sigma_in_copula
        st.session_state.sigma_in_research = sigma_in_research
        st.session_state.sigma_coefficient = sigma_coefficient
        st.session_state.sigma_value_ui = sigma_value_ui
        st.session_state.anchor_observed_weight = anchor_observed_weight
        st.session_state.raw_draw_mode = raw_draw_mode
        st.session_state.n_agents = n_agents
        # Note: simulation_mode now comes from Page 1 (st.session_state.sim_params.simulation_mode)
        if simulation_mode == "Single Run":
            st.session_state.seed = seed
        else:
            st.session_state.n_runs = n_runs
            st.session_state.base_seed = base_seed
        st.session_state.show_individual_agents = show_individual_agents
        st.session_state.save_results = save_results
        
        # Summary info
        if st.sidebar.button("📊 Show Parameter Summary"):
            total_params = len(get_all_global_parameters())
            applicable_count = len(all_applicable)
            st.sidebar.success(f"✅ {applicable_count}/{total_params} parameters applicable ({applicable_count/total_params:.0%})")
    
    # Run simulation button
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🚀 Run Simulation", type="primary", width="stretch"):
        # Check if simulation is already running
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
            
        if st.session_state.simulation_running:
            st.warning("⚠️ A simulation is already running. Please wait for it to complete.")
        else:
            st.session_state.simulation_running = True
            try:
                if st.session_state.sim_params.simulation_mode == "Single Run":
                    run_simulation_from_sidebar()
                    st.session_state.mc_results = None
                else:
                    mc_summary, mc_detailed, output_log = run_monte_carlo_study()
                    if mc_summary is not None:
                        st.session_state.mc_results = {
                            'summary': mc_summary,
                            'detailed': mc_detailed,
                            'log': output_log
                        }
                        st.session_state.simulation_results = None
                        st.session_state.page = 'results'
                        st.success("✅ Monte Carlo results saved to session state. Redirecting to results page...")
                        st.rerun()
                    else:
                        st.error("❌ Monte Carlo simulation returned no results")
            finally:
                st.session_state.simulation_running = False
