# app/pages/decision_tabs/disclose_income.py
"""
Disclose Income decision tab configuration.

Decision 1: Disclose income for Fixed status at time of registration/review.
Uses a two-stage mediation model when specified (research spec mode).
"""
import streamlit as st
import yaml
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
        'di_intercept': config.get('intercept', 0.1),
        'di_wopb': anchor_weights.get('observed_prosocial', 0.25),
        'di_wpb': anchor_weights.get('prosocial_weight', 0.50),
        'di_sigma_enabled': stochastic.get('sigma_value', 0) > 0,
        'di_sigma_strategy': stochastic.get('sigma_strategy', 'overall'),
        'di_scale_factor': stochastic.get('scale_factor', 0.1),
        'di_income_mode': config.get('income_mode', 'categorical'),
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def render_disclose_income_tab():
    """Render disclose_income specific configuration."""
    initialize_disclose_income_session_state()
    config = load_disclose_income_config()
    
    st.markdown('<h3 class="section-header">📋 Disclose Income Configuration</h3>', unsafe_allow_html=True)
    
    st.info("""
    **Decision 1: Disclose Income** uses a two-stage mediation model:
    - **Equation 1**: Prosocial Behavior (PB) from personality traits
    - **Equation 2**: Disclosure Intention (DI) from PB and direct effects
    
    Output: "Y" if DI > 0 after stochastic draw, "N" otherwise.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Income Mode
        st.markdown('<h4 class="subsection-header">Income Mode</h4>', unsafe_allow_html=True)
        
        current_mode = st.session_state.get('di_income_mode', 'categorical')
        mode_options = ["categorical", "continuous"]
        mode_index = mode_options.index(current_mode) if current_mode in mode_options else 0
        
        income_mode = st.radio(
            "Income Mode for Disclosure Model",
            mode_options,
            index=mode_index,
            help="""
            **Categorical**: income_high = 1 if Assigned Allowance Level > 3
            **Continuous**: income_high = 1 if agent's income > population median
            """,
            key="di_income_mode_widget"
        )
        
        if income_mode != st.session_state.get('di_income_mode'):
            st.session_state.di_income_mode = income_mode
            save_disclose_income_config({'income_mode': income_mode})
            st.toast(f"✅ Income mode set to: {income_mode}", icon="💾")
        
        # Intercept (β0)
        st.markdown('<h4 class="subsection-header">Intercept (β₀)</h4>', unsafe_allow_html=True)
        
        current_intercept = config.get('intercept', 0.1)
        
        new_intercept = st.number_input(
            "Baseline disclosure tendency",
            min_value=-1.0,
            max_value=1.0,
            value=float(current_intercept),
            step=0.01,
            format="%.3f",
            help="β₀ in the disclosure intention equation. Default: 0.1",
            key="di_intercept_widget"
        )
        
        if abs(new_intercept - current_intercept) > 0.001:
            st.session_state.di_intercept = new_intercept
            save_disclose_income_config({'intercept': float(new_intercept)})
            st.toast(f"✅ Intercept set to: {new_intercept:.3f}", icon="💾")
    
    with col2:
        # Anchor Weights
        st.markdown('<h4 class="subsection-header">Anchor Weights</h4>', unsafe_allow_html=True)
        
        anchor_weights = config.get('anchor_weights', {})
        current_wopb = anchor_weights.get('observed_prosocial', 0.25)
        current_wpb = anchor_weights.get('prosocial_weight', 0.50)
        
        # WOPB - Weight for observed vs calculated prosocial behavior
        new_wopb = st.slider(
            "W_OPB: Observed vs Calculated PB weight",
            min_value=0.0,
            max_value=1.0,
            value=float(current_wopb),
            step=0.01,
            help="anchored_PB = WOPB × observed_PB + (1-WOPB) × calculated_PB. Default: 0.25",
            key="di_wopb_widget"
        )
        st.caption(f"Calculated PB weight: {1 - new_wopb:.2f}")
        
        if abs(new_wopb - current_wopb) > 0.001:
            st.session_state.di_wopb = new_wopb
            save_disclose_income_config({'anchor_weights.observed_prosocial': float(new_wopb)})
            st.toast(f"✅ WOPB set to: {new_wopb:.2f}", icon="💾")
        
        # WPB - Weight for prosocial effect in disclosure equation
        new_wpb = st.slider(
            "W_PB: Prosocial behavior effect weight",
            min_value=0.0,
            max_value=1.0,
            value=float(current_wpb),
            step=0.01,
            help="DI = β₀ + (1-WPB)×direct_effects + WPB×(PB×income_high). Default: 0.50",
            key="di_wpb_widget"
        )
        st.caption(f"Direct effects weight: {1 - new_wpb:.2f}")
        
        if abs(new_wpb - current_wpb) > 0.001:
            st.session_state.di_wpb = new_wpb
            save_disclose_income_config({'anchor_weights.prosocial_weight': float(new_wpb)})
            st.toast(f"✅ WPB set to: {new_wpb:.2f}", icon="💾")
    
    # Stochastic Component Section
    st.markdown("---")
    st.markdown('<h4 class="subsection-header">Stochastic Component</h4>', unsafe_allow_html=True)
    
    stochastic = config.get('stochastic', {})
    current_sigma_value = stochastic.get('sigma_value', 0)
    current_strategy = stochastic.get('sigma_strategy', 'overall')
    current_scale = stochastic.get('scale_factor', 0.1)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Enable/disable stochastic
        sigma_enabled = st.checkbox(
            "Enable stochastic component",
            value=current_sigma_value > 0,
            help="When enabled, adds Normal(DI, σ×scale) noise to disclosure intention",
            key="di_sigma_enabled_widget"
        )
        
        if sigma_enabled:
            # Sigma strategy
            strategy_options = ["overall", "quintile"]
            strategy_index = strategy_options.index(current_strategy) if current_strategy in strategy_options else 0
            
            sigma_strategy = st.radio(
                "σ Strategy",
                strategy_options,
                index=strategy_index,
                help="""
                **overall**: Use overall SD from 280 participants (σ = 9.90)
                **quintile**: Use income-quintile-specific SD
                """,
                key="di_sigma_strategy_widget"
            )
            
            if sigma_strategy != current_strategy:
                save_disclose_income_config({'stochastic.sigma_strategy': sigma_strategy})
                st.toast(f"✅ σ strategy set to: {sigma_strategy}", icon="💾")
    
    with col4:
        if sigma_enabled:
            # Scale factor
            new_scale = st.slider(
                "σ Scale Factor",
                min_value=0.01,
                max_value=0.5,
                value=float(current_scale),
                step=0.01,
                help="Scales raw σ to match DI range. Effective σ = raw_σ × scale_factor",
                key="di_scale_factor_widget"
            )
            
            # Show effective sigma
            raw_sigma = 9.899547  # Overall SD
            effective_sigma = raw_sigma * new_scale
            st.metric("Effective σ", f"{effective_sigma:.4f}")
            
            if abs(new_scale - current_scale) > 0.001:
                save_disclose_income_config({'stochastic.scale_factor': float(new_scale)})
                st.toast(f"✅ Scale factor set to: {new_scale:.2f}", icon="💾")
            
            # Update sigma_value based on enabled state
            if current_sigma_value == 0:
                save_disclose_income_config({'stochastic.sigma_value': raw_sigma})
        else:
            st.info("ℹ️ Stochastic disabled - using deterministic DI values")
            if current_sigma_value > 0:
                save_disclose_income_config({'stochastic.sigma_value': 0})
    
    # Model Formula Display
    st.markdown("---")
    render_formula_display(config)
    
    # Simulation buttons
    st.markdown("---")
    try:
        from app.pages.decision_execution import render_simulation_buttons
        selected_decs = getattr(st.session_state.decision_params, 'selected_decisions', [])
        render_simulation_buttons(
            decision_name="disclose_income",
            selected_decisions=selected_decs
        )
    except Exception as e:
        st.error(f"Error rendering simulation buttons: {e}")


def render_formula_display(config):
    """Render the mathematical model formula."""
    with st.expander("📐 Mathematical Model", expanded=False):
        st.markdown("### Equation 1: Prosocial Behavior (PB_i)")
        st.latex(r"""
        PB_i = 0.0238 \times z_{Agreeable} + 0.0165 \times z_{Openness} + 0.0295 \times z_{HH} + 0.0677 \times z_{Religious}
        """)
        
        st.markdown("### Anchoring with Observed Behavior")
        wopb = config.get('anchor_weights', {}).get('observed_prosocial', 0.25)
        st.latex(f"""
        anchored\_PB = {wopb:.2f} \\times z_{{obs\_PB}} + {1-wopb:.2f} \\times PB_i
        """)
        
        st.markdown("### Equation 2: Disclosure Intention (DI_i)")
        wpb = config.get('anchor_weights', {}).get('prosocial_weight', 0.50)
        beta0 = config.get('intercept', 0.1)
        st.latex(f"""
        DI_i = {beta0:.2f} + {1-wpb:.2f} \\times [direct\_effects] + {wpb:.2f} \\times (anchored\_PB \\times I_{{high}})
        """)
        
        st.markdown("**Direct Effects:**")
        st.latex(r"""
        0.0067 \times z_E + 0.0174 \times z_N + 0.0164 \times z_{HH} - 0.0090 \times z_I
        """)
        
        st.markdown("### Final Decision")
        st.markdown("""
        - If stochastic enabled: `draw ~ Normal(DI_i, σ × scale_factor)`
        - **disclose_income = "Y"** if draw > 0, else **"N"**
        """)
        
        st.markdown("### Variable Definitions")
        st.markdown("""
        | Variable | Definition |
        |----------|------------|
        | z_Agreeable | Z-scored Agreeableness |
        | z_Openness | Z-scored Openness to Experience |
        | z_HH | Z-scored Honesty-Humility |
        | z_Religious | Z-scored religiosity composite |
        | z_E | Z-scored Extraversion |
        | z_N | Z-scored Neuroticism |
        | z_I | Z-scored income |
        | I_high | 1 if income > median, 0 otherwise |
        | obs_PB | Observed prosocial behavior (TWT+Sospeso) |
        """)
