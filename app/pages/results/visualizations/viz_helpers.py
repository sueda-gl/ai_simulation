# app/pages/results/visualizations/viz_helpers.py
"""
Helper functions for decision visualizations.
Contains reusable utilities for rendering controls and formatting.
"""
import streamlit as st


def render_probability_controls(decision_name, df):
    """Render probability controls for random Y/N decisions directly under their display
    
    Note: This updates the DEFAULT configuration (same keys as Page 2 Overview tab).
    To run a new simulation with updated settings, go to Page 2 → Overview → Run Complete Simulation.
    """
    
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES
    
    # Check if this is a random decision that needs controls
    default_value = DEFAULT_DECISION_VALUES.get(decision_name)
    
    if isinstance(default_value, dict) and default_value.get("type") == "random_probability":
        st.markdown("**⚙️ Probability Settings (Read-Only):**")
        
        options = default_value.get("options", ["Y", "N"])
        description = default_value.get("description", "Probability")
        
        # Use _default_ key (same as Page 2 Overview tab) for consistency
        prob_key = f"{decision_name}_default_probability_y"
        current_prob = st.session_state.get(prob_key, default_value.get("probability_y", 0.5))
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Show configured probability as read-only metrics
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.metric(f"P({options[0]})", f"{current_prob:.0%}", 
                         help=f"Configured probability of {options[0]}")
            with subcol2:
                st.metric(f"P({options[1]})", f"{1-current_prob:.0%}",
                         help=f"Configured probability of {options[1]}")
            
        with col2:
            # Show ratio
            st.metric("Ratio", f"{current_prob:.0%} : {1-current_prob:.0%}")
            st.caption(f"{options[0]} : {options[1]}")
        
        with col3:
            # Show current distribution in results if available
            if decision_name in df.columns:
                decision_counts = df[decision_name].value_counts()
                if len(decision_counts) >= 2:
                    # Get the actual distribution from results
                    if options[0] in decision_counts.index:
                        actual_y_count = decision_counts[options[0]]
                    else:
                        actual_y_count = 0
                    
                    if options[1] in decision_counts.index:
                        actual_n_count = decision_counts[options[1]]  
                    else:
                        actual_n_count = 0
                    
                    total_count = actual_y_count + actual_n_count
                    if total_count > 0:
                        actual_y_ratio = actual_y_count / total_count
                        st.metric("Actual Results", f"{actual_y_ratio:.0%} : {1-actual_y_ratio:.0%}")
                        st.caption(f"From simulation")
        
        # Show helpful message about where to modify settings
        st.caption("💡 To modify these settings: Go to **Page 2 → Overview Tab**")


def get_dynamic_description(decision_name):
    """Get dynamic description for decisions showing current probability settings"""
    
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES, DEFAULT_DECISION_DESCRIPTIONS
    
    default_value = DEFAULT_DECISION_VALUES.get(decision_name)
    
    # For parametric random decisions, show current probability
    if isinstance(default_value, dict) and default_value.get("type") == "random_probability":
        options = default_value.get("options", ["Y", "N"])
        # Use _default_ key (consistent with Page 2 Overview tab)
        prob_key = f"{decision_name}_default_probability_y"
        current_prob = st.session_state.get(prob_key, default_value.get("probability_y", 0.5))
        
        return f"{current_prob:.0%} chance of {options[0]}, {1-current_prob:.0%} chance of {options[1]}"
    
    # For other decisions, use static description
    return DEFAULT_DECISION_DESCRIPTIONS.get(decision_name, "Standard default behavior")


def _render_missing_visualization(decision_title: str) -> None:
    """Fallback when a specific visualization is not implemented."""
    st.info(f"No specialized visualization implemented for {decision_title}.")

