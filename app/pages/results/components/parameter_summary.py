import streamlit as st
import yaml
from pathlib import Path
from app.models import get_decision_global_parameters, get_all_global_parameters, ALL_DECISIONS


def render_parameter_applicability_summary():
    """Render parameter applicability summary for the run"""
    # Get selected decisions in chronological order
    selected_set = set(st.session_state.decision_params.selected_decisions)
    selected_decisions = [d for d in ALL_DECISIONS if d in selected_set]
    
    if selected_decisions:
        # Calculate overall applicability
        total_applicable = get_decision_global_parameters(selected_decisions)
        all_global_params = get_all_global_parameters()
        total_not_applicable = all_global_params - total_applicable
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Parameters", len(all_global_params))
        with col2:
            st.metric("✅ Applicable", len(total_applicable))
        with col3:
            st.metric("❌ Not Applicable", len(total_not_applicable))
        with col4:
            applicability_pct = len(total_applicable) / len(all_global_params) * 100 if all_global_params else 0
            st.metric("📈 Efficiency", f"{applicability_pct:.0f}%")
        
            # Show which parameters were actually used vs unused
        col_used, col_unused = st.columns(2)
        
        with col_used:
            st.markdown("### ✅ Parameters Used in This Simulation")
            if total_applicable:
                for param in sorted(total_applicable):
                    st.markdown(f"  • {param.replace('_', ' ').title()}")
            else:
                st.caption("No parameters were applicable for the selected decisions.")
        
        with col_unused:
            st.markdown("### ❌ Parameters Not Used in This Simulation")
            if total_not_applicable:
                for param in sorted(total_not_applicable):
                    st.markdown(f"  • {param.replace('_', ' ').title()}")
            else:
                st.caption("All parameters were used in this simulation.")
        
        # Show decision-specific breakdown
        st.markdown("### 📊 Parameter Usage by Decision")
        
        try:
            decisions_path = Path(__file__).resolve().parents[3] / "config" / "decisions.yaml"
            with open(decisions_path, 'r') as f:
                decisions_config = yaml.safe_load(f)
            
            for decision in selected_decisions:
                decision_config = decisions_config.get(decision, {})
                decision_params = set(decision_config.get('uses_global_parameters', []))
                not_used = all_global_params - decision_params
                efficiency = len(decision_params) / len(all_global_params) * 100 if all_global_params else 0
                
                with st.container():
                    col_title, col_metrics = st.columns([2, 3])
                    
                    with col_title:
                        st.markdown(f"**{decision.replace('_', ' ').title()}**")
                    
                    with col_metrics:
                        sub_col1, sub_col2, sub_col3 = st.columns(3)
                        with sub_col1:
                            st.metric("Uses", len(decision_params), label_visibility="collapsed")
                        with sub_col2:
                            st.metric("Doesn't Use", len(not_used), label_visibility="collapsed")
                        with sub_col3:
                            st.metric("Efficiency", f"{efficiency:.0f}%", label_visibility="collapsed")
                    
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error loading decision configurations: {e}")
    else:
        st.caption("No decisions were selected for this simulation.")
