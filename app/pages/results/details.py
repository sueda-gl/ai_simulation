# app/pages/results/details.py
"""
Detail views and export functionality for simulation results.
"""
import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from app.models import get_decision_global_parameters, get_all_global_parameters


def render_parameter_applicability_summary():
    """Render parameter applicability summary for the run"""
    selected_decisions = st.session_state.decision_params.selected_decisions
    
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


def render_individual_agent_details(df):
    """Render individual agent details section"""
    # Check if this is dependent variable mode
    is_depvar_mode = len(df.columns) == 1 and 'donation_default' in df.columns
    
    if not is_depvar_mode:
        st.subheader("🔍 Individual Agent Details")
        
        # Agent selection
        agent_id = st.selectbox(
            "Select Agent to Examine",
            options=range(len(df)),
            format_func=lambda x: f"Agent {x+1}"
        )
        
        agent_data = df.iloc[agent_id]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("** Agent Traits**")
            trait_data = {}
            for col in ['Honesty_Humility', 'Assigned Allowance Level', 'Study Program', 
                       'Group_experiment', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']:
                if col in agent_data:
                    trait_data[col] = agent_data[col]
            
            trait_df = pd.DataFrame(list(trait_data.items()), columns=['Trait', 'Value'])
            # Convert all values to strings to avoid PyArrow serialization issues
            trait_df['Value'] = trait_df['Value'].astype(str)
            st.dataframe(trait_df, hide_index=True)
        
        with col2:
            st.markdown("**🎯 Agent Decisions**")
            decision_data = {}
            for col in df.columns:
                if col not in ['Assigned Allowance Level', 'Group_experiment', 'Honesty_Humility', 
                              'Study Program', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']:
                    decision_data[col] = agent_data[col]
            
            decision_df = pd.DataFrame(list(decision_data.items()), columns=['Decision', 'Value'])
            if 'donation_default' in decision_data:
                decision_df.loc[decision_df['Decision'] == 'donation_default', 'Value'] = \
                    f"{decision_data['donation_default']:.1%}"
            if 'donation_default_raw' in decision_data:
                decision_df.loc[decision_df['Decision'] == 'donation_default_raw', 'Value'] = \
                    f"{decision_data['donation_default_raw']:.1%}"
            # Convert all values to strings to avoid PyArrow serialization issues
            decision_df['Value'] = decision_df['Value'].astype(str)
            st.dataframe(decision_df, hide_index=True)
    else:
        st.caption("Individual agent details not available in dependent variable resampling mode (no trait information)")


def render_export_section(df):
    """Render the export/download section"""
    st.subheader("💾 Export Results")
    
    # Show a quick verification metric from the SAME DataFrame used for export
    # This helps detect any mismatch between charts and the exported numbers
    if 'donation_default' in df.columns:
        try:
            export_mean = pd.to_numeric(df['donation_default'], errors='coerce').mean()
            colm1, colm2, colm3 = st.columns(3)
            with colm1:
                st.metric("Export Mean (donation_default)", f"{export_mean:.1%}")
            with colm2:
                st.caption("Computed from the exact DataFrame below")
            with colm3:
                st.caption(f"Rows: {len(df):,}")
        except Exception:
            pass
    
    # CRITICAL: Show all donation-related columns in the export to help diagnose confusion
    donation_cols = [col for col in df.columns if 'donation' in col.lower()]
    if donation_cols:
        with st.expander("📊 Donation Columns in Export", expanded=True):
            st.markdown("**This CSV contains the following donation-related columns:**")
            for col in donation_cols:
                try:
                    col_mean = pd.to_numeric(df[col], errors='coerce').mean()
                    if pd.notna(col_mean):
                        st.write(f"• **{col}**: mean = {col_mean:.4f} ({col_mean:.1%})")
                    else:
                        st.write(f"• **{col}**: (non-numeric)")
                except Exception:
                    st.write(f"• **{col}**: (error computing mean)")
            
            st.markdown("---")
            st.info("✅ **Use `donation_default` for the final processed donation rate** (shown in charts above)")
            if 'donation_default_raw' in donation_cols:
                st.caption("⚠️ `donation_default_raw` is the pre-truncation draw (can be negative)")
            if 'donation_default_raw_pos' in donation_cols:
                st.caption("⚠️ `donation_default_raw_pos` is the floored (non-negative) draw on 0-100 scale")
            if 'final_donation_rate' in donation_cols:
                st.caption("⚠️ `final_donation_rate` is a separate decision (slider/default value), not the computed rate")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"enhanced_simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel export
        try:
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Results')
            
            excel_data = buffer.getvalue()
            st.download_button(
                label="📊 Download Excel",
                data=excel_data,
                file_name=f"enhanced_simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.caption("⚠️ Excel export requires openpyxl")
            st.caption("Install with: pip install openpyxl")
    
    with col3:
        if st.button("🔄 Clear Results"):
            st.session_state.simulation_results = None
            st.rerun()
