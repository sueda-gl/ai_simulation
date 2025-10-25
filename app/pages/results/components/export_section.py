import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO
from app.models import initialize_session_state


def render_export_section(df, results_dict=None, using_selected_config=False):
    """Render the export/download section (simplified)"""
    # Remove 'raw', 'index', 'consumption_frequency', 'actual_allowance', 'income', 'customer_type', and 'enriched_requests_count' columns before any processing
    columns_to_exclude = ['raw', 'index', 'consumption_frequency', 'actual_allowance', 'income', 'customer_type', 'enriched_requests_count']
    
    if df is not None:
        df = df[[col for col in df.columns if not any(excl in col.lower() for excl in columns_to_exclude)]]
    if results_dict is not None:
        results_dict = {
            key: config_df[[col for col in config_df.columns if not any(excl in col.lower() for excl in columns_to_exclude)]]
            for key, config_df in results_dict.items()
        }

    st.subheader("💾 Export Results")

    trait_columns = ['Honesty_Humility', 'Assigned Allowance Level', 'Study Program', 
                     'Group_experiment', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
    
    is_donation_only_run = (
        hasattr(st.session_state, 'custom_decisions') and 
        st.session_state.custom_decisions == ['donation_default'] and
        hasattr(st.session_state, 'default_decisions') and
        len(st.session_state.default_decisions) == 0
    )
    
    if is_donation_only_run:
        columns_to_keep = [col for col in df.columns if 'donation' in col.lower() or col in trait_columns]
        df = df[columns_to_keep]
        if results_dict:
            results_dict = {
                key: config_df[[col for col in config_df.columns if 'donation' in col.lower() or col in trait_columns]]
                for key, config_df in results_dict.items()
            }

    try:
        buffer = BytesIO()
        export_all_configs = results_dict is not None and len(results_dict) > 1 and not using_selected_config

        if export_all_configs:
            first_config_df = next(iter(results_dict.values()))
            available_traits = [col for col in trait_columns if col in first_config_df.columns]
            combined_df = first_config_df[available_traits].copy()
            
            # Add agent_id if it exists
            if 'agent_id' in first_config_df.columns:
                combined_df['Agent ID'] = first_config_df['agent_id'].values

            green_columns = []
            
            if not is_donation_only_run:
                decision_cols_first = [col for col in first_config_df.columns if col not in trait_columns and col != 'agent_id']
                for col in decision_cols_first:
                    if 'donation_default' not in col:
                        combined_df[col] = first_config_df[col].values
            
            for config_key, config_df in results_dict.items():
                if not config_df.empty:
                    decision_cols = [col for col in config_df.columns if col not in trait_columns and col != 'agent_id']
                    for col in decision_cols:
                        if 'donation_default' in col:
                            config_suffix = config_key.replace('_', ' ').title().replace(' ', '_')
                            new_col_name = f"{col}_{config_suffix}"
                            combined_df[new_col_name] = config_df[col].values
                            green_columns.append(new_col_name)
            
            # Reorder columns to put Agent ID first
            if 'Agent ID' in combined_df.columns:
                cols = ['Agent ID'] + [col for col in combined_df.columns if col != 'Agent ID']
                combined_df = combined_df[cols]

            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                combined_df.to_excel(writer, index=False, sheet_name='All Configurations')
                from openpyxl.styles import PatternFill
                worksheet = writer.sheets['All Configurations']
                green_fill = PatternFill(start_color='90EE90', end_color='90EE90', fill_type='solid')
                header_row = list(combined_df.columns)
                for col_name in green_columns:
                    if col_name in header_row:
                        col_idx = header_row.index(col_name) + 1
                        for row_idx in range(1, len(combined_df) + 2):
                            worksheet.cell(row=row_idx, column=col_idx).fill = green_fill
            
            excel_label = f"📊 Download Excel (All {len(results_dict)} Configs)"
            excel_filename = f"enhanced_simulation_all_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        else:
            df_export = df.copy()
            # Rename agent_id to 'Agent ID' for clarity
            if 'agent_id' in df_export.columns:
                df_export = df_export.rename(columns={'agent_id': 'Agent ID'})
            
            # Reorder columns to put Agent ID first
            if 'Agent ID' in df_export.columns:
                cols = ['Agent ID'] + [col for col in df_export.columns if col != 'Agent ID']
                df_export = df_export[cols]
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Results')
            excel_label = "📊 Download Excel"
            excel_filename = f"enhanced_simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        st.download_button(
            label=excel_label,
            data=buffer.getvalue(),
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except ImportError:
        st.caption("⚠️ Excel export requires openpyxl")

    if st.button("🔄 Clear Results"):
        # Clear all session state to reset the entire application
        keys_to_delete = [key for key in st.session_state.keys()]
        for key in keys_to_delete:
            del st.session_state[key]
        
        # Reinitialize session state with default values
        initialize_session_state()
        
        # Stay on results page to show "no results" message
        st.session_state.page = 'results'
        
        # Force page reload
        st.rerun()
