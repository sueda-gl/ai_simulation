import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO
from app.models import initialize_session_state
from app.utils.timestamp_utils import TimestampConverter
from app.reports import (
    apply_export_price_formatting,
    build_agent_level_dataframe,
    build_transaction_level_dataframe,
    to_xlsx_bytes,
)


def _is_compare_all_mode(results_dict):
    """
    Check if results_dict contains configurations from multiple population modes.
    
    Returns True if we have configurations from different population types
    (copula, research_spec, research_baseline).
    """
    if results_dict is None or len(results_dict) <= 1:
        return False
    
    keys = list(results_dict.keys())
    
    has_copula = any(k.startswith('copula') for k in keys)
    has_research_spec = any(k.startswith('research_spec') for k in keys)
    has_research_baseline = any(k.startswith('research_baseline') for k in keys)
    
    # It's "Compare all" if we have configurations from at least 2 different population modes
    population_count = sum([has_copula, has_research_spec, has_research_baseline])
    return population_count >= 2


def render_export_section(df, results_dict=None, using_selected_config=False):
    """Render the export/download section (simplified)"""
    from app.pages.results.run_context import RunContext
    ctx = RunContext.from_session()  # the run's own shape (R28)
    # Remove 'raw', 'index', 'consumption_frequency', 'actual_allowance', 'income', 'customer_type', and 'enriched_requests_count' columns before any processing
    # Use exact column name matching to avoid filtering out 'disclose_income' when we only want to exclude 'income'
    columns_to_exclude = ['raw', 'index', 'consumption_frequency', 'enriched_requests_count']
    
    if df is not None:
        df = df[[col for col in df.columns if col not in columns_to_exclude]]
    if results_dict is not None:
        results_dict = {
            key: config_df[[col for col in config_df.columns if col not in columns_to_exclude]]
            for key, config_df in results_dict.items()
        }

    st.subheader("💾 Export Results")
    
    # Check if this is a donation-only run (special simplified export)
    trait_columns = ['Honesty_Humility', 'Assigned Allowance Level', 'Study Program', 
                     'Group_experiment', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
    
    is_donation_only_run = ctx.is_individual_run('donation_default')

    if is_donation_only_run:
        # DONATION-ONLY EXPORT: Simplified version with just donation and traits
        # Filter main df to only include donation columns
        columns_to_keep = [col for col in df.columns 
                          if (col == 'donation_default' or col in trait_columns or col == 'agent_id')]
        df = df[columns_to_keep]
        
        # Filter results_dict to only include donation columns
        if results_dict is not None:
            results_dict = {
                key: config_df[[col for col in config_df.columns 
                               if (col == 'donation_default' or col in trait_columns or col == 'agent_id')]]
                for key, config_df in results_dict.items()
            }
        
        # Check if we have multiple configurations to compare.
        # For donation-only runs, the using_selected_config flag (which may reflect a stale
        # saved config) is irrelevant -- always export all computed configs.
        export_all_configs = results_dict is not None and len(results_dict) > 1
        
        # Check if this is "Compare all" mode (different population modes with different agents)
        is_compare_all = _is_compare_all_mode(results_dict)
        
        if export_all_configs and is_compare_all:
            st.markdown(f"""
            **Donation Default Results Export (Compare All Mode - {len(results_dict)} Configurations):**
            - Each population mode (Copula, Research Spec, Research Baseline) has its own columns
            - Agent ID, traits, and donation rates for Categorical and Continuous income modes
            - **Note:** Each row contains 3 different agents (one per population mode), each with their correct traits
            """)
        elif export_all_configs:
            st.markdown(f"""
            **Donation Default Results Export (All {len(results_dict)} Configurations):**
            - Agent ID, trait columns, and donation_default rate for each configuration
            - All configurations combined in one file for comparison
            """)
        else:
            st.markdown("""
            **Donation Default Results Export:**
            - Agent ID, trait columns, and donation_default rate
            """)
        
        try:
            if export_all_configs and is_compare_all:
                # COMPARE ALL MODE: Each population mode gets its own sheet
                # Each sheet contains traits + both categorical and continuous donation columns
                
                population_modes = [
                    ('copula', 'Copula'),
                    ('research_spec', 'ResSpec'),
                    ('research_baseline', 'ResBase')
                ]
                
                sheets_data = {}  # Store DataFrames for each sheet
                
                for pop_key, pop_prefix in population_modes:
                    # Find the DataFrames for this population mode
                    cat_key = f"{pop_key}_categorical"
                    cont_key = f"{pop_key}_continuous"
                    
                    cat_df = results_dict.get(cat_key)
                    cont_df = results_dict.get(cont_key)
                    
                    # Use whichever DataFrame is available for traits (they have the same agents)
                    base_df = cat_df if cat_df is not None and not cat_df.empty else cont_df
                    
                    if base_df is None or base_df.empty:
                        continue
                    
                    # Build DataFrame for this population mode
                    sheet_data = {}
                    
                    # Add Agent ID
                    if 'agent_id' in base_df.columns:
                        sheet_data['Agent_ID'] = base_df['agent_id'].values
                    else:
                        sheet_data['Agent_ID'] = list(range(1, len(base_df) + 1))
                    
                    # Add trait columns
                    for trait in trait_columns:
                        if trait in base_df.columns:
                            # Use shorter column names for readability
                            short_trait = trait.replace('Assigned Allowance Level', 'Income_Level')
                            short_trait = short_trait.replace('TWT+Sospeso [=AW2+AX2]{Periods 1+2}', 'TWT_Sospeso')
                            short_trait = short_trait.replace('Group_experiment', 'Group')
                            short_trait = short_trait.replace('Study Program', 'Study_Program')
                            sheet_data[short_trait] = base_df[trait].values
                    
                    # Add donation columns for each income mode
                    if cat_df is not None and not cat_df.empty and 'donation_default' in cat_df.columns:
                        sheet_data['donation_Categorical'] = cat_df['donation_default'].values
                    
                    if cont_df is not None and not cont_df.empty and 'donation_default' in cont_df.columns:
                        sheet_data['donation_Continuous'] = cont_df['donation_default'].values
                    
                    # Create DataFrame for this sheet
                    sheet_df = pd.DataFrame(sheet_data)
                    sheets_data[pop_prefix] = sheet_df
                
                if not sheets_data:
                    st.warning("⚠️ No data available for export")
                    return
                
                # Write each population mode to its own sheet,
                # applying 2-decimal formatting to price/rate columns
                excel_bytes = to_xlsx_bytes(sheets_data, apply_export_price_formatting)
                
                # Show metrics
                total_sheets = len(sheets_data)
                first_sheet_df = next(iter(sheets_data.values()))
                n_agents = len(first_sheet_df)
                n_columns = len(first_sheet_df.columns)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sheets", total_sheets)
                with col2:
                    st.metric("Agents per Sheet", n_agents)
                with col3:
                    st.metric("Columns per Sheet", n_columns)
                
                excel_label = f"📊 Download Donation Excel ({total_sheets} Sheets)"
                excel_filename = f"donation_compare_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                
                st.download_button(
                    label=excel_label,
                    data=excel_bytes,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Each population mode has its own sheet with traits and donation rates for both income modes"
                )
                
                # Show preview with explanation
                with st.expander("📋 Preview Donation Data (first 5 rows per sheet)"):
                    st.info("""
                    **Sheet Structure:**
                    - **Copula**: Synthetic agents generated from copula
                    - **ResSpec**: Original 280 participants (random sample)
                    - **ResBase**: Original 280 participants (sequential order)
                    
                    Each sheet contains Agent ID, traits, and donation rates for both Categorical and Continuous income modes.
                    """)
                    
                    for sheet_name, sheet_df in sheets_data.items():
                        st.markdown(f"**{sheet_name} Sheet:**")
                        st.dataframe(sheet_df.head(), use_container_width=True)
                        st.caption(f"Columns: {', '.join(sheet_df.columns)}")
            
            elif export_all_configs:
                # SAME POPULATION MODE: Multiple income modes with same agents
                # Safe to combine by row since agents are identical
                first_config_df = next(iter(results_dict.values()))
                available_traits = [col for col in trait_columns if col in first_config_df.columns]
                combined_df = first_config_df[available_traits].copy()
                
                # Add agent_id if it exists
                if 'agent_id' in first_config_df.columns:
                    combined_df['Agent ID'] = first_config_df['agent_id'].values
                
                # Add donation_default from each configuration
                for config_key, config_df in results_dict.items():
                    if not config_df.empty and 'donation_default' in config_df.columns:
                        config_suffix = config_key.replace('_', ' ').title().replace(' ', '_')
                        new_col_name = f"donation_default_{config_suffix}"
                        combined_df[new_col_name] = config_df['donation_default'].values
                
                # Reorder columns to put Agent ID first
                if 'Agent ID' in combined_df.columns:
                    cols = ['Agent ID'] + [col for col in combined_df.columns if col != 'Agent ID']
                    combined_df = combined_df[cols]
                
                # Apply 2-decimal formatting to price/rate columns
                excel_bytes = to_xlsx_bytes(
                    {'All Configurations': combined_df}, apply_export_price_formatting
                )
                
                st.metric("Total Agents", len(combined_df))
                
                excel_label = f"📊 Download Donation Excel (All {len(results_dict)} Configs)"
                excel_filename = f"donation_all_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                
                st.download_button(
                    label=excel_label,
                    data=excel_bytes,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help=f"Donation results with {len(results_dict)} configurations for comparison"
                )
                
                # Show preview
                with st.expander("📋 Preview Donation Data (first 5 rows)"):
                    st.dataframe(combined_df.head(), use_container_width=True)
                    st.caption(f"**Columns ({len(combined_df.columns)})**: {', '.join(combined_df.columns[:10])}{'...' if len(combined_df.columns) > 10 else ''}")
            
            else:
                # SINGLE CONFIG: Simple export with just one configuration
                df_export = df.copy()
                
                # Rename agent_id to 'Agent ID' for clarity
                if 'agent_id' in df_export.columns:
                    df_export = df_export.rename(columns={'agent_id': 'Agent ID'})
                
                # Reorder columns to put Agent ID first
                if 'Agent ID' in df_export.columns:
                    cols = ['Agent ID'] + [col for col in df_export.columns if col != 'Agent ID']
                    df_export = df_export[cols]
                
                # Apply 2-decimal formatting to price/rate columns
                excel_bytes = to_xlsx_bytes(
                    {'Donation Results': df_export}, apply_export_price_formatting
                )
                
                st.metric("Total Agents", len(df_export))
                
                excel_filename = f"donation_default_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                st.download_button(
                    label="📊 Download Donation Default Excel",
                    data=excel_bytes,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Donation results with agent traits and donation rates"
                )
                
                # Show preview
                with st.expander("📋 Preview Donation Data (first 5 rows)"):
                    st.dataframe(df_export.head(), use_container_width=True)
                    st.caption(f"**Columns ({len(df_export.columns)})**: {', '.join(df_export.columns)}")
        
        except ImportError:
            st.caption("⚠️ Excel export requires openpyxl")
    
    # Check if this is a disclose_income-only run (special simplified export)
    is_disclose_income_only_run = ctx.is_individual_run('disclose_income')

    # Check if this is a disclose_documents-only run (focused export, mirrors disclose_income)
    is_disclose_documents_only_run = ctx.is_individual_run('disclose_documents')

    # Check if this is an individual Decision 4 (rejected_transaction_defaults) MODEL
    # run: no purchase requests exist, so there is no transaction-level Excel here -
    # only the Decision 4 agent-level workbook is offered.
    is_rtd_only_run = (
        ctx.is_individual_run('rejected_transaction_defaults') and
        df is not None and 'rtd_choice_length' in df.columns
    )

    if is_disclose_income_only_run:
        # DISCLOSE INCOME-ONLY EXPORT: Simplified version with all 19 disclose income columns
        from app.pages.results.visualizations.disclosure_viz import (
            _prepare_disclose_income_excel_data,
            _apply_price_formatting_disclosure
        )
        
        # Check if we have multiple configurations to compare.
        # For disclose_income-only runs, the using_selected_config flag (which is based on
        # donation_default config state) is irrelevant -- always export all computed configs.
        export_all_configs = results_dict is not None and len(results_dict) > 1
        
        # Check if this is "Compare all" mode (different population modes with different agents)
        is_compare_all = _is_compare_all_mode(results_dict)
        
        if export_all_configs and is_compare_all:
            st.markdown(f"""
            **Disclose Income Results Export (Compare All Mode - {len(results_dict)} Configurations):**
            - Each configuration (population mode + income mode) has its own sheet
            - All 19 disclose income columns including traits, weights, and calculated values
            - PB_i, DI_i, and disclose_income values are specific to each configuration
            """)
        elif export_all_configs:
            st.markdown(f"""
            **Disclose Income Results Export (Compare Both - {len(results_dict)} Configurations):**
            - Each income mode (Categorical/Continuous) has its own sheet
            - All 19 disclose income columns with correct PB_i, DI_i, and disclose_income values
            """)
        else:
            st.markdown("""
            **Disclose Income Results Export:**
            - All 19 disclose income columns including traits, weights, and calculated values
            """)
        
        try:
            buffer = BytesIO()
            
            if export_all_configs and is_compare_all:
                # COMPARE ALL MODE: Each configuration gets its own sheet
                # Each sheet contains all 19 disclose income columns with correct values for that config
                # This ensures PB_i, DI_i, and disclose_income are accurate for each income mode
                
                # Define all possible configuration keys and their sheet names
                config_sheet_mapping = [
                    ('copula_categorical', 'Copula_Cat'),
                    ('copula_continuous', 'Copula_Cont'),
                    ('research_spec_categorical', 'ResSpec_Cat'),
                    ('research_spec_continuous', 'ResSpec_Cont'),
                    ('research_baseline_categorical', 'ResBase_Cat'),
                    ('research_baseline_continuous', 'ResBase_Cont')
                ]
                
                sheets_data = {}  # Store DataFrames for each sheet
                
                for config_key, sheet_name in config_sheet_mapping:
                    config_df = results_dict.get(config_key)
                    
                    if config_df is None or config_df.empty:
                        continue
                    
                    # Prepare the export DataFrame using the standard function
                    # This gives us all 19 columns with correct PB_i, DI_i, disclose_income for this config
                    sheet_df = _prepare_disclose_income_excel_data(config_df)
                    
                    if sheet_df is None:
                        continue
                    
                    sheets_data[sheet_name] = sheet_df
                
                if not sheets_data:
                    st.warning("⚠️ No data available for export")
                else:
                    # Write each population mode to its own sheet
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        for sheet_name, sheet_df in sheets_data.items():
                            sheet_df.to_excel(writer, index=False, sheet_name=sheet_name)
                            _apply_price_formatting_disclosure(writer, sheet_name, sheet_df)
                    
                    # Show metrics
                    total_sheets = len(sheets_data)
                    first_sheet_df = next(iter(sheets_data.values()))
                    n_agents = len(first_sheet_df)
                    n_columns = len(first_sheet_df.columns)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sheets", total_sheets)
                    with col2:
                        st.metric("Agents per Sheet", n_agents)
                    with col3:
                        st.metric("Columns per Sheet", n_columns)
                    
                    excel_label = f"📊 Download Disclose Income Excel ({total_sheets} Sheets)"
                    excel_filename = f"disclose_income_compare_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    
                    st.download_button(
                        label=excel_label,
                        data=buffer.getvalue(),
                        file_name=excel_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Each configuration (population mode + income mode) has its own sheet"
                    )
                    
                    # Show preview
                    with st.expander("📋 Preview Disclose Income Data (first 5 rows per sheet)"):
                        st.info("""
                        **Sheet Structure (separate sheets for each configuration):**
                        - **Copula_Cat / Copula_Cont**: Synthetic agents with Categorical/Continuous income
                        - **ResSpec_Cat / ResSpec_Cont**: Research Specification with Categorical/Continuous income
                        - **ResBase_Cat / ResBase_Cont**: Research Baseline with Categorical/Continuous income
                        
                        Each sheet contains all 19 disclose income columns with PB_i, DI_i, and disclose_income values specific to that configuration.
                        """)
                        
                        for sheet_name, sheet_df in sheets_data.items():
                            st.markdown(f"**{sheet_name} Sheet:**")
                            st.dataframe(sheet_df.head(), use_container_width=True)
                            st.caption(f"Columns: {', '.join(sheet_df.columns[:10])}{'...' if len(sheet_df.columns) > 10 else ''}")
            
            elif export_all_configs:
                # COMPARE BOTH MODE: Separate sheets for each income mode
                # Each sheet has complete 19 columns with correct PB_i, DI_i, disclose_income values
                
                sheets_data = {}
                
                for config_key, config_df in results_dict.items():
                    if config_df is None or config_df.empty:
                        continue
                    
                    # Create readable sheet name
                    if 'categorical' in config_key.lower():
                        sheet_name = 'Categorical'
                    elif 'continuous' in config_key.lower():
                        sheet_name = 'Continuous'
                    else:
                        sheet_name = config_key.replace('_', ' ').title()
                    
                    # Prepare the export DataFrame with all 19 columns for this config
                    sheet_df = _prepare_disclose_income_excel_data(config_df)
                    
                    if sheet_df is not None:
                        sheets_data[sheet_name] = sheet_df
                
                if not sheets_data:
                    st.warning("⚠️ Unable to prepare Excel data")
                else:
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        for sheet_name, sheet_df in sheets_data.items():
                            sheet_df.to_excel(writer, index=False, sheet_name=sheet_name)
                            _apply_price_formatting_disclosure(writer, sheet_name, sheet_df)
                    
                    # Show metrics
                    total_sheets = len(sheets_data)
                    first_sheet_df = next(iter(sheets_data.values()))
                    n_agents = len(first_sheet_df)
                    n_columns = len(first_sheet_df.columns)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sheets", total_sheets)
                    with col2:
                        st.metric("Agents per Sheet", n_agents)
                    with col3:
                        st.metric("Columns per Sheet", n_columns)
                    
                    excel_label = f"📊 Download Disclose Income Excel ({total_sheets} Sheets)"
                    excel_filename = f"disclose_income_compare_both_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    
                    st.download_button(
                        label=excel_label,
                        data=buffer.getvalue(),
                        file_name=excel_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Each income mode has its own sheet with all 19 disclose income columns"
                    )
                    
                    # Show preview
                    with st.expander("📋 Preview Disclose Income Data (first 5 rows per sheet)"):
                        st.info("""
                        **Sheet Structure:**
                        - **Categorical**: Results with categorical income treatment
                        - **Continuous**: Results with continuous income treatment
                        
                        Each sheet contains all 19 columns with PB_i, DI_i, and disclose_income values specific to that income mode.
                        """)
                        for sheet_name, sheet_df in sheets_data.items():
                            st.markdown(f"**{sheet_name} Sheet:**")
                            st.dataframe(sheet_df.head(), use_container_width=True)
                            st.caption(f"Columns: {', '.join(sheet_df.columns[:10])}{'...' if len(sheet_df.columns) > 10 else ''}")
            
            else:
                # SINGLE CONFIG: Simple export with just one configuration
                export_df = _prepare_disclose_income_excel_data(df)
                
                if export_df is None:
                    st.warning("⚠️ Unable to prepare Excel data. Some required columns may be missing.")
                else:
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        export_df.to_excel(writer, index=False, sheet_name='Disclose Income Results')
                        _apply_price_formatting_disclosure(writer, 'Disclose Income Results', export_df)
                    
                    st.metric("Total Agents", len(export_df))
                    
                    excel_filename = f"disclose_income_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    st.download_button(
                        label="📊 Download Disclose Income Excel",
                        data=buffer.getvalue(),
                        file_name=excel_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Disclose income results with all 19 columns"
                    )
                    
                    # Show preview
                    with st.expander("📋 Preview Disclose Income Data (first 5 rows)"):
                        st.dataframe(export_df.head(), use_container_width=True)
                        st.caption(f"**Columns ({len(export_df.columns)})**: {', '.join(export_df.columns)}")
        
        except ImportError:
            st.caption("⚠️ Excel export requires openpyxl")

    elif is_disclose_documents_only_run:
        # DISCLOSE DOCUMENTS-ONLY EXPORT: focused DD Excel (mirror of the disclose_income-only export).
        # A DD-only run has no purchases (transaction-level export would be empty) and no other
        # decisions (agent-level export would be sparse), so we export a focused DD sheet instead.
        from app.pages.results.visualizations.disclosure_viz import (
            _prepare_disclose_documents_excel_data,
            _apply_price_formatting_disclosure,
        )
        export_all_configs = results_dict is not None and len(results_dict) > 1
        is_compare_all = _is_compare_all_mode(results_dict)

        if export_all_configs and is_compare_all:
            st.markdown(f"**Disclose Documents Results Export (Compare All - {len(results_dict)} configurations):** one sheet per population × income mode, full privacy-calculus calculation chain.")
        elif export_all_configs:
            st.markdown(f"**Disclose Documents Results Export (Compare Both - {len(results_dict)} configurations):** one sheet per income mode.")
        else:
            st.markdown("**Disclose Documents Results Export:** full privacy-calculus calculation chain (z-scores, weighted_dd, score, decision, customer type).")

        try:
            buffer = BytesIO()
            if export_all_configs:
                if is_compare_all:
                    config_sheet_mapping = [
                        ('copula_categorical', 'Copula_Cat'), ('copula_continuous', 'Copula_Cont'),
                        ('research_spec_categorical', 'ResSpec_Cat'), ('research_spec_continuous', 'ResSpec_Cont'),
                        ('research_baseline_categorical', 'ResBase_Cat'), ('research_baseline_continuous', 'ResBase_Cont'),
                    ]
                    items = [(results_dict.get(k), name) for k, name in config_sheet_mapping]
                else:
                    items = [(cdf, ('Categorical' if 'categorical' in k.lower() else ('Continuous' if 'continuous' in k.lower() else k)))
                             for k, cdf in results_dict.items()]
                sheets_data = {}
                for cdf, name in items:
                    if cdf is None or cdf.empty:
                        continue
                    sdf = _prepare_disclose_documents_excel_data(cdf)
                    if sdf is not None:
                        sheets_data[name] = sdf
                if not sheets_data:
                    st.warning("⚠️ No disclose documents data available for export")
                else:
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        for name, sdf in sheets_data.items():
                            sdf.to_excel(writer, index=False, sheet_name=name)
                            _apply_price_formatting_disclosure(writer, name, sdf)
                    first = next(iter(sheets_data.values()))
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Sheets", len(sheets_data))
                    with c2: st.metric("Agents per Sheet", len(first))
                    with c3: st.metric("Columns per Sheet", len(first.columns))
                    st.download_button(
                        label=f"📄 Download Agent Disclose Documents Data ({len(sheets_data)} Sheets)",
                        data=buffer.getvalue(),
                        file_name=f"disclose_documents_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Each configuration (population × income mode) has its own sheet",
                    )
                    with st.expander("📋 Preview Disclose Documents Data (first 5 rows per sheet)"):
                        for name, sdf in sheets_data.items():
                            st.markdown(f"**{name} Sheet:**")
                            st.dataframe(sdf.head(), use_container_width=True)
            else:
                export_df = _prepare_disclose_documents_excel_data(df)
                if export_df is None or export_df.empty:
                    st.warning("⚠️ No disclose documents data available for export")
                else:
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        export_df.to_excel(writer, index=False, sheet_name='Disclose Documents')
                        _apply_price_formatting_disclosure(writer, 'Disclose Documents', export_df)
                    c1, c2 = st.columns(2)
                    with c1: st.metric("Agents", len(export_df))
                    with c2: st.metric("Columns", len(export_df.columns))
                    st.download_button(
                        label="📄 Download Agent Disclose Documents Data",
                        data=buffer.getvalue(),
                        file_name=f"disclose_documents_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Disclose documents results with the full calculation chain",
                    )
                    with st.expander("📋 Preview Disclose Documents Data (first 5 rows)"):
                        st.dataframe(export_df.head(), use_container_width=True)
                        st.caption(f"**Columns ({len(export_df.columns)})**: {', '.join(export_df.columns)}")
        except ImportError:
            st.caption("⚠️ Excel export requires openpyxl")

    elif is_rtd_only_run:
        # DECISION 4-ONLY EXPORT: agent-level workbook only. An individual Decision 4
        # run produces no purchase requests, so no transaction-level file is offered.
        from app.pages.results.visualizations.transaction_viz import (
            _prepare_rtd_model_export, _rtd_active_element, _RTD_ELEMENT_SHEETS,
        )
        active_element = _rtd_active_element()
        export_all_configs = results_dict is not None and len(results_dict) > 1

        if active_element:
            st.markdown(
                f"**Decision 4 Results Export ({_RTD_ELEMENT_SHEETS[active_element]} element):** "
                "one row per agent with the element's independent variables, its score "
                "and the resulting option sequence."
            )
        else:
            st.markdown(
                "**Decision 4 (Rejected Transaction Defaults) Results Export:** one row "
                "per agent with the Decision 4 element results - one self-contained "
                "sheet per element with its independent variables, score, intermediate "
                "distributions and the resulting option sequence."
            )
        if export_all_configs:
            st.caption(f"{len(results_dict)} configurations - sheet names are prefixed "
                       "with the configuration.")

        try:
            def _rtd_sheets_for(frame):
                sheets = _prepare_rtd_model_export(frame) or {}
                if active_element:
                    name = _RTD_ELEMENT_SHEETS[active_element]
                    sheets = {name: sheets[name]} if name in sheets else {}
                return sheets

            if export_all_configs:
                config_labels = {
                    'copula_categorical': 'Copula_Cat', 'copula_continuous': 'Copula_Cont',
                    'research_spec_categorical': 'ResSpec_Cat', 'research_spec_continuous': 'ResSpec_Cont',
                    'research_baseline_categorical': 'ResBase_Cat', 'research_baseline_continuous': 'ResBase_Cont',
                    'categorical': 'Cat', 'continuous': 'Cont',
                }
                sheets_data = {}
                for config_key, config_df in results_dict.items():
                    if (config_df is None or config_df.empty
                            or 'rtd_choice_length' not in config_df.columns):
                        continue
                    prefix = config_labels.get(config_key, str(config_key)[:12])
                    for name, sheet_df in _rtd_sheets_for(config_df).items():
                        sheets_data[f"{prefix} {name}"[:31]] = sheet_df
            else:
                sheets_data = _rtd_sheets_for(df)

            if not sheets_data:
                st.warning("⚠️ No Decision 4 data available for export")
            else:
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    for sheet_name, sheet_df in sheets_data.items():
                        sheet_df.to_excel(writer, index=False, sheet_name=sheet_name)
                first = next(iter(sheets_data.values()))
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Agents", len(first))
                with c2:
                    st.metric("Sheets", len(sheets_data))
                st.download_button(
                    label="📊 Download Decision 4 Agent-Level Excel",
                    data=buffer.getvalue(),
                    file_name=f"rejected_transaction_defaults_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="One row per agent with the Decision 4 element results",
                    key="rtd_export_section_download",
                )
                with st.expander("📋 Preview Decision 4 Data (first 5 rows per sheet)"):
                    for sheet_name, sheet_df in sheets_data.items():
                        st.markdown(f"**{sheet_name} Sheet:**")
                        st.dataframe(sheet_df.head().astype(str), use_container_width=True)
                        st.caption(f"Columns: {', '.join(sheet_df.columns)}")
        except ImportError:
            st.caption("⚠️ Excel export requires openpyxl")

    elif not is_donation_only_run:
        # FULL TWO-LEVEL EXPORT: For all other simulations
        st.markdown("""
        **Two separate Excel files are available for download:**
        - **Agent-Level Excel**: One row per agent with all agent-level decisions and summary statistics
        - **Transaction-Level Excel**: One row per purchase request with detailed transaction information
        """)
        
        # Get vendor data if available
        vendors_data = None
        if hasattr(df, 'attrs') and 'vendors' in df.attrs:
            vendors_data = df.attrs['vendors']
        
        try:
            # Pricing parameters used to be read off st.session_state.sim_params
            # inside the builders; the page reads them here and passes the VALUES in.
            sim_params = getattr(st.session_state, 'sim_params', None)
            vendor_price_min = getattr(sim_params, 'vendor_price_min', 50.0)
            vendor_price_max = getattr(sim_params, 'vendor_price_max', 150.0)

            # Build agent-level and transaction-level DataFrames
            agent_df = build_agent_level_dataframe(
                df,
                vendors_data=vendors_data,
                vendor_price_min=vendor_price_min,
                vendor_price_max=vendor_price_max,
            )
            transaction_df = build_transaction_level_dataframe(
                df,
                vendors_data=vendors_data,
                # TimestampConverter takes its base time, period duration and
                # period count from session state, so it is built here - exactly
                # as the builder used to build it - and passed in.
                ts_converter=TimestampConverter(),
                market_price=getattr(sim_params, 'market_price', 100.0),
                platform_markup=getattr(sim_params, 'platform_markup', 0.1),
                price_range=getattr(sim_params, 'price_range', 0.25),
                duration_hours=getattr(sim_params, 'duration_hours', 1.0),
                vendor_price_min=vendor_price_min,
                vendor_price_max=vendor_price_max,
            )

            # Show summary statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Agents", len(agent_df))
                st.caption("Rows in Agent-Level file")
            with col2:
                st.metric("Total Transactions", len(transaction_df))
                st.caption("Rows in Transaction-Level file")
            
            # Create two separate Excel files
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Agent-Level Excel (2-decimal formatting on price columns)
            agent_bytes = to_xlsx_bytes(
                {'Agent Level': agent_df}, apply_export_price_formatting
            )
            
            # Transaction-Level Excel (2-decimal formatting on price columns)
            transaction_bytes = to_xlsx_bytes(
                {'Transaction Level': transaction_df}, apply_export_price_formatting
            )
            
            # Download buttons for separate files
            st.markdown("### 📥 Download Files")
            col1, col2 = st.columns(2)
            
            with col1:
                agent_filename = f"simulation_agent_level_{timestamp}.xlsx"
                st.download_button(
                    label="📊 Download Agent-Level Excel",
                    data=agent_bytes,
                    file_name=agent_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help=f"Agent-level data: {len(agent_df)} agents × {len(agent_df.columns)} columns"
                )
            
            with col2:
                transaction_filename = f"simulation_transaction_level_{timestamp}.xlsx"
                st.download_button(
                    label="📊 Download Transaction-Level Excel",
                    data=transaction_bytes,
                    file_name=transaction_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help=f"Transaction-level data: {len(transaction_df)} transactions × {len(transaction_df.columns)} columns"
                )
            
            # Show preview of what's in each file
            st.markdown("### 📋 Data Preview")
            
            with st.expander("👥 Preview Agent-Level Data (first 5 rows)"):
                st.dataframe(agent_df.head(), use_container_width=True)
                st.caption(f"**Columns ({len(agent_df.columns)})**: {', '.join(agent_df.columns[:15])}{'...' if len(agent_df.columns) > 15 else ''}")
            
            with st.expander("🔄 Preview Transaction-Level Data (first 5 rows)"):
                st.dataframe(transaction_df.head(), use_container_width=True)
                st.caption(f"**Columns ({len(transaction_df.columns)})**: {', '.join(transaction_df.columns[:15])}{'...' if len(transaction_df.columns) > 15 else ''}")
            
        except Exception as e:
            st.error(f"Error creating Excel export: {str(e)}")
            st.caption("⚠️ Please ensure all required data is available. If the problem persists, contact support.")
            import traceback
            st.caption(f"Error details: {traceback.format_exc()}")
            
            # Fallback: show raw data
            with st.expander("🔍 View Raw Data (for debugging)"):
                st.dataframe(df, use_container_width=True)

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
