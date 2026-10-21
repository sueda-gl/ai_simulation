import streamlit as st
from datetime import datetime
from src.contract.columns import rename_consensus_columns
from app.models import initialize_session_state
from app.reports import timestamps
from app.reports.timestamps import DEFAULT_DURATION_HOURS, DEFAULT_PERIODS
from app.reports import (
    apply_export_price_formatting,
    build_agent_level_dataframe,
    build_disclose_documents_results_xlsx,
    build_disclose_income_results_xlsx,
    build_disclosure_sheets_xlsx,
    build_transaction_level_dataframe,
    disclose_documents_export_sheets,
    disclose_income_compare_all_sheets,
    disclose_income_compare_both_sheets,
    DONATION_ONLY_EXPORT_TRAIT_COLUMNS,
    donation_all_configs_frame,
    donation_compare_all_sheets,
    donation_single_export_frame,
    prepare_disclose_documents_excel_data,
    prepare_disclose_income_excel_data,
    rtd_config_sheet_prefix,
    rtd_element_subset,
    rtd_model_xlsx_bytes,
    rtd_prefixed_sheet_name,
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


_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _xlsx_download_button(label, data, file_name, help):
    """One (unkeyed) Excel download button of the export section."""
    st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=_XLSX_MIME,
        help=help
    )


def _column_list(columns, max_columns=None):
    """The column names joined by ', ': all of them, or the first `max_columns`
    followed by '...' when there are more."""
    if max_columns is None:
        return ', '.join(columns)
    return f"{', '.join(columns[:max_columns])}{'...' if len(columns) > max_columns else ''}"


def _preview_frame(frame, max_columns=None):
    """Preview of one export frame: its first 5 rows and its column list."""
    st.dataframe(frame.head(), use_container_width=True)
    st.caption(f"**Columns ({len(frame.columns)})**: {_column_list(frame.columns, max_columns)}")


def _preview_sheets(sheets, caption=True, max_columns=None, as_text=False):
    """Preview of a multi-sheet export: per sheet its name, its first 5 rows
    (as text when `as_text`) and, when `caption`, its column list."""
    for sheet_name, sheet_df in sheets.items():
        st.markdown(f"**{sheet_name} Sheet:**")
        st.dataframe(sheet_df.head().astype(str) if as_text else sheet_df.head(), use_container_width=True)
        if caption:
            st.caption(f"Columns: {_column_list(sheet_df.columns, max_columns)}")


def _metrics_row(metrics):
    """(label, value) metrics: one on its own, several in equal columns."""
    if len(metrics) == 1:
        st.metric(*metrics[0])
    else:
        for column, (name, value) in zip(st.columns(len(metrics)), metrics):
            with column:
                st.metric(name, value)


def _download_with_preview(metrics, label, data, file_name, help, preview_title, preview):
    """The block the export section repeats per export: a row of metrics, the
    Excel download button and a preview expander filled by `preview()`."""
    _metrics_row(metrics)
    _xlsx_download_button(label, data, file_name, help)
    with st.expander(preview_title):
        preview()


def render_export_section(df, results_dict=None, using_selected_config=False):
    """Render the export/download section (simplified)"""
    from app.pages.results.run_context import RunContext
    ctx = RunContext.from_session()  # the run's own shape, not the live widgets
    # Drop the bookkeeping columns before any processing. Exact column-name
    # matching, so 'disclose_income' is never caught by an 'income' substring.
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
    trait_columns = DONATION_ONLY_EXPORT_TRAIT_COLUMNS
    
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
                sheets_data = donation_compare_all_sheets(results_dict)
                
                if not sheets_data:
                    st.warning("⚠️ No data available for export")
                    return
                
                # Write each population mode to its own sheet,
                # applying 2-decimal formatting to price/rate columns
                excel_bytes = to_xlsx_bytes(sheets_data, apply_export_price_formatting)
                
                # Show metrics
                total_sheets = len(sheets_data)
                first_sheet_df = next(iter(sheets_data.values()))
                
                # Show preview with explanation
                def preview():
                    st.info("""
                    **Sheet Structure:**
                    - **Copula**: Synthetic agents generated from copula
                    - **ResSpec**: Original 280 participants (random subset when agents < 280; bootstrap with replacement when agents > 280)
                    - **ResBase**: Original 280 participants (original order; cycling when agents ≠ 280)
                    
                    Each sheet contains Agent ID, traits, and donation rates for both Categorical and Continuous income modes.
                    """)
                    _preview_sheets(sheets_data)
                
                _download_with_preview(
                    [("Sheets", total_sheets), ("Agents per Sheet", len(first_sheet_df)),
                     ("Columns per Sheet", len(first_sheet_df.columns))],
                    label=f"📊 Download Donation Excel ({total_sheets} Sheets)",
                    data=excel_bytes,
                    file_name=f"donation_compare_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    help="Each population mode has its own sheet with traits and donation rates for both income modes",
                    preview_title="📋 Preview Donation Data (first 5 rows per sheet)",
                    preview=preview,
                )
            
            elif export_all_configs:
                # SAME POPULATION MODE: Multiple income modes with same agents
                # Safe to combine by row since agents are identical
                combined_df = donation_all_configs_frame(results_dict)
                
                # Apply 2-decimal formatting to price/rate columns
                excel_bytes = to_xlsx_bytes(
                    {'All Configurations': combined_df}, apply_export_price_formatting
                )
                
                _download_with_preview(
                    [("Total Agents", len(combined_df))],
                    label=f"📊 Download Donation Excel (All {len(results_dict)} Configs)",
                    data=excel_bytes,
                    file_name=f"donation_all_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    help=f"Donation results with {len(results_dict)} configurations for comparison",
                    preview_title="📋 Preview Donation Data (first 5 rows)",
                    preview=lambda: _preview_frame(combined_df, 10),
                )
            
            else:
                # SINGLE CONFIG: Simple export with just one configuration
                df_export = donation_single_export_frame(df)
                
                # Apply 2-decimal formatting to price/rate columns
                excel_bytes = to_xlsx_bytes(
                    {'Donation Results': df_export}, apply_export_price_formatting
                )
                
                _download_with_preview(
                    [("Total Agents", len(df_export))],
                    label="📊 Download Donation Default Excel",
                    data=excel_bytes,
                    file_name=f"donation_default_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    help="Donation results with agent traits and donation rates",
                    preview_title="📋 Preview Donation Data (first 5 rows)",
                    preview=lambda: _preview_frame(df_export),
                )
        
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
            if export_all_configs and is_compare_all:
                # COMPARE ALL MODE: Each configuration gets its own sheet
                # Each sheet contains all 19 disclose income columns with correct values for that config
                # This ensures PB_i, DI_i, and disclose_income are accurate for each income mode
                sheets_data = disclose_income_compare_all_sheets(results_dict)
                
                if not sheets_data:
                    st.warning("⚠️ No data available for export")
                else:
                    # Write each population mode to its own sheet
                    excel_bytes = build_disclosure_sheets_xlsx(sheets_data)
                    
                    # Show metrics
                    total_sheets = len(sheets_data)
                    first_sheet_df = next(iter(sheets_data.values()))
                    
                    # Show preview
                    def preview():
                        st.info("""
                        **Sheet Structure (separate sheets for each configuration):**
                        - **Copula_Cat / Copula_Cont**: Synthetic agents with Categorical/Continuous income
                        - **ResSpec_Cat / ResSpec_Cont**: Research Specification with Categorical/Continuous income
                        - **ResBase_Cat / ResBase_Cont**: Research Baseline with Categorical/Continuous income
                        
                        Each sheet contains all 19 disclose income columns with PB_i, DI_i, and disclose_income values specific to that configuration.
                        """)
                        _preview_sheets(sheets_data, max_columns=10)
                    
                    _download_with_preview(
                        [("Sheets", total_sheets), ("Agents per Sheet", len(first_sheet_df)),
                         ("Columns per Sheet", len(first_sheet_df.columns))],
                        label=f"📊 Download Disclose Income Excel ({total_sheets} Sheets)",
                        data=excel_bytes,
                        file_name=f"disclose_income_compare_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        help="Each configuration (population mode + income mode) has its own sheet",
                        preview_title="📋 Preview Disclose Income Data (first 5 rows per sheet)",
                        preview=preview,
                    )
            
            elif export_all_configs:
                # COMPARE BOTH MODE: Separate sheets for each income mode
                # Each sheet has complete 19 columns with correct PB_i, DI_i, disclose_income values
                sheets_data = disclose_income_compare_both_sheets(results_dict)
                
                if not sheets_data:
                    st.warning("⚠️ Unable to prepare Excel data")
                else:
                    excel_bytes = build_disclosure_sheets_xlsx(sheets_data)
                    
                    # Show metrics
                    total_sheets = len(sheets_data)
                    first_sheet_df = next(iter(sheets_data.values()))
                    
                    # Show preview
                    def preview():
                        st.info("""
                        **Sheet Structure:**
                        - **Categorical**: Results with categorical income treatment
                        - **Continuous**: Results with continuous income treatment
                        
                        Each sheet contains all 19 columns with PB_i, DI_i, and disclose_income values specific to that income mode.
                        """)
                        _preview_sheets(sheets_data, max_columns=10)
                    
                    _download_with_preview(
                        [("Sheets", total_sheets), ("Agents per Sheet", len(first_sheet_df)),
                         ("Columns per Sheet", len(first_sheet_df.columns))],
                        label=f"📊 Download Disclose Income Excel ({total_sheets} Sheets)",
                        data=excel_bytes,
                        file_name=f"disclose_income_compare_both_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        help="Each income mode has its own sheet with all 19 disclose income columns",
                        preview_title="📋 Preview Disclose Income Data (first 5 rows per sheet)",
                        preview=preview,
                    )
            
            else:
                # SINGLE CONFIG: Simple export with just one configuration
                export_df = prepare_disclose_income_excel_data(df)
                
                if export_df is None:
                    st.warning("⚠️ Unable to prepare Excel data. Some required columns may be missing.")
                else:
                    excel_bytes = build_disclose_income_results_xlsx(export_df)
                    
                    _download_with_preview(
                        [("Total Agents", len(export_df))],
                        label="📊 Download Disclose Income Excel",
                        data=excel_bytes,
                        file_name=f"disclose_income_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        help="Disclose income results with all 19 columns",
                        preview_title="📋 Preview Disclose Income Data (first 5 rows)",
                        preview=lambda: _preview_frame(export_df),
                    )
        
        except ImportError:
            st.caption("⚠️ Excel export requires openpyxl")

    elif is_disclose_documents_only_run:
        # DISCLOSE DOCUMENTS-ONLY EXPORT: focused DD Excel (mirror of the disclose_income-only export).
        # A DD-only run has no purchases (transaction-level export would be empty) and no other
        # decisions (agent-level export would be sparse), so we export a focused DD sheet instead.
        export_all_configs = results_dict is not None and len(results_dict) > 1
        is_compare_all = _is_compare_all_mode(results_dict)

        if export_all_configs and is_compare_all:
            st.markdown(f"**Disclose Documents Results Export (Compare All - {len(results_dict)} configurations):** one sheet per population × income mode, full privacy-calculus calculation chain.")
        elif export_all_configs:
            st.markdown(f"**Disclose Documents Results Export (Compare Both - {len(results_dict)} configurations):** one sheet per income mode.")
        else:
            st.markdown("**Disclose Documents Results Export:** full privacy-calculus calculation chain (z-scores, weighted_dd, score, decision, customer type).")

        try:
            if export_all_configs:
                sheets_data = disclose_documents_export_sheets(results_dict, is_compare_all)
                if not sheets_data:
                    st.warning("⚠️ No disclose documents data available for export")
                else:
                    excel_bytes = build_disclosure_sheets_xlsx(sheets_data)
                    first = next(iter(sheets_data.values()))
                    _download_with_preview(
                        [("Sheets", len(sheets_data)), ("Agents per Sheet", len(first)),
                         ("Columns per Sheet", len(first.columns))],
                        label=f"📄 Download Agent Disclose Documents Data ({len(sheets_data)} Sheets)",
                        data=excel_bytes,
                        file_name=f"disclose_documents_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        help="Each configuration (population × income mode) has its own sheet",
                        preview_title="📋 Preview Disclose Documents Data (first 5 rows per sheet)",
                        preview=lambda: _preview_sheets(sheets_data, caption=False),
                    )
            else:
                export_df = prepare_disclose_documents_excel_data(df)
                if export_df is None or export_df.empty:
                    st.warning("⚠️ No disclose documents data available for export")
                else:
                    excel_bytes = build_disclose_documents_results_xlsx(export_df)
                    _download_with_preview(
                        [("Agents", len(export_df)), ("Columns", len(export_df.columns))],
                        label="📄 Download Agent Disclose Documents Data",
                        data=excel_bytes,
                        file_name=f"disclose_documents_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        help="Disclose documents results with the full calculation chain",
                        preview_title="📋 Preview Disclose Documents Data (first 5 rows)",
                        preview=lambda: _preview_frame(export_df),
                    )
        except ImportError:
            st.caption("⚠️ Excel export requires openpyxl")

    elif is_rtd_only_run:
        # DECISION 4-ONLY EXPORT: agent-level workbook only. An individual Decision 4
        # run produces no purchase requests, so no transaction-level file is offered.
        from app.pages.results.visualizations.transaction_viz import (
            _prepare_rtd_model_export, _rtd_active_element, _rtd_selected_config_key,
        )
        from app.reports.rtd import RTD_AGGREGATION_ELEMENT, RTD_ELEMENT_SHEETS
        active_element = _rtd_active_element()
        # Once a configuration has been selected with "Use This Config" the export
        # covers ONLY that configuration, mirroring the results
        # display (transaction_viz.render_rtd_comparison_results).
        selected_key = _rtd_selected_config_key(results_dict or {})
        if selected_key is not None:
            results_dict = {selected_key: results_dict[selected_key]}
            df = results_dict[selected_key]
        export_all_configs = results_dict is not None and len(results_dict) > 1

        if active_element == RTD_AGGREGATION_ELEMENT:
            st.markdown(
                "**Decision 4 Integrated Default List Export:** one row per agent with "
                "every input and z-score, each element's score, segment and list, the "
                "integrated ranking with its tie-break diagnostics and the integrated "
                "default list."
            )
        elif active_element:
            st.markdown(
                f"**Decision 4 Results Export ({RTD_ELEMENT_SHEETS[active_element]} element):** "
                "one row per agent with the element's independent variables and the "
                "z-scores its equation uses, its score, segments and the resulting "
                "option sequence."
            )
        else:
            st.markdown(
                "**Decision 4 (Rejected Transaction Defaults) Results Export:** one row "
                "per agent with the Decision 4 element results - the integrated default "
                "list sheet with every input, z-score, element score and segment, then "
                "one self-contained sheet per element."
            )
        if selected_key is not None:
            st.caption("Selected configuration only.")
        if export_all_configs:
            st.caption(f"{len(results_dict)} configurations - sheet names are prefixed "
                       "with the configuration.")

        try:
            def _rtd_sheets_for(frame):
                return rtd_element_subset(_prepare_rtd_model_export(frame) or {}, active_element)

            if export_all_configs:
                sheets_data = {}
                for config_key, config_df in results_dict.items():
                    if (config_df is None or config_df.empty
                            or 'rtd_choice_length' not in config_df.columns):
                        continue
                    prefix = rtd_config_sheet_prefix(config_key)
                    for name, sheet_df in _rtd_sheets_for(config_df).items():
                        sheets_data[rtd_prefixed_sheet_name(prefix, name)] = sheet_df
            else:
                sheets_data = _rtd_sheets_for(df)

            if not sheets_data:
                st.warning("⚠️ No Decision 4 data available for export")
            else:
                excel_bytes = rtd_model_xlsx_bytes(sheets_data)
                first = next(iter(sheets_data.values()))
                # the one keyed download of the section, so it is drawn here
                _metrics_row([("Agents", len(first)), ("Sheets", len(sheets_data))])
                st.download_button(
                    label="📊 Download Decision 4 Agent-Level Excel",
                    data=excel_bytes,
                    file_name=f"rejected_transaction_defaults_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime=_XLSX_MIME,
                    help="One row per agent with the Decision 4 element results",
                    key="rtd_export_section_download",
                )
                with st.expander("📋 Preview Decision 4 Data (first 5 rows per sheet)"):
                    _preview_sheets(sheets_data, as_text=True)
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
                # The three session values the builder's TimestampConverter used
                # to read for itself: today's midnight (through the module, so a
                # patched `timestamps.midnight_of` is honoured) and Page 1's
                # period duration and period count.
                base_time=timestamps.midnight_of(datetime.now()),
                duration_hours=float(getattr(sim_params, 'duration_hours', DEFAULT_DURATION_HOURS)),
                periods=int(getattr(sim_params, 'periods', DEFAULT_PERIODS)),
                market_price=getattr(sim_params, 'market_price', 100.0),
                platform_markup=getattr(sim_params, 'platform_markup', 0.1),
                price_range=getattr(sim_params, 'price_range', 0.25),
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
                _xlsx_download_button(
                    "📊 Download Agent-Level Excel",
                    agent_bytes,
                    f"simulation_agent_level_{timestamp}.xlsx",
                    f"Agent-level data: {len(agent_df)} agents × {len(agent_df.columns)} columns"
                )
            
            with col2:
                _xlsx_download_button(
                    "📊 Download Transaction-Level Excel",
                    transaction_bytes,
                    f"simulation_transaction_level_{timestamp}.xlsx",
                    f"Transaction-level data: {len(transaction_df)} transactions × {len(transaction_df.columns)} columns"
                )
            
            # Show preview of what's in each file
            st.markdown("### 📋 Data Preview")
            
            with st.expander("👥 Preview Agent-Level Data (first 5 rows)"):
                _preview_frame(agent_df, 15)
            
            with st.expander("🔄 Preview Transaction-Level Data (first 5 rows)"):
                _preview_frame(transaction_df, 15)
            
        except Exception as e:
            st.error(f"Error creating Excel export: {str(e)}")
            st.caption("⚠️ Please ensure all required data is available. If the problem persists, contact support.")
            import traceback
            st.caption(f"Error details: {traceback.format_exc()}")
            
            # Fallback: show raw data
            with st.expander("🔍 View Raw Data (for debugging)"):
                st.dataframe(rename_consensus_columns(df), use_container_width=True)

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
