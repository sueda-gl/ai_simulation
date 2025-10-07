# app/pages/results/decision_visualizations.py
"""
Decision-specific visualization functions for the Enhanced AI Agent Simulation.
Contains all render_* functions for different decision types.
"""
import streamlit as st
import pandas as pd
import plotly.express as px


def render_decision_results(df, decision_name, decision_title):
    """Render detailed results for a specific decision using decision-specific visualizations"""
    if decision_name not in df.columns:
        st.warning(f"No results available for {decision_title}")
        return
    
    decision_data = df[decision_name]
    
    # Get the appropriate visualization function for this decision
    viz_function = DECISION_VISUALIZATIONS.get(decision_name)
    
    # Call the specific visualization function
    viz_function(df, decision_name, decision_title, decision_data)


# =============================================================================
# DECISION-SPECIFIC VISUALIZATION FUNCTIONS
# =============================================================================

def render_donation_default(df, decision_name, decision_title, decision_data):
    """Visualization for donation_default - placeholder until specialized view is added"""
    try:
        numeric_data = pd.to_numeric(decision_data, errors='coerce')
        if not numeric_data.isna().all():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Agents", f"{len(decision_data):,}")
            with col2:
                st.metric("Mean", f"{numeric_data.mean():.1%}")
            with col3:
                st.metric("Std Dev", f"{numeric_data.std():.2%}")
            with col4:
                st.metric("Range", f"{numeric_data.min():.1%} - {numeric_data.max():.1%}")
            col_plot, col_stats = st.columns([2, 1])
            with col_plot:
                fig = px.histogram(
                    df,
                    x=decision_name,
                    nbins=30,
                    title=f"Distribution of {decision_title}",
                    labels={decision_name: decision_title, 'count': 'Number of Agents'}
                )
                fig.update_layout(
                    showlegend=False,
                    xaxis_tickformat='.0%'
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_stats:
                st.markdown("**📈 Statistics**")
                stats = numeric_data.describe()
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
                    'Value': [f"{stats[key]:.1%}" for key in ['mean', 'std', 'min', 'max', '50%', '25%', '75%']]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("Data not numeric; specialized visualization not available yet.")
    except Exception:
        st.info("Unable to render donation_default with placeholder visualization.")


def render_final_donation_rate(df, decision_name, decision_title, decision_data):
    """Visualization for final_donation_rate with 3-case logic for donation configs"""
    
    # CASE 3: Check if a donation configuration has been selected
    has_selected_config = hasattr(st.session_state, 'selected_donation_config')
    
    # CASE 1: Check if exactly one donation config exists (auto-use it)
    is_single_donation_run = (
        hasattr(st.session_state, 'custom_decisions') and 
        st.session_state.custom_decisions == ['donation_default'] and
        hasattr(st.session_state, 'default_decisions') and
        len(st.session_state.default_decisions) == 0
    )
    
    # If this is a single donation run with exactly one result, treat it as "only config available"
    has_only_one_config = False
    if is_single_donation_run and hasattr(st.session_state, 'simulation_results'):
        results_dict = st.session_state.simulation_results
        if results_dict and len(results_dict) == 1:
            has_only_one_config = True
            only_config_key = list(results_dict.keys())[0]
            only_config_df = results_dict[only_config_key]
    
    # Decision logic: Use distribution if selected config OR only one config available
    use_distribution = (has_selected_config or has_only_one_config) and 'donation_default' in df.columns
    
    if use_distribution:
        # Show the actual donation distribution - distinguish between cases
        if has_selected_config:
            st.success("📊 **Using Distribution from Selected Donation Configuration**")
            st.caption("✅ The final_donation_rate values in your export match the donation_default distribution shown below")
        elif has_only_one_config:
            st.success("📊 **Using Distribution from Only Available Donation Configuration**")
            st.caption("✅ Only one donation configuration was generated - final_donation_rate values match donation_default")
        
        donation_data = df['donation_default']
        
        # Top section: Distribution statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Agents", f"{len(donation_data):,}")
        
        with col2:
            st.metric("Mean Rate", f"{donation_data.mean():.1%}")
        
        with col3:
            st.metric("Median Rate", f"{donation_data.median():.1%}")
        
        with col4:
            st.metric("Std Dev", f"{donation_data.std():.1%}")
        
        # Distribution visualization
        st.markdown("---")
        st.markdown("**📊 Donation Rate Distribution:**")
        
        col_hist, col_stats = st.columns([2, 1])
        
        with col_hist:
            # Histogram showing the distribution - match overview chart settings for consistency
            fig = px.histogram(
                df,
                x='donation_default',
                title="Distribution of Donation Rates Across Agents",
                labels={'donation_default': 'Donation Rate', 'count': 'Number of Agents'},
                nbins=30,  # Match overview chart
                marginal="box"  # Match overview chart
            )
            fig.update_layout(
                xaxis_tickformat='.0%',
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_stats:
            st.markdown("**📈 Distribution Stats:**")
            st.write(f"• **Min**: {donation_data.min():.1%}")
            st.write(f"• **25th %ile**: {donation_data.quantile(0.25):.1%}")
            st.write(f"• **50th %ile**: {donation_data.quantile(0.50):.1%}")
            st.write(f"• **75th %ile**: {donation_data.quantile(0.75):.1%}")
            st.write(f"• **Max**: {donation_data.max():.1%}")
            st.write(f"• **Range**: {donation_data.max() - donation_data.min():.1%}")
            
            st.markdown("---")
            st.markdown("**ℹ️ Source:**")
            if hasattr(st.session_state, 'selected_donation_config'):
                config = st.session_state.selected_donation_config
                st.caption(f"Population: {config['population_mode']}")
                st.caption(f"Income: {config['income_spec_mode']}")
    
    else:
        # Fall back to slider if no donation_default data available
        st.info("💡 **No donation configuration selected** - Using simple rate configuration")
        st.caption("Select a donation configuration on Page 2 to see the full distribution")
        
        # Initialize session state for donation rate (default 10%)
        slider_key = "final_donation_rate_slider"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = 0.10  # 10% as default
        
        # Top section: Current settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Agents", f"{len(decision_data):,}")
        
        with col2:
            current_rate = st.session_state[slider_key]
            st.metric("Current Rate", f"{current_rate:.1%}")
        
        with col3:
            st.metric("Default", "10%")
        
        # Main configuration section
        st.markdown("---")
        st.markdown("**🎛️ Configure Final Donation Rate:**")
        
        col_slider, col_info = st.columns([2, 1])
        
        with col_slider:
            # Donation rate slider (0% to 100%)
            donation_rate = st.slider(
                "Final Donation Rate",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state[slider_key],
                step=0.01,
                key=slider_key,
                help="Set the final donation rate as a percentage"
            )
            
            # Show the current percentage value
            st.write(f"**Selected Rate: {donation_rate:.1%}**")
            
            # Apply button to update the rate
            if st.button("🔄 Apply New Rate", type="primary", help="Update final donation rate for future simulations"):
                # Update session state for decision execution
                st.session_state["final_donation_rate_config"] = donation_rate
                st.success(f"✅ Final donation rate updated to: {donation_rate:.1%}")
                st.info("💡 Run a new simulation to see the changes take effect")
        
        with col_info:
            st.markdown("**📋 Rate Information:**")
            st.write(f"• **Selected**: {donation_rate:.1%}")
            st.write(f"• **Default**: 10%")
            st.write(f"• **Range**: 0% - 100%")
            
            if donation_rate == 0.10:
                st.success("✅ Using default rate")
            elif donation_rate < 0.10:
                st.info(f"📉 {abs(donation_rate - 0.10):.1%} below default")
            else:
                st.info(f"📈 {donation_rate - 0.10:.1%} above default")


def render_disclose_income(df, decision_name, decision_title, decision_data):
    """Visualization for disclose_income - binary Y/N choice"""
    
    # Binary choice metrics
    col1, col2, col3, col4 = st.columns(4)
    
    value_counts = decision_data.value_counts()
    total = len(decision_data)
    
    with col1:
        st.metric("Total Agents", f"{total:,}")
    with col2:
        yes_count = value_counts.get('Y', 0)
        st.metric("Disclosed (Y)", f"{yes_count:,} ({yes_count/total:.1%})")
    with col3:
        no_count = value_counts.get('N', 0)
        st.metric("Not Disclosed (N)", f"{no_count:,} ({no_count/total:.1%})")
    with col4:
        st.metric("Disclosure Rate", f"{yes_count/total:.1%}")
    
    # Binary choice visualization - pie chart
    col_plot, col_stats = st.columns([2, 1])
    
    with col_plot:
        if len(value_counts) > 0:
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"{decision_title} Distribution",
                color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'}  # Green for Yes, Red for No
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col_stats:
        st.markdown("**📊 Choice Breakdown**")
        breakdown_df = pd.DataFrame({
            'Choice': value_counts.index,
            'Count': value_counts.values,
            'Percentage': [f"{(count/total)*100:.1f}%" for count in value_counts.values]
        })
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)


def render_disclose_documents(df, decision_name, decision_title, decision_data):
    """Visualization for disclose_documents - binary Y/N choice with NA handling
    
    This decision only applies to agents qualified for discount (income < threshold).
    Agents not qualified will have "NA" value.
    """
    
    # Separate NA (not applicable) from Y/N choices
    value_counts = decision_data.value_counts()
    total_agents = len(decision_data)
    
    na_count = value_counts.get('NA', 0)
    qualified_agents = total_agents - na_count
    
    # Show overall metrics
    st.markdown("### Eligibility & Application")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Agents", f"{total_agents:,}")
    with col2:
        st.metric("Qualified for Discount", f"{qualified_agents:,}", 
                  help="Agents with income < discount threshold")
    with col3:
        st.metric("Not Qualified (NA)", f"{na_count:,}", 
                  help="Agents with income ≥ discount threshold (decision does not apply)")
    
    # If there are qualified agents, show their Y/N choices
    if qualified_agents > 0:
        st.markdown("### Qualified Agents' Choices")
        st.caption(f"📊 Among the {qualified_agents:,} agents qualified for discount (income < threshold)")
        
        # Binary choice metrics for qualified agents only
        col1, col2, col3, col4 = st.columns(4)
        
        yes_count = value_counts.get('Y', 0)
        no_count = value_counts.get('N', 0)
        
        with col1:
            st.metric("Qualified Agents", f"{qualified_agents:,}")
        with col2:
            st.metric("Disclosed (Y)", f"{yes_count:,} ({yes_count/qualified_agents:.1%})")
        with col3:
            st.metric("Not Disclosed (N)", f"{no_count:,} ({no_count/qualified_agents:.1%})")
        with col4:
            st.metric("Disclosure Rate", f"{yes_count/qualified_agents:.1%}",
                      help="Percentage of qualified agents who disclosed documents")
        
        # Binary choice visualization - pie chart (only Y/N, excluding NA)
        col_plot, col_stats = st.columns([2, 1])
        
        with col_plot:
            # Filter out NA for the pie chart
            qualified_counts = {k: v for k, v in value_counts.items() if k != 'NA'}
            if len(qualified_counts) > 0:
                fig = px.pie(
                    values=list(qualified_counts.values()),
                    names=list(qualified_counts.keys()),
                    title=f"{decision_title} - Qualified Agents Only",
                    color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'}  # Green for Yes, Red for No
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col_stats:
            st.markdown("**📊 Choice Breakdown (Qualified)**")
            qualified_breakdown = pd.DataFrame({
                'Choice': list(qualified_counts.keys()),
                'Count': list(qualified_counts.values()),
                'Percentage': [f"{(count/qualified_agents)*100:.1f}%" for count in qualified_counts.values()]
            })
            st.dataframe(qualified_breakdown, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No agents qualified for discount (all agents have income ≥ threshold)")
    
    # Show full breakdown including NA
    st.markdown("### Complete Breakdown (All Agents)")
    full_breakdown = pd.DataFrame({
        'Status': value_counts.index,
        'Count': value_counts.values,
        'Percentage': [f"{(count/total_agents)*100:.1f}%" for count in value_counts.values]
    })
    st.dataframe(full_breakdown, use_container_width=True, hide_index=True)


def render_purchase_vs_bid(df, decision_name, decision_title, decision_data):
    """Visualization for purchase_vs_bid - binary Purchase Now/bid choice"""
    
    # Purchase Now vs Bid metrics
    col1, col2, col3, col4 = st.columns(4)
    
    value_counts = decision_data.value_counts()
    total = len(decision_data)
    
    with col1:
        st.metric("Total Agents", f"{total:,}")
    with col2:
        purchase_count = value_counts.get('Purchase Now', 0)
        st.metric("Purchase Now", f"{purchase_count:,} ({purchase_count/total:.1%})")
    with col3:
        bid_count = value_counts.get('bid', 0)
        st.metric("Bid", f"{bid_count:,} ({bid_count/total:.1%})")
    with col4:
        st.metric("Purchase Now Rate", f"{purchase_count/total:.1%}")
    
    # Purchase Now vs Bid visualization - donut chart
    col_plot, col_stats = st.columns([2, 1])
    
    with col_plot:
        if len(value_counts) > 0:
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"{decision_title} Distribution",
                hole=0.4,  # Donut chart
                color_discrete_map={'Purchase Now': '#4CAF50', 'bid': '#FF9800'}  # Green for Purchase Now, Orange for bid
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col_stats:
        st.markdown("**🛒 Transaction Choices**")
        breakdown_df = pd.DataFrame({
            'Choice': value_counts.index,
            'Count': value_counts.values,
            'Percentage': [f"{(count/total)*100:.1f}%" for count in value_counts.values]
        })
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)


def render_rejected_transaction_defaults(df, decision_name, decision_title, decision_data):
    """Visualization for rejected_transaction_defaults with interactive radio buttons"""
    
    # Define the 5 options
    options = [
        ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
        ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"),
        ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"), 
        ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
        ("forgo_transaction", "Option 5: Forgo the purchase request")
    ]
    
    option_names = dict(options)
    
    # Check the actual simulation execution mode from session state
    simulation_mode = "unknown"
    if hasattr(st.session_state, 'sim_params') and hasattr(st.session_state.sim_params, 'simulation_execution_mode'):
        simulation_mode = st.session_state.sim_params.simulation_execution_mode
    
    # Get current default from results or session state
    value_counts = decision_data.value_counts()
    current_default = value_counts.index[0] if len(value_counts) > 0 else "forgo_transaction"
    
    # Initialize session state for radio button selection
    # Use the actual result from simulation, not a separate radio key
    radio_key = "rejected_transaction_defaults_option"
    if radio_key not in st.session_state:
        # Initialize with what was actually used in the simulation
        st.session_state[radio_key] = current_default
    
    # Top section: Current results display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    with col2:
        st.metric("Simulation Mode", simulation_mode.title())
    
    with col3:
        # Show what's configured for NEXT simulation (from session state)
        configured_option = st.session_state.get(radio_key, current_default)
        display_name = option_names.get(configured_option, configured_option)
        st.metric("Configured Default", display_name.split(":")[0])  # Just "Option X"
    
    with col4:
        # Show consistency of CURRENT results
        if len(value_counts) > 0:
            percentage = (value_counts.iloc[0] / len(decision_data)) * 100
            st.metric("Results Consistency", f"{percentage:.1f}%")
        else:
            st.metric("Results Consistency", "N/A")
    
    # Main configuration section with radio buttons
    st.markdown("---")
    st.markdown("**🎛️ Configure Default Behavior for Future Simulations:**")
    
    col_radio, col_viz = st.columns([1, 1])
    
    with col_radio:
        st.markdown("**Select Default Option:**")
        
        # Create radio buttons for the 5 options
        selected_option = st.radio(
            "When transactions are rejected, agents should:",
            options=[opt[0] for opt in options],
            format_func=lambda x: option_names[x],
            key=radio_key,
            help="Choose the default behavior for rejected transactions"
        )
        
        # Show description based on simulation mode
        if simulation_mode == "snapshot":
            st.success(f"✅ **Snapshot Mode**: All agents will use '{option_names[selected_option]}' when transactions are rejected")
        elif simulation_mode == "live":
            st.info(f"🔴 **Live Mode**: '{option_names[selected_option]}' will be the fallback default, but agents will be asked in real-time")
        else:
            st.caption(f"Selected: {option_names[selected_option]}")
        
        # Apply button to re-run simulation with new default
        if st.button("🔄 Apply New Default", type="primary", help="Re-run simulation with this default", key="apply_rejected_defaults"):
            # The radio button already updates session state automatically via its key
            # Now re-run the simulation
            from app.simulation import run_simulation_from_sidebar
            
            with st.spinner(f"Re-running simulation with {option_names[selected_option]}..."):
                # Clear current results to force fresh display
                st.session_state.simulation_results = None
                run_simulation_from_sidebar()
                st.success(f"✅ Simulation complete with: {option_names[selected_option]}")
                st.rerun()
    
    with col_viz:
        # Show current results visualization
        if len(value_counts) > 0:
            # Create readable labels for the chart
            readable_labels = [option_names.get(opt, opt) for opt in value_counts.index]
            
            fig = px.pie(
                values=value_counts.values,
                names=readable_labels,
                title="Current Simulation Results",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True, key="rejected_transaction_defaults_chart")
        else:
            st.info("No simulation data available")
    

def render_vendor_choice_weights(df, decision_name, decision_title, decision_data):
    """Visualization for vendor_choice_weights with interactive parameter selection"""
    
    # Define the 4 vendor choice parameters
    parameters = [
        ("price", "Price", "Cost of the product/service"),
        ("quality", "Quality", "Quality rating and reviews"),
        ("proximity", "Proximity", "Distance and convenience"),
        ("sustainability", "Sustainability", "Environmental and social impact")
    ]
    
    param_names = {param[0]: param[1] for param in parameters}
    param_descriptions = {param[0]: param[2] for param in parameters}
    
    # Initialize session state for parameter selection
    # Try to infer from actual results what was used
    selection_key = "vendor_choice_weights_selection"
    if selection_key not in st.session_state:
        # Try to infer selection from the actual weights in the data
        if not decision_data.empty and isinstance(decision_data.iloc[0], dict):
            # The data contains weight dictionaries
            sample_weights = decision_data.iloc[0]
            # Find which parameters have non-zero weights
            inferred_selection = [key for key, weight in sample_weights.items() if weight > 0]
            if inferred_selection:
                st.session_state[selection_key] = inferred_selection
            else:
                # Default to all if no inference possible
                st.session_state[selection_key] = ["price", "quality", "proximity", "sustainability"]
        else:
            # Default to all if data format doesn't match
            st.session_state[selection_key] = ["price", "quality", "proximity", "sustainability"]
    
    # Top section: Current results display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    with col2:
        # Show number of selected parameters
        num_selected = len(st.session_state[selection_key])
        st.metric("Active Parameters", f"{num_selected}/4")
    
    with col3:
        # Show current weight per parameter
        if num_selected > 0:
            weight_per_param = 100 / num_selected
            st.metric("Weight Each", f"{weight_per_param:.1f}%")
        else:
            st.metric("Weight Each", "0%")
    
    with col4:
        # Show current configuration
        if num_selected == 4:
            st.metric("Configuration", "All Factors")
        elif num_selected == 1:
            st.metric("Configuration", "Single Factor")
        else:
            st.metric("Configuration", f"{num_selected} Factors")
    
    # Main configuration section
    st.markdown("---")
    st.markdown("**🎛️ Configure Vendor Choice Parameters:**")
    
    col_selection, col_viz = st.columns([1, 1])
    
    with col_selection:
        st.markdown("**Select Parameters to Include:**")
        
        # Create checkboxes for each parameter with direct state management
        selected_params = []
        
        for param_key, param_name, param_desc in parameters:
            # Use a unique key for each checkbox that's independent
            checkbox_key = f"vendor_weights_{param_key}_checkbox"
            
            # Initialize checkbox state from session state if not exists
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = param_key in st.session_state[selection_key]
            
            # Create checkbox
            is_selected = st.checkbox(
                f"{param_name}",
                key=checkbox_key,
                help=param_desc
            )
            
            # Add to selected params if checked
            if is_selected:
                selected_params.append(param_key)
        
        # Update the main selection state only if it has changed
        if set(selected_params) != set(st.session_state[selection_key]):
            st.session_state[selection_key] = selected_params
        
        # Calculate and display weights
        if len(selected_params) > 0:
            weight_per_param = 1.0 / len(selected_params)
            
            st.markdown("**📊 Calculated Weights:**")
            
            # Show weight distribution
            weight_data = []
            for param_key in selected_params:
                weight_data.append({
                    'Parameter': param_names[param_key],
                    'Weight': f"{weight_per_param:.1%}",
                    'Decimal': f"{weight_per_param:.3f}"
                })
            
            if weight_data:
                weight_df = pd.DataFrame(weight_data)
                st.dataframe(weight_df, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Please select at least one parameter")
        
        # Apply button to update the weights
        if len(selected_params) > 0:
            if st.button("🔄 Apply New Weights", type="primary", help="Update vendor choice weights for future simulations"):
                # Calculate weights dictionary
                weight_per_param = 1.0 / len(selected_params)
                new_weights = {}
                
                # Set weights for selected parameters
                for param_key in selected_params:
                    new_weights[param_key] = weight_per_param
                
                # Set zero weights for unselected parameters
                for param_key, _, _ in parameters:
                    if param_key not in selected_params:
                        new_weights[param_key] = 0.0
                
                # Update session state for decision execution
                st.session_state["vendor_choice_weights_config"] = new_weights
                st.success(f"✅ Weights updated! {len(selected_params)} parameters with {weight_per_param:.1%} each")
                st.info("💡 Run a new simulation to see the changes take effect")
    
    with col_viz:
        # Show current weights visualization
        if len(selected_params) > 0:
            # Create pie chart showing weight distribution
            weight_per_param = 1.0 / len(selected_params)
            
            fig = px.pie(
                values=[weight_per_param] * len(selected_params),
                names=[param_names[param] for param in selected_params],
                title="Vendor Choice Weight Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True, key="vendor_choice_weights_chart")
            
            # Show summary
            st.markdown("**📋 Weight Summary:**")
            summary_text = []
            for param_key in selected_params:
                summary_text.append(f"• {param_names[param_key]}: {weight_per_param:.1%}")
            
            if len(selected_params) < 4:
                summary_text.append("")
                summary_text.append("**Excluded:**")
                for param_key, param_name, _ in parameters:
                    if param_key not in selected_params:
                        summary_text.append(f"• {param_name}: 0%")
            
            st.markdown("\n".join(summary_text))
        else:
            st.info("Select parameters to see weight distribution")
    

def render_consumption_quantity(df, decision_name, decision_title, decision_data):
    """Visualization for consumption_quantity - quantity analysis"""
    st.markdown("**Random within consumption limit**")


def render_consumption_frequency(df, decision_name, decision_title, decision_data):
    """Visualization for consumption_frequency - frequency analysis"""
    st.markdown("**Consumption quantity divided by period duration**")


def render_vendor_selection(df, decision_name, decision_title, decision_data):
    """Visualization for vendor_selection - vendor choice analysis"""
    st.markdown("**Deterministic based on weights**")


def render_bid_value(df, decision_name, decision_title, decision_data):
    """Visualization for bid_value with bidding price range formula"""
    
    # Get parameters from session state (from Page 1)
    if hasattr(st.session_state, 'sim_params'):
        vendor_price = getattr(st.session_state.sim_params, 'market_price', 100.0)  # Default €100
        platform_markup = getattr(st.session_state.sim_params, 'platform_markup', 0.1)  # Default 10%
        price_range = getattr(st.session_state.sim_params, 'price_range', 0.25)  # Default 25%
    else:
        # Fallback defaults
        vendor_price = 100.0
        platform_markup = 0.1
        price_range = 0.25
    
    # Calculate bidding range using the formula
    baseline_price = (1 + platform_markup) * vendor_price  # Pc = (1+m) × vendor_price
    min_bid_price = (1 - price_range) * baseline_price      # Pmb = (1-r) × Pc
    max_bid_price = (1 + price_range) * baseline_price      # Ppn = (1+r) × Pc
    
    # Top section: Current parameters and calculated range
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    with col2:
        st.metric("Vendor Price", f"€{vendor_price:.2f}")
    
    with col3:
        st.metric("Baseline Price (Pc)", f"€{baseline_price:.2f}")
    
    with col4:
        st.metric("Range Parameter (r)", f"{price_range:.1%}")
    
    # Bidding range display
    st.markdown("---")
    st.markdown("**📊 Bidding Range Formula**")
    
    col_formula, col_range = st.columns([1, 1])
    
    with col_formula:
        st.markdown("**Formula Components:**")
        st.write(f"• **Vendor Price**: €{vendor_price:.2f}")
        st.write(f"• **Platform Markup (m)**: {platform_markup:.1%}")
        st.write(f"• **Range Parameter (r)**: {price_range:.1%}")
        st.write("")
        st.markdown("**Calculations:**")
        st.write(f"• **Baseline Price (Pc)**: (1 + {platform_markup:.1%}) × €{vendor_price:.2f} = €{baseline_price:.2f}")
        st.write(f"• **Min Bid (Pmb)**: (1 - {price_range:.1%}) × €{baseline_price:.2f} = €{min_bid_price:.2f}")
        st.write(f"• **Max Bid (Ppn)**: (1 + {price_range:.1%}) × €{baseline_price:.2f} = €{max_bid_price:.2f}")
    
    with col_range:
        st.markdown("**📈 Bidding Range:**")
        
        # Visual range display
        range_width = max_bid_price - min_bid_price
        
        # Create metrics for the range
        col_min, col_max = st.columns(2)
        with col_min:
            st.metric("Minimum Bid", f"€{min_bid_price:.2f}")
        with col_max:
            st.metric("Maximum Bid", f"€{max_bid_price:.2f}")
        
        st.metric("Range Width", f"€{range_width:.2f}")
        
        # Show the range notation
        st.success(f"**Bidding Range**: [€{min_bid_price:.2f}, €{max_bid_price:.2f})")
        st.caption("Range notation: [minimum, maximum)")
    
    # Configuration section
    st.markdown("---")
    st.markdown("**🎛️ Bidding Behavior:**")
    
    st.info("**Default Behavior**: Random bid amount within the calculated range")
    st.caption("💡 Agents will select random bid values between the minimum and maximum bid prices")
    
    # Show example bids
    if st.button("🎲 Show Example Bids", help="Generate sample bid values within the range"):
        import random
        st.markdown("**🎯 Example Bid Values:**")
        
        # Generate 5 random example bids
        example_bids = []
        for i in range(5):
            random_bid = random.uniform(min_bid_price, max_bid_price)
            example_bids.append(f"€{random_bid:.2f}")
        
        st.write(f"Sample bids: {', '.join(example_bids)}")
        st.caption(f"All values fall within [€{min_bid_price:.2f}, €{max_bid_price:.2f})")
    
    # Current simulation results summary (if available)
    if not decision_data.empty:
        st.markdown("---")
        st.markdown("**📊 Current Simulation Summary:**")
        
        try:
            # Try to get numeric data for analysis
            numeric_data = pd.to_numeric(decision_data, errors='coerce')
            if not numeric_data.isna().all():
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                
                with col_stats1:
                    st.metric("Mean Bid", f"€{numeric_data.mean():.2f}")
                
                with col_stats2:
                    st.metric("Min Bid", f"€{numeric_data.min():.2f}")
                
                with col_stats3:
                    st.metric("Max Bid", f"€{numeric_data.max():.2f}")
            else:
                st.caption("Current results show non-numeric bid data")
        except:
            st.caption("Current results show complex bid data structure")



def render_rejected_transaction_option(df, decision_name, decision_title, decision_data):
    """Visualization for rejected_transaction_option with interactive radio buttons"""
    
    # Define the 5 options
    options = [
        ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
        ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"),
        ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"), 
        ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
        ("forgo_transaction", "Option 5: Forgo the purchase request")
    ]
    
    option_names = dict(options)
    
    # Get current option from results or session state
    value_counts = decision_data.value_counts()
    current_option = value_counts.index[0] if len(value_counts) > 0 else "forgo_transaction"
    
    # Initialize session state for radio button selection
    # Use the actual result from simulation, not a separate radio key
    radio_key = "rejected_transaction_option_selection"
    if radio_key not in st.session_state:
        # Initialize with what was actually used in the simulation
        st.session_state[radio_key] = current_option
    
    # Top section: Current results display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    with col2:
        # Show what's configured for NEXT simulation (from session state)
        configured_option = st.session_state.get(radio_key, current_option)
        display_name = option_names.get(configured_option, configured_option)
        st.metric("Configured Option", display_name.split(":")[0])  # Just "Option X"
    
    with col3:
        # Show most common from CURRENT results
        most_common_option = option_names.get(current_option, current_option)
        st.metric("Most Common Result", most_common_option.split(":")[0])
    
    with col4:
        if len(value_counts) > 0:
            percentage = (value_counts.iloc[0] / len(decision_data)) * 100
            st.metric("Result Frequency", f"{percentage:.1f}%")
    
    # Main configuration section with radio buttons
    st.markdown("---")
    st.markdown("**🎛️ Configure Specific Option for Future Simulations:**")
    
    col_radio, col_viz = st.columns([1, 1])
    
    with col_radio:
        st.markdown("**Select Specific Option:**")
        
        # Create radio buttons for the 5 options
        selected_option = st.radio(
            "When this specific transaction is rejected, agents should:",
            options=[opt[0] for opt in options],
            format_func=lambda x: option_names[x],
            key=radio_key,
            help="Choose the specific behavior for this rejected transaction scenario"
        )
        
        
        # Apply button to re-run simulation with new option
        if st.button("🔄 Apply New Option", type="primary", help="Re-run simulation with this option", key="apply_rejected_option"):
            # The radio button already updates session state automatically via its key
            # Now re-run the simulation
            from app.simulation import run_simulation_from_sidebar
            
            with st.spinner(f"Re-running simulation with {option_names[selected_option]}..."):
                # Clear current results to force fresh display
                st.session_state.simulation_results = None
                run_simulation_from_sidebar()
                st.success(f"✅ Simulation complete with: {option_names[selected_option]}")
                st.rerun()
    
    with col_viz:
        # Show current results visualization
        if len(value_counts) > 0:
            # Create readable labels for the chart
            readable_labels = [option_names.get(opt, opt) for opt in value_counts.index]
            
            fig = px.pie(
                values=value_counts.values,
                names=readable_labels,
                title="Current Simulation Results",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True, key="rejected_transaction_option_chart")
        else:
            st.info("No simulation data available")
    

def render_rejected_bid_value(df, decision_name, decision_title, decision_data):
    """Visualization for rejected_bid_value"""
    st.markdown("**Not relevant given choice of Option 5**")


def _render_missing_visualization(decision_title: str) -> None:
    """Fallback when a specific visualization is not implemented."""
    st.info(f"No specialized visualization implemented for {decision_title}.")


def render_probability_controls(decision_name, df):
    """Render probability controls for random Y/N decisions directly under their display"""
    
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES
    
    # Check if this is a random decision that needs controls
    default_value = DEFAULT_DECISION_VALUES.get(decision_name)
    
    if isinstance(default_value, dict) and default_value.get("type") == "random_probability":
        st.markdown("**🎛️ Adjust Probability Settings:**")
        
        options = default_value.get("options", ["Y", "N"])
        description = default_value.get("description", "Probability")
        current_prob = st.session_state.get(f"{decision_name}_probability_y", default_value.get("probability_y", 0.5))
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Use the actual probability key directly for the slider
            new_prob = st.slider(
                f"P({options[0]}) - {description}",
                min_value=0.0, max_value=1.0, value=current_prob, step=0.01,
                help=f"Probability of {options[0]} vs {options[1]}",
                key=f"{decision_name}_probability_y"
            )
            
        with col2:
            st.metric("Ratio", f"{new_prob:.0%} : {1-new_prob:.0%}")
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
                        st.metric("Current", f"{actual_y_ratio:.0%} : {1-actual_y_ratio:.0%}")
                        st.caption("Actual Results")
        
        # Always show re-run button for clarity
        if st.button(
            f"🔄 Re-run with P({options[0]})={st.session_state[f'{decision_name}_probability_y']:.0%}", 
            key=f"rerun_{decision_name}",
            type="primary",
            help="Re-run simulation with new probability settings"
        ):
            st.info(f"Re-running simulation with {options[0]} probability = {st.session_state[f'{decision_name}_probability_y']:.0%}...")
            # Clear existing results to force a fresh simulation
            if hasattr(st.session_state, 'simulation_results'):
                st.session_state.simulation_results = None
            # Trigger re-run by calling the combined simulation again
            from app.pages.decision_execution import run_combined_simulation
            if hasattr(st.session_state, 'custom_decisions') and hasattr(st.session_state, 'default_decisions'):
                selected_decisions = st.session_state.custom_decisions
                run_combined_simulation(selected_decisions)
                # Force page refresh after simulation
                st.rerun()


def get_dynamic_description(decision_name):
    """Get dynamic description for decisions showing current probability settings"""
    
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES, DEFAULT_DECISION_DESCRIPTIONS
    
    default_value = DEFAULT_DECISION_VALUES.get(decision_name)
    
    # For parametric random decisions, show current probability
    if isinstance(default_value, dict) and default_value.get("type") == "random_probability":
        options = default_value.get("options", ["Y", "N"])
        current_prob = st.session_state.get(f"{decision_name}_probability_y", default_value.get("probability_y", 0.5))
        
        return f"{current_prob:.0%} chance of {options[0]}, {1-current_prob:.0%} chance of {options[1]}"
    
    # For other decisions, use static description
    return DEFAULT_DECISION_DESCRIPTIONS.get(decision_name, "Standard default behavior")


# =============================================================================
# DECISION VISUALIZATION REGISTRY
# =============================================================================

DECISION_VISUALIZATIONS = {
    # Donation decisions
    'donation_default': render_donation_default,
    'donation_default_raw': render_donation_default,  # Same as donation_default
    'final_donation_rate': render_final_donation_rate,
    
    # Disclosure decisions
    'disclose_income': render_disclose_income,
    'disclose_documents': render_disclose_documents,
    
    # Transaction decisions
    'rejected_transaction_defaults': render_rejected_transaction_defaults,
    'purchase_vs_bid': render_purchase_vs_bid,
    'rejected_transaction_option': render_rejected_transaction_option,
    'rejected_bid_value': render_rejected_bid_value,
    
    # Vendor decisions
    'vendor_choice_weights': render_vendor_choice_weights,
    'vendor_selection': render_vendor_selection,
    
    # Consumption decisions
    'consumption_quantity': render_consumption_quantity,
    'consumption_frequency': render_consumption_frequency,
    
    # Bidding decisions
    'bid_value': render_bid_value,
    
    # Note: Any decision not in this registry will automatically use render_generic_decision
}
