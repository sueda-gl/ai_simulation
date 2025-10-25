# app/pages/results/visualizations/vendor_viz.py
"""
Vendor-related visualization functions.
Handles vendor_choice_weights and vendor_selection decisions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px


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
    
    # Use _default_ key (same as Page 2 Overview tab) for consistency
    selection_key = f"{decision_name}_default_params"
    
    # Initialize if not exists (try to infer from actual results)
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
        st.markdown("**⚙️ Selected Parameters (Read-Only):**")
        
        # Get selected parameters from session state
        selected_params = st.session_state.get(selection_key, [])
        
        # Display active parameters
        if len(selected_params) > 0:
            st.success(f"✅ **Active Parameters:**")
            for param_key in selected_params:
                st.write(f"• {param_names[param_key]} - {param_descriptions[param_key]}")
        else:
            st.warning("⚠️ No parameters selected")
        
        # Show excluded parameters if any
        excluded_params = [param for param, _, _ in parameters if param not in selected_params]
        if excluded_params:
            st.markdown("**Excluded:**")
            for param_key in excluded_params:
                st.caption(f"• {param_names[param_key]}")
        
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
        
        # Show helpful message
        st.caption("💡 To modify these settings: Go to **Page 2 → Overview Tab**")
    
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


def render_vendor_selection(df, decision_name, decision_title, decision_data):
    """Visualization for vendor_selection - shows vendor distribution and selection logic"""
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    # Count unique vendors selected (excluding NaN)
    vendor_counts = decision_data.dropna().value_counts()
    
    with col2:
        num_vendors_used = len(vendor_counts)
        st.metric("Vendors Selected", f"{num_vendors_used}")
    
    with col3:
        if len(vendor_counts) > 0:
            most_common_vendor = vendor_counts.idxmax()
            most_common_count = vendor_counts.max()
            st.metric("Most Popular Vendor", f"Vendor {int(most_common_vendor)}")
        else:
            st.metric("Most Popular Vendor", "N/A")
    
    with col4:
        agents_with_selection = decision_data.notna().sum()
        pct_with_selection = agents_with_selection / len(decision_data) * 100
        st.metric("Agents with Selection", f"{pct_with_selection:.1f}%")
    
    # Check if only 1 vendor exists
    if num_vendors_used == 1 and len(vendor_counts) == 1:
        st.info(f"""
        ℹ️ **Single Vendor Simulation**: Only 1 vendor was configured on Page 1, so all agents select that vendor.
        
        💡 **To see vendor selection in action**: 
        1. Go to **Page 1 → Market & Vendor Configuration**
        2. Change **Number of Vendors (N)** from 1 to 3 or 5
        3. Re-run the simulation
        
        With multiple vendors, agents will select different vendors based on weighted composite scores.
        """)
    
    # Vendor distribution visualization
    st.markdown("---")
    st.markdown("**📊 Vendor Selection Distribution:**")
    
    if len(vendor_counts) > 0:
        col_plot, col_stats = st.columns([2, 1])
        
        with col_plot:
            # Bar chart showing vendor distribution
            fig = px.bar(
                x=[f"Vendor {int(vid)}" for vid in vendor_counts.index],
                y=vendor_counts.values,
                title="Number of Agents Selecting Each Vendor",
                labels={'x': 'Vendor', 'y': 'Number of Agents'}
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Vendor",
                yaxis_title="Number of Agents"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_stats:
            st.markdown("**📈 Selection Breakdown:**")
            breakdown_df = pd.DataFrame({
                'Vendor': [f"Vendor {int(vid)}" for vid in vendor_counts.index],
                'Agents': vendor_counts.values,
                'Percentage': [f"{(count/agents_with_selection)*100:.1f}%" for count in vendor_counts.values]
            })
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
    else:
        st.info("No vendor selections found (agents may have 0 purchases)")
    
    # Vendor Data Section (only for multiple vendors)
    if num_vendors_used > 1:
        st.markdown("---")
        st.markdown("**🏪 Vendor Data & Selection Analysis:**")
        st.caption("Understanding why certain vendors were selected or not selected")
        
        # Try multiple ways to get vendor data
        vendors_data = None
        
        # Method 1: Check session_state.vendors (generated during simulation)
        if hasattr(st.session_state, 'vendors') and st.session_state.vendors:
            vendors_data = st.session_state.vendors
        
        # Method 2: Check simulation_results
        if not vendors_data and 'simulation_results' in st.session_state:
            results = st.session_state.simulation_results
            if isinstance(results, dict):
                # Try different possible locations
                vendors_data = results.get('vendors') or results.get('config', {}).get('vendors')
        
        # Method 3: Try to infer from DataFrame metadata if available
        if not vendors_data and hasattr(df, 'attrs') and 'vendors' in df.attrs:
            vendors_data = df.attrs['vendors']
        
        if vendors_data and isinstance(vendors_data, list) and len(vendors_data) > 0:
            # Calculate average proximity scores from all agents' proximity data
            avg_proximity_per_vendor = {}
            
            if 'vendor_proximity_scores' in df.columns:
                # Extract all proximity scores
                all_proximity_scores = df['vendor_proximity_scores'].dropna()
                
                if len(all_proximity_scores) > 0:
                    # Initialize accumulators
                    proximity_sums = {}
                    proximity_counts = {}
                    
                    for proximity_dict in all_proximity_scores:
                        if isinstance(proximity_dict, dict):
                            for vendor_key, proximity_value in proximity_dict.items():
                                # vendor_key is a string like "1", "2", etc.
                                if vendor_key not in proximity_sums:
                                    proximity_sums[vendor_key] = 0.0
                                    proximity_counts[vendor_key] = 0
                                proximity_sums[vendor_key] += float(proximity_value)
                                proximity_counts[vendor_key] += 1
                    
                    # Calculate averages
                    for vendor_key in proximity_sums:
                        if proximity_counts[vendor_key] > 0:
                            avg_proximity_per_vendor[int(vendor_key)] = proximity_sums[vendor_key] / proximity_counts[vendor_key]
            
            # Create vendor comparison table
            vendor_table_data = []
            
            for idx, vendor in enumerate(vendors_data, 1):
                vendor_id = vendor.get('vendor_id', idx)
                
                # Get selection count for this vendor
                selection_count = 0
                if vendor_id in vendor_counts.index:
                    selection_count = int(vendor_counts[vendor_id])
                
                # Get average proximity for this vendor
                avg_proximity = avg_proximity_per_vendor.get(vendor_id, None)
                proximity_display = f"{avg_proximity:.1f}" if avg_proximity is not None else "N/A"
                
                vendor_table_data.append({
                    'Vendor ID': f"Vendor {vendor_id}",
                    'Price ($)': f"${vendor.get('price', 0):.2f}",
                    'Quality': vendor.get('quality', 'N/A'),
                    'Sustainability': vendor.get('sustainability', 'N/A'),
                    'Avg Proximity': proximity_display,
                    'Times Selected': selection_count,
                    'Selection %': f"{(selection_count / agents_with_selection * 100) if agents_with_selection > 0 else 0:.1f}%",
                    'Status': '✅ Selected' if selection_count > 0 else '❌ Not Selected'
                })
            
            vendor_df = pd.DataFrame(vendor_table_data)
            
            st.markdown("**📋 Vendor Attributes & Selection Results:**")
            st.dataframe(vendor_df, use_container_width=True, hide_index=True)
            
            # Score comparison visualization
            st.markdown("**📊 Vendor Attribute Comparison:**")
            
            col_price, col_quality = st.columns(2)
            col_sust, col_prox = st.columns(2)
            
            with col_price:
                # Price comparison (inverted - lower is better)
                price_fig = px.bar(
                    vendor_df,
                    x='Vendor ID',
                    y=[100 - float(p.replace('$', '').replace(',', '')) for p in vendor_df['Price ($)']],
                    title="Price Score (Higher = Lower Price)",
                    labels={'y': 'Score (Inverted)', 'x': ''}
                )
                price_fig.update_layout(showlegend=False, height=250)
                st.plotly_chart(price_fig, use_container_width=True)
            
            with col_quality:
                # Quality comparison
                quality_vals = [v if isinstance(v, int) else 0 for v in vendor_df['Quality']]
                qual_fig = px.bar(
                    vendor_df,
                    x='Vendor ID',
                    y=quality_vals,
                    title="Quality Score (1-5)",
                    labels={'y': 'Quality', 'x': ''}
                )
                qual_fig.update_layout(showlegend=False, height=250)
                st.plotly_chart(qual_fig, use_container_width=True)
            
            with col_sust:
                # Sustainability comparison
                sust_vals = [v if isinstance(v, int) else 0 for v in vendor_df['Sustainability']]
                sust_fig = px.bar(
                    vendor_df,
                    x='Vendor ID',
                    y=sust_vals,
                    title="Sustainability Score (1-5)",
                    labels={'y': 'Sustainability', 'x': ''}
                )
                sust_fig.update_layout(showlegend=False, height=250)
                st.plotly_chart(sust_fig, use_container_width=True)
            
            with col_prox:
                # Proximity comparison (average across all agents)
                proximity_vals = []
                for row in vendor_table_data:
                    prox_str = row['Avg Proximity']
                    if prox_str != "N/A":
                        proximity_vals.append(float(prox_str))
                    else:
                        proximity_vals.append(0.0)
                
                prox_fig = px.bar(
                    vendor_df,
                    x='Vendor ID',
                    y=proximity_vals,
                    title="Avg Proximity Score (0-100)",
                    labels={'y': 'Proximity', 'x': ''}
                )
                prox_fig.update_layout(showlegend=False, height=250)
                st.plotly_chart(prox_fig, use_container_width=True)
            
            st.caption("""
            💡 **Note**: Proximity scores vary by agent (each agent has different proximity to each vendor). 
            The chart above shows the **average proximity** across all agents.
            """)
        else:
            st.info("ℹ️ Vendor attribute data not available. This section shows detailed vendor data in multi-vendor simulations.")
    
    # Explanation of how vendor selection works
    with st.expander("ℹ️ How Vendor Selection Works (Default Behavior)", expanded=False):
        st.markdown("""
        **Vendor Selection Default Logic:**
        
        For each agent:
        1. **Get Vendor Pool**: Vendors have attributes:
           - **Price**: From Page 1 configuration
           - **Quality**: Random integer in [1, 5] (generated once per vendor)
           - **Sustainability**: Random integer in [1, 5] (generated once per vendor)
           - **Proximity**: Random value in [0, 100] (generated once per customer-vendor dyad)
        
        2. **Get Weights**: From vendor_choice_weights decision (configured on Page 2 Overview)
           - Example: {price: 0.5, quality: 0.5, proximity: 0.0, sustainability: 0.0}
        
        3. **Standardize Attributes** to [0, 1]:
           - Price: Normalized and **inverted** (lower price = higher score)
           - Quality: (value - 1) / 4
           - Sustainability: (value - 1) / 4
           - Proximity: value / 100
        
        4. **Calculate Composite Score** for each vendor:
           ```
           score = w_price × norm_price + w_quality × norm_quality + 
                   w_proximity × norm_proximity + w_sustainability × norm_sustainability
           ```
        
        5. **Select Best Vendor**: Vendor with highest composite score
        
        6. **Apply to All Requests**: All purchase requests from the same agent get the same vendorID
        
        **Result**: Deterministic selection based on weighted preferences
        """)
    
    # Show configured weights (read-only)
    if 'vendor_choice_weights' in df.columns:
        st.markdown("---")
        st.markdown("**⚙️ Configured Vendor Choice Weights (Read-Only):**")
        
        # Get weights from first agent (all should have same weights)
        sample_weights = df['vendor_choice_weights'].iloc[0]
        
        if isinstance(sample_weights, dict):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Show active weights
                active_weights = {k: v for k, v in sample_weights.items() if v > 0}
                
                if active_weights:
                    st.success("✅ **Active Parameters:**")
                    for param, weight in active_weights.items():
                        st.write(f"• {param.title()}: {weight:.2%}")
                else:
                    st.warning("No parameters selected")
            
            with col2:
                # Show pie chart if multiple weights
                if len(active_weights) > 1:
                    fig = px.pie(
                        values=list(active_weights.values()),
                        names=[k.title() for k in active_weights.keys()],
                        title="Weight Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif len(active_weights) == 1:
                    st.info(f"Single factor: {list(active_weights.keys())[0].title()}")
        
        st.caption("💡 To modify weights: Go to **Page 2 → Overview Tab → Vendor Choice Weights**")

