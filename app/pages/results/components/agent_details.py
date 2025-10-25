import streamlit as st
import pandas as pd


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
                    f"{decision_data['donation_default']:.2%}"
            # Convert all values to strings to avoid PyArrow serialization issues
            decision_df['Value'] = decision_df['Value'].astype(str)
            st.dataframe(decision_df, hide_index=True)
    else:
        st.caption("Individual agent details not available in dependent variable resampling mode (no trait information)")
