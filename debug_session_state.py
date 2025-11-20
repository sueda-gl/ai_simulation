# Debug script to check what's in session state keys for default decisions
import streamlit as st

# This would be called from your app to debug
def debug_default_decision_keys():
    st.write("### Default Decision Keys Debug")
    
    default_decision_keys = [
        'disclose_income_default_probability_y',
        'disclose_documents_default_probability_y',
        'purchase_vs_bid_default_probability_y',
        'vendor_choice_weights_default_params',
        'rejected_transaction_defaults_priority_template',
        'rejected_transaction_option_default_selection'
    ]
    
    for key in default_decision_keys:
        if key in st.session_state:
            st.success(f"✅ {key} = {st.session_state[key]}")
        else:
            st.error(f"❌ {key} NOT FOUND")
    
    st.write(f"Total keys in session_state: {len(st.session_state.keys())}")
