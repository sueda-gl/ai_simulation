# app/state/__init__.py
"""Session-state stores for the Streamlit app.

`saved_configs` owns the unified saved-decision-configuration store
(`st.session_state.selected_decision_configs`) that pins seed, agent count,
population mode and per-decision parameters for a decision the user explicitly
selected with "Use This Config".

`registry` lists every `st.session_state` key the app uses (plain data, no
Streamlit). `widgets` draws keyed widgets whose value lives in session state
(`stateful`) and keeps the per-run widget render log.
"""
