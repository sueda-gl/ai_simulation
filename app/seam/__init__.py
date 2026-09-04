# app/seam/__init__.py
"""
The seam between the Streamlit session state and the engine.

    snapshot    - take_snapshot(st.session_state) -> SessionSnapshot (frozen copy)
    config_repo - DecisionsConfig: config/decisions.yaml + simulation.yaml, read-only
    sentinels   - the sigma constants the seam needs (from the config, not literals)
    build_plan  - build_run_plan(snapshot, config_repo) -> RunPlan  (pure)
    execute     - execute(plan) -> {result_key: DataFrame}          (no Streamlit)

Only ``app/simulation.py`` (the adapter) touches ``st``; the modules above
never import Streamlit so a plan can be built and run from a test or a script.
"""
