# app/simulation.py
"""
Simulation execution logic for the Enhanced AI Agent Simulation.
Handles both single runs and Monte Carlo studies.

Single runs go through the seam:

    snapshot  = take_snapshot(st.session_state)          # read ONCE per click
    plan      = build_run_plan(snapshot, config_repo)     # pure (app/seam/build_plan.py)
    messages  -> st.info / st.caption / st.success, in plan order
    results   = execute(plan)                             # engine only (app/seam/execute.py)
    saved-config hashes re-checked, then results / _run_metadata /
    vendors / page are stored and the app reruns.

This module is the only place that touches ``st``; the seam and the engine
never import Streamlit.  ``run_monte_carlo_study`` repeats one sub-run of the
single-run plan in process (app/seam/mc.py), so a Monte-Carlo repetition uses
exactly the settings of a single run with the same seed - including a selected
saved configuration's pinned agent count and population; only the seed varies
per repetition.  The per-run statistics and the summary come from
app/reports/mc.py, which scripts/run_mc_study.py uses as well.
"""
import copy
import dataclasses
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contract.plan import RunMetadata, RunPlan, RunRequest, UiMessage, deep_merge_patch
from app.seam.build_plan import (
    build_rejected_transaction_patch,
    build_run_plan,
)
from app.seam.config_repo import get_config_repo
from app.seam.execute import execute, verify_saved_expectations
from app.seam.mc import (
    pinned_config_caption,
    population_mode_name,
    run_mc_repetition,
    select_mc_sub_run,
)
from app.seam.snapshot import take_snapshot
from app.reports import mc as mc_report
from src.contract.columns import rename_consensus_columns


# =============================================================================
# COMPAT - the one private mapping the tests import directly
# =============================================================================

def _apply_rejected_transaction_config(orchestrator, pop_mode: str, inc_mode: str = None):
    """
    Apply Decision 4 (rejected_transaction_defaults) configuration to an
    orchestrator/engine IN PLACE, from the current session state.

    Compat wrapper (used by tests/test_rejected_transaction_defaults.py) over the
    seam's pure patch builder:
    ``app.seam.build_plan.build_rejected_transaction_patch``.
    """
    if not hasattr(orchestrator, 'config') or 'rejected_transaction_defaults' not in orchestrator.config:
        return

    snapshot = take_snapshot(st.session_state)
    patch = build_rejected_transaction_patch(snapshot, get_config_repo(), pop_mode, inc_mode)
    deep_merge_patch(orchestrator.config['rejected_transaction_defaults'], patch)


def run_monte_carlo_study(request: Optional[RunRequest] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """Run a Monte-Carlo study in process and return (summary, detailed, log).

    ``request`` is the click's RunRequest (the run buttons hand it over); the
    decisions it names drive the plan.  The study repeats one sub-run of the
    single-run plan ``n_runs`` times with seeds ``base_seed + i``
    (``app.seam.mc.run_mc_repetition``) and computes the per-run statistics and
    the summary with ``app.reports.mc`` - the same numbers ``scripts/run_mc_study.py``
    writes.  Nothing is written to disk.  ``summary`` is None when the study did
    not run ``donation_default`` (there is no summary row then)."""
    try:
        # Check if we're in comparison mode
        is_pop_comparison = st.session_state.population_mode == "Compare all"
        is_income_comparison = st.session_state.get('income_spec_mode', 'categorical only') == "Compare both"
        
        # Map UI modes to the internal population / income names
        population_mode_map = {
            'Copula (synthetic)': 'copula',
            'Research Specification': 'documentation',
            'Research Baseline': 'baseline'
        }
        
        income_mode_map = {
            'categorical only': 'categorical',
            'continuous only': 'continuous'
        }
        
        # Determine which mode combinations to run
        if is_pop_comparison:
            # Run all 3 population modes
            pop_modes = [
                ('copula', 'Copula'),
                ('documentation', 'Research Spec'),
                ('baseline', 'Baseline')
            ]
        else:
            pop_key = st.session_state.population_mode
            pop_label = pop_key.replace(' (synthetic)', '').replace(' ', '_')
            pop_modes = [(population_mode_map.get(pop_key, 'copula'), pop_label)]
        
        if is_income_comparison:
            # Run both income modes (only if not doing population comparison)
            income_modes = [
                ('categorical', 'Categorical'),
                ('continuous', 'Continuous')
            ]
        else:
            income_key = st.session_state.get('income_spec_mode', 'categorical only')
            income_label = income_key.replace(' only', '').title()
            income_modes = [(income_mode_map.get(income_key, 'categorical'), income_label)]
        
        # For comparison mode, use the first mode and show a message
        pop_mode_arg, pop_label = pop_modes[0]
        income_mode_arg, income_label = income_modes[0]

        # The SAME plan a single run builds (every Page-1 / Page-2 setting, each
        # decision's own income mode, saved configs) - Monte Carlo repeats one of its
        # sub-runs with seed base_seed + i (app/seam/mc.py).
        from app.state.saved_configs import get_simulation_seed_from_configs
        seed_resolution = get_simulation_seed_from_configs()
        pinned_config = seed_resolution[2] == 'configs'
        plan = build_plan_from_session(seed_resolution, request)
        mc_sub_run = select_mc_sub_run(plan, pop_mode_arg, income_mode_arg)
        if mc_sub_run.income_mode != income_mode_arg:
            # e.g. a decision-only run whose tab chose its own income mode
            income_mode_arg = mc_sub_run.income_mode
            income_label = income_mode_arg.title()
        if mc_sub_run.population != pop_mode_arg:
            # a selected ("Use This Config") saved configuration pins the population
            # of a complete run, exactly as in a single run (to reproduce the pinned run)
            pop_mode_arg = mc_sub_run.population
            pop_label = population_mode_name(pop_mode_arg).replace(' (synthetic)', '').replace(' ', '_')
        # ... and the agent count of every run (Page 1's only without a pinned config)
        n_agents = mc_sub_run.n_agents

        if len({(s.population, s.income_mode) for s in plan.sub_runs}) > 1:
            st.warning(f"⚠️ **Comparison Mode Limitation**: Monte Carlo will run with **{pop_label} + {income_label}** mode only")
            st.info("""
            💡 **To compare multiple modes with Monte Carlo:**
            1. Run Monte Carlo with current mode
            2. Export/save results  
            3. Change to different mode (e.g., Research Specification)
            4. Run Monte Carlo again
            5. Compare the exported results
            
            This approach gives you better control and avoids extremely long run times.
            """)
        
        n_runs = int(st.session_state.n_runs)
        base_seed = int(st.session_state.base_seed)
        st.info(f"🔄 Starting Monte-Carlo study with {n_runs} runs of {n_agents} agents each...")
        if pinned_config:
            st.caption(pinned_config_caption(mc_sub_run))
        st.caption(f"📊 Mode: {pop_label} + {income_label}")
        
        # Show estimated time
        estimated_time_per_run = 2
        total_estimated_time = n_runs * estimated_time_per_run
        st.caption(f"⏱️ Estimated time: ~{total_estimated_time} seconds ({total_estimated_time/60:.1f} minutes)")
        
        # Create progress tracking elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # The repetitions: seed base_seed + i, each one exactly a single run with
        # that seed (app/seam/mc.py); written files say rtd_integrated_*, so the
        # statistics are taken on the renamed frame, as the CLI study takes them
        run_stats = []
        log_lines = [
            f"Starting Monte-Carlo study with {n_runs} repetitions",
            f"Population size: {n_agents} agents per run",
            f"Base seed: {base_seed}",
            f"Mode: {pop_label} + {income_label}",
            "-" * 50,
        ]
        for i in range(n_runs):
            seed = base_seed + i
            status_text.text(f"🔄 Progress: Run {i + 1}/{n_runs}")
            df = run_mc_repetition(mc_sub_run, seed, custom_decisions=plan.metadata.custom_decisions)
            df = rename_consensus_columns(df)
            run_stats.append(mc_report.run_statistics(df, i, seed))
            log_lines.append(f"Run {i + 1:3d}/{n_runs}: seed={seed} ... {mc_report.run_outcome_text(df)}")
            progress_bar.progress((i + 1) / n_runs)

        mc_detailed = mc_report.detailed_frame(run_stats)
        summary_stats = mc_report.summary_statistics(mc_detailed)
        mc_summary = mc_report.summary_frame(summary_stats, n_agents)
        log_lines.append("=" * 50)
        log_lines.append(f"Monte-Carlo study completed: {n_runs} successful runs")
        if 'donation_default' in summary_stats:
            stats = summary_stats['donation_default']
            log_lines += [
                f"Donation Default Summary (across {stats['runs']} runs):",
                f"  Mean: {stats['mean']:.6f}",
                f"  Std:  {stats['std']:.6f}",
                f"  95% CI: [{stats['p2.5']:.6f}, {stats['p97.5']:.6f}]",
            ]

        progress_bar.progress(1.0)
        status_text.success("✅ Monte-Carlo study completed!")

        # No donation_default in the study: no summary row (the caller reports it)
        if len(mc_summary) == 0:
            mc_summary = None
        return mc_summary, mc_detailed, "\n".join(log_lines)
                
    except Exception as e:
        st.error(f"❌ Monte-Carlo study failed: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return None, None, None


def _decision_title(decision_name: str) -> str:
    return decision_name.replace('_', ' ').title()


def _emit(message: UiMessage) -> None:
    """Render one plan message with the matching st.* call."""
    {
        'info': st.info,
        'caption': st.caption,
        'success': st.success,
        'warning': st.warning,
        'error': st.error,
    }[message.kind](message.text)


def _run_metadata_dict(metadata: RunMetadata) -> dict:
    """st.session_state._run_metadata - what was ACTUALLY run (the results page
    describes the run that happened, not the next one being configured)."""
    return {
        'is_comparison': metadata.is_comparison,
        'result_keys': list(metadata.result_keys),
        'num_results': len(metadata.result_keys),
        'effective_income_mode': metadata.effective_income_mode,
        'effective_population_mode': metadata.effective_population_mode,
        'custom_decisions': list(metadata.custom_decisions),
        'default_decisions': list(metadata.default_decisions),
        'seed': metadata.seed,
        'n_agents': metadata.n_agents,
        # per result key, the income mode each income-dependent decision ran with
        'decision_income_modes': {key: dict(modes) for key, modes in metadata.decision_income_modes.items()},
        'rtd_compare_both_fallback': metadata.rtd_compare_both_fallback,
        'started_at': metadata.started_at,
        # the Page 1 parameters, Decision 4 element flag and donation anchor weight
        # the run used - the results page reads these, never the live widgets
        'simulation_params': dict(metadata.simulation_params),
        'rtd_run_element': metadata.rtd_run_element,
        'anchor_observed_weight': metadata.anchor_observed_weight,
        # the default-decision settings the run used (the Overview tab's values)
        'default_decision_settings': copy.deepcopy(metadata.default_decision_settings),
    }


def run_started_at() -> datetime:
    """The moment a run starts: the base time every export's timestamp column is
    relative to (``RunMetadata.started_at``).  The one place the clock is read for
    that purpose, so a test can freeze it here."""
    return datetime.now()


def build_plan_from_session(seed_resolution=None, request: Optional[RunRequest] = None) -> RunPlan:
    """Snapshot the session ONCE and build the plan for this click (one read per
    click, so nothing the run writes can feed back into it).

    seed_resolution: (seed, n_agents, source) when the caller already resolved it
    (``get_simulation_seed_from_configs``); resolved here otherwise.
    request: the click's RunRequest (which decisions run, custom and default
    lists); without one the builder reads those from the snapshot."""
    from src.contract.defaults import DEFAULT_DECISION_VALUES
    from app.state.saved_configs import get_simulation_seed_from_configs

    snapshot = take_snapshot(st.session_state)
    if seed_resolution is None:
        seed_resolution = get_simulation_seed_from_configs()
    plan = build_run_plan(
        snapshot,
        get_config_repo(),
        default_decision_values=DEFAULT_DECISION_VALUES,
        seed_resolution=seed_resolution,
        request=request,
    )
    # The run starts now: every export's timestamps are relative to this moment.
    metadata = dataclasses.replace(plan.metadata, started_at=run_started_at().isoformat())
    return dataclasses.replace(plan, metadata=metadata)


def run_full_simulation(request: Optional[RunRequest] = None):
    """
    Run a single simulation: snapshot -> plan -> messages -> execute ->
    saved-config check -> store results and navigate to the results page.

    ``request`` is the click's RunRequest (built by the run buttons).  The run
    never writes a session key before its results are stored; the population /
    income modes it used travel in _run_metadata, so the results page describes
    the run that happened, not the next one being configured.
    """
    try:
        with st.spinner("🔄 Running simulation..."):
            plan = build_plan_from_session(request=request)

            for message in plan.messages:
                _emit(message)

            results = execute(plan, config_repo=get_config_repo())

            # A pinned decision must reproduce the run the user selected
            failed = verify_saved_expectations(plan.saved_expectations, results)
            if failed:
                for expectation in failed:
                    st.error(
                        f"Saved configuration for {_decision_title(expectation.decision)} "
                        "could not be reproduced: results differ from the saved run."
                    )
                return

            # Store results
            st.session_state.simulation_results = results

            # Store run metadata - this captures WHAT WAS ACTUALLY RUN
            st.session_state._run_metadata = _run_metadata_dict(plan.metadata)

            # Extract and store vendor data from first DataFrame
            for df in results.values():
                if hasattr(df, 'attrs') and 'vendors' in df.attrs:
                    st.session_state.vendors = df.attrs['vendors']
                    break

            st.session_state.page = 'results'
            st.rerun()

    except Exception as e:
        st.error(f"❌ Simulation failed: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
