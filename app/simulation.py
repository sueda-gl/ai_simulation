# app/simulation.py
"""
Simulation execution logic for the Enhanced AI Agent Simulation.
Handles both single runs and Monte Carlo studies.

Single runs go through the seam:

    snapshot  = take_snapshot(st.session_state)          # read ONCE (R15)
    plan      = build_run_plan(snapshot, config_repo)     # pure (app/seam/build_plan.py)
    messages  -> st.info / st.caption / st.success, in plan order
    results   = execute(plan)                             # engine only (app/seam/execute.py)
    saved-config hashes re-checked (R14), then results / _run_metadata /
    vendors / page are stored and the app reruns.

This module is the only place that touches ``st``; the seam and the engine
never import Streamlit.  ``run_monte_carlo_study`` runs the scripts as a
subprocess, handing them the single-run plan's sub-run as a plan file
(app/seam/mc.py), so a Monte-Carlo repetition uses exactly the settings of a
single run with the same seed - including a selected saved configuration's
pinned agent count and population; only the seed varies per repetition.
"""
import streamlit as st
import pandas as pd
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contract.plan import RunMetadata, RunPlan, UiMessage, deep_merge_patch
from app.models import ALL_DECISIONS
from app.seam.build_plan import (
    build_rejected_transaction_patch,
    build_run_plan,
    collect_decision_settings as _collect_decision_settings,
    get_pop_type,
    get_population_mode_from_result_key,
)
from app.seam.config_repo import get_config_repo
from app.seam.execute import execute, verify_saved_expectations
from app.seam.mc import (
    pinned_config_caption,
    population_mode_name,
    select_mc_sub_run,
    write_mc_plan_file,
)
from app.seam.snapshot import take_snapshot


# =============================================================================
# COMPAT - the one private mapping the tests import directly
# =============================================================================

def _apply_rejected_transaction_config(orchestrator, pop_mode: str, inc_mode: str = None):
    """
    Apply Decision 4 (rejected_transaction_defaults) configuration to an
    orchestrator/engine IN PLACE, from the current session state.

    Compat wrapper (tests/test_rejected_transaction_defaults.py,
    tests/test_rtd_batch4_ui.py) over the seam's pure patch builder:
    ``app.seam.build_plan.build_rejected_transaction_patch``.
    """
    if not hasattr(orchestrator, 'config') or 'rejected_transaction_defaults' not in orchestrator.config:
        return

    snapshot = take_snapshot(st.session_state)
    patch = build_rejected_transaction_patch(snapshot, get_config_repo(), pop_mode, inc_mode)
    deep_merge_patch(orchestrator.config['rejected_transaction_defaults'], patch)


def run_monte_carlo_study() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """Run Monte-Carlo study and return results."""
    try:
        # Check if we're in comparison mode
        is_pop_comparison = st.session_state.population_mode == "Compare all"
        is_income_comparison = st.session_state.get('income_spec_mode', 'categorical only') == "Compare both"
        
        # Map UI modes to script arguments
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
        # sub-runs with seed base_seed + i (app/seam/mc.py; Q-29 fixed 2026-10-07).
        from app.state.saved_configs import get_simulation_seed_from_configs
        seed_resolution = get_simulation_seed_from_configs()
        pinned_config = seed_resolution[2] == 'configs'
        plan = build_plan_from_session(seed_resolution)
        mc_sub_run = select_mc_sub_run(plan, pop_mode_arg, income_mode_arg)
        if mc_sub_run.income_mode != income_mode_arg:
            # e.g. a decision-only run whose tab chose its own income mode
            income_mode_arg = mc_sub_run.income_mode
            income_label = income_mode_arg.title()
        if mc_sub_run.population != pop_mode_arg:
            # a selected ("Use This Config") saved configuration pins the population
            # of a complete run, exactly as in a single run (R14)
            pop_mode_arg = mc_sub_run.population
            pop_label = population_mode_name(pop_mode_arg).replace(' (synthetic)', '').replace(' ', '_')
        # ... and the agent count of every run (Page 1's only without a pinned config)
        n_agents = mc_sub_run.n_agents
        plan_file = write_mc_plan_file(
            Path(__file__).resolve().parents[1] / 'outputs'
            / f"mc_plan_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.pkl",
            mc_sub_run, plan.metadata.custom_decisions)

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
        
        st.info(f"🔄 Starting Monte-Carlo study with {st.session_state.n_runs} runs of {n_agents} agents each...")
        if pinned_config:
            st.caption(pinned_config_caption(mc_sub_run))
        st.caption(f"📊 Mode: {pop_label} + {income_label}")
        
        # Show estimated time
        estimated_time_per_run = 2
        total_estimated_time = st.session_state.n_runs * estimated_time_per_run
        st.caption(f"⏱️ Estimated time: ~{total_estimated_time} seconds ({total_estimated_time/60:.1f} minutes)")
        
        # Build command
        cmd = [
            sys.executable, 'scripts/run_mc_study.py',
            '--agents', str(n_agents),
            '--runs', str(st.session_state.n_runs),
            '--base-seed', str(st.session_state.base_seed),
            '--anchor-observed', str(st.session_state.anchor_observed_weight),
            '--population-mode', pop_mode_arg,
            '--income-mode', income_mode_arg,
            '--plan-file', str(plan_file),
        ]
        
        # Handle multiple decisions for Monte Carlo
        if len(st.session_state.decision_params.selected_decisions) < len(ALL_DECISIONS):
            # Pass each selected decision as a separate argument
            for decision in st.session_state.decision_params.selected_decisions:
                cmd.extend(['--decision', decision])
        
        # Change to project directory to ensure scripts can be found
        cwd = Path(__file__).resolve().parents[1]
        
        # Debug: print command and environment
        with st.expander("🔧 Debug Information", expanded=True):
            st.code(' '.join(cmd))
            st.caption(f"Working directory: {cwd}")
            st.caption(f"Python executable: {sys.executable}")
            st.caption(f"Population mode: {st.session_state.population_mode} → {pop_mode_arg}")
            st.caption(f"Income mode: {st.session_state.get('income_spec_mode', 'categorical only')} → {income_mode_arg}")
            st.caption(f"Selected decisions: {st.session_state.decision_params.selected_decisions}")
            st.caption(f"Number of runs: {st.session_state.n_runs}")
            st.caption(f"Agents per run: {n_agents}")
        
        # Create progress tracking elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        output_container = st.container()
        
        # Run with real-time output capture using Popen instead of run
        status_text.text("🚀 Launching Monte-Carlo simulations...")
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd),
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Collect output
        stdout_lines = []
        stderr_lines = []
        last_update_time = time.time()
        
        # Monitor the process
        while True:
            # Read any available output (blocking until we get a line or EOF)
            line = process.stdout.readline()
            if line:
                stdout_lines.append(line.strip())
                
                # Parse progress from output
                if "Run" in line and "/" in line:
                    try:
                        # Extract run number (e.g., "Run  10/100:")
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "/" in part:
                                current_run = int(parts[i-1])
                                total_runs = int(part.split("/")[1].split(":")[0])
                                progress = current_run / total_runs
                                progress_bar.progress(progress)
                                status_text.text(f"🔄 Progress: Run {current_run}/{total_runs}")
                                break
                    except:
                        pass
                
                # Show last few lines of output
                if time.time() - last_update_time > 0.5:  # Update every 0.5 seconds
                    with output_container.container():
                        st.text("📊 Recent output:")
                        st.code('\n'.join(stdout_lines[-5:]))
                    last_update_time = time.time()
            
            # Check if process is done AND we've read all output
            # Empty line from readline() means EOF when process is done
            poll = process.poll()
            if poll is not None and not line:
                # Process finished and no more output
                break
            
            # If we got an empty line but process still running, continue
            if not line and poll is None:
                time.sleep(0.1)
                continue
        
        # Get any remaining stderr
        _, remaining_stderr = process.communicate()
        if remaining_stderr:
            stderr_lines.extend(remaining_stderr.strip().split('\n'))
        
        # Debug: Show total lines captured
        st.info(f"🔍 Total lines captured from stdout: {len(stdout_lines)}")
        
        # Join all output
        stdout = '\n'.join(stdout_lines)
        stderr = '\n'.join(stderr_lines)
        
        # Show final output
        if stdout:
            with st.expander("📋 Monte Carlo Output", expanded=True):
                st.text(stdout)
        else:
            st.warning("⚠️ No stdout output captured!")
        
        if stderr:
            with st.expander("⚠️ Monte Carlo Errors", expanded=True):
                st.text(stderr)
        
        # Debug: Show return code
        st.info(f"🔍 Process return code: {process.returncode}")
        
        if process.returncode == 0:
            # Parse output to find result files
            output_lines = stdout.strip().split('\n') if stdout else []
            summary_file = None
            detailed_file = None
            
            # Debug: Show what we're parsing
            st.info(f"🔍 Parsing {len(output_lines)} lines of output...")
            
            for i, line in enumerate(output_lines):
                if 'Summary saved to:' in line:
                    summary_file = line.split('Summary saved to:')[1].strip()
                    st.success(f"✅ Found summary file in line {i+1}: {summary_file}")
                elif 'Detailed results saved to:' in line:
                    detailed_file = line.split('Detailed results saved to:')[1].strip()
                    st.success(f"✅ Found detailed file in line {i+1}: {detailed_file}")
            
            progress_bar.progress(1.0)
            status_text.success("✅ Monte-Carlo study completed!")
            
            # Debug: Show what files were detected
            st.info(f"📁 Detected files - Summary: {summary_file}, Detailed: {detailed_file}")
            
            # Load results - handle relative paths
            if summary_file and not Path(summary_file).is_absolute():
                summary_file = str(cwd / summary_file)
            if detailed_file and not Path(detailed_file).is_absolute():
                detailed_file = str(cwd / detailed_file)
            
            # Debug: Show resolved paths
            st.info(f"📍 Resolved paths - Summary: {summary_file}, Detailed: {detailed_file}")
            
            # Check if files exist before loading
            summary_exists = Path(summary_file).exists() if summary_file else False
            detailed_exists = Path(detailed_file).exists() if detailed_file else False
            
            st.info(f"✅ File existence - Summary: {summary_exists}, Detailed: {detailed_exists}")
            
            # Load results
            mc_summary = pd.read_csv(summary_file) if summary_file and summary_exists else None
            mc_detailed = pd.read_csv(detailed_file) if detailed_file and detailed_exists else None
            
            # If files not found, show debug info
            if mc_summary is None:
                if not summary_file:
                    st.error("❌ Summary file path not detected in output!")
                elif not summary_exists:
                    st.error(f"❌ Summary file not found at: {summary_file}")
                    # List files in outputs directory
                    outputs_dir = cwd / "outputs"
                    if outputs_dir.exists():
                        recent_files = sorted(outputs_dir.glob("mc_summary*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
                        if recent_files:
                            st.warning(f"📂 Recent MC summary files found:")
                            for f in recent_files:
                                st.caption(f"  - {f.name}")
                    
            if mc_detailed is None:
                if not detailed_file:
                    st.warning("⚠️ Detailed file path not detected in output (this is optional)")
                elif not detailed_exists:
                    st.warning(f"⚠️ Detailed file not found at: {detailed_file}")
            
            # Show loaded data shape
            if mc_summary is not None:
                st.success(f"✅ Loaded summary: {mc_summary.shape}")
            if mc_detailed is not None:
                st.success(f"✅ Loaded detailed: {mc_detailed.shape}")
            
            return mc_summary, mc_detailed, stdout
        else:
            st.error(f"❌ Monte-Carlo study failed with return code: {process.returncode}")
            st.error(f"Error output: {stderr}")
            return None, None, None
                
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
    """st.session_state._run_metadata - what was ACTUALLY run (R28)."""
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
    }


def build_plan_from_session(seed_resolution=None) -> RunPlan:
    """Snapshot the session ONCE and build the plan for this click (R15).

    seed_resolution: (seed, n_agents, source) when the caller already resolved it
    (``get_simulation_seed_from_configs``); resolved here otherwise."""
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES
    from app.state.saved_configs import get_simulation_seed_from_configs

    snapshot = take_snapshot(st.session_state)
    if seed_resolution is None:
        seed_resolution = get_simulation_seed_from_configs()
    return build_run_plan(
        snapshot,
        get_config_repo(),
        default_decision_values=DEFAULT_DECISION_VALUES,
        seed_resolution=seed_resolution,
    )


def run_full_simulation():
    """
    Run a single simulation: snapshot -> plan -> messages -> execute ->
    saved-config check -> store results and navigate to the results page.

    The run never writes a session key before its results are stored (R15);
    the population / income modes it used travel in _run_metadata (R28).
    """
    try:
        with st.spinner("🔄 Running simulation..."):
            plan = build_plan_from_session()

            for message in plan.messages:
                _emit(message)

            results = execute(plan, config_repo=get_config_repo())

            # R14: a pinned decision must reproduce the run the user selected
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

            # Restore ONLY selected_decisions BEFORE st.rerun() if there's a pending
            # restoration (set by run_individual_decision / run_combined_simulation).
            # custom_decisions / default_decisions stay: they describe THIS run.
            if hasattr(st.session_state, '_pending_decisions_restore'):
                restore_data = st.session_state._pending_decisions_restore
                st.session_state.decision_params.selected_decisions = restore_data['selected_decisions']
                del st.session_state._pending_decisions_restore

            st.session_state.page = 'results'
            st.rerun()

    except Exception as e:
        st.error(f"❌ Simulation failed: {str(e)}")
        import traceback
        st.text(traceback.format_exc())


def collect_decision_settings():
    """Collect current default decision settings from session state (probabilities, selections, etc.)

    Delegates to the seam (app.seam.build_plan.collect_decision_settings) on a
    snapshot of the session state; the registry of defaults still lives in
    app.pages.decision_execution.DEFAULT_DECISION_VALUES.
    """
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES
    return _collect_decision_settings(take_snapshot(st.session_state), DEFAULT_DECISION_VALUES)
