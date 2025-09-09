# app/simulation.py
"""
Simulation execution logic for the Enhanced AI Agent Simulation.
Handles both single runs and Monte Carlo studies.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator import Orchestrator
from src.orchestrator_doc_mode import OrchestratorDocMode
from src.orchestrator_depvar import OrchestratorDepVar
from app.models import ALL_DECISIONS


def run_monte_carlo_study() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """Run Monte-Carlo study and return results."""
    try:
        # Create a container for real-time updates
        status_container = st.container()
        
        st.info(f"🔄 Starting Monte-Carlo study with {st.session_state.n_runs} runs of {st.session_state.n_agents} agents each...")
        
        # Show estimated time
        estimated_time_per_run = 2  # seconds, rough estimate
        total_estimated_time = st.session_state.n_runs * estimated_time_per_run
        st.caption(f"⏱️ Estimated time: ~{total_estimated_time} seconds ({total_estimated_time/60:.1f} minutes)")
        
        # Build command
        cmd = [
            sys.executable, 'scripts/run_mc_study.py',
            '--agents', str(st.session_state.n_agents),
            '--runs', str(st.session_state.n_runs),
            '--base-seed', str(st.session_state.base_seed),
            '--anchor-observed', str(st.session_state.anchor_observed_weight)
        ]
        
        # Handle multiple decisions for Monte Carlo
        if len(st.session_state.decision_params.selected_decisions) < len(ALL_DECISIONS):
            # Pass each selected decision as a separate argument
            for decision in st.session_state.decision_params.selected_decisions:
                cmd.extend(['--decision', decision])
        
        # Debug: print command
        with st.expander("🔧 Debug Information", expanded=False):
            st.code(' '.join(cmd))
        
        # Create progress tracking elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        output_container = st.container()
        
        # Change to project directory to ensure scripts can be found
        cwd = Path(__file__).resolve().parents[1]
        
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
            # Check if process is still running
            poll = process.poll()
            
            # Read any available output
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
            
            # If process finished, break
            if poll is not None:
                break
            
            # Small delay to prevent CPU spinning
            time.sleep(0.1)
        
        # Get any remaining output
        remaining_stdout, remaining_stderr = process.communicate()
        if remaining_stdout:
            stdout_lines.extend(remaining_stdout.strip().split('\n'))
        if remaining_stderr:
            stderr_lines.extend(remaining_stderr.strip().split('\n'))
        
        # Join all output
        stdout = '\n'.join(stdout_lines)
        stderr = '\n'.join(stderr_lines)
        
        # Show final output
        if stdout:
            with st.expander("📋 Monte Carlo Output", expanded=False):
                st.text(stdout)
        
        if stderr:
            with st.expander("⚠️ Monte Carlo Errors", expanded=True):
                st.text(stderr)
        
        if process.returncode == 0:
            # Parse output to find result files
            output_lines = stdout.strip().split('\n')
            summary_file = None
            detailed_file = None
            
            for line in output_lines:
                if 'Summary saved to:' in line:
                    summary_file = line.split('Summary saved to:')[1].strip()
                elif 'Detailed results saved to:' in line:
                    detailed_file = line.split('Detailed results saved to:')[1].strip()
            
            progress_bar.progress(1.0)
            status_text.success("✅ Monte-Carlo study completed!")
            
            # Load results - handle relative paths
            if summary_file and not Path(summary_file).is_absolute():
                summary_file = str(cwd / summary_file)
            if detailed_file and not Path(detailed_file).is_absolute():
                detailed_file = str(cwd / detailed_file)
            
            # Load results
            mc_summary = pd.read_csv(summary_file) if summary_file and Path(summary_file).exists() else None
            mc_detailed = pd.read_csv(detailed_file) if detailed_file and Path(detailed_file).exists() else None
            
            # If files not found, show debug info
            if mc_summary is None and summary_file:
                st.warning(f"Summary file not found at: {summary_file}")
            if mc_detailed is None and detailed_file:
                st.warning(f"Detailed file not found at: {detailed_file}")
            
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


def run_simulation_from_sidebar():
    """Run simulation using original app.py logic"""
    try:
        with st.spinner("🔄 Generating synthetic agents and running simulation..."):
            # Helper to run simulation with chosen orchestrator and income mode
            def _run(pop_mode: str, inc_mode: str):
                # Initialize appropriate orchestrator
                if pop_mode == "documentation":
                    orchestrator = OrchestratorDocMode()
                elif pop_mode == "depvar":
                    orchestrator = OrchestratorDepVar()
                    # Set raw output mode for dependent variable resampling
                    orchestrator.set_raw_output(st.session_state.raw_draw_mode)
                else:  # copula
                    orchestrator = Orchestrator()
                
                # Override income specification in config based on choice (not for depvar mode)
                if hasattr(orchestrator, 'config') and 'donation_default' in orchestrator.config:
                    if pop_mode != "depvar":  # depvar mode doesn't use these settings
                        orchestrator.config['donation_default']['regression']['income_mode'] = inc_mode
                        # Set stochastic flag for copula mode if checkbox is enabled
                        if pop_mode == "copula":
                            orchestrator.config['donation_default']['stochastic']['in_copula'] = st.session_state.sigma_in_copula
                        # Apply selected sigma value
                        orchestrator.config['donation_default']['stochastic']['sigma_value'] = st.session_state.sigma_value_ui
                        # Apply chosen anchor weights
                        orchestrator.config['donation_default']['anchor_weights']['observed'] = st.session_state.anchor_observed_weight
                        orchestrator.config['donation_default']['anchor_weights']['predicted'] = 1 - st.session_state.anchor_observed_weight
                        # Set raw output flag if applicable
                        if pop_mode == "documentation" or (pop_mode == "copula" and st.session_state.sigma_in_copula):
                            orchestrator.config['donation_default']['stochastic']['raw_output'] = st.session_state.raw_draw_mode


                
                # Handle multiple decisions
                decision_param = None if len(st.session_state.decision_params.selected_decisions) == len(ALL_DECISIONS) else st.session_state.decision_params.selected_decisions
                return orchestrator.run_simulation(
                    n_agents=st.session_state.n_agents,
                    seed=st.session_state.seed if st.session_state.sim_params.simulation_mode == "Single Run" else st.session_state.base_seed,
                    single_decision=decision_param
                )
            
            # Run based on population and income specification modes
            results = {}
            
            if st.session_state.population_mode == "Compare both":
                # Compare population modes
                for pop_name, pop_type in [("copula", "copula"), ("doc_mode", "documentation")]:
                    if st.session_state.income_spec_mode == "Compare both":
                        results[f"{pop_name}_categorical"] = _run(pop_type, "categorical")
                        results[f"{pop_name}_continuous"] = _run(pop_type, "continuous")
                    elif st.session_state.income_spec_mode == "continuous only":
                        results[f"{pop_name}_continuous"] = _run(pop_type, "continuous")
                    else:  # categorical only
                        results[f"{pop_name}_categorical"] = _run(pop_type, "categorical")
            elif st.session_state.population_mode == "Dependent variable resampling":
                # Dependent variable mode - only one result regardless of income spec
                results["depvar"] = _run("depvar", "categorical")  # income mode is ignored
            else:
                # Single population mode
                pop_type = "documentation" if "Research" in st.session_state.population_mode else "copula"
                if st.session_state.income_spec_mode == "Compare both":
                    results["categorical"] = _run(pop_type, "categorical")
                    results["continuous"] = _run(pop_type, "continuous")
                elif st.session_state.income_spec_mode == "continuous only":
                    results["continuous"] = _run(pop_type, "continuous")
                else:  # categorical only
                    results["categorical"] = _run(pop_type, "categorical")
            
            # Save results if requested
            if st.session_state.save_results:
                output_dir = Path("outputs")
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Create decision suffix for filename
                if len(st.session_state.decision_params.selected_decisions) == len(ALL_DECISIONS):
                    decision_suffix = "_all"
                elif len(st.session_state.decision_params.selected_decisions) == 1:
                    decision_suffix = f"_{st.session_state.decision_params.selected_decisions[0]}"
                else:
                    decision_suffix = f"_{len(st.session_state.decision_params.selected_decisions)}decisions"
                
                for mode, df in results.items():
                    filename = f"enhanced_simulation_{mode}_seed{st.session_state.seed if st.session_state.sim_params.simulation_mode == 'Single Run' else st.session_state.base_seed}_agents{st.session_state.n_agents}{decision_suffix}_{timestamp}.parquet"
                    filepath = output_dir / filename
                    df.to_parquet(filepath, index=False)
                
                st.sidebar.caption(f"✅ Results saved with timestamp {timestamp}")
            
            st.session_state.simulation_results = results
            st.session_state.page = 'results'
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Simulation failed: {str(e)}")
        import traceback
        st.text(traceback.format_exc())


def run_simulation():
    """Run simulation with current parameters"""
    st.session_state.page = 'results'
    # Simulation will be triggered on results page
