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
        # Check if we're in comparison mode
        is_pop_comparison = st.session_state.population_mode == "Compare all"
        is_income_comparison = st.session_state.get('income_spec_mode', 'categorical only') == "Compare both"
        
        # Map UI modes to script arguments
        population_mode_map = {
            'Copula (synthetic)': 'copula',
            'Research Specification': 'documentation',
            'Research Baseline': 'baseline',
            'Dependent variable resampling': 'depvar'
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
        
        if len(pop_modes) > 1 or len(income_modes) > 1:
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
        
        st.info(f"🔄 Starting Monte-Carlo study with {st.session_state.n_runs} runs of {st.session_state.n_agents} agents each...")
        st.caption(f"📊 Mode: {pop_label} + {income_label}")
        
        # Show estimated time
        estimated_time_per_run = 2
        total_estimated_time = st.session_state.n_runs * estimated_time_per_run
        st.caption(f"⏱️ Estimated time: ~{total_estimated_time} seconds ({total_estimated_time/60:.1f} minutes)")
        
        # Build command
        cmd = [
            sys.executable, 'scripts/run_mc_study.py',
            '--agents', str(st.session_state.n_agents),
            '--runs', str(st.session_state.n_runs),
            '--base-seed', str(st.session_state.base_seed),
            '--anchor-observed', str(st.session_state.anchor_observed_weight),
            '--population-mode', pop_mode_arg,
            '--income-mode', income_mode_arg
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
            st.caption(f"Agents per run: {st.session_state.n_agents}")
        
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


def run_simulation_from_sidebar():
    """Run simulation using original app.py logic"""
    try:
        with st.spinner("🔄 Generating synthetic agents and running simulation..."):
            # Helper to run simulation with chosen orchestrator and income mode
            def _run(pop_mode: str, inc_mode: str, prob_settings=None, agents_df=None):
                # Initialize appropriate orchestrator
                if pop_mode == "documentation":
                    orchestrator = OrchestratorDocMode()
                elif pop_mode == "depvar":
                    orchestrator = OrchestratorDepVar()
                elif pop_mode == "baseline":
                    from src.orchestrator_baseline import OrchestratorBaseline
                    orchestrator = OrchestratorBaseline()
                else:  # copula
                    orchestrator = Orchestrator()
                
                # Override income specification in config based on choice (not for depvar mode)
                if hasattr(orchestrator, 'config') and 'donation_default' in orchestrator.config:
                    if pop_mode != "depvar":  # depvar mode doesn't use these settings
                        # Set income_mode in both legacy and new locations for compatibility
                        orchestrator.config['donation_default']['regression']['income_mode'] = inc_mode
                        if 'regression_coefficients' not in orchestrator.config['donation_default']:
                            orchestrator.config['donation_default']['regression_coefficients'] = {}
                        orchestrator.config['donation_default']['regression_coefficients']['income_mode'] = inc_mode
                        # Set stochastic flag for copula mode if checkbox is enabled
                        if pop_mode == "copula":
                            orchestrator.config['donation_default']['stochastic']['in_copula'] = st.session_state.sigma_in_copula
                        
                        # Apply sigma value based on mode and user preferences
                        if pop_mode == "documentation" and not st.session_state.sigma_in_research:
                            # Research mode with sigma disabled - set to 0
                            orchestrator.config['donation_default']['stochastic']['sigma_value'] = 0.0
                        else:
                            # Apply selected sigma value
                            orchestrator.config['donation_default']['stochastic']['sigma_value'] = st.session_state.sigma_value_ui
                        # Apply chosen anchor weights
                        orchestrator.config['donation_default']['anchor_weights']['observed'] = st.session_state.anchor_observed_weight
                        orchestrator.config['donation_default']['anchor_weights']['predicted'] = 1 - st.session_state.anchor_observed_weight
                        
                        if hasattr(st.session_state, 'selected_donation_config'):
                            apply_selected_donation_config(orchestrator, pop_mode, inc_mode)
                        # Fallback: Apply custom regression coefficients if they exist
                        elif hasattr(st.session_state, 'custom_coefficients') and 'donation_default' in st.session_state.custom_coefficients:
                            custom_coeffs = st.session_state.custom_coefficients['donation_default']
                            # Update regression_coefficients in the config
                            if 'regression_coefficients' not in orchestrator.config['donation_default']:
                                orchestrator.config['donation_default']['regression_coefficients'] = {}
                            orchestrator.config['donation_default']['regression_coefficients'].update(custom_coeffs)
                            
                            # Ensure the income mode matches the selected specification
                            orchestrator.config['donation_default']['regression_coefficients']['income_mode'] = inc_mode
                        # NEW FALLBACK: Use current session state coefficients if no custom coefficients are set
                        else:
                            # Ensure session state coefficients are loaded from YAML
                            from app.models import load_donation_coefficients_from_yaml
                            if 'donation_coeff_intercept' not in st.session_state:
                                load_donation_coefficients_from_yaml()
                            
                            # Collect current coefficients from session state (loaded from YAML on app start)
                            from app.pages.decision_execution import get_current_coefficients
                            current_coeffs = get_current_coefficients()
                            current_coeffs['income_mode'] = inc_mode
                            
                            # Update orchestrator config with current session state coefficients
                            if 'regression_coefficients' not in orchestrator.config['donation_default']:
                                orchestrator.config['donation_default']['regression_coefficients'] = {}
                            orchestrator.config['donation_default']['regression_coefficients'].update(current_coeffs)
                
                # CRITICAL: Override YAML defaults with Page 1 UI parameters
                # This ensures user-configured values from Page 1 take precedence over config/simulation.yaml
                if hasattr(orchestrator, 'simulation_config'):
                    if 'simulation' not in orchestrator.simulation_config:
                        orchestrator.simulation_config['simulation'] = {}
                    
                    # Copy ALL Page 1 parameters from session state to override YAML defaults
                    sim_params = st.session_state.sim_params
                    
                    # Income distribution parameters - CRITICAL for disclose_documents eligibility
                    orchestrator.simulation_config['simulation']['income_distribution'] = sim_params.income_distribution
                    orchestrator.simulation_config['simulation']['discount_income_threshold'] = sim_params.discount_income_threshold
                    
                    # Lognormal parameters
                    orchestrator.simulation_config['simulation']['lognormal_mu'] = sim_params.lognormal_mu
                    orchestrator.simulation_config['simulation']['lognormal_sigma'] = sim_params.lognormal_sigma
                    orchestrator.simulation_config['simulation']['lognormal_min'] = sim_params.lognormal_min
                    orchestrator.simulation_config['simulation']['lognormal_max'] = sim_params.lognormal_max
                    
                    # Generalised Gamma parameters
                    orchestrator.simulation_config['simulation']['gg_k'] = sim_params.gg_k
                    orchestrator.simulation_config['simulation']['gg_c'] = sim_params.gg_c
                    orchestrator.simulation_config['simulation']['gg_lambda'] = sim_params.gg_lambda
                    orchestrator.simulation_config['simulation']['gg_min'] = sim_params.gg_min
                    orchestrator.simulation_config['simulation']['gg_max'] = sim_params.gg_max
                    
                    # Dagum parameters
                    orchestrator.simulation_config['simulation']['dagum_a'] = sim_params.dagum_a
                    orchestrator.simulation_config['simulation']['dagum_p'] = sim_params.dagum_p
                    orchestrator.simulation_config['simulation']['dagum_b'] = sim_params.dagum_b
                    orchestrator.simulation_config['simulation']['dagum_min'] = sim_params.dagum_min
                    orchestrator.simulation_config['simulation']['dagum_max'] = sim_params.dagum_max
                    
                    # Market parameters - used by bid_value and other decisions
                    orchestrator.simulation_config['simulation']['market_price'] = sim_params.market_price
                    orchestrator.simulation_config['simulation']['platform_markup'] = sim_params.platform_markup
                    orchestrator.simulation_config['simulation']['price_range'] = sim_params.price_range
                    orchestrator.simulation_config['simulation']['bidding_percentage'] = sim_params.bidding_percentage
                    orchestrator.simulation_config['simulation']['num_vendors'] = sim_params.num_vendors
                    
                    # Vendor configuration parameters - CRITICAL for vendor generation
                    # Without these, orchestrator falls back to YAML defaults regardless of UI settings
                    orchestrator.simulation_config['simulation']['vendor_config_mode'] = sim_params.vendor_config_mode
                    orchestrator.simulation_config['simulation']['vendor_price_source'] = sim_params.vendor_price_source
                    
                    # Vendor pricing parameters (for random generation)
                    orchestrator.simulation_config['simulation']['vendor_price_min'] = sim_params.vendor_price_min
                    orchestrator.simulation_config['simulation']['vendor_price_max'] = sim_params.vendor_price_max
                    
                    # Vendor products parameters (for random generation)
                    # These control quantity_offered per vendor, which determines total market supply
                    orchestrator.simulation_config['simulation']['vendor_products_min'] = sim_params.vendor_products_min
                    orchestrator.simulation_config['simulation']['vendor_products_max'] = sim_params.vendor_products_max
                    orchestrator.simulation_config['simulation']['vendor_products_avg'] = sim_params.vendor_products_avg
                    
                    # Vendor carryover parameters
                    orchestrator.simulation_config['simulation']['vendor_carryover_probability'] = sim_params.vendor_carryover_probability
                    orchestrator.simulation_config['simulation']['override_carryover'] = sim_params.override_carryover
                    orchestrator.simulation_config['simulation']['global_carryover'] = sim_params.global_carryover
                    
                    # Vendor configuration data (if uploaded via CSV)
                    if hasattr(sim_params, 'vendor_config_data') and sim_params.vendor_config_data is not None:
                        orchestrator.simulation_config['simulation']['vendor_config_data'] = sim_params.vendor_config_data
                    
                    # Legacy vendor parameters (for backward compatibility)
                    orchestrator.simulation_config['simulation']['products_per_vendor'] = sim_params.products_per_vendor
                    orchestrator.simulation_config['simulation']['carryover'] = sim_params.carryover
                    if hasattr(sim_params, 'vendor_prices') and sim_params.vendor_prices is not None:
                        orchestrator.simulation_config['simulation']['vendor_prices'] = sim_params.vendor_prices
                    
                    # Time parameters
                    orchestrator.simulation_config['simulation']['periods'] = sim_params.periods
                    orchestrator.simulation_config['simulation']['duration_hours'] = sim_params.duration_hours
                    
                    # Income categories
                    orchestrator.simulation_config['simulation']['num_discount_categories'] = sim_params.num_discount_categories
                    orchestrator.simulation_config['simulation']['num_fixed_categories'] = sim_params.num_fixed_categories
                    
                    # Consumption parameters
                    orchestrator.simulation_config['simulation']['max_purchases_per_term'] = sim_params.max_purchases_per_term
                
                # Ensure all orchestrators have decision settings available
                if prob_settings:
                    # For orchestrators with simulation_config
                    if hasattr(orchestrator, 'simulation_config'):
                        # Store both as 'random_decisions' (for backward compatibility) 
                        # and 'default_decisions' (for all decision types)
                        orchestrator.simulation_config['random_decisions'] = prob_settings
                        orchestrator.simulation_config['default_decisions'] = prob_settings
                    else:
                        # For orchestrators without simulation_config, create minimal config
                        orchestrator.simulation_config = {
                            'random_decisions': prob_settings,
                            'default_decisions': prob_settings
                        }
                
                # Pass purchasing limits to orchestrator if enabled
                if st.session_state.sim_params.apply_purchasing_limits:
                    if hasattr(orchestrator, 'simulation_config'):
                        orchestrator.simulation_config['purchasing_limits'] = st.session_state.sim_params.purchasing_limits
                    else:
                        orchestrator.simulation_config = {
                            'purchasing_limits': st.session_state.sim_params.purchasing_limits
                        }
                
                # Also pass information about which decisions are custom vs default
                # This helps decision modules know whether to use custom parameters or configured defaults
                if hasattr(st.session_state, 'custom_decisions') and hasattr(st.session_state, 'default_decisions'):
                    if not hasattr(orchestrator, 'simulation_config'):
                        orchestrator.simulation_config = {}
                    orchestrator.simulation_config['custom_decisions'] = st.session_state.custom_decisions
                    orchestrator.simulation_config['default_decisions_list'] = st.session_state.default_decisions
                
                # Handle multiple decisions
                decision_param = None if len(st.session_state.decision_params.selected_decisions) == len(ALL_DECISIONS) else st.session_state.decision_params.selected_decisions
                # Determine correct seed (prefer input widget value to avoid sync issues)
                if st.session_state.sim_params.simulation_mode == "Single Run":
                    seed_val = st.session_state.get('seed_input', st.session_state.seed)
                else:
                    seed_val = st.session_state.get('base_seed_input', st.session_state.base_seed)
                
                return orchestrator.run_simulation(
                    n_agents=st.session_state.n_agents,
                    seed=seed_val,
                    single_decision=decision_param,
                    agents_df=agents_df  # Pass pre-sampled agents for consistency
                )
            
            # CRITICAL FIX: Apply selected donation configuration BEFORE determining result variants
            # This ensures we generate only the selected configuration, not all variants
            if hasattr(st.session_state, 'selected_donation_config'):
                config = st.session_state.selected_donation_config
                
                # Show user that we're using selected configuration
                st.info(f"🎯 Using selected donation configuration: {config['population_mode']} + {config['income_spec_mode']}")
                
                # Override session state variables that control result generation
                if not hasattr(st.session_state, '_original_population_mode'):
                    st.session_state._original_population_mode = st.session_state.population_mode
                    st.session_state._original_income_spec_mode = st.session_state.income_spec_mode
                
                # Override to match selected configuration - this controls how many results are generated
                st.session_state.population_mode = config['population_mode']
                st.session_state.income_spec_mode = config['income_spec_mode']
            
            # Collect current decision settings for all default decisions
            random_decision_probabilities = collect_decision_settings()
            
            # Debug: Show decision settings being applied
            if random_decision_probabilities:
                setting_info = []
                for decision, settings in random_decision_probabilities.items():
                    decision_type = settings.get('type')
                    if decision_type == 'random_probability':
                        prob_y = settings['probability_y']
                        options = settings['options']
                        setting_info.append(f"{decision}: {prob_y:.0%} {options[0]} / {1-prob_y:.0%} {options[1]}")
                    elif decision_type == 'prioritized_selection':
                        priority_template = settings.get('priority_template', [])
                        if len(priority_template) == 1:
                            setting_info.append(f"{decision}: {priority_template[0]} only")
                        else:
                            setting_info.append(f"{decision}: {len(priority_template)} priorities")
                    elif decision_type == 'checkbox_selection':
                        selected = settings.get('selected_params', [])
                        setting_info.append(f"{decision}: {len(selected)} params selected ({', '.join(selected)})")
                    elif decision_type == 'radio_selection':
                        selected = settings.get('selected_option', 'unknown')
                        setting_info.append(f"{decision}: {selected}")
                    elif decision_type == 'numeric':
                        value = settings.get('value', 0)
                        # Format as percentage if it's between 0 and 1
                        try:
                            float_value = float(value)
                            if 0 <= float_value <= 1:
                                setting_info.append(f"{decision}: {float_value:.1%}")
                            else:
                                setting_info.append(f"{decision}: {float_value}")
                        except (ValueError, TypeError):
                            # If value can't be converted to float, just display as is
                            setting_info.append(f"{decision}: {value}")
                    elif decision_type == 'placeholder':
                        value = settings.get('value', 'default')
                        # These are placeholder strings like "RANDOM_WITHIN_LIMIT", "NA", etc.
                        setting_info.append(f"{decision}: {value}")
                
                if setting_info:
                    st.success(f"🎲 Using configured defaults: {', '.join(setting_info)}")
                    # Also print to console for debugging
                    # print(f"[DEBUG] Decision settings: {random_decision_probabilities}")
            
            # CRITICAL FIX: Pre-sample agents ONCE for all configurations
            # This ensures agent alignment across different income/population modes
            agents_df = None
            n_agents = st.session_state.n_agents
            seed = st.session_state.seed if st.session_state.sim_params.simulation_mode == "Single Run" else st.session_state.base_seed
            
            # Determine which type of agents to use based on population mode
            if st.session_state.population_mode == "Dependent variable resampling":
                # DepVar mode doesn't use agents (only resamples outcomes)
                agents_df = None
            elif st.session_state.population_mode in ["Research Specification", "Research Baseline"]:
                # Load original 280 participants for research modes
                from src.orchestrator_baseline import OrchestratorBaseline
                temp_orchestrator = OrchestratorBaseline()
                if n_agents <= len(temp_orchestrator.original_data):
                    agents_df = temp_orchestrator.original_data.iloc[:n_agents].copy()
                else:
                    # Bootstrap sample if more agents requested than available
                    rng = np.random.default_rng(seed)
                    indices = rng.choice(len(temp_orchestrator.original_data), size=n_agents, replace=True)
                    agents_df = temp_orchestrator.original_data.iloc[indices].copy()
                    agents_df.index = range(len(agents_df))
                st.info(f"📊 Using {len(agents_df)} agents from original participant data")
            elif st.session_state.population_mode == "Compare all":
                # For comparison mode, we need to decide: use copula or research participants?
                # Default to copula for synthetic diversity
                from src.trait_engine import TraitEngine
                trait_engine = TraitEngine()
                agents_df = trait_engine.sample(n_agents, seed)
                st.info(f"🎲 Sampled {len(agents_df)} synthetic agents from copula for comparison")
            else:  # Copula (synthetic)
                # Sample from copula for single copula mode
                from src.trait_engine import TraitEngine
                trait_engine = TraitEngine()
                agents_df = trait_engine.sample(n_agents, seed)
                st.info(f"🎲 Sampled {len(agents_df)} synthetic agents from copula")
            
            # Run based on population and income specification modes
            results = {}
            
            if st.session_state.population_mode == "Compare all":
                # Compare all three population modes
                for pop_name, pop_type in [("copula", "copula"), ("research_spec", "documentation"), ("research_baseline", "baseline")]:
                    if st.session_state.income_spec_mode == "Compare both":
                        results[f"{pop_name}_categorical"] = _run(pop_type, "categorical", random_decision_probabilities, agents_df)
                        results[f"{pop_name}_continuous"] = _run(pop_type, "continuous", random_decision_probabilities, agents_df)
                    elif st.session_state.income_spec_mode == "continuous only":
                        results[f"{pop_name}_continuous"] = _run(pop_type, "continuous", random_decision_probabilities, agents_df)
                    else:  # categorical only
                        results[f"{pop_name}_categorical"] = _run(pop_type, "categorical", random_decision_probabilities, agents_df)
            elif st.session_state.population_mode == "Dependent variable resampling":
                # Dependent variable mode - only one result regardless of income spec
                results["depvar"] = _run("depvar", "categorical", random_decision_probabilities, agents_df)  # income mode is ignored, agents_df is None
            else:
                # Single population mode
                if st.session_state.population_mode == "Research Specification":
                    pop_type = "documentation"
                elif st.session_state.population_mode == "Research Baseline":
                    pop_type = "baseline"
                else:  # Copula (synthetic)
                    pop_type = "copula"
                    
                if st.session_state.income_spec_mode == "Compare both":
                    results["categorical"] = _run(pop_type, "categorical", random_decision_probabilities, agents_df)
                    results["continuous"] = _run(pop_type, "continuous", random_decision_probabilities, agents_df)
                elif st.session_state.income_spec_mode == "continuous only":
                    results["continuous"] = _run(pop_type, "continuous", random_decision_probabilities, agents_df)
                else:  # categorical only
                    results["categorical"] = _run(pop_type, "categorical", random_decision_probabilities, agents_df)
            
            # COMMENTED OUT: Auto-save to parquet file (redundant with Results page Excel export)
            # Uncomment if needed for batch processing or programmatic access
            # if st.session_state.save_results:
            #     output_dir = Path("outputs")
            #     output_dir.mkdir(exist_ok=True)
            #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #     # Create decision suffix for filename
            #     if len(st.session_state.decision_params.selected_decisions) == len(ALL_DECISIONS):
            #         decision_suffix = "_all"
            #     elif len(st.session_state.decision_params.selected_decisions) == 1:
            #         decision_suffix = f"_{st.session_state.decision_params.selected_decisions[0]}"
            #     else:
            #         decision_suffix = f"_{len(st.session_state.decision_params.selected_decisions)}decisions"
            #     
            #     for mode, df in results.items():
            #         filename = f"enhanced_simulation_{mode}_seed{st.session_state.seed if st.session_state.sim_params.simulation_mode == 'Single Run' else st.session_state.base_seed}_agents{st.session_state.n_agents}{decision_suffix}_{timestamp}.parquet"
            #         filepath = output_dir / filename
            #         
            #         # Prepare DataFrame for parquet saving
            #         # Parquet can't handle complex nested structures, so convert purchase_requests to JSON
            #         df_to_save = df.copy()
            #         if 'purchase_requests' in df_to_save.columns:
            #             import json
            #             df_to_save['purchase_requests'] = df_to_save['purchase_requests'].apply(
            #                 lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x)
            #             )
            #         
            #         df_to_save.to_parquet(filepath, index=False)
            #     
            #     st.sidebar.caption(f"✅ Results saved with timestamp {timestamp}")
            
            st.session_state.simulation_results = results
            
            # Extract and store vendor data from first DataFrame for easy access in results visualization
            # (All DataFrames should have the same vendor data in their attrs)
            for df in results.values():
                if hasattr(df, 'attrs') and 'vendors' in df.attrs:
                    st.session_state.vendors = df.attrs['vendors']
                    break
            
            # Add a flag to indicate we're using selected configuration
            if hasattr(st.session_state, 'selected_donation_config'):
                st.session_state._using_selected_config = True
            
            # DON'T restore session state here - we need the results page to see the selected configuration values
            # The restoration will happen when navigating away from results
            
            st.session_state.page = 'results'
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Simulation failed: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        
        # CLEANUP: Restore original session state even on error
        if hasattr(st.session_state, '_original_population_mode'):
            st.session_state.population_mode = st.session_state._original_population_mode
            st.session_state.income_spec_mode = st.session_state._original_income_spec_mode
            delattr(st.session_state, '_original_population_mode')
            delattr(st.session_state, '_original_income_spec_mode')


def collect_decision_settings():
    """Collect current default decision settings from session state (probabilities, selections, etc.)
    
    IMPORTANT: This function checks both session state AND _persistent_defaults (shadow state).
    The _persistent_defaults dictionary is used by default_config.py widgets to preserve values
    across page navigation. We must check it here to ensure configured values are used even when
    Page 2 hasn't rendered (e.g., when running simulation from Results page).
    """
    
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES
    
    decision_settings = {}
    
    # Get the persistent defaults dictionary (shadow state from default_config.py)
    # This is critical for preserving user-configured values when Page 2 hasn't rendered
    persistent_defaults = st.session_state.get('_persistent_defaults', {})
    
    # Check each decision for configured settings
    for decision_name, default_value in DEFAULT_DECISION_VALUES.items():
        if isinstance(default_value, dict):
            decision_type = default_value.get("type")
            
            # Handle random probability decisions (disclose_income, disclose_documents, purchase_vs_bid)
            if decision_type == "random_probability":
                # Priority order for probability values:
                # 1. Post-simulation adjustment from Results page ({decision_name}_probability_y)
                # 2. Pre-configured default from Overview tab - persistent storage (_persistent_defaults)
                # 3. Pre-configured default from Overview tab - session state ({decision_name}_default_probability_y)
                # 4. Hard-coded default from DEFAULT_DECISION_VALUES
                
                post_sim_key = f"{decision_name}_probability_y"
                pre_config_key = f"{decision_name}_default_probability_y"
                hardcoded_default = default_value.get("probability_y", 0.5)
                
                # Check in priority order
                if post_sim_key in st.session_state:
                    current_prob = st.session_state[post_sim_key]
                elif pre_config_key in persistent_defaults:
                    # CRITICAL: Check persistent storage BEFORE session state
                    # Session state keys might be reset to defaults by initialization scripts during reruns
                    current_prob = persistent_defaults[pre_config_key]
                elif pre_config_key in st.session_state:
                    current_prob = st.session_state[pre_config_key]
                else:
                    current_prob = hardcoded_default
                
                # Debug print to verify source
                # if pre_config_key in persistent_defaults:
                #    print(f"[DEBUG] {decision_name}: Using persistent value {current_prob}")
                
                decision_settings[decision_name] = {
                    "probability_y": current_prob,
                    "options": default_value.get("options", ["Y", "N"]),
                    "type": "random_probability"
                }
            
            # Handle checkbox selection decisions (vendor_choice_weights)
            elif decision_type == "checkbox_selection":
                # Priority order:
                # 1. Post-simulation adjustment from Results page (vendor_choice_weights_selection)
                # 2. Pre-configured default from Overview tab - persistent storage (_persistent_defaults)
                # 3. Pre-configured default from Overview tab - session state (vendor_choice_weights_default_params)
                # 4. Hard-coded default from DEFAULT_DECISION_VALUES
                
                post_sim_key = f"{decision_name}_selection"  # e.g., "vendor_choice_weights_selection"
                pre_config_key = f"{decision_name}_default_params"  # e.g., "vendor_choice_weights_default_params"
                hardcoded_default = default_value.get("default_selection", [])
                
                # Check in priority order
                if post_sim_key in st.session_state:
                    selected_params = st.session_state[post_sim_key]
                elif pre_config_key in persistent_defaults:
                    # CRITICAL: Check persistent storage BEFORE session state
                    selected_params = persistent_defaults[pre_config_key]
                elif pre_config_key in st.session_state:
                    selected_params = st.session_state[pre_config_key]
                else:
                    selected_params = hardcoded_default
                
                # Calculate equal weights for selected parameters
                if len(selected_params) > 0:
                    weight_per_param = 1.0 / len(selected_params)
                    weights = {}
                    
                    # Set weights for all parameters
                    for param_key in default_value.get("parameters", {}).keys():
                        if param_key in selected_params:
                            weights[param_key] = weight_per_param
                        else:
                            weights[param_key] = 0.0
                else:
                    # Fallback to equal weights if nothing selected
                    params = list(default_value.get("parameters", {}).keys())
                    weight_per_param = 1.0 / len(params) if params else 0.25
                    weights = {param: weight_per_param for param in params}
                
                decision_settings[decision_name] = {
                    "selected_params": selected_params,
                    "weights": weights,
                    "type": "checkbox_selection"
                }
            
            # Handle prioritized selection decisions (rejected_transaction_defaults with priority lists)
            elif decision_type == "prioritized_selection":
                # Priority order:
                # 1. Pre-configured priority template from Overview tab - persistent storage (_persistent_defaults)
                # 2. Pre-configured priority template from Overview tab - session state ({decision_name}_priority_template)
                # 3. Hard-coded default from DEFAULT_DECISION_VALUES
                
                pre_config_key = f"{decision_name}_priority_template"
                hardcoded_default = default_value.get("priority_template", ["forgo_transaction"])
                
                # Check in priority order
                if pre_config_key in persistent_defaults:
                    # CRITICAL: Check persistent storage BEFORE session state
                    priority_template = persistent_defaults[pre_config_key]
                elif pre_config_key in st.session_state:
                    priority_template = st.session_state[pre_config_key]
                else:
                    priority_template = hardcoded_default
                
                decision_settings[decision_name] = {
                    "priority_template": priority_template,
                    "type": "prioritized_selection"
                }
            
            # Handle radio selection decisions (rejected_transaction_option)
            elif decision_type == "radio_selection":
                # Priority order:
                # 1. Post-simulation adjustment from Results page (specific to each decision)
                # 2. Pre-configured default from Overview tab - persistent storage (_persistent_defaults)
                # 3. Pre-configured default from Overview tab - session state ({decision_name}_default_selection)
                # 4. Hard-coded default from DEFAULT_DECISION_VALUES
                
                # Map decision names to their post-simulation keys
                post_sim_keys = {
                    "rejected_transaction_defaults": "rejected_transaction_defaults_option",
                    "rejected_transaction_option": "rejected_transaction_option_selection"
                }
                
                post_sim_key = post_sim_keys.get(decision_name)
                pre_config_key = f"{decision_name}_default_selection"
                hardcoded_default = default_value.get("default_option", "")
                
                # Check in priority order
                if post_sim_key and post_sim_key in st.session_state:
                    selected_option = st.session_state[post_sim_key]
                elif pre_config_key in persistent_defaults:
                    # CRITICAL: Check persistent storage BEFORE session state
                    selected_option = persistent_defaults[pre_config_key]
                elif pre_config_key in st.session_state:
                    selected_option = st.session_state[pre_config_key]
                else:
                    selected_option = hardcoded_default
                
                decision_settings[decision_name] = {
                    "selected_option": selected_option,
                    "type": "radio_selection"
                }
        
        else:
            # Handle simple values (numeric or string placeholders)
            pre_config_key = f"{decision_name}_default_value"
            
            # Check in priority order: persistent_defaults → session state → hardcoded default
            if pre_config_key in persistent_defaults:
                configured_value = persistent_defaults[pre_config_key]
            elif pre_config_key in st.session_state:
                configured_value = st.session_state[pre_config_key]
            else:
                configured_value = default_value
            
            # Determine if it's numeric or a placeholder string
            if isinstance(configured_value, (int, float)):
                # print(f"[DEBUG] collect_settings: {decision_name} (numeric) = {configured_value}")
                decision_settings[decision_name] = {
                    "value": configured_value,
                    "type": "numeric"
                }
            else:
                # It's a placeholder string like "RANDOM_WITHIN_LIMIT", "NA", etc.
                # print(f"[DEBUG] collect_settings: {decision_name} (placeholder) = {configured_value}")
                decision_settings[decision_name] = {
                    "value": configured_value,
                    "type": "placeholder"
                }
    
    return decision_settings


def run_simulation():
    """Run simulation with current parameters"""
    st.session_state.page = 'results'
    # Simulation will be triggered on results page


def apply_selected_donation_config(orchestrator, pop_mode, inc_mode):
    """Apply the selected donation configuration to the orchestrator"""
    
    config = st.session_state.selected_donation_config
    
    # Override coefficients
    if 'regression_coefficients' not in orchestrator.config['donation_default']:
        orchestrator.config['donation_default']['regression_coefficients'] = {}
    
    # Apply all coefficients from selected configuration
    orchestrator.config['donation_default']['regression_coefficients'].update(config['coefficients'])
    
    # Set income mode to match current simulation mode (not necessarily the selected one)
    # This allows users to run different income modes with the same coefficient set
    orchestrator.config['donation_default']['regression_coefficients']['income_mode'] = inc_mode
    
    # Apply stochastic parameters
    stoch_params = config['stochastic_params']
    
    # Update stochastic settings
    # CRITICAL: The decision module uses 'in_copula' key, not 'sigma_in_copula'
    # We must set 'in_copula' for the stochastic behavior to apply correctly
    orchestrator.config['donation_default']['stochastic'].update({
        'sigma_value': stoch_params['stochastic']['sigma_value'],
        'sigma_coefficient': stoch_params['stochastic']['sigma_coefficient'],
        'in_copula': stoch_params['stochastic']['sigma_in_copula'],  # Map to correct key
        'sigma_in_research': stoch_params['stochastic']['sigma_in_research']
    })
    
    # Update anchor weights
    orchestrator.config['donation_default']['anchor_weights'].update(stoch_params['anchor_weights'])
    
    # Override session state variables to match selected configuration
    # This ensures the UI reflects the selected configuration
    st.session_state.sigma_value_ui = stoch_params['stochastic']['sigma_value']
    st.session_state.sigma_coefficient = stoch_params['stochastic']['sigma_coefficient']
    st.session_state.sigma_in_copula = stoch_params['stochastic']['sigma_in_copula']
    st.session_state.sigma_in_research = stoch_params['stochastic']['sigma_in_research']
    st.session_state.anchor_observed_weight = stoch_params['anchor_weights']['observed']
    
    # Override coefficient session state variables
    coeffs = config['coefficients']
    st.session_state.donation_coeff_intercept = coeffs['intercept']
    st.session_state.donation_coeff_hh = coeffs['beta_hh']
    st.session_state.donation_coeff_linear = coeffs['beta_income_linear']
    
    # Group coefficients
    for group, coeff in coeffs['beta_group'].items():
        if group == 'MidSub':
            st.session_state.donation_coeff_midsub = coeff
        elif group == 'NoSub':
            st.session_state.donation_coeff_nosub = coeff
        elif group == 'FullSub':
            st.session_state.donation_coeff_fullsub = coeff
    
    # Income quintile coefficients
    for quintile, coeff in coeffs['beta_income_q'].items():
        if quintile == 'Q1':
            st.session_state.donation_coeff_q1 = coeff
        elif quintile == 'Q2':
            st.session_state.donation_coeff_q2 = coeff
        elif quintile == 'Q3':
            st.session_state.donation_coeff_q3 = coeff
        elif quintile == 'Q4':
            st.session_state.donation_coeff_q4 = coeff
        elif quintile == 'Q5':
            st.session_state.donation_coeff_q5 = coeff
        elif quintile == 'Q4_Q5':  # Legacy support
            st.session_state.donation_coeff_q45 = coeff
    
    # Study programme coefficients
    for study, coeff in coeffs['beta_study'].items():
        if study == 'Incoming':
            st.session_state.donation_coeff_incoming = coeff
        elif study == 'Law5yr':
            st.session_state.donation_coeff_law = coeff
        elif study == 'UG3yr':
            st.session_state.donation_coeff_ug = coeff
        elif study == 'Grad2yr':
            st.session_state.donation_coeff_grad = coeff
