# src/orchestrator_baseline.py
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
import importlib

# Import the merged data directly
from src.validate_traits import merged
from src.build_master_traits import get_master_trait_list

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "decisions.yaml"
SIMULATION_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "simulation.yaml"

# Import default decision values
try:
    from app.pages.decision_execution import get_actual_default_value
except ImportError:
    # Fallback if import fails
    def get_actual_default_value(decision_name, sim_params=None):
        return "NA"

class OrchestratorBaseline:
    """
    Orchestrator for Research Baseline mode - uses original participants with NO stochastic component.
    
    Key differences from OrchestratorDocMode:
    - Uses original 280 participants (same as DocMode)
    - NO stochastic component - returns anchor/model values directly
    - Uses research models but without stochastic draws:
      * disclose_income: Uses two-stage mediation model, DI value directly (no Normal draw)
      * donation_default: Uses regression model, anchor value directly (no Normal draw)
    - Sigma is forced to 0 / pop_context='baseline' to disable stochastic draws
    """
    
    def __init__(self):
        # Load decision configuration
        with open(CONFIG_PATH, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load global simulation configuration (Page 1 parameters)
        with open(SIMULATION_CONFIG_PATH, 'r') as f:
            self.simulation_config = yaml.safe_load(f)
        
        # Get required traits and load original data
        self.traits = get_master_trait_list()
        self.original_data = merged[self.traits].copy().dropna()
        print(f"Research Baseline mode: Using {len(self.original_data)} original participants (no stochastic component)")
        
        # Set population context for decision modules
        self.pop_context = 'baseline'
        
        # Define decision order (same as regular orchestrator)
        self.decision_order = [
            'disclose_income',           # 1
            'disclose_documents',        # 2  
            'donation_default',          # 3
            'rejected_transaction_defaults',  # 4
            'vendor_choice_weights',     # 5
            'purchasing_quantity',      # 6
            'enrich_purchase_requests',  # 6b (NEW!) - Enriches each purchase request with transaction decisions
            'purchasing_frequency',     # 7
            'vendor_selection',          # 8
            'purchase_vs_bid',           # 9 (now deprecated - kept for backward compatibility)
            'bid_value',                 # 10 (now deprecated - kept for backward compatibility)
            'rejected_transaction_option',  # 11
            'rejected_bid_value',        # 12
            'final_donation_rate'        # 13
        ]
        
        # Load decision modules - use regular versions (no stochastic) EXCEPT for disclose_income
        # disclose_income uses stochastic module with pop_context='baseline' to get research model without stochastic draw
        self.decision_modules = {}
        for decision_name in self.decision_order:
            try:
                if decision_name == 'disclose_income':
                    # Use stochastic version for disclose_income (it handles baseline mode via pop_context)
                    module = importlib.import_module(f'src.decisions.{decision_name}_stochastic')
                    self.decision_modules[decision_name] = getattr(module, f'{decision_name}_stochastic')
                else:
                    module = importlib.import_module(f'src.decisions.{decision_name}')
                    self.decision_modules[decision_name] = getattr(module, decision_name)
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load decision module {decision_name}: {e}")
    
    def run_simulation(self, n_agents: int, seed: int, 
                      single_decision: Optional[Union[str, List[str]]] = None,
                      agents_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run simulation for n_agents using original participants with no stochastic component.
        
        If n_agents <= 280, use first n_agents participants.
        If n_agents > 280, bootstrap sample from the 280 participants.
        
        Args:
            agents_df: Optional pre-loaded agents DataFrame. If provided, skip loading participants.
                      This ensures the same agents are used across multiple configurations.
        """
        # Determine which decisions to run
        if single_decision:
            if isinstance(single_decision, str):
                if single_decision not in self.decision_order:
                    raise ValueError(f"Unknown decision: {single_decision}")
                decisions_to_run = [single_decision]
            elif isinstance(single_decision, list):
                for decision in single_decision:
                    if decision not in self.decision_order:
                        raise ValueError(f"Unknown decision: {decision}")
                decisions_to_run = [d for d in self.decision_order if d in single_decision]
            else:
                raise ValueError("single_decision must be a string or list of strings")
        else:
            decisions_to_run = self.decision_order
        
        # Create separate RNG for setup tasks (vendors, bootstrap sampling)
        rng_setup = np.random.default_rng(seed)
        
        # Store seed in simulation_config for access by decision modules
        # This allows decision modules to access the global seed (e.g. for consistent vendor locations)
        self.simulation_config['simulation_seed'] = seed
        
        # Generate vendor attributes using setup RNG
        self._initialize_vendors(rng_setup)
        
        # Reset vendor capacity tracking for this simulation run
        if 'vendor_remaining_capacity' in self.simulation_config:
            del self.simulation_config['vendor_remaining_capacity']
        
        # Load/sample agents if not provided
        if agents_df is None:
            if n_agents <= len(self.original_data):
                # Use first n_agents participants
                agents_df = self.original_data.iloc[:n_agents].copy()
            else:
                # Bootstrap sample to reach n_agents using setup RNG
                indices = rng_setup.choice(len(self.original_data), size=n_agents, replace=True)
                agents_df = self.original_data.iloc[indices].copy()
                agents_df.index = range(len(agents_df))  # Reset index
        
        # Create dedicated RNG for agent processing (independent of setup)
        # All modes use same seed here, ensuring identical agent RNG derivation
        rng_global = np.random.default_rng(seed + 1000000)
        
        # Create decision index mapping for deterministic RNG seeding
        # This ensures each decision gets the same RNG regardless of which decisions run
        decision_index = {name: i for i, name in enumerate(self.decision_order)}
        
        # Import income utility for pre-generation
        from src.decisions.income_utils import get_agent_income
        
        # ============================================================================
        # TWO-PASS APPROACH FOR INCOME MEDIAN COMPUTATION
        # Pass 1: Generate all incomes first to compute population median
        # Pass 2: Run decisions (which can now access the median for continuous mode)
        # ============================================================================
        
        # PASS 1: Generate all agent incomes and compute population median
        rng_pass1 = np.random.default_rng(seed + 1000000)
        all_incomes = []
        agent_base_seeds = []  # Store seeds for reuse in Pass 2
        
        for idx, row in agents_df.iterrows():
            # Generate the same base seed that will be used in Pass 2
            agent_base_seed = rng_pass1.integers(1e9)
            agent_base_seeds.append(agent_base_seed)
            
            # Create temporary agent state for income generation
            temp_state = row.to_dict()
            income_rng = np.random.default_rng(agent_base_seed + 999999)
            income = get_agent_income(temp_state, self.simulation_config, income_rng)
            all_incomes.append(income)
        
        # Compute and store population statistics for continuous income mode
        self.simulation_config['income_median'] = float(np.median(all_incomes))
        self.simulation_config['income_stats'] = {
            'mean': float(np.mean(all_incomes)),
            'sd': float(np.std(all_incomes))
        }
        print(f"[Baseline] Computed income median: ${self.simulation_config['income_median']:,.2f}")
        print(f"[Baseline] Computed income stats: mean=${self.simulation_config['income_stats']['mean']:,.2f}, sd=${self.simulation_config['income_stats']['sd']:,.2f}")
        
        # ============================================================================
        # SINGLE-PASS APPROACH FOR DISCLOSE_INCOME
        # No re-standardization of composite variables - use values as-is
        # Natural variation in bootstrap samples is preserved as legitimate
        # ============================================================================
        
        if 'disclose_income' in decisions_to_run and 'disclose_income' in self.decision_modules:
            disclose_income_params = self.config.get('disclose_income', {})
            di_income_mode = disclose_income_params.get('income_mode', 'categorical')
            print(f"[Baseline] disclose_income using single-pass (no re-standardization), income_mode: {di_income_mode}")
            
            if 'continuous' in str(di_income_mode).lower():
                from src.decisions.disclose_income_stochastic import compute_continuous_de_stats
                di_cont_stats = compute_continuous_de_stats(agents_df, all_incomes, disclose_income_params, self.simulation_config)
                self.simulation_config['di_cont_de_stats'] = di_cont_stats
                print(f"[Baseline] Computed continuous DE stats: mean={di_cont_stats['mean']:.6f}, sd={di_cont_stats['sd']:.6f}")
        
        # Process agents and run decisions (single-pass)
        results = []
        
        for list_idx, (idx, row) in enumerate(agents_df.iterrows()):
            agent_state = row.to_dict()
            
            # Add agent ID and index to agent_state (CRITICAL for customer_id in purchase_requests)
            agent_state['index'] = idx
            agent_state['agent_id'] = idx + 1  # Agent IDs start at 1
            
            # Use the same base seed from income pass
            agent_base_seed = agent_base_seeds[list_idx]
            
            # Pre-generate income with the SAME RNG as income pass (ensures consistency)
            income_rng = np.random.default_rng(agent_base_seed + 999999)
            get_agent_income(agent_state, self.simulation_config, income_rng)
            
            # Copy agent_state AFTER income generation so agent_results includes
            # 'income' and 'actual_allowance' fields added by get_agent_income()
            agent_results = agent_state.copy()
            
            # Run each decision in order
            for decision_name in decisions_to_run:
                if decision_name in self.decision_modules:
                    decision_func = self.decision_modules[decision_name]
                    decision_params = self.config.get(decision_name, {})
                    
                    # For donation_default, force sigma to 0 to disable stochastic component
                    if decision_name == 'donation_default':
                        decision_params_copy = decision_params.copy()
                        if 'stochastic' in decision_params_copy:
                            decision_params_copy['stochastic'] = decision_params_copy['stochastic'].copy()
                            decision_params_copy['stochastic']['sigma_value'] = 0.0  # Force no stochastic component
                        else:
                            decision_params_copy['stochastic'] = {'sigma_value': 0.0}
                        decision_params = decision_params_copy
                    
                    # CRITICAL FIX: Create decision-specific RNG
                    # This ensures each decision sees the same RNG state regardless of
                    # which other decisions are running
                    decision_seed = agent_base_seed + decision_index[decision_name] * 1000
                    decision_rng = np.random.default_rng(decision_seed)
                    
                    try:
                        # Call decision function with baseline context
                        # Pass pop_context to decisions that support it (donation_default, disclose_income)
                        if decision_name in ('donation_default', 'disclose_income'):
                            decision_output = decision_func(
                                agent_state, 
                                decision_params, 
                                decision_rng,
                                simulation_config=self.simulation_config,
                                pop_context=self.pop_context
                            )
                        else:
                            decision_output = decision_func(
                                agent_state, 
                                decision_params, 
                                decision_rng,
                                simulation_config=self.simulation_config
                            )
                        
                        # Update agent state with decision outputs
                        if isinstance(decision_output, dict):
                            agent_results.update(decision_output)
                            agent_state.update(decision_output)
                        
                    except Exception as e:
                        print(f"Error in decision {decision_name} for agent {idx}: {e}")
                        # Use default value for failed decisions
                        default_field = decision_params.get('output_field', decision_name)
                        default_value = get_actual_default_value(decision_name)
                        agent_results[default_field] = default_value
                        agent_state[default_field] = default_value
                else:
                    # Decision module not loaded - use default
                    decision_params = self.config.get(decision_name, {})
                    default_field = decision_params.get('output_field', decision_name)
                    default_value = get_actual_default_value(decision_name)
                    agent_results[default_field] = default_value
                    agent_state[default_field] = default_value
            
            results.append(agent_results)
        
        # Create DataFrame with results
        results_df = pd.DataFrame(results)

        # Attach vendor data to DataFrame attrs for access in results visualization
        if 'vendors' in self.simulation_config:
            results_df.attrs['vendors'] = self.simulation_config['vendors']

        return results_df
    
    def _initialize_vendors(self, rng: np.random.Generator):
        """
        Generate vendor attributes once per simulation.
        
        Creates vendors with quality, sustainability, price (randomized), and quantity attributes.
        Shared implementation with main Orchestrator.
        """
        from src.vendor_attribute_generator import generate_vendor_attributes
        
        # Get vendor configuration from simulation_config
        if 'simulation' not in self.simulation_config:
            return
        
        sim_config = self.simulation_config['simulation']
        num_vendors = sim_config.get('num_vendors', 1)
        
        # Get price range for randomization
        price_min = sim_config.get('vendor_price_min', 50.0)
        price_max = sim_config.get('vendor_price_max', 150.0)
        
        # Get quantity range for randomization
        quantity_min = sim_config.get('vendor_products_min', 50)
        quantity_max = sim_config.get('vendor_products_max', 150)
        
        # Get number of periods for per-period quantity generation
        num_periods = sim_config.get('periods', 1)
        
        # Get vendor prices (for backward compatibility if specified)
        vendor_prices = []
        use_explicit_prices = False
        
        if 'vendor_prices' in sim_config and sim_config['vendor_prices']:
            vendor_prices = sim_config['vendor_prices']
            use_explicit_prices = True
        else:
            market_price = sim_config.get('market_price', 100.0)
            vendor_prices = [market_price] * num_vendors
        
        # Ensure we have enough prices
        while len(vendor_prices) < num_vendors:
            vendor_prices.append(sim_config.get('market_price', 100.0))
        
        # Generate vendor attributes with randomization
        vendors = generate_vendor_attributes(
            num_vendors=num_vendors,
            vendor_prices=vendor_prices,
            rng=rng,
            price_min=None if use_explicit_prices else price_min,
            price_max=None if use_explicit_prices else price_max,
            quantity_min=quantity_min,
            quantity_max=quantity_max,
            num_periods=num_periods  # NEW: Pass number of periods for per-period quantity generation
        )
        
        # Store in simulation_config
        self.simulation_config['vendors'] = vendors
        
        print(f"[Baseline] Generated {len(vendors)} vendors with randomized attributes")
