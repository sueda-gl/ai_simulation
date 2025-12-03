# src/orchestrator_doc_mode.py
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

class OrchestratorDocMode:
    """
    Orchestrator for documentation mode - uses original participants instead of copula sampling.
    
    Key differences from regular Orchestrator:
    - No TraitEngine/copula sampling
    - Works with original 280 participants from merged dataset
    - Uses stochastic version of decision modules
    - Can bootstrap participants to reach desired n_agents
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
        print(f"Documentation mode: Using {len(self.original_data)} original participants")
        
        # Set population context for decision modules
        self.pop_context = 'documentation'
        
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
        
        # Load decision modules - use stochastic version where available
        self.decision_modules = {}
        for decision_name in self.decision_order:
            try:
                # First try to load stochastic version
                try:
                    module = importlib.import_module(f'src.decisions.{decision_name}_stochastic')
                    self.decision_modules[decision_name] = getattr(module, f'{decision_name}_stochastic')
                except (ImportError, AttributeError):
                    # Fall back to regular version
                    module = importlib.import_module(f'src.decisions.{decision_name}')
                    self.decision_modules[decision_name] = getattr(module, decision_name)
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load decision module {decision_name}: {e}")
    
    def run_simulation(self, n_agents: int, seed: int, 
                      single_decision: Optional[Union[str, List[str]]] = None, 
                      outcome_draws: int = 1,
                      agents_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run simulation using original participants with bootstrap sampling.
        
        If n_agents > original participants, bootstrap with replacement.
        If n_agents <= original participants, sample without replacement.
        
        Args:
            agents_df: Optional pre-loaded agents DataFrame. If provided, skip loading participants.
                      This ensures the same agents are used across multiple configurations.
        """
        # Create separate RNG for setup tasks (vendors, bootstrap sampling)
        rng_setup = np.random.default_rng(seed)
        
        # Generate vendor attributes using setup RNG
        self._initialize_vendors(rng_setup)
        
        # Reset vendor capacity tracking for this simulation run
        if 'vendor_remaining_capacity' in self.simulation_config:
            del self.simulation_config['vendor_remaining_capacity']
        
        # Sample agents from original data using setup RNG if not provided
        if agents_df is None:
            n_original = len(self.original_data)
            
            if n_agents > n_original:
                # Bootstrap with replacement on participants then repeat draws
                indices = rng_setup.choice(n_original, size=n_agents, replace=True)
                agents_df = self.original_data.iloc[indices].reset_index(drop=True)
            else:
                indices = rng_setup.choice(n_original, size=n_agents, replace=False)
                agents_df = self.original_data.iloc[indices].reset_index(drop=True)
        
        # Create dedicated RNG for agent processing (independent of setup)
        # All modes use same seed here, ensuring identical agent RNG derivation
        rng_global = np.random.default_rng(seed + 1000000)
        
        # Determine which decisions to run
        if single_decision:
            if isinstance(single_decision, str):
                # Single decision
                if single_decision not in self.decision_order:
                    raise ValueError(f"Unknown decision: {single_decision}")
                decisions_to_run = [single_decision]
            elif isinstance(single_decision, list):
                # Multiple decisions
                for decision in single_decision:
                    if decision not in self.decision_order:
                        raise ValueError(f"Unknown decision: {decision}")
                # Run decisions in the order they appear in decision_order
                decisions_to_run = [d for d in self.decision_order if d in single_decision]
            else:
                raise ValueError("single_decision must be a string or list of strings")
        else:
            decisions_to_run = self.decision_order
        
        # Create decision index mapping for deterministic RNG seeding
        # This ensures each decision gets the same RNG regardless of which decisions run
        decision_index = {name: i for i, name in enumerate(self.decision_order)}
        
        # Import income utility for pre-generation
        from src.decisions.income_utils import get_agent_income
        
        # Process each agent
        results = []
        
        for idx, row in agents_df.iterrows():
            for rep in range(outcome_draws):  # repeat dependent-var draw
                agent_state = row.to_dict()
                
                # Add agent ID and index to agent_state (CRITICAL for customer_id in purchase_requests)
                agent_state['index'] = idx
                agent_state['agent_id'] = idx + 1  # Agent IDs start at 1
                
                if outcome_draws>1:
                    agent_state['draw_id']=rep+1
                
                # Create base seed for this agent (deterministic based on agent index and rep)
                agent_base_seed = rng_global.integers(1e9)
                
                # CRITICAL FIX: Pre-generate income with a CONSISTENT RNG
                # This ensures income is the same regardless of which decisions run
                income_rng = np.random.default_rng(agent_base_seed + 999999)
                get_agent_income(agent_state, self.simulation_config, income_rng)

                # Execute decisions with decision-specific RNGs
                for decision_name in decisions_to_run:
                    if decision_name in self.decision_modules:
                        params = self.config.get(decision_name, {})
                        
                        # CRITICAL FIX: Create decision-specific RNG
                        # This ensures each decision sees the same RNG state regardless of
                        # which other decisions are running
                        decision_seed = agent_base_seed + decision_index[decision_name] * 1000
                        decision_rng = np.random.default_rng(decision_seed)
                        
                        if decision_name == 'donation_default':
                            decision_output = self.decision_modules[decision_name](
                                agent_state, params, decision_rng, pop_context=self.pop_context, 
                                simulation_config=self.simulation_config
                            )
                        else:
                            decision_output = self.decision_modules[decision_name](
                                agent_state, params, decision_rng, 
                                simulation_config=self.simulation_config
                            )
                        agent_state.update(decision_output)
                results.append(agent_state)
        
        df = pd.DataFrame(results)

        # If global-max rescaling is requested, apply it here
        donation_col = 'donation_default_raw_pos'
        if 'donation_default' not in df.columns and donation_col in df.columns:
            global_max = df[donation_col].max()
            if global_max == 0:
                df['donation_default'] = 0.0
            else:
                df['donation_default'] = (df[donation_col] / global_max).clip(0,1)
        
        # Attach vendor data to DataFrame attrs for access in results visualization
        if 'vendors' in self.simulation_config:
            df.attrs['vendors'] = self.simulation_config['vendors']
        
        return df
    
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
        
        print(f"[DocMode] Generated {len(vendors)} vendors with randomized attributes")
    
    def get_available_decisions(self) -> List[str]:
        """Return list of available decision modules."""
        return list(self.decision_modules.keys())
