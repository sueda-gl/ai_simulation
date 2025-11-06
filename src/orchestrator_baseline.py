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
    - NO stochastic component - returns anchor values directly
    - Uses regular decision modules (not stochastic versions)
    - Sigma is forced to 0 to disable any stochastic draws
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
            'consumption_quantity',      # 6
            'enrich_purchase_requests',  # 6b (NEW!) - Enriches each purchase request with transaction decisions
            'consumption_frequency',     # 7
            'vendor_selection',          # 8
            'purchase_vs_bid',           # 9 (now deprecated - kept for backward compatibility)
            'bid_value',                 # 10 (now deprecated - kept for backward compatibility)
            'rejected_transaction_option',  # 11
            'rejected_bid_value',        # 12
            'final_donation_rate'        # 13
        ]
        
        # Load decision modules - use regular versions (no stochastic)
        self.decision_modules = {}
        for decision_name in self.decision_order:
            try:
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
        
        # Process each agent
        results = []
        
        for idx, row in agents_df.iterrows():
            agent_state = row.to_dict()
            
            # Add agent ID and index to agent_state (CRITICAL for customer_id in purchase_requests)
            agent_state['index'] = idx
            agent_state['agent_id'] = idx + 1  # Agent IDs start at 1
            
            agent_results = agent_state.copy()
            
            # Create child RNG for this agent (consistent with other orchestrators)
            agent_rng = np.random.default_rng(rng_global.integers(1e9))
            
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
                    
                    try:
                        # Call decision function with baseline context
                        # Only pass pop_context to donation_default (which expects it)
                        if decision_name == 'donation_default':
                            decision_output = decision_func(
                                agent_state, 
                                decision_params, 
                                agent_rng,
                                simulation_config=self.simulation_config,
                                pop_context=self.pop_context
                            )
                        else:
                            decision_output = decision_func(
                                agent_state, 
                                decision_params, 
                                agent_rng,
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
            quantity_max=quantity_max
        )
        
        # Store in simulation_config
        self.simulation_config['vendors'] = vendors
        
        print(f"[Baseline] Generated {len(vendors)} vendors with randomized attributes")
