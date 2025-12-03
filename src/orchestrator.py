# src/orchestrator.py
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
import importlib

from src.trait_engine import TraitEngine

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "decisions.yaml"
SIMULATION_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "simulation.yaml"

class Orchestrator:
    """
    Coordinates trait sampling and decision execution.
    
    Supports both full-run (all 13 decisions) and single-decision modes.
    Each agent maintains state that accumulates across decisions.
    """
    
    def __init__(self):
        # Load decision configuration
        with open(CONFIG_PATH, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load global simulation configuration
        with open(SIMULATION_CONFIG_PATH, 'r') as f:
            self.simulation_config = yaml.safe_load(f)
        
        # Initialize trait engine
        self.trait_engine = TraitEngine()
        
        # Set population context for decision modules
        self.pop_context = 'copula'
        
        # Define decision order (1-13 as specified)
        self.decision_order = [
            'disclose_income',           # 1
            'disclose_documents',        # 2  
            'donation_default',          # 3
            'rejected_transaction_defaults',  # 4
            'vendor_choice_weights',     # 5
            'purchasing_quantity',       # 6
            'purchasing_frequency',      # 7
            'vendor_selection',          # 8
            'purchase_vs_bid',           # 9 (now deprecated - kept for backward compatibility)
            'bid_value',                 # 10 (now deprecated - kept for backward compatibility)
            'rejected_transaction_option',  # 11
            'rejected_bid_value',        # 12
            'final_donation_rate'        # 13
        ]
        
        # Load decision modules dynamically
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
        Run simulation for n_agents with specified seed.
        
        If single_decision is provided:
        - If it's a string, only run that decision
        - If it's a list of strings, run those decisions in order
        Otherwise run all decisions in order.
        
        Args:
            agents_df: Optional pre-sampled agents DataFrame. If provided, skip trait sampling.
                      This ensures the same agents are used across multiple configurations.
        """
        # Sample synthetic agents FIRST (copula uses its own internal RNG)
        # UNLESS pre-sampled agents are provided (for multi-config consistency)
        if agents_df is None:
            agents_df = self.trait_engine.sample(n_agents, seed)
        else:
            # Validate that provided agents have the required traits
            required_traits = set(self.trait_engine.get_available_traits())
            provided_traits = set(agents_df.columns)
            if not required_traits.issubset(provided_traits):
                missing = required_traits - provided_traits
                raise ValueError(f"Provided agents_df missing required traits: {missing}")
        
        # Create separate RNG for setup tasks (vendors, etc.)
        rng_setup = np.random.default_rng(seed)
        self._initialize_vendors(rng_setup)
        
        # Reset vendor capacity tracking for this simulation run
        if 'vendor_remaining_capacity' in self.simulation_config:
            del self.simulation_config['vendor_remaining_capacity']
        
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
        
        # Process each agent
        results = []
        
        for idx, row in agents_df.iterrows():
            # Initialize agent state with traits
            agent_state = row.to_dict()
            
            # Add agent ID and index to agent_state (CRITICAL for customer_id in purchase_requests)
            agent_state['index'] = idx
            agent_state['agent_id'] = idx + 1  # Agent IDs start at 1
            
            # Execute decisions in order - use per-decision RNG for consistency
            # This ensures each decision gets the same random state regardless of
            # which other decisions are run (critical for individual vs full simulation runs)
            for decision_idx, decision_name in enumerate(self.decision_order):
                # Create per-decision RNG using DETERMINISTIC seed based on:
                # - Global seed (same for all runs with same seed)
                # - Agent index (deterministic from data order)
                # - Decision index (fixed position in decision_order)
                # This ensures IDENTICAL RNG state for each decision regardless of
                # which other decisions are run (critical for individual vs full runs)
                deterministic_seed = seed + (idx * 100000) + (decision_idx * 1000)
                decision_rng = np.random.default_rng(deterministic_seed)
                
                # Skip decisions not in the run list (decision_idx is consistent regardless)
                if decision_name not in decisions_to_run:
                    continue
                
                if decision_name in self.decision_modules:
                    # Get parameters for this decision
                    params = self.config.get(decision_name, {})
                    
                    # Execute decision module
                    # Pass pop_context to modules that support it (donation_default)
                    # Pass simulation_config to all modules for global parameters
                    if decision_name == 'donation_default':
                        decision_output = self.decision_modules[decision_name](
                            agent_state, params, decision_rng, pop_context=self.pop_context, simulation_config=self.simulation_config
                        )
                    else:
                        decision_output = self.decision_modules[decision_name](
                            agent_state, params, decision_rng, simulation_config=self.simulation_config
                        )
                    
                    # Update agent state with decision outputs
                    agent_state.update(decision_output)
                else:
                    print(f"Warning: No module found for decision {decision_name}")
            
            results.append(agent_state)
        
        # Create DataFrame with results
        results_df = pd.DataFrame(results)

        # Attach vendor data to DataFrame attrs for access in results visualization
        if 'vendors' in self.simulation_config:
            results_df.attrs['vendors'] = self.simulation_config['vendors']

        return results_df
    
    def _initialize_vendors(self, rng: np.random.Generator):
        """
        Generate vendor attributes once per simulation.
        
        Creates vendors with:
        - vendor_id: Sequential ID (1, 2, 3, ...)
        - price: Randomized within [vendor_price_min, vendor_price_max]
        - quality: Random integer in [1, 5]
        - sustainability: Random integer in [1, 5]
        - quantity_offered: Random integer in [vendor_products_min, vendor_products_max]
        
        Proximity is NOT generated here - it's customer-vendor specific
        and generated per agent in vendor_selection decision.
        
        Args:
            rng: Random number generator for reproducibility
        """
        from src.vendor_attribute_generator import generate_vendor_attributes
        
        # Get vendor configuration from simulation_config
        if 'simulation' not in self.simulation_config:
            return  # No simulation config available
        
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
        # If explicit vendor_prices are provided, use those instead of randomizing
        vendor_prices = []
        use_explicit_prices = False
        
        if 'vendor_prices' in sim_config and sim_config['vendor_prices']:
            vendor_prices = sim_config['vendor_prices']
            use_explicit_prices = True
        else:
            # Will randomize prices, but need a list for function signature
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
            price_min=None if use_explicit_prices else price_min,  # Only randomize if not using explicit prices
            price_max=None if use_explicit_prices else price_max,
            quantity_min=quantity_min,
            quantity_max=quantity_max,
            num_periods=num_periods  # NEW: Pass number of periods for per-period quantity generation
        )
        
        # Store in simulation_config for access by decision modules
        self.simulation_config['vendors'] = vendors
        
        # print(f"[DEBUG] Generated {len(vendors)} vendors with attributes:")
        # for vendor in vendors:
        #     print(f"  Vendor {vendor['vendor_id']}: price=${vendor['price']:.2f}, "
        #           f"quality={vendor['quality']}, sustainability={vendor['sustainability']}")
    
    def get_available_decisions(self) -> List[str]:
        """Return list of available decision modules."""
        return list(self.decision_modules.keys())