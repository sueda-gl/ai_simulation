# src/orchestrator.py
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import importlib

from src.trait_engine import TraitEngine

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "decisions.yaml"

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
        
        # Initialize trait engine
        self.trait_engine = TraitEngine()
        
        # Define decision order (1-13 as specified)
        self.decision_order = [
            'disclose_income',           # 1
            'disclose_documents',        # 2  
            'donation_default',          # 3
            'rejected_transaction_defaults',  # 4
            'vendor_choice_weights',     # 5
            'consumption_quantity',      # 6
            'consumption_frequency',     # 7
            'vendor_selection',          # 8
            'purchase_vs_bid',           # 9
            'bid_value',                 # 10
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
                      single_decision: Optional[str] = None) -> pd.DataFrame:
        """
        Run simulation for n_agents with specified seed.
        
        If single_decision is provided, only run that decision.
        Otherwise run all decisions in order.
        """
        # Sample synthetic agents
        agents_df = self.trait_engine.sample(n_agents, seed)
        
        # Determine which decisions to run
        if single_decision:
            if single_decision not in self.decision_order:
                raise ValueError(f"Unknown decision: {single_decision}")
            decisions_to_run = [single_decision]
        else:
            decisions_to_run = self.decision_order
        
        # Process each agent
        results = []
        rng_global = np.random.default_rng(seed)
        
        for idx, row in agents_df.iterrows():
            # Initialize agent state with traits
            agent_state = row.to_dict()
            
            # Create child RNG for this agent
            agent_rng = np.random.default_rng(rng_global.integers(1e9))
            
            # Execute decisions in order
            for decision_name in decisions_to_run:
                if decision_name in self.decision_modules:
                    # Get parameters for this decision
                    params = self.config.get(decision_name, {})
                    
                    # Execute decision module
                    decision_output = self.decision_modules[decision_name](
                        agent_state, params, agent_rng
                    )
                    
                    # Update agent state with decision outputs
                    agent_state.update(decision_output)
                else:
                    print(f"Warning: No module found for decision {decision_name}")
            
            results.append(agent_state)
        
        return pd.DataFrame(results)
    
    def get_available_decisions(self) -> List[str]:
        """Return list of available decision modules."""
        return list(self.decision_modules.keys())