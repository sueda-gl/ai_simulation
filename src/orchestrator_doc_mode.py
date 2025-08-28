# src/orchestrator_doc_mode.py
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import importlib

# Import the merged data directly
from src.validate_traits import merged
from src.build_master_traits import get_master_trait_list

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "decisions.yaml"

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
            'consumption_quantity',      # 6
            'consumption_frequency',     # 7
            'vendor_selection',          # 8
            'purchase_vs_bid',           # 9
            'bid_value',                 # 10
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
                      single_decision: Optional[str] = None, 
                      outcome_draws: int = 1) -> pd.DataFrame:
        """
        Run simulation using original participants with bootstrap sampling.
        
        If n_agents > original participants, bootstrap with replacement.
        If n_agents <= original participants, sample without replacement.
        """
        # Sample agents from original data
        rng = np.random.default_rng(seed)
        n_original = len(self.original_data)
        
        if n_agents > n_original:
            # Bootstrap with replacement on participants then repeat draws
            indices = rng.choice(n_original, size=n_agents, replace=True)
            agents_df = self.original_data.iloc[indices].reset_index(drop=True)
        else:
            indices = rng.choice(n_original, size=n_agents, replace=False)
            agents_df = self.original_data.iloc[indices].reset_index(drop=True)
        
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
            for rep in range(outcome_draws):  # repeat dependent-var draw
                agent_state = row.to_dict()
                if outcome_draws>1:
                    agent_state['draw_id']=rep+1
                agent_rng = np.random.default_rng(rng_global.integers(1e9))

                # we postpone decision execution until after loop to capture repeats
                # so move decision loop inside rep
                for decision_name in decisions_to_run:
                    if decision_name in self.decision_modules:
                        params = self.config.get(decision_name, {})
                        if decision_name == 'donation_default':
                            decision_output = self.decision_modules[decision_name](
                                agent_state, params, agent_rng, pop_context=self.pop_context
                            )
                        else:
                            decision_output = self.decision_modules[decision_name](
                                agent_state, params, agent_rng)
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
        return df
    
    def get_available_decisions(self) -> List[str]:
        """Return list of available decision modules."""
        return list(self.decision_modules.keys())
