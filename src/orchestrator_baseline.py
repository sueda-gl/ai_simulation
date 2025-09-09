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
            'consumption_frequency',     # 7
            'vendor_selection',          # 8
            'purchase_vs_bid',           # 9
            'bid_value',                 # 10
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
                      single_decision: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Run simulation for n_agents using original participants with no stochastic component.
        
        If n_agents <= 280, use first n_agents participants.
        If n_agents > 280, bootstrap sample from the 280 participants.
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
        
        # Sample agents from original data
        rng = np.random.default_rng(seed)
        
        if n_agents <= len(self.original_data):
            # Use first n_agents participants
            agents_df = self.original_data.iloc[:n_agents].copy()
        else:
            # Bootstrap sample to reach n_agents
            indices = rng.choice(len(self.original_data), size=n_agents, replace=True)
            agents_df = self.original_data.iloc[indices].copy()
            agents_df.index = range(len(agents_df))  # Reset index
        
        # Process each agent
        results = []
        
        for idx, row in agents_df.iterrows():
            agent_state = row.to_dict()
            agent_results = agent_state.copy()
            
            # Generate unique seed for this agent
            agent_seed = seed + idx
            agent_rng = np.random.default_rng(agent_seed)
            
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
                        decision_output = decision_func(
                            agent_state, 
                            decision_params, 
                            agent_rng,
                            simulation_config=None,
                            pop_context=self.pop_context
                        )
                        
                        # Update agent state with decision outputs
                        if isinstance(decision_output, dict):
                            agent_results.update(decision_output)
                            agent_state.update(decision_output)
                        
                    except Exception as e:
                        print(f"Error in decision {decision_name} for agent {idx}: {e}")
                        # Use default value for failed decisions
                        default_field = decision_params.get('output_field', decision_name)
                        default_value = decision_params.get('default_value', 'NA')
                        agent_results[default_field] = default_value
                        agent_state[default_field] = default_value
                else:
                    # Decision module not loaded - use default
                    decision_params = self.config.get(decision_name, {})
                    default_field = decision_params.get('output_field', decision_name)
                    default_value = decision_params.get('default_value', 'NA')
                    agent_results[default_field] = default_value
                    agent_state[default_field] = default_value
            
            results.append(agent_results)
        
        return pd.DataFrame(results)
