# src/orchestrator_depvar.py
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List

# Import the merged data directly
from src.validate_traits import merged
from src.build_master_traits import get_master_trait_list

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "decisions.yaml"

class OrchestratorDepVar:
    """
    Orchestrator for dependent variable resampling - no trait sampling.
    
    Key approach:
    - Compute donation rates once for all 280 original participants
    - Store these 280 values as the empirical distribution
    - Generate larger populations by bootstrap resampling these values
    - No trait information is preserved - only the outcome distribution
    """
    
    def __init__(self):
        # Load decision configuration
        with open(CONFIG_PATH, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get required traits and load original data
        self.traits = get_master_trait_list()
        self.original_data = merged[self.traits].copy().dropna()
        print(f"Dependent variable mode: Computing donation rates for {len(self.original_data)} participants")
        
        # Compute donation rates for all 280 participants once
        self._compute_empirical_distribution()
    
    def _compute_empirical_distribution(self):
        """Compute donation rates for all original participants."""
        # Import decision module - use stochastic version for realistic variation
        try:
            from src.decisions.donation_default_stochastic import donation_default_stochastic
            decision_func = donation_default_stochastic
            print("Using stochastic version for donation computation")
        except ImportError:
            from src.decisions.donation_default import donation_default
            decision_func = donation_default
            print("Using deterministic version for donation computation")
        
        # Fixed seed for reproducible empirical distribution
        rng = np.random.default_rng(0)
        params = self.config.get('donation_default', {})
        
        # Compute donation rate for each participant
        donation_rates = []
        raw_donation_rates = []
        
        # Enable raw output for pre-computation
        params_with_raw = params.copy()
        params_with_raw.setdefault('stochastic', {})['raw_output'] = True
        
        for idx, row in self.original_data.iterrows():
            agent_state = row.to_dict()
            
            # Generate donation rate
            decision_output = decision_func(
                agent_state, 
                params_with_raw, 
                rng,
                pop_context='documentation'  # Use documentation mode logic
            )
            
            donation_rates.append(decision_output['donation_default'])
            if 'donation_default_raw' in decision_output:
                raw_donation_rates.append(decision_output['donation_default_raw'])
            else:
                raw_donation_rates.append(decision_output['donation_default'])
        
        # Store empirical distributions
        self.empirical_donations = np.array(donation_rates)
        self.empirical_donations_raw = np.array(raw_donation_rates)
        
        # Compute summary statistics
        self.empirical_stats = {
            'mean': self.empirical_donations.mean(),
            'std': self.empirical_donations.std(),
            'min': self.empirical_donations.min(),
            'max': self.empirical_donations.max(),
            'median': np.median(self.empirical_donations),
            'n_zeros': (self.empirical_donations == 0).sum()
        }
        
        print(f"Empirical distribution computed:")
        print(f"  Mean: {self.empirical_stats['mean']:.4f}")
        print(f"  Std:  {self.empirical_stats['std']:.4f}")
        print(f"  Range: [{self.empirical_stats['min']:.4f}, {self.empirical_stats['max']:.4f}]")
        print(f"  Zeros: {self.empirical_stats['n_zeros']}")
        
        # Also show raw stats
        print(f"Raw (pre-truncation) distribution:")
        print(f"  Mean: {self.empirical_donations_raw.mean():.4f}")
        print(f"  Std:  {self.empirical_donations_raw.std():.4f}")
        print(f"  Range: [{self.empirical_donations_raw.min():.4f}, {self.empirical_donations_raw.max():.4f}]")
        
        # Default to processed output
        self.use_raw_output = False
    
    def set_raw_output(self, use_raw: bool):
        """Set whether to use raw (pre-truncation) output."""
        self.use_raw_output = use_raw
        if use_raw:
            print("Dependent variable mode: Using RAW (pre-truncation) distribution")
    
    def run_simulation(self, n_agents: int, seed: int, 
                      single_decision: Optional[str] = None) -> pd.DataFrame:
        """
        Generate n_agents by resampling from empirical donation distribution.
        
        Args:
            n_agents: Number of agents to generate
            seed: Random seed for reproducible sampling
            single_decision: If specified, must be 'donation_default' (only decision supported)
        
        Returns:
            DataFrame with single column 'donation_default' containing resampled values
        """
        if single_decision and single_decision != 'donation_default':
            raise ValueError(f"Dependent variable mode only supports 'donation_default', not '{single_decision}'")
        
        # Bootstrap resample from empirical distribution
        rng = np.random.default_rng(seed)
        
        # Choose which distribution to resample from
        if self.use_raw_output:
            source_donations = self.empirical_donations_raw
            column_name = 'donation_default_raw'
        else:
            source_donations = self.empirical_donations
            column_name = 'donation_default'
        
        resampled_donations = rng.choice(
            source_donations, 
            size=n_agents, 
            replace=True  # Bootstrap with replacement
        )
        
        # Return as DataFrame with appropriate column name
        return pd.DataFrame({
            column_name: resampled_donations
        })
    
    def get_available_decisions(self) -> List[str]:
        """Return list of available decision modules."""
        # Only donation_default is supported in this mode
        return ['donation_default']
    
    def get_empirical_stats(self) -> dict:
        """Return statistics of the empirical distribution."""
        if self.use_raw_output:
            # Compute stats for raw distribution
            return {
                'mean': self.empirical_donations_raw.mean(),
                'std': self.empirical_donations_raw.std(),
                'min': self.empirical_donations_raw.min(),
                'max': self.empirical_donations_raw.max(),
                'median': np.median(self.empirical_donations_raw),
                'n_zeros': (self.empirical_donations_raw == 0).sum()
            }
        return self.empirical_stats.copy()
    
    def get_empirical_distribution(self) -> np.ndarray:
        """Return the original 280 donation rates."""
        if self.use_raw_output:
            return self.empirical_donations_raw.copy()
        return self.empirical_donations.copy()
