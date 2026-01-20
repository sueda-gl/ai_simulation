#!/usr/bin/env python3
"""
Debug test to match frontend behavior exactly.
Tests different seed values and configurations.
"""
import sys
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.decisions.disclose_income_stochastic import (
    compute_pass1_values,
    compute_pass2_anchored_pb,
    disclose_income_stochastic
)
from src.validate_traits import merged
from src.build_master_traits import get_master_trait_list

# Load config
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "decisions.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

disclose_income_params = config.get('disclose_income', {})

# Load original 280 participants
traits = get_master_trait_list()
original_data = merged[traits].copy().dropna()

def run_test_with_params(income_mode: str, n_agents: int, seed: int, 
                         intercept: float, wopb: float, wpb: float,
                         sigma_value: float, scale_factor: float,
                         random_sample: bool = True):
    """Run test with specific parameters."""
    
    # Create setup RNG (like orchestrator)
    rng_setup = np.random.default_rng(seed)
    
    # Sample agents (matching orchestrator logic)
    n_original = len(original_data)
    if n_agents > n_original:
        if random_sample:
            indices = rng_setup.choice(n_original, size=n_agents, replace=True)
        else:
            # Deterministic wrapping for baseline
            full_repeats = n_agents // n_original
            remainder = n_agents % n_original
            indices = list(range(n_original)) * full_repeats + list(range(remainder))
    else:
        if random_sample:
            indices = rng_setup.choice(n_original, size=n_agents, replace=False)
        else:
            indices = list(range(n_agents))
    
    agents_df = original_data.iloc[indices].reset_index(drop=True)
    
    # Create params
    params = disclose_income_params.copy()
    params['income_mode'] = income_mode
    params['intercept'] = intercept
    params['anchor_weights'] = {
        'observed_prosocial': wopb,
        'prosocial_weight': wpb
    }
    params['stochastic'] = {
        'sigma_value': sigma_value,
        'scale_factor': scale_factor,
        'sigma_strategy': 'overall',
        'sigma_overall': 9.899547
    }
    
    # Generate incomes using same RNG pattern as orchestrator
    # Orchestrator uses: rng_pass1 = np.random.default_rng(seed + 1000000)
    rng_pass1 = np.random.default_rng(seed + 1000000)
    all_incomes = []
    agent_base_seeds = []
    
    for idx, row in agents_df.iterrows():
        agent_base_seed = rng_pass1.integers(1e9)
        agent_base_seeds.append(agent_base_seed)
        
        # Simple income generation
        income_rng = np.random.default_rng(agent_base_seed + 999999)
        income = income_rng.lognormal(mean=10.5, sigma=0.5) * 100
        all_incomes.append(income)
    
    income_median = float(np.median(all_incomes))
    
    simulation_config = {
        'income_median': income_median
    }
    
    # Three-pass approach for correct population standardization
    
    # Pass 1: Compute weighted_prosocial and direct_effect
    pass1_values_list = []
    simulation_config['disclose_income_cache'] = {}
    income_idx = 0
    for idx, row in agents_df.iterrows():
        agent_state = row.to_dict()
        agent_state['income'] = all_incomes[income_idx]
        pass1_values = compute_pass1_values(agent_state, params, simulation_config)
        pass1_values_list.append(pass1_values)
        simulation_config['disclose_income_cache'][income_idx] = pass1_values.copy()
        income_idx += 1
    
    # Compute Pass 1 statistics
    wp_values = [v['weighted_prosocial'] for v in pass1_values_list]
    de_values = [v['direct_effect'] for v in pass1_values_list]
    wp_stats = {'mean': float(np.mean(wp_values)), 'sd': float(np.std(wp_values))}
    de_stats = {'mean': float(np.mean(de_values)), 'sd': float(np.std(de_values))}
    
    # Pass 2: Compute anchored_pb using correct z_weighted_prosocial
    anchored_pb_values = []
    for cache_idx in range(len(pass1_values_list)):
        pass1_values = simulation_config['disclose_income_cache'][cache_idx]
        anchored_pb = compute_pass2_anchored_pb(pass1_values, wp_stats, params)
        anchored_pb_values.append(anchored_pb)
        simulation_config['disclose_income_cache'][cache_idx]['anchored_pb'] = anchored_pb
    
    # Compute Pass 2 statistics  
    ap_stats = {'mean': float(np.mean(anchored_pb_values)), 'sd': float(np.std(anchored_pb_values))}
    
    simulation_config['disclose_income_population_stats'] = {
        'weighted_prosocial': wp_stats,
        'direct_effect': de_stats,
        'anchored_pb': ap_stats
    }
    
    # Pass 3: Run decisions using cached values
    results = []
    agent_idx = 0
    rng_global = np.random.default_rng(seed + 1000000)  # Reset to match orchestrator
    
    for idx, row in agents_df.iterrows():
        agent_base_seed = rng_global.integers(1e9)  # This should match agent_base_seeds
        
        agent_state = row.to_dict()
        agent_state['income'] = all_incomes[agent_idx]
        agent_state['_cache_index'] = agent_idx  # Pass cache index for cached values
        
        # Decision-specific RNG (like orchestrator: decision_seed = agent_base_seed + decision_index[decision_name] * 1000)
        # disclose_income is index 0, so decision_seed = agent_base_seed + 0
        decision_rng = np.random.default_rng(agent_base_seed)
        
        result = disclose_income_stochastic(agent_state, params, decision_rng, simulation_config)
        results.append(result['disclose_income'])
        agent_idx += 1
    
    y_count = sum(1 for r in results if r == 'Y')
    return y_count, len(results), y_count / len(results) * 100


if __name__ == "__main__":
    print("=" * 80)
    print("DEBUGGING DISCLOSE INCOME - MATCHING FRONTEND PARAMETERS")
    print("=" * 80)
    
    # Test with different configurations
    configs = [
        # (name, n_agents, seed, intercept, wopb, wpb, sigma_value, scale_factor)
        ("Default (my test)", 1000, 42, 0.1, 0.25, 0.5, 9.8995, 0.13),
        ("Seed 0", 1000, 0, 0.1, 0.25, 0.5, 9.8995, 0.13),
        ("Seed 1", 1000, 1, 0.1, 0.25, 0.5, 9.8995, 0.13),
        ("No stochastic", 1000, 42, 0.1, 0.25, 0.5, 0, 0),
        ("100 agents", 100, 42, 0.1, 0.25, 0.5, 9.8995, 0.13),
        ("280 agents", 280, 42, 0.1, 0.25, 0.5, 9.8995, 0.13),
        ("Lower intercept 0.0", 1000, 42, 0.0, 0.25, 0.5, 9.8995, 0.13),
        ("Higher scale 1.0", 1000, 42, 0.1, 0.25, 0.5, 9.8995, 1.0),
    ]
    
    print(f"\n{'Configuration':<25} {'Categorical %':<15} {'Continuous %':<15} {'Diff':<10}")
    print("-" * 70)
    
    for name, n_agents, seed, intercept, wopb, wpb, sigma_val, scale in configs:
        cat_y, cat_total, cat_pct = run_test_with_params(
            'categorical', n_agents, seed, intercept, wopb, wpb, sigma_val, scale
        )
        cont_y, cont_total, cont_pct = run_test_with_params(
            'continuous', n_agents, seed, intercept, wopb, wpb, sigma_val, scale
        )
        diff = abs(cat_pct - cont_pct)
        print(f"{name:<25} {cat_pct:<15.2f} {cont_pct:<15.2f} {diff:<10.2f}")
    
    print("\n" + "=" * 80)
    print("CHECKING SPECIFIC SEEDS TO FIND 51.5%")
    print("=" * 80)
    
    # Try to find what seed produces ~51.5%
    for seed in range(10):
        cat_y, cat_total, cat_pct = run_test_with_params(
            'categorical', 1000, seed, 0.1, 0.25, 0.5, 9.8995, 0.13
        )
        cont_y, cont_total, cont_pct = run_test_with_params(
            'continuous', 1000, seed, 0.1, 0.25, 0.5, 9.8995, 0.13
        )
        if 51 <= cat_pct <= 52.5 or 51 <= cont_pct <= 52.5:
            print(f"Seed {seed}: Categorical={cat_pct:.2f}%, Continuous={cont_pct:.2f}%")
