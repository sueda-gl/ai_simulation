#!/usr/bin/env python3
"""
Direct test of disclose_income_stochastic for 1000 agents.
Compares categorical vs continuous modes using default frontend values.
"""
import sys
import numpy as np
import yaml
from pathlib import Path

# Add project root to path
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

# Print default values being used
print("=" * 70)
print("DEFAULT VALUES FROM CONFIG (decisions.yaml)")
print("=" * 70)
print(f"Intercept (β₀): {disclose_income_params.get('intercept', 0.1)}")
print(f"WOPB (observed_prosocial weight): {disclose_income_params.get('anchor_weights', {}).get('observed_prosocial', 0.25)}")
print(f"WPB (prosocial_weight): {disclose_income_params.get('anchor_weights', {}).get('prosocial_weight', 0.50)}")
print(f"Sigma value: {disclose_income_params.get('stochastic', {}).get('sigma_value', 0)}")
print(f"Scale factor: {disclose_income_params.get('stochastic', {}).get('scale_factor', 0.1)}")
print(f"Current income_mode in config: {disclose_income_params.get('income_mode', 'categorical')}")
print()

# Load original 280 participants
traits = get_master_trait_list()
original_data = merged[traits].copy().dropna()
print(f"Loaded {len(original_data)} original participants")

# Settings
N_AGENTS = 1000
SEED = 42

def run_test(income_mode: str, n_agents: int, seed: int):
    """Run disclose_income test for given income mode."""
    print(f"\n{'=' * 70}")
    print(f"TESTING: {income_mode.upper()} MODE")
    print(f"{'=' * 70}")
    
    # Create RNG
    rng = np.random.default_rng(seed)
    
    # Bootstrap sample agents
    n_original = len(original_data)
    if n_agents > n_original:
        indices = rng.choice(n_original, size=n_agents, replace=True)
    else:
        indices = rng.choice(n_original, size=n_agents, replace=False)
    
    agents_df = original_data.iloc[indices].reset_index(drop=True)
    
    # Create params with specific income mode
    params = disclose_income_params.copy()
    params['income_mode'] = income_mode
    
    print(f"Using income_mode: {params['income_mode']}")
    
    # Generate incomes for continuous mode
    income_rng = np.random.default_rng(seed + 999999)
    all_incomes = []
    for idx, row in agents_df.iterrows():
        # Simple income generation (similar to what the orchestrator does)
        income = income_rng.lognormal(mean=10.5, sigma=0.5) * 100  # Rough approximation
        all_incomes.append(income)
    
    income_median = float(np.median(all_incomes))
    income_mean = float(np.mean(all_incomes))
    income_sd = float(np.std(all_incomes))
    print(f"Generated income median: ${income_median:,.2f}")
    print(f"Generated income stats: mean=${income_mean:,.2f}, sd=${income_sd:,.2f}")
    
    # Create simulation config
    simulation_config = {
        'income_median': income_median,
        'income_stats': {
            'mean': income_mean,
            'sd': income_sd
        }
    }
    
    # ========================================================================
    # THREE-PASS APPROACH FOR CORRECT POPULATION STANDARDIZATION
    # ========================================================================
    
    # ========== PASS 1: Compute weighted_prosocial and direct_effect ==========
    print("\nPass 1: Computing weighted_prosocial and direct_effect...")
    
    simulation_config['disclose_income_cache'] = {}
    pass1_values_list = []
    
    for idx, row in agents_df.iterrows():
        agent_state = row.to_dict()
        agent_state['income'] = all_incomes[idx]
        
        pass1_values = compute_pass1_values(
            agent_state, 
            params,
            simulation_config
        )
        pass1_values_list.append(pass1_values)
        simulation_config['disclose_income_cache'][idx] = pass1_values.copy()
    
    # Compute Pass 1 statistics
    wp_values = [v['weighted_prosocial'] for v in pass1_values_list]
    de_values = [v['direct_effect'] for v in pass1_values_list]
    income_highs = [v['income_high'] for v in pass1_values_list]
    
    wp_stats = {'mean': float(np.mean(wp_values)), 'sd': float(np.std(wp_values))}
    de_stats = {'mean': float(np.mean(de_values)), 'sd': float(np.std(de_values))}
    
    print(f"  weighted_prosocial: mean={wp_stats['mean']:.6f}, sd={wp_stats['sd']:.6f}")
    print(f"  direct_effect: mean={de_stats['mean']:.6f}, sd={de_stats['sd']:.6f}")
    print(f"  income_high count: {sum(income_highs)} / {len(income_highs)} ({100*sum(income_highs)/len(income_highs):.1f}%)")
    
    # ========== PASS 2: Compute anchored_pb using correct z_weighted_prosocial ==========
    print("\nPass 2: Computing anchored_pb with z-scored weighted_prosocial...")
    
    anchored_pb_values = []
    for cache_idx in range(len(pass1_values_list)):
        pass1_values = simulation_config['disclose_income_cache'][cache_idx]
        anchored_pb = compute_pass2_anchored_pb(pass1_values, wp_stats, params)
        anchored_pb_values.append(anchored_pb)
        simulation_config['disclose_income_cache'][cache_idx]['anchored_pb'] = anchored_pb
    
    # Compute Pass 2 statistics
    ap_stats = {'mean': float(np.mean(anchored_pb_values)), 'sd': float(np.std(anchored_pb_values))}
    
    print(f"  anchored_pb: mean={ap_stats['mean']:.6f}, sd={ap_stats['sd']:.6f}")
    
    # Store all population stats
    simulation_config['disclose_income_population_stats'] = {
        'weighted_prosocial': wp_stats,
        'direct_effect': de_stats,
        'anchored_pb': ap_stats
    }
    
    # ========================================================================
    # PASS 3: Run disclose_income_stochastic for all agents using cached values
    # ========================================================================
    print("\nPass 3: Running disclose_income_stochastic for all agents...")
    
    results = []
    decision_rng = np.random.default_rng(seed + 1000)  # Separate RNG for decisions
    
    for idx, row in agents_df.iterrows():
        agent_state = row.to_dict()
        agent_state['income'] = all_incomes[idx]
        agent_state['_cache_index'] = idx  # Pass cache index for cached values
        
        # Create agent-specific RNG (similar to orchestrator)
        agent_rng = np.random.default_rng(decision_rng.integers(1e9))
        
        result = disclose_income_stochastic(
            agent_state,
            params,
            agent_rng,
            simulation_config
        )
        results.append(result['disclose_income'])
    
    # Count results
    y_count = sum(1 for r in results if r == 'Y')
    n_count = sum(1 for r in results if r == 'N')
    
    print(f"\n{'=' * 40}")
    print(f"RESULTS FOR {income_mode.upper()} MODE:")
    print(f"{'=' * 40}")
    print(f"  Total agents: {len(results)}")
    print(f"  Disclose Y: {y_count} ({100*y_count/len(results):.2f}%)")
    print(f"  Disclose N: {n_count} ({100*n_count/len(results):.2f}%)")
    
    return y_count, n_count, len(results)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DISCLOSE INCOME DIRECT TEST - 1000 AGENTS")
    print("=" * 70)
    
    # Run categorical test
    cat_y, cat_n, cat_total = run_test('categorical', N_AGENTS, SEED)
    
    # Run continuous test
    cont_y, cont_n, cont_total = run_test('continuous', N_AGENTS, SEED)
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<15} {'Y Count':<10} {'N Count':<10} {'Y %':<10} {'N %':<10}")
    print("-" * 55)
    print(f"{'Categorical':<15} {cat_y:<10} {cat_n:<10} {100*cat_y/cat_total:<10.2f} {100*cat_n/cat_total:<10.2f}")
    print(f"{'Continuous':<15} {cont_y:<10} {cont_n:<10} {100*cont_y/cont_total:<10.2f} {100*cont_n/cont_total:<10.2f}")
    print("-" * 55)
    print(f"Difference in Y%: {abs(100*cat_y/cat_total - 100*cont_y/cont_total):.2f}%")
