#!/usr/bin/env python3
"""
Test using the actual orchestrator to match frontend behavior exactly.
"""
import sys
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.orchestrator_doc_mode import OrchestratorDocMode

def run_with_orchestrator(income_mode: str, n_agents: int, seed: int):
    """Run using the actual orchestrator."""
    
    # Create orchestrator
    orchestrator = OrchestratorDocMode()
    
    # Set income mode in config
    orchestrator.config['disclose_income']['income_mode'] = income_mode
    
    # Print what we're using
    print(f"\nRunning with income_mode={income_mode}")
    print(f"  Config income_mode: {orchestrator.config['disclose_income']['income_mode']}")
    print(f"  Intercept: {orchestrator.config['disclose_income'].get('intercept', 0.1)}")
    print(f"  Sigma: {orchestrator.config['disclose_income'].get('stochastic', {}).get('sigma_value', 0)}")
    print(f"  Scale: {orchestrator.config['disclose_income'].get('stochastic', {}).get('scale_factor', 0.1)}")
    
    # Run simulation
    df = orchestrator.run_simulation(
        n_agents=n_agents,
        seed=seed,
        single_decision=['disclose_income']
    )
    
    # Count results
    if 'disclose_income' in df.columns:
        y_count = (df['disclose_income'] == 'Y').sum()
        n_count = (df['disclose_income'] == 'N').sum()
        total = len(df)
        pct = 100 * y_count / total
        print(f"  Results: Y={y_count} ({pct:.2f}%), N={n_count}")
        return y_count, total, pct
    else:
        print(f"  ERROR: disclose_income column not found!")
        print(f"  Available columns: {df.columns.tolist()}")
        return 0, 0, 0


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING WITH ACTUAL ORCHESTRATOR")
    print("=" * 80)
    
    N_AGENTS = 1000
    SEED = 42
    
    print(f"\nSettings: n_agents={N_AGENTS}, seed={SEED}")
    
    # Test categorical
    print("\n" + "=" * 40)
    print("CATEGORICAL MODE")
    print("=" * 40)
    cat_y, cat_total, cat_pct = run_with_orchestrator('categorical', N_AGENTS, SEED)
    
    # Test continuous
    print("\n" + "=" * 40)
    print("CONTINUOUS MODE")
    print("=" * 40)
    cont_y, cont_total, cont_pct = run_with_orchestrator('continuous', N_AGENTS, SEED)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Categorical: {cat_pct:.2f}% Y")
    print(f"Continuous:  {cont_pct:.2f}% Y")
    print(f"Difference:  {abs(cat_pct - cont_pct):.2f}%")
    
    # Also test with different seeds
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT SEEDS")
    print("=" * 80)
    
    for seed in [0, 1, 42, 100]:
        cat_y, cat_total, cat_pct = run_with_orchestrator('categorical', N_AGENTS, seed)
        cont_y, cont_total, cont_pct = run_with_orchestrator('continuous', N_AGENTS, seed)
        print(f"Seed {seed}: Categorical={cat_pct:.2f}%, Continuous={cont_pct:.2f}%, Diff={abs(cat_pct-cont_pct):.2f}%")
