#!/usr/bin/env python3
"""
Test script to run disclose_income decision from backend
using Research Specification mode (OrchestratorDocMode).

This tests both categorical and continuous income modes
to compare with frontend results.
"""

import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.orchestrator_doc_mode import OrchestratorDocMode

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "decisions.yaml"


def run_disclose_income_test(income_mode: str, n_agents: int = 100, seed: int = 42):
    """
    Run disclose_income decision with specified income mode.
    
    Args:
        income_mode: 'Categorical only' or 'Continuous only'
        n_agents: Number of agents to simulate
        seed: Random seed for reproducibility
    """
    print(f"\n{'='*60}")
    print(f"Running disclose_income with income_mode = '{income_mode}'")
    print(f"Agents: {n_agents}, Seed: {seed}")
    print(f"{'='*60}\n")
    
    # Create orchestrator (Research Specification mode)
    orchestrator = OrchestratorDocMode()
    
    # Print which decision module is loaded
    print(f"Decision module loaded: {orchestrator.decision_modules.get('disclose_income')}")
    
    # Load and display current YAML config for disclose_income
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    di_config = config.get('disclose_income', {})
    print(f"\n--- Current YAML Configuration ---")
    print(f"income_mode: {di_config.get('income_mode')}")
    print(f"intercept: {di_config.get('intercept')}")
    print(f"anchor_weights: {di_config.get('anchor_weights')}")
    print(f"stochastic: {di_config.get('stochastic')}")
    
    # Override income_mode in orchestrator's config
    orchestrator.config['disclose_income']['income_mode'] = income_mode
    print(f"\n--- Overriding income_mode to: {income_mode} ---")
    
    # Run simulation for disclose_income only
    print(f"\nRunning simulation...")
    results_df = orchestrator.run_simulation(
        n_agents=n_agents,
        seed=seed,
        single_decision=['disclose_income']
    )
    
    # Analyze results
    print(f"\n--- Results ---")
    
    if 'disclose_income' in results_df.columns:
        value_counts = results_df['disclose_income'].value_counts()
        total = len(results_df)
        
        print(f"Total agents: {total}")
        print(f"\nDisclose Income Distribution:")
        for value, count in value_counts.items():
            pct = count / total * 100
            print(f"  {value}: {count} ({pct:.1f}%)")
        
        # Calculate Y percentage
        y_count = value_counts.get('Y', 0)
        n_count = value_counts.get('N', 0)
        y_pct = y_count / total * 100 if total > 0 else 0
        
        print(f"\n  Y percentage: {y_pct:.2f}%")
        print(f"  N percentage: {100 - y_pct:.2f}%")
        
        # Check if results look like 50/50 random (default slider behavior)
        if 45 <= y_pct <= 55:
            print(f"\n⚠️  WARNING: Results are close to 50/50 - might be using default random slider!")
        else:
            print(f"\n✓ Results are NOT 50/50 - likely using research model")
    else:
        print("ERROR: 'disclose_income' column not found in results!")
        print(f"Available columns: {list(results_df.columns)}")
    
    return results_df


def check_orchestrator_module_loading():
    """Check which module is loaded for disclose_income in different orchestrators."""
    print("\n" + "="*60)
    print("CHECKING WHICH MODULES ARE LOADED")
    print("="*60)
    
    # Check OrchestratorDocMode
    from src.orchestrator_doc_mode import OrchestratorDocMode
    doc_orch = OrchestratorDocMode()
    doc_module = doc_orch.decision_modules.get('disclose_income')
    print(f"\nOrchestratorDocMode (Research Spec):")
    print(f"  Module: {doc_module}")
    print(f"  Module name: {doc_module.__name__ if doc_module else 'None'}")
    
    # Check regular Orchestrator
    from src.orchestrator import Orchestrator
    reg_orch = Orchestrator()
    reg_module = reg_orch.decision_modules.get('disclose_income')
    print(f"\nOrchestrator (Copula):")
    print(f"  Module: {reg_module}")
    print(f"  Module name: {reg_module.__name__ if reg_module else 'None'}")
    
    # Check OrchestratorBaseline
    from src.orchestrator_baseline import OrchestratorBaseline
    base_orch = OrchestratorBaseline()
    base_module = base_orch.decision_modules.get('disclose_income')
    print(f"\nOrchestratorBaseline (Research Baseline):")
    print(f"  Module: {base_module}")
    print(f"  Module name: {base_module.__name__ if base_module else 'None'}")


def main():
    """Main entry point."""
    print("\n" + "#"*60)
    print("# DISCLOSE INCOME BACKEND TEST")
    print("#"*60)
    
    # First, check which modules are loaded
    check_orchestrator_module_loading()
    
    # Test parameters
    n_agents = 100
    seed = 42
    
    # Run with Categorical mode
    results_cat = run_disclose_income_test(
        income_mode='Categorical only',
        n_agents=n_agents,
        seed=seed
    )
    
    # Run with Continuous mode
    results_cont = run_disclose_income_test(
        income_mode='Continuous only',
        n_agents=n_agents,
        seed=seed
    )
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    cat_y_pct = (results_cat['disclose_income'] == 'Y').mean() * 100
    cont_y_pct = (results_cont['disclose_income'] == 'Y').mean() * 100
    
    print(f"\nCategorical mode - Y%: {cat_y_pct:.2f}%")
    print(f"Continuous mode  - Y%: {cont_y_pct:.2f}%")
    print(f"Difference: {abs(cat_y_pct - cont_y_pct):.2f} percentage points")
    
    if abs(cat_y_pct - 50) < 5 and abs(cont_y_pct - 50) < 5:
        print("\n⚠️  BOTH results are close to 50% - likely using default random behavior!")
    elif abs(cat_y_pct - cont_y_pct) < 1:
        print("\n⚠️  Results are nearly identical - income mode may not be affecting output")
    else:
        print("\n✓ Results differ between modes - research model appears to be working")
    
    print("\n" + "#"*60)
    print("# TEST COMPLETE")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
