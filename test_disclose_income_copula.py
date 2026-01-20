#!/usr/bin/env python3
"""
Test script to run disclose_income decision using COPULA mode
to demonstrate the difference from Research Specification mode.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.orchestrator import Orchestrator
from src.trait_engine import TraitEngine


def run_copula_disclose_income(n_agents: int = 100, seed: int = 42):
    """Run disclose_income with Copula orchestrator (Copula mode)."""
    
    print("\n" + "="*60)
    print("COPULA MODE TEST (Regular Orchestrator)")
    print("="*60)
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Check which module is loaded
    module = orchestrator.decision_modules.get('disclose_income')
    print(f"\nDecision module loaded: {module}")
    print(f"Module name: {module.__name__ if module else 'None'}")
    
    # Sample agents using copula
    trait_engine = TraitEngine()
    agents_df = trait_engine.sample(n_agents, seed)
    
    print(f"\nRunning simulation with {n_agents} agents, seed={seed}...")
    
    # Run simulation
    results_df = orchestrator.run_simulation(
        n_agents=n_agents,
        seed=seed,
        single_decision=['disclose_income'],
        agents_df=agents_df
    )
    
    # Analyze results
    if 'disclose_income' in results_df.columns:
        value_counts = results_df['disclose_income'].value_counts()
        total = len(results_df)
        
        print(f"\n--- Results ---")
        print(f"Total agents: {total}")
        print(f"\nDisclose Income Distribution:")
        for value, count in value_counts.items():
            pct = count / total * 100
            print(f"  {value}: {count} ({pct:.1f}%)")
        
        y_pct = (results_df['disclose_income'] == 'Y').mean() * 100
        print(f"\n  Y percentage: {y_pct:.2f}%")
        
        if 45 <= y_pct <= 55:
            print(f"\n⚠️  Results are close to 50/50 - using DEFAULT RANDOM slider!")
        else:
            print(f"\n✓ Results are NOT 50/50 - research model might be in use")
    
    return results_df


def main():
    print("\n" + "#"*60)
    print("# COPULA MODE DISCLOSE INCOME TEST")
    print("# This uses the regular Orchestrator (not DocMode)")
    print("#"*60)
    
    results = run_copula_disclose_income(n_agents=100, seed=42)
    
    print("\n" + "#"*60)
    print("# KEY FINDING:")
    print("# If Copula mode shows ~50/50, but Research Spec shows 58/42,")
    print("# it confirms Copula uses the WRONG module (simple random).")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
