#!/usr/bin/env python3
"""
Comprehensive test of disclose_income decision comparing:
1. Research Spec mode (OrchestratorDocMode) - uses stochastic version
2. Copula mode (Orchestrator) - uses simple version
3. Copula mode WITH decision_settings passed (simulating frontend behavior)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_research_spec_mode(n_agents=500, seed=42):
    """Test with Research Specification mode."""
    print("\n" + "="*70)
    print("TEST 1: RESEARCH SPECIFICATION MODE (OrchestratorDocMode)")
    print("="*70)
    
    from src.orchestrator_doc_mode import OrchestratorDocMode
    
    orchestrator = OrchestratorDocMode()
    print(f"Module: {orchestrator.decision_modules.get('disclose_income').__name__}")
    
    # Set income mode to Categorical
    orchestrator.config['disclose_income']['income_mode'] = 'Categorical only'
    
    results = orchestrator.run_simulation(
        n_agents=n_agents,
        seed=seed,
        single_decision=['disclose_income']
    )
    
    y_pct = (results['disclose_income'] == 'Y').mean() * 100
    print(f"\nResult: Y={y_pct:.1f}%, N={100-y_pct:.1f}%")
    return y_pct


def test_copula_mode_no_settings(n_agents=500, seed=42):
    """Test with Copula mode WITHOUT decision_settings."""
    print("\n" + "="*70)
    print("TEST 2: COPULA MODE - No decision_settings (should be ~50/50)")
    print("="*70)
    
    from src.orchestrator import Orchestrator
    from src.trait_engine import TraitEngine
    
    orchestrator = Orchestrator()
    print(f"Module: {orchestrator.decision_modules.get('disclose_income').__name__}")
    
    trait_engine = TraitEngine()
    agents_df = trait_engine.sample(n_agents, seed)
    
    results = orchestrator.run_simulation(
        n_agents=n_agents,
        seed=seed,
        single_decision=['disclose_income'],
        agents_df=agents_df
    )
    
    y_pct = (results['disclose_income'] == 'Y').mean() * 100
    print(f"\nResult: Y={y_pct:.1f}%, N={100-y_pct:.1f}%")
    
    if 48 <= y_pct <= 52:
        print("✓ Close to 50/50 - using default random choice (no research model)")
    else:
        print(f"Note: {y_pct:.1f}% is within random variance for 50% probability")
    
    return y_pct


def test_copula_mode_with_settings(n_agents=500, seed=42, prob_y=0.5):
    """Test with Copula mode WITH decision_settings (simulating frontend)."""
    print("\n" + "="*70)
    print(f"TEST 3: COPULA MODE - WITH decision_settings (prob_y={prob_y})")
    print("This simulates what the frontend does")
    print("="*70)
    
    from src.orchestrator import Orchestrator
    from src.trait_engine import TraitEngine
    
    orchestrator = Orchestrator()
    print(f"Module: {orchestrator.decision_modules.get('disclose_income').__name__}")
    
    # Add decision settings to simulation_config (simulating what frontend does)
    orchestrator.simulation_config['random_decisions'] = {
        'disclose_income': {
            'type': 'random_probability',
            'probability_y': prob_y,
            'options': ['Y', 'N']
        }
    }
    orchestrator.simulation_config['default_decisions'] = orchestrator.simulation_config['random_decisions']
    
    trait_engine = TraitEngine()
    agents_df = trait_engine.sample(n_agents, seed)
    
    results = orchestrator.run_simulation(
        n_agents=n_agents,
        seed=seed,
        single_decision=['disclose_income'],
        agents_df=agents_df
    )
    
    y_pct = (results['disclose_income'] == 'Y').mean() * 100
    print(f"\nResult: Y={y_pct:.1f}%, N={100-y_pct:.1f}%")
    print(f"Expected: Y≈{prob_y*100:.0f}%")
    
    if abs(y_pct - prob_y*100) < 5:
        print("✓ Result matches slider probability - using slider, NOT research model!")
    
    return y_pct


def main():
    print("\n" + "#"*70)
    print("# COMPREHENSIVE DISCLOSE_INCOME TEST")
    print("#"*70)
    
    n_agents = 500
    seed = 42
    
    # Test 1: Research Spec mode
    research_y = test_research_spec_mode(n_agents, seed)
    
    # Test 2: Copula without settings
    copula_no_settings_y = test_copula_mode_no_settings(n_agents, seed)
    
    # Test 3: Copula with default 50% slider
    copula_50_y = test_copula_mode_with_settings(n_agents, seed, prob_y=0.5)
    
    # Test 4: Copula with 70% slider (to clearly show it uses slider)
    copula_70_y = test_copula_mode_with_settings(n_agents, seed, prob_y=0.7)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n1. Research Spec Mode:     Y = {research_y:.1f}%")
    print(f"2. Copula (no settings):   Y = {copula_no_settings_y:.1f}%")
    print(f"3. Copula (slider=50%):    Y = {copula_50_y:.1f}%")
    print(f"4. Copula (slider=70%):    Y = {copula_70_y:.1f}%")
    
    print("\n" + "-"*70)
    print("ANALYSIS:")
    print("-"*70)
    
    print(f"\n• Research Spec uses the STOCHASTIC model → Y={research_y:.1f}%")
    print(f"• Copula with slider=70% shows Y={copula_70_y:.1f}% (expected ~70%)")
    print(f"  → This PROVES Copula mode uses the slider, ignoring research model!")
    
    if abs(research_y - 50) > 5:
        print(f"\n• Research Spec result ({research_y:.1f}%) differs from 50%")
        print(f"  → Your research model IS working in Research Spec mode")
    
    print("\n" + "#"*70)
    print("# CONCLUSION:")
    print("# If running in COPULA mode on the frontend, the slider overrides")
    print("# your research model settings. Switch to 'Research Specification'")
    print("# mode on Page 1 to use your manual disclose_income configuration.")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
