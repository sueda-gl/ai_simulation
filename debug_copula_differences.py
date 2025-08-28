#!/usr/bin/env python3
"""
Diagnostic script to identify differences in copula calculations between environments.
Run this on both local and deployed versions to compare results.
"""

import numpy as np
import pandas as pd
import sys
import json
import platform
from pathlib import Path
import hashlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from src.orchestrator import Orchestrator
from src.trait_engine import TraitEngine
from src.orchestrator_doc_mode import OrchestratorDocMode

def get_file_hash(filepath):
    """Get MD5 hash of a file."""
    if Path(filepath).exists():
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    return "FILE_NOT_FOUND"

def run_diagnostics():
    """Run comprehensive diagnostics for copula calculations."""
    
    print("=" * 80)
    print("COPULA CALCULATION DIAGNOSTICS")
    print("=" * 80)
    
    # 1. Environment info
    print("\n1. ENVIRONMENT INFORMATION:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    
    # Check import versions
    try:
        import scipy
        print(f"SciPy version: {scipy.__version__}")
    except:
        print("SciPy: NOT INSTALLED")
    
    try:
        import joblib
        print(f"Joblib version: {joblib.__version__}")
    except:
        print("Joblib: NOT INSTALLED")
    
    # 2. File integrity checks
    print("\n2. FILE INTEGRITY CHECKS:")
    critical_files = [
        "config/trait_model.pkl",
        "config/decisions.yaml",
        "src/trait_engine.py",
        "src/orchestrator.py",
        "src/decisions/donation_default.py"
    ]
    
    for filepath in critical_files:
        file_hash = get_file_hash(filepath)
        print(f"{filepath}: {file_hash}")
    
    # 3. Trait model inspection
    print("\n3. TRAIT MODEL INSPECTION:")
    try:
        trait_engine = TraitEngine()
        print(f"Number of traits: {len(trait_engine.traits)}")
        print(f"Traits: {trait_engine.traits}")
        print(f"Correlation matrix shape: {trait_engine.Sigma.shape}")
        print(f"Correlation matrix diagonal mean: {np.diag(trait_engine.Sigma).mean():.6f}")
        print(f"Correlation matrix off-diagonal abs mean: {np.abs(trait_engine.Sigma - np.diag(np.diag(trait_engine.Sigma))).mean():.6f}")
        
        # Check eigenvalues
        eigenvals = np.linalg.eigvals(trait_engine.Sigma)
        print(f"Min eigenvalue: {eigenvals.min():.6f}")
        print(f"Max eigenvalue: {eigenvals.max():.6f}")
        
    except Exception as e:
        print(f"ERROR loading trait model: {e}")
    
    # 4. Sample generation test
    print("\n4. SAMPLE GENERATION TEST:")
    seeds_to_test = [42, 123, 999]
    n_test_agents = 10
    
    for seed in seeds_to_test:
        print(f"\nSeed {seed}:")
        try:
            # Test copula sampling
            orchestrator = Orchestrator()
            df_copula = orchestrator.run_simulation(n_agents=n_test_agents, seed=seed, single_decision='donation_default')
            
            print(f"  Copula mode:")
            print(f"    - donation_default mean: {df_copula['donation_default'].mean():.6f}")
            print(f"    - donation_default std: {df_copula['donation_default'].std():.6f}")
            print(f"    - donation_default min: {df_copula['donation_default'].min():.6f}")
            print(f"    - donation_default max: {df_copula['donation_default'].max():.6f}")
            
            # Show first 3 agent values
            print(f"    - First 3 values: {df_copula['donation_default'].head(3).tolist()}")
            
            # Check key traits for first agent
            first_agent = df_copula.iloc[0]
            print(f"    - First agent traits:")
            print(f"      * Honesty_Humility: {first_agent['Honesty_Humility']:.6f}")
            print(f"      * Income level: {first_agent['Assigned Allowance Level']}")
            print(f"      * Study Program: {first_agent['Study Program']}")
            print(f"      * Group: {first_agent['Group_experiment']}")
            print(f"      * TWT+Sospeso: {first_agent['TWT+Sospeso [=AW2+AX2]{Periods 1+2}']:.6f}")
            
        except Exception as e:
            print(f"  ERROR in copula mode: {e}")
        
        try:
            # Test documentation mode
            orchestrator_doc = OrchestratorDocMode()
            df_doc = orchestrator_doc.run_simulation(n_agents=n_test_agents, seed=seed, single_decision='donation_default')
            
            print(f"  Documentation mode:")
            print(f"    - donation_default mean: {df_doc['donation_default'].mean():.6f}")
            print(f"    - donation_default std: {df_doc['donation_default'].std():.6f}")
            
        except Exception as e:
            print(f"  ERROR in documentation mode: {e}")
    
    # 5. Configuration check
    print("\n5. CONFIGURATION CHECK:")
    try:
        with open("config/decisions.yaml", 'r') as f:
            import yaml
            config = yaml.safe_load(f)
            
        donation_config = config.get('donation_default', {})
        print(f"Donation default configuration:")
        print(f"  - anchor_weights: {donation_config.get('anchor_weights', {})}")
        print(f"  - stochastic.in_copula: {donation_config.get('stochastic', {}).get('in_copula', 'NOT SET')}")
        print(f"  - stochastic.sigma_value: {donation_config.get('stochastic', {}).get('sigma_value', 'NOT SET')}")
        print(f"  - regression.income_mode: {donation_config.get('regression', {}).get('income_mode', 'NOT SET')}")
        
    except Exception as e:
        print(f"ERROR loading config: {e}")
    
    # 6. Random state verification
    print("\n6. RANDOM STATE VERIFICATION:")
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    vals1 = [rng1.random() for _ in range(5)]
    vals2 = [rng2.random() for _ in range(5)]
    print(f"RNG consistency check: {'PASS' if vals1 == vals2 else 'FAIL'}")
    print(f"First 5 random values with seed=42: {vals1}")
    
    print("\n" + "=" * 80)
    print("END OF DIAGNOSTICS")
    print("=" * 80)

if __name__ == "__main__":
    run_diagnostics()
