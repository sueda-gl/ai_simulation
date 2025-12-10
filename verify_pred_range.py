
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import data loader
from src.validate_traits import merged
from src.build_master_traits import get_master_trait_list

def verify_prediction_range():
    print("="*80)
    print("VERIFYING EMPIRICAL PREDICTION RANGE")
    print("="*80)

    # 1. Load Data
    traits = get_master_trait_list()
    df = merged[traits].copy().dropna()
    print(f"Loaded {len(df)} original participants")

    # 2. Load Coefficients
    config_path = Path("config/decisions.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    coeffs = config['donation_default']['regression_coefficients']['categorical']
    print("\nRegression Coefficients (Categorical):")
    print(f"Intercept: {coeffs['intercept']}")
    
    # 3. Calculate Predicted Score for EACH agent
    predictions = []
    
    hh_mean = 3.3922
    hh_std = 0.5587
    
    print("\nCalculating predictions...")
    
    for idx, row in df.iterrows():
        # Start with intercept
        pred = coeffs['intercept']
        
        # Group Effect
        group = row['Group_experiment']
        # Map HighSub to FullSub
        group_mapped = 'FullSub' if group == 'HighSub' else group
        pred += coeffs['beta_group'].get(group_mapped, 0.0)
        
        # Income Effect (Categorical)
        income_level = int(row['Assigned Allowance Level'])
        income_quintiles = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'Q5'}
        income_q = income_quintiles.get(income_level, 'Q5')
        pred += coeffs['beta_income_q'].get(income_q, 0.0)
        
        # Study Program Effect
        program = row['Study Program']
        study_cat = 'Grad2yr' # default
        if any(p in program.upper() for p in ['INCOMING', 'EXCHANGE']):
            study_cat = 'Incoming'
        elif any(p in program.upper() for p in ['LAW', 'CLMG']):
            study_cat = 'Law5yr'
        elif any(p in program.upper() for p in ['BESS', 'BIEM', 'BIG', 'BAI', 'BEMACS']):
            study_cat = 'UG3yr'
            
        pred += coeffs['beta_study'].get(study_cat, 0.0)
        
        # HH Effect (z-scored)
        hh_raw = row['Honesty_Humility']
        hh_z = (hh_raw - hh_mean) / hh_std
        pred += coeffs['beta_hh'] * hh_z
        
        predictions.append(pred)

    # 4. Analyze Results
    predictions = np.array(predictions)
    
    actual_min = predictions.min()
    actual_max = predictions.max()
    
    hardcoded_min = -4.0778
    hardcoded_max = 7.2030
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Calculated Min: {actual_min:.6f}")
    print(f"Hardcoded Min:  {hardcoded_min:.6f}")
    print(f"Difference:     {actual_min - hardcoded_min:.6f}")
    
    print("-" * 40)
    
    print(f"Calculated Max: {actual_max:.6f}")
    print(f"Hardcoded Max:  {hardcoded_max:.6f}")
    print(f"Difference:     {actual_max - hardcoded_max:.6f}")
    print("-" * 40)
    
    if abs(actual_min - hardcoded_min) < 0.001 and abs(actual_max - hardcoded_max) < 0.001:
        print("✅ VERIFIED: Hardcoded values match the empirical data exactly!")
    else:
        print("❌ DISCREPANCY: Hardcoded values DO NOT match the current data/coefficients.")
        print("Possible reasons:")
        print("1. Coefficients in config have changed")
        print("2. HH Mean/Std used for z-scoring differs")
        print("3. Underlying data has changed")

if __name__ == "__main__":
    verify_prediction_range()

