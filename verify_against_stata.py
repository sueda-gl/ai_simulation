# verify_against_stata.py
"""
Run the disclose_income model on the original 280 participants (deterministic, no bootstrap)
and compare against Stata results.
"""
import pandas as pd
import numpy as np
import yaml
from src.validate_traits import merged
from src.build_master_traits import get_master_trait_list
from src.decisions.disclose_income_stochastic import (
    compute_pass1_values, compute_anchored_pb
)

# Load config
with open('config/decisions.yaml', 'r') as f:
    config = yaml.safe_load(f)
params = config['disclose_income']

# Get the original 280 (no bootstrap)
traits = get_master_trait_list()
df = merged[traits].copy().dropna()
print(f"Processing {len(df)} participants\n")

# Get composite z-scoring stats (same fixed stats used in simulation)
composite_z = params.get('composite_z_scoring', {})
de_mean = composite_z.get('weighted_disclosure_categorical', {}).get('mean', 0)
de_sd = composite_z.get('weighted_disclosure_categorical', {}).get('sd', 0.0344991747)
ap_mean = composite_z.get('anchored_pb', {}).get('mean', 0)
ap_sd = composite_z.get('anchored_pb', {}).get('sd', 0.2594725501)

anchor_weights = params.get('anchor_weights', {})
WPB = anchor_weights.get('prosocial_weight', 0.50)
beta_0 = params.get('intercept', 0.0)

results = []
for idx, row in df.iterrows():
    agent_state = row.to_dict()

    # Pass 1: traits, weighted_prosocial, direct_effect
    pass1 = compute_pass1_values(agent_state, params, simulation_config=None)

    # Anchored PB (deterministic, no stochastic)
    anchored_pb = compute_anchored_pb(pass1, params)

    # Z-score composites using fixed stats from config
    z_direct_effect = (pass1['direct_effect'] - de_mean) / de_sd if de_sd > 0 else pass1['direct_effect']
    z_anchored_pb = (anchored_pb - ap_mean) / ap_sd if ap_sd > 0 else anchored_pb

    # Final DI equation
    prosocial_effect = z_anchored_pb * pass1['income_high']
    di_i = beta_0 + (1 - WPB) * z_direct_effect + WPB * prosocial_effect

    # Y/N decision
    disclose = "Y" if di_i > 0 else "N"

    results.append({
        'participant_id': agent_state.get('Participant ID', idx),
        'allowance_level': agent_state.get('Assigned Allowance Level'),
        'z_agreeable': pass1['z_agreeable'],
        'weighted_prosocial': pass1['weighted_prosocial'],
        'z_weighted_prosocial': pass1['z_weighted_prosocial'],
        'direct_effect': pass1['direct_effect'],
        'anchored_pb': anchored_pb,
        'z_direct_effect': z_direct_effect,
        'z_anchored_pb': z_anchored_pb,
        'income_high': pass1['income_high'],
        'di_i': di_i,
        'disclose': disclose,
    })

results_df = pd.DataFrame(results)
results_df.to_csv('data/python_verification.csv', index=False)

# ============================================================
# PRINT THE SAME SUMMARY TABLES AS STATA
# ============================================================

print("=" * 50)
print("  OVERALL DISCLOSURE RATE")
print("=" * 50)
counts = results_df['disclose'].value_counts().sort_index()
for val, cnt in counts.items():
    print(f"  {val}: {cnt} ({cnt/len(results_df)*100:.2f}%)")
print(f"  Total: {len(results_df)}\n")

print("=" * 50)
print("  DISCLOSURE BY INCOME LEVEL")
print("=" * 50)
cross = pd.crosstab(results_df['allowance_level'], results_df['disclose'], margins=True)
cross_pct = pd.crosstab(results_df['allowance_level'], results_df['disclose'], normalize='index') * 100
print("\nCounts:")
print(cross)
print("\nRow percentages:")
print(cross_pct.round(2))
print()

print("=" * 50)
print("  DI_i DISTRIBUTION")
print("=" * 50)
di = results_df['di_i']
print(f"  Obs:      {len(di)}")
print(f"  Mean:     {di.mean():.7f}")
print(f"  Std dev:  {di.std():.7f}")
print(f"  Min:      {di.min():.7f}")
print(f"  Max:      {di.max():.7f}")
print(f"  Median:   {di.median():.7f}")
print()

print("=" * 50)
print("  DI_i BY INCOME LEVEL")
print("=" * 50)
for level in sorted(results_df['allowance_level'].unique()):
    sub = results_df[results_df['allowance_level'] == level]['di_i']
    print(f"\n  Level {int(level)}:  Obs={len(sub)}, Mean={sub.mean():.7f}, SD={sub.std():.7f}, Min={sub.min():.7f}, Max={sub.max():.7f}")

print()
print("=" * 50)
print("  FIRST 10 PARTICIPANTS")
print("=" * 50)
print(results_df[['participant_id', 'allowance_level', 'z_agreeable', 'weighted_prosocial', 'anchored_pb', 'direct_effect', 'di_i', 'disclose']].head(10).to_string(index=False))

# ============================================================
# COMPARE WITH STATA RESULTS
# ============================================================
print("\n")
print("=" * 50)
print("  COMPARISON WITH STATA")
print("=" * 50)
try:
    stata = pd.read_csv('data/stata_results.csv')
    # Match on participant_id
    merged_compare = results_df.merge(stata, on='participant_id', suffixes=('_python', '_stata'))
    
    for col in ['z_agreeable', 'weighted_prosocial', 'anchored_pb', 'direct_effect', 'di_i']:
        py_col = f'{col}_python'
        st_col = f'{col}_stata'
        if py_col in merged_compare.columns and st_col in merged_compare.columns:
            diff = (merged_compare[py_col] - merged_compare[st_col]).abs()
            print(f"  {col:25s}  max_diff = {diff.max():.10f}  mean_diff = {diff.mean():.10f}")
    
    # Check if Y/N decisions match
    yn_match = (merged_compare['disclose_python'] == merged_compare['disclose_stata']).sum()
    print(f"\n  Y/N decisions match: {yn_match} / {len(merged_compare)} ({yn_match/len(merged_compare)*100:.1f}%)")
    
except FileNotFoundError:
    print("  stata_results.csv not found - skipping comparison")
except Exception as e:
    print(f"  Error during comparison: {e}")

print("\nPython results saved to data/python_verification.csv")
