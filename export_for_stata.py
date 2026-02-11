# export_for_stata.py
"""
Export the original 280 participants to a clean CSV for Stata verification.

Renames columns to Stata-friendly names (no spaces, no special characters).
Prints summary statistics so you can cross-check in Stata with `summarize`.
"""
import pandas as pd
from src.validate_traits import merged
from src.build_master_traits import get_master_trait_list

traits = get_master_trait_list()
df = merged[['Participant ID'] + traits].copy().dropna()

# Rename columns to Stata-friendly names
rename_map = {
    'Participant ID': 'participant_id',
    'Agreeable': 'agreeable',
    'Assigned Allowance Level': 'allowance_level',
    'ExtraversionBig5': 'extraversion',
    'Group_experiment': 'group_experiment',
    'Honesty_Humility': 'honesty_humility',
    'NeuroticismBig5': 'neuroticism',
    'OpennessBig5': 'openness',
    'ReligiousAffiliation': 'religious_affiliation',
    'ReligiousService': 'religious_service',
    'Study Program': 'study_program',
    'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': 'twt_sospeso',
}
df = df.rename(columns=rename_map)

print(f"Exporting {len(df)} participants")
print(f"Columns: {list(df.columns)}\n")

# Print summary stats for cross-checking with Stata's `summarize`
numeric_cols = df.select_dtypes(include='number').columns
print("=" * 70)
print("SUMMARY STATISTICS (compare with Stata's `summarize` output)")
print("=" * 70)
for col in numeric_cols:
    print(f"\n  {col}:")
    print(f"    N    = {df[col].count()}")
    print(f"    Mean = {df[col].mean():.6f}")
    print(f"    SD   = {df[col].std():.6f}")
    print(f"    Min  = {df[col].min()}")
    print(f"    Max  = {df[col].max()}")

# Save
output_path = 'data/stata_verification.csv'
df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
