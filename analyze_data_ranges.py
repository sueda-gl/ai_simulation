#!/usr/bin/env python3
"""Analyze data ranges and distributions for proper scaling"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('.')
from src.validate_traits import merged
from src.build_master_traits import get_master_trait_list

# Get the merged data
traits = get_master_trait_list()
df = merged[traits].copy().dropna()

print("📊 DATA ANALYSIS FOR DONATION_DEFAULT SCALING")
print("=" * 80)

# 1. Analyze TWT+Sospeso (observed prosocial behavior)
print("\n1. TWT+Sospeso [=AW2+AX2]{Periods 1+2} Analysis:")
twt_sospeso = df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
print(f"   Count: {len(twt_sospeso)}")
print(f"   Mean: {twt_sospeso.mean():.4f}")
print(f"   Std: {twt_sospeso.std():.4f}")
print(f"   Min: {twt_sospeso.min():.4f}")
print(f"   Max: {twt_sospeso.max():.4f}")
print(f"   Range: [{twt_sospeso.min():.4f}, {twt_sospeso.max():.4f}]")

# 2. Analyze Group_experiment values
print("\n2. Group_experiment unique values:")
print(f"   {df['Group_experiment'].value_counts().to_dict()}")

# 3. Analyze Study Program values
print("\n3. Study Program unique values:")
study_programs = df['Study Program'].unique()
print(f"   Total unique programs: {len(study_programs)}")
print("   Sample programs:")
for i, prog in enumerate(sorted(study_programs)[:10]):
    print(f"      {i+1}. {prog}")
print("   ...")

# 4. Analyze Assigned Allowance Level
print("\n4. Assigned Allowance Level distribution:")
allowance_dist = df['Assigned Allowance Level'].value_counts().sort_index()
print(f"   {allowance_dist.to_dict()}")

# 5. Calculate income quintiles based on actual distribution
print("\n5. Income Quintile Mapping (based on actual distribution):")
# Since we have 5 allowance levels, let's see how they map to quintiles
# Direct mapping since levels are already 1-5
print("   Direct mapping (1→Q1, 2→Q2, etc.)")

# 6. SD by income quintile
print("\n6. Standard Deviation of TWT+Sospeso by Income Level:")
for level in sorted(df['Assigned Allowance Level'].unique()):
    level_data = df[df['Assigned Allowance Level'] == level]['TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
    if len(level_data) > 1:
        print(f"   Level {level}: SD = {level_data.std():.4f} (n={len(level_data)})")

# 7. Honesty-Humility stats
print("\n7. Honesty_Humility Analysis:")
hh = df['Honesty_Humility']
print(f"   Mean: {hh.mean():.4f}")
print(f"   Std: {hh.std():.4f}")
print(f"   Min: {hh.min():.4f}")
print(f"   Max: {hh.max():.4f}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("1. Group mapping: Use the actual values from data (likely HighSub/MidSub/NoSub)")
print("2. Income quintiles: Use pd.qcut on actual distribution, not direct 1→Q1 mapping")
print("3. Study programs: Need proper categorization mapping to Incoming/Law5yr/UG3yr/Grad2yr")
print("4. TWT+Sospeso range for scaling: Use the min/max from actual data")
print("5. Overall SD for stochastic component:", twt_sospeso.std())
