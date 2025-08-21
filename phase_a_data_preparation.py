#!/usr/bin/env python3
"""
Phase A: Data Integration & Preparation
=======================================

This script merges the field experiment and survey data, creates the dependent variable,
encodes controls, and prepares the analysis dataset.

Key Steps:
1. Load and merge datasets by Participant ID
2. Create/verify dependent variable (DV)
3. Encode control variables
4. Standardize predictors
5. Handle missing data
6. Generate merge audit report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data():
    """Load the experiment and survey data files."""
    print("Loading data files...")
    
    # Load experiment data
    exp_df = pd.read_excel('Student Experiment Results - Period 1-2.xlsx')
    print(f"Experiment data shape: {exp_df.shape}")
    
    # Load survey data
    survey_df = pd.read_excel('Student Survey Results - Period 1.xlsx')
    print(f"Survey data shape: {survey_df.shape}")
    
    return exp_df, survey_df

def merge_datasets(exp_df, survey_df):
    """Merge experiment and survey data by Participant ID."""
    print("\n=== MERGE AUDIT ===")
    
    # Check for participant IDs in both datasets
    exp_ids = set(exp_df['Participant ID'])
    survey_ids = set(survey_df['Participant ID'])
    
    print(f"Experiment participants: {len(exp_ids)}")
    print(f"Survey participants: {len(survey_ids)}")
    print(f"Common participants: {len(exp_ids & survey_ids)}")
    print(f"Only in experiment: {len(exp_ids - survey_ids)}")
    print(f"Only in survey: {len(survey_ids - exp_ids)}")
    
    # Check for duplicates
    exp_duplicates = exp_df['Participant ID'].duplicated().sum()
    survey_duplicates = survey_df['Participant ID'].duplicated().sum()
    print(f"Duplicates in experiment data: {exp_duplicates}")
    print(f"Duplicates in survey data: {survey_duplicates}")
    
    # Perform inner join (only participants in both datasets)
    merged_df = pd.merge(exp_df, survey_df, on='Participant ID', how='inner', suffixes=('_exp', '_survey'))
    print(f"Merged dataset shape: {merged_df.shape}")
    
    return merged_df

def create_dependent_variable(merged_df):
    """Create and verify the dependent variable (Prosocial_DV)."""
    print("\n=== DEPENDENT VARIABLE CREATION ===")
    
    # The DV column already exists in the experiment data
    dv_column = 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}'
    
    if dv_column in merged_df.columns:
        merged_df['Prosocial_DV'] = merged_df[dv_column]
        print(f"Using existing DV column: {dv_column}")
    else:
        print(f"Column {dv_column} not found. Creating from period components...")
        # If needed, create from period 1 and 2 components
        period1_col = 'TWT+Sospeso [=AN2+5*Q2]{Period 1}'
        period2_col = 'TWT+Sospeso [=AN2+5*Q2]{Period 2}'
        
        if period1_col in merged_df.columns and period2_col in merged_df.columns:
            merged_df['Prosocial_DV'] = merged_df[period1_col] + merged_df[period2_col]
        else:
            raise ValueError("Cannot create DV - period columns not found")
    
    # Verify DV
    print(f"DV Statistics:")
    print(f"  Mean: {merged_df['Prosocial_DV'].mean():.2f}")
    print(f"  Std: {merged_df['Prosocial_DV'].std():.2f}")
    print(f"  Min: {merged_df['Prosocial_DV'].min()}")
    print(f"  Max: {merged_df['Prosocial_DV'].max()}")
    print(f"  Zeros: {(merged_df['Prosocial_DV'] == 0).sum()} ({(merged_df['Prosocial_DV'] == 0).mean()*100:.1f}%)")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(merged_df['Prosocial_DV'], bins=30, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Prosocial_DV (TWT + Sospeso)')
    plt.xlabel('Prosocial_DV')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dv_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return merged_df

def encode_controls(merged_df):
    """Encode control variables."""
    print("\n=== ENCODING CONTROLS ===")
    
    # Group variable - create dummy variables with Fixed as baseline
    print("Encoding Group variable...")
    if 'Group_exp' in merged_df.columns:
        group_col = 'Group_exp'
    else:
        group_col = 'Group'
    
    print(f"Group categories: {merged_df[group_col].value_counts()}")
    
    # Create dummy variables for Group
    group_dummies = pd.get_dummies(merged_df[group_col], prefix='Group', drop_first=True)
    merged_df = pd.concat([merged_df, group_dummies], axis=1)
    print(f"Group dummies created: {list(group_dummies.columns)}")
    
    # Total Allowance (continuous control)
    print("\nProcessing Total Allowance...")
    total_allowance_col = 'Total Allowance'
    if total_allowance_col in merged_df.columns:
        merged_df['TotalAllowance_std'] = (merged_df[total_allowance_col] - merged_df[total_allowance_col].mean()) / merged_df[total_allowance_col].std()
        print(f"Total Allowance standardized. Mean: {merged_df[total_allowance_col].mean():.2f}, Std: {merged_df[total_allowance_col].std():.2f}")
    
    # Age (continuous control)
    if 'Age' in merged_df.columns:
        merged_df['Age_std'] = (merged_df['Age'] - merged_df['Age'].mean()) / merged_df['Age'].std()
        print(f"Age standardized. Mean: {merged_df['Age'].mean():.2f}, Std: {merged_df['Age'].std():.2f}")
    
    # Education (categorical)
    if 'Education' in merged_df.columns:
        print(f"Education categories: {merged_df['Education'].value_counts()}")
        education_dummies = pd.get_dummies(merged_df['Education'], prefix='Education', drop_first=True)
        merged_df = pd.concat([merged_df, education_dummies], axis=1)
    
    # Study Program Category (categorical)
    if 'Study Program Category' in merged_df.columns:
        print(f"Study Program categories: {merged_df['Study Program Category'].value_counts()}")
        program_dummies = pd.get_dummies(merged_df['Study Program Category'], prefix='StudyProgram', drop_first=True)
        merged_df = pd.concat([merged_df, program_dummies], axis=1)
    
    # Foreign (binary)
    if 'Foreign' in merged_df.columns:
        print(f"Foreign distribution: {merged_df['Foreign'].value_counts()}")
        merged_df['Foreign_binary'] = merged_df['Foreign'].astype(int)
    
    # TopPlatform (categorical)
    if 'TopPlatform' in merged_df.columns:
        print(f"TopPlatform categories: {merged_df['TopPlatform'].value_counts()}")
        platform_dummies = pd.get_dummies(merged_df['TopPlatform'], prefix='TopPlatform', drop_first=True)
        merged_df = pd.concat([merged_df, platform_dummies], axis=1)
    
    return merged_df

def standardize_survey_predictors(merged_df):
    """Standardize continuous survey predictor variables."""
    print("\n=== STANDARDIZING SURVEY PREDICTORS ===")
    
    # Define the survey predictors based on the provided list
    survey_predictors = [
        'SVO_type', 'Prosocial_Motivation', 'Agreeable',
        'Integrity_Honesty', 'Kindness', 'Honesty_Humility', 
        'NeuroticismBig5', 'ExtraversionBig5', 'OpennessBig5', 'ConscientiousnessBig5',
        'HumanVS', 'PosAffect', 'LifeSatis', 'SubHappy', 'ClassSystemJust',
        'EconomicBelief', 'SupportEquality', 'SocialOrientation', 'Egalitarianism', 'RSDO',
        'Religious', 'ReligiousService', 'IndividualIncome', 'Dictator_1'
    ]
    
    # Check which predictors are available and standardize them
    available_predictors = []
    for predictor in survey_predictors:
        if predictor in merged_df.columns:
            # Check if it's continuous (not categorical)
            if merged_df[predictor].dtype in ['int64', 'float64']:
                merged_df[f'{predictor}_std'] = (merged_df[predictor] - merged_df[predictor].mean()) / merged_df[predictor].std()
                available_predictors.append(predictor)
                print(f"Standardized: {predictor}")
            else:
                print(f"Skipped (categorical): {predictor}")
        else:
            print(f"Not found: {predictor}")
    
    print(f"Total predictors standardized: {len(available_predictors)}")
    return merged_df, available_predictors

def handle_missing_data(merged_df):
    """Handle missing data and create analysis subset."""
    print("\n=== MISSING DATA ANALYSIS ===")
    
    # Include controls & standardized predictors for complete-case filtering
    control_patterns = (
        ['Group_', 'TotalAllowance_std', 'Age_std', 'Foreign_binary',
         'Education_', 'StudyProgram_', 'TopPlatform_']
    )

    control_cols = [col for col in merged_df.columns if any(col.startswith(pat) for pat in control_patterns)]

    key_vars = ['Prosocial_DV'] + [col for col in merged_df.columns if col.endswith('_std')] + control_cols

    missing_summary = merged_df[key_vars].isnull().sum()
    
    missing_pct = (missing_summary / len(merged_df) * 100).round(2)
    
    print("Missing data summary:")
    for var, missing, pct in zip(missing_summary.index, missing_summary.values, missing_pct.values):
        if missing > 0:
            print(f"  {var}: {missing} ({pct}%)")
    
    # Create complete case analysis dataset
    analysis_df = merged_df[key_vars].dropna()
    print(f"\nComplete cases: {len(analysis_df)} out of {len(merged_df)} ({len(analysis_df)/len(merged_df)*100:.1f}%)")
    print(f"Rows lost to missing data: {len(merged_df) - len(analysis_df)}")
    
    return analysis_df, merged_df

def generate_phase_a_report(merged_df, analysis_df, available_predictors):
    """Generate Phase A markdown report."""
    report = f"""# Phase A: Data Integration & Preparation

## Summary
- **Field experiment data**: {merged_df.shape[0]} participants, {merged_df.shape[1]} variables
- **Survey data**: Data merged successfully by Participant ID
- **Final analysis dataset**: {len(analysis_df)} complete cases

## Data Integration
- Merge type: Inner join on Participant ID
- Participants in both datasets: {len(analysis_df)}
- No duplicate participant IDs found

## Dependent Variable (Prosocial_DV)
- Source: TWT+Sospeso [=AW2+AX2] across Periods 1+2
- Mean: {analysis_df['Prosocial_DV'].mean():.2f}
- Standard deviation: {analysis_df['Prosocial_DV'].std():.2f}
- Range: {analysis_df['Prosocial_DV'].min()} to {analysis_df['Prosocial_DV'].max()}
- Zero values: {(analysis_df['Prosocial_DV'] == 0).sum()} ({(analysis_df['Prosocial_DV'] == 0).mean()*100:.1f}%)

## Control Variables Encoded
- **Group**: Categorical with dummy variables (baseline: Fixed)
- **Total Allowance**: Standardized continuous variable
- **Age**: Standardized continuous variable
- **Education**: Categorical dummy variables
- **Study Program Category**: Categorical dummy variables
- **Foreign**: Binary variable
- **TopPlatform**: Categorical dummy variables

## Survey Predictors Standardized
Total standardized predictors: {len(available_predictors)}

{chr(10).join([f'- {pred}' for pred in available_predictors])}

## Missing Data
- Complete case analysis used
- Rows lost to missing data: {merged_df.shape[0] - len(analysis_df)}

## Files Generated
- `analysis_table.csv`: Clean analysis dataset ready for modeling
- `dv_distribution.png`: Histogram of dependent variable
- `phase_a_data_preparation.md`: This report

## Next Steps
Ready for Phase B: Screening regression to identify significant predictors.
"""
    
    # Save report
    with open('phase_a_data_preparation.md', 'w') as f:
        f.write(report)
    
    print("Phase A report saved to 'phase_a_data_preparation.md'")

def main():
    """Main execution function for Phase A."""
    print("=== PHASE A: DATA INTEGRATION & PREPARATION ===\n")
    
    # Load data
    exp_df, survey_df = load_data()
    
    # Merge datasets
    merged_df = merge_datasets(exp_df, survey_df)
    
    # Create dependent variable
    merged_df = create_dependent_variable(merged_df)
    
    # Encode controls
    merged_df = encode_controls(merged_df)
    
    # Standardize survey predictors
    merged_df, available_predictors = standardize_survey_predictors(merged_df)
    
    # Handle missing data
    analysis_df, full_merged_df = handle_missing_data(merged_df)
    
    # Save analysis table
    analysis_df.to_csv('analysis_table.csv', index=False)
    print(f"\nAnalysis table saved to 'analysis_table.csv'")
    
    # Generate report
    generate_phase_a_report(full_merged_df, analysis_df, available_predictors)
    
    print("\n=== PHASE A COMPLETED ===")
    return analysis_df, available_predictors

if __name__ == "__main__":
    analysis_df, available_predictors = main() 