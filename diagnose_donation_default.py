#!/usr/bin/env python3
"""
Diagnostic script to analyze donation_default calculation for all 6 modes.

The 6 modes are combinations of:
- Population modes: Copula (synthetic), Documentation (original + stochastic), Baseline (original, no stochastic)
- Income modes: Categorical (quintiles), Continuous (linear)

This script specifically addresses the professor's concern:
"Some agents with TWT+Sospeso = 0 have higher donation rates than other agents with TWT+Sospeso = 0"

EXPLANATION OF WHY THIS HAPPENS:
The donation rate formula is:
  anchor = 0.75 × s100_observed + 0.25 × s100_predicted
  donation_rate = (anchor + shift_value) / 100

When TWT+Sospeso = 0:
  s100_observed = 0  (because it's scaled from 0-112 to 0-100)
  anchor = 0.25 × s100_predicted

So the donation rate depends ENTIRELY on the predicted prosocial score, which is computed from:
1. Group effect (HighSub/FullSub, MidSub, NoSub)
2. Income effect (Q1-Q5 categorical, or linear continuous)
3. Study program effect (Incoming, Law5yr, UG3yr, Grad2yr)
4. Honesty-Humility score

This is BY DESIGN: even if someone didn't donate in the experiment (TWT+Sospeso=0),
their other characteristics can still predict some prosocial tendency.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load configuration
import yaml
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "decisions.yaml"

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

donation_config = config['donation_default']

# Constants from the code
OBS_MIN = 0.0
OBS_MAX = 112.0
PRED_MIN = -4.0778
PRED_MAX = 7.2030
HH_MEAN = 3.3922
HH_STD = 0.5587

def calculate_donation_default_step_by_step(
    twt_sospeso: float,
    group: str,
    income_level: int,
    study_program: str,
    hh_score: float,
    income_mode: str = 'categorical',
    sigma: float = 0.0,
    shift_value: float = -4.0,
    verbose: bool = True
) -> dict:
    """
    Calculate donation default rate step by step, showing all intermediate values.
    
    Args:
        twt_sospeso: TWT+Sospeso [=AW2+AX2]{Periods 1+2} value (observed prosocial behavior)
        group: Group_experiment (FullSub, MidSub, NoSub, HighSub)
        income_level: Assigned Allowance Level (1-5)
        study_program: Study Program name
        hh_score: Honesty_Humility score
        income_mode: 'categorical' or 'continuous'
        sigma: Standard deviation for stochastic component (0 = deterministic)
        shift_value: Adjustment shift value
        verbose: Print step-by-step details
    
    Returns:
        Dictionary with all intermediate values and final donation rate
    """
    
    # Get coefficients based on income mode
    if income_mode == 'continuous':
        coeffs = donation_config['regression_coefficients']['continuous']
    else:
        coeffs = donation_config['regression_coefficients']['categorical']
    
    results = {
        'twt_sospeso': twt_sospeso,
        'group': group,
        'income_level': income_level,
        'study_program': study_program,
        'hh_score': hh_score,
        'income_mode': income_mode,
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"DONATION DEFAULT CALCULATION (Income Mode: {income_mode.upper()})")
        print(f"{'='*80}")
        print(f"Agent Profile:")
        print(f"  - TWT+Sospeso: {twt_sospeso}")
        print(f"  - Group: {group}")
        print(f"  - Income Level: {income_level}")
        print(f"  - Study Program: {study_program}")
        print(f"  - Honesty-Humility: {hh_score}")
        print(f"\n{'='*80}")
    
    # STEP 1: Compute predicted prosocial
    if verbose:
        print("\n📊 STEP 1: Compute Predicted Prosocial Score")
        print("-" * 60)
    
    # Start with intercept
    predicted = coeffs.get('intercept', 0.0)
    if verbose:
        print(f"  Base intercept: {predicted:.6f}")
    
    # Add group effect (map HighSub to FullSub)
    group_mapped = 'FullSub' if group == 'HighSub' else group
    beta_group = coeffs.get('beta_group', {})
    group_effect = beta_group.get(group_mapped, 0.0)
    predicted += group_effect
    if verbose:
        print(f"  + Group effect ({group_mapped}): {group_effect:.6f}")
        print(f"    → Running total: {predicted:.6f}")
    results['group_effect'] = group_effect
    
    # Add income effect
    if income_mode == 'continuous':
        # Linear income effect (use actual allowance values for levels 1-5)
        allowance_map = {1: 16, 2: 32, 3: 72, 4: 128, 5: 200}
        actual_allowance = allowance_map.get(income_level, 100)
        beta_lin = coeffs.get('beta_income_linear', 0.0)
        income_effect = beta_lin * actual_allowance
        predicted += income_effect
        if verbose:
            print(f"  + Income effect (continuous): {beta_lin:.6f} × {actual_allowance} = {income_effect:.6f}")
            print(f"    → Running total: {predicted:.6f}")
    else:
        # Categorical income effect
        income_quintiles = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'Q5'}
        income_q = income_quintiles.get(int(income_level), 'Q5')
        beta_income_q = coeffs.get('beta_income_q', {})
        income_effect = beta_income_q.get(income_q, 0.0)
        predicted += income_effect
        if verbose:
            print(f"  + Income effect ({income_q}): {income_effect:.6f}")
            print(f"    → Running total: {predicted:.6f}")
    results['income_effect'] = income_effect
    
    # Add study program effect
    study_category = 'Grad2yr'  # default
    if any(prog in study_program.upper() for prog in ['INCOMING', 'EXCHANGE']):
        study_category = 'Incoming'
    elif any(prog in study_program.upper() for prog in ['LAW', 'CLMG']):
        study_category = 'Law5yr'
    elif any(prog in study_program.upper() for prog in ['BESS', 'BIEM', 'BIG', 'BAI', 'BEMACS', 'BIEF']):
        study_category = 'UG3yr'
    
    beta_study = coeffs.get('beta_study', {})
    study_effect = beta_study.get(study_category, 0.0)
    predicted += study_effect
    if verbose:
        print(f"  + Study effect ({study_category}): {study_effect:.6f}")
        print(f"    → Running total: {predicted:.6f}")
    results['study_category'] = study_category
    results['study_effect'] = study_effect
    
    # Add Honesty-Humility effect (z-scored)
    hh_zscore = (hh_score - HH_MEAN) / HH_STD
    beta_hh = coeffs.get('beta_hh', 0.0)
    hh_effect = beta_hh * hh_zscore
    predicted += hh_effect
    if verbose:
        print(f"  + HH effect: {beta_hh:.6f} × z({hh_score:.2f}) = {beta_hh:.6f} × {hh_zscore:.4f} = {hh_effect:.6f}")
        print(f"    → Final predicted: {predicted:.6f}")
    results['hh_zscore'] = hh_zscore
    results['hh_effect'] = hh_effect
    results['predicted_raw'] = predicted
    
    # STEP 2: Scale to 0-100
    if verbose:
        print(f"\n📊 STEP 2: Scale to 0-100")
        print("-" * 60)
    
    s100_observed = 100 * (twt_sospeso - OBS_MIN) / (OBS_MAX - OBS_MIN)
    s100_predicted = 100 * (predicted - PRED_MIN) / (PRED_MAX - PRED_MIN)
    
    # Clip to [0, 100]
    s100_observed = np.clip(s100_observed, 0, 100)
    s100_predicted = np.clip(s100_predicted, 0, 100)
    
    if verbose:
        print(f"  Observed scaling: 100 × ({twt_sospeso} - {OBS_MIN}) / ({OBS_MAX} - {OBS_MIN})")
        print(f"    s100_observed = {s100_observed:.4f}")
        print(f"  Predicted scaling: 100 × ({predicted:.4f} - {PRED_MIN}) / ({PRED_MAX} - {PRED_MIN})")
        print(f"    s100_predicted = {s100_predicted:.4f}")
    
    results['s100_observed'] = s100_observed
    results['s100_predicted'] = s100_predicted
    
    # STEP 3: Compute anchor
    if verbose:
        print(f"\n📊 STEP 3: Compute Anchor (weighted average)")
        print("-" * 60)
    
    weights = donation_config['anchor_weights']
    s100_anchor = weights['observed'] * s100_observed + weights['predicted'] * s100_predicted
    
    if verbose:
        print(f"  anchor = {weights['observed']} × s100_observed + {weights['predicted']} × s100_predicted")
        print(f"  anchor = {weights['observed']} × {s100_observed:.4f} + {weights['predicted']} × {s100_predicted:.4f}")
        print(f"  anchor = {s100_anchor:.4f}")
    
    results['anchor'] = s100_anchor
    
    # STEP 3b: Apply shift adjustment
    if verbose:
        print(f"\n📊 STEP 3b: Apply Shift Adjustment")
        print("-" * 60)
    
    adjusted_anchor = s100_anchor + shift_value
    if verbose:
        print(f"  adjusted_anchor = anchor + shift_value = {s100_anchor:.4f} + ({shift_value}) = {adjusted_anchor:.4f}")
    
    results['adjusted_anchor'] = adjusted_anchor
    
    # STEP 4: Stochastic component (if sigma > 0)
    if verbose:
        print(f"\n📊 STEP 4: Stochastic Component")
        print("-" * 60)
    
    if sigma > 0:
        # In actual code, this would be a random draw
        draw_0_100 = adjusted_anchor  # For demonstration, showing deterministic case
        if verbose:
            print(f"  With sigma={sigma}, would draw from Normal({adjusted_anchor:.4f}, {sigma:.4f})")
            print(f"  (Using deterministic value for this demo)")
    else:
        draw_0_100 = adjusted_anchor
        if verbose:
            print(f"  Sigma = 0, no stochastic component")
            print(f"  draw_0_100 = {draw_0_100:.4f}")
    
    # STEP 5: Floor at 0
    if verbose:
        print(f"\n📊 STEP 5: Floor negative values at 0")
        print("-" * 60)
    
    draw_0_100_floored = max(draw_0_100, 0.0)
    if verbose:
        print(f"  draw_0_100_floored = max({draw_0_100:.4f}, 0) = {draw_0_100_floored:.4f}")
    
    # STEP 6: Convert to [0,1] proportion
    if verbose:
        print(f"\n📊 STEP 6: Convert to [0,1] proportion")
        print("-" * 60)
    
    donation_rate = np.clip(draw_0_100_floored / 100.0, 0.0, 1.0)
    if verbose:
        print(f"  donation_rate = {draw_0_100_floored:.4f} / 100 = {donation_rate:.4f}")
        print(f"  Final donation_default = {donation_rate:.4f} ({donation_rate*100:.2f}%)")
    
    results['donation_rate'] = donation_rate
    
    return results


def analyze_zero_twt_sospeso_variation():
    """
    Analyze why agents with TWT+Sospeso = 0 can have different donation rates.
    """
    print("\n" + "="*80)
    print("ANALYSIS: WHY DO AGENTS WITH TWT+SOSPESO = 0 HAVE DIFFERENT DONATION RATES?")
    print("="*80)
    
    print("""
When TWT+Sospeso = 0:
  - s100_observed = 0
  - anchor = 0.75 × 0 + 0.25 × s100_predicted = 0.25 × s100_predicted
  
So the donation rate depends ONLY on the predicted prosocial score, which varies based on:
  1. Group (experimental condition)
  2. Income level (quintile or continuous)
  3. Study program
  4. Honesty-Humility score

Let's see some examples:
""")
    
    # Create example agents with TWT+Sospeso = 0 but different other characteristics
    test_cases = [
        {
            'twt_sospeso': 0, 'group': 'NoSub', 'income_level': 1,
            'study_program': 'Incoming', 'hh_score': 2.5,
            'description': 'NoSub, Low Income (Q1), Incoming Student, Low HH'
        },
        {
            'twt_sospeso': 0, 'group': 'MidSub', 'income_level': 3,
            'study_program': 'CLEAM', 'hh_score': 3.4,
            'description': 'MidSub, Middle Income (Q3), Graduate Student, Average HH'
        },
        {
            'twt_sospeso': 0, 'group': 'FullSub', 'income_level': 5,
            'study_program': 'CLEF', 'hh_score': 4.2,
            'description': 'FullSub, High Income (Q5), Graduate Student, High HH'
        },
    ]
    
    print("-" * 80)
    print("Example agents with TWT+Sospeso = 0 (Categorical Income Mode):")
    print("-" * 80)
    
    results_list = []
    for i, case in enumerate(test_cases, 1):
        print(f"\n🧑 Agent {i}: {case['description']}")
        result = calculate_donation_default_step_by_step(
            twt_sospeso=case['twt_sospeso'],
            group=case['group'],
            income_level=case['income_level'],
            study_program=case['study_program'],
            hh_score=case['hh_score'],
            income_mode='categorical',
            shift_value=-4.0,
            verbose=True
        )
        results_list.append({
            'Agent': f"Agent {i}",
            'Description': case['description'],
            'TWT+Sospeso': case['twt_sospeso'],
            'Predicted Raw': result['predicted_raw'],
            's100_predicted': result['s100_predicted'],
            'Anchor': result['anchor'],
            'Donation Rate': result['donation_rate']
        })
    
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    df = pd.DataFrame(results_list)
    print(df.to_string(index=False))
    
    print("""
⚠️  KEY INSIGHT:
Even though all agents have TWT+Sospeso = 0 (no observed prosocial behavior),
their donation rates vary from {:.1f}% to {:.1f}%.

This is EXPECTED BEHAVIOR because:
1. The anchor formula gives 25% weight to PREDICTED prosocial behavior
2. Predicted behavior is computed from other agent characteristics
3. Different groups, incomes, study programs, and HH scores → different predictions

This reflects the RESEARCH DESIGN: even if someone didn't donate in the experiment,
their demographic and personality characteristics can predict some donation tendency.
""".format(min(r['Donation Rate'] for r in results_list) * 100,
           max(r['Donation Rate'] for r in results_list) * 100))
    
    return df


def show_coefficient_summary():
    """Show a summary of all coefficients used in the calculation."""
    
    print("\n" + "="*80)
    print("COEFFICIENT SUMMARY FROM config/decisions.yaml")
    print("="*80)
    
    for mode in ['categorical', 'continuous']:
        print(f"\n📋 {mode.upper()} INCOME MODE")
        print("-" * 40)
        
        coeffs = donation_config['regression_coefficients'][mode]
        
        print(f"  Intercept: {coeffs.get('intercept', 0):.6f}")
        
        print("\n  Group Effects (reference: FullSub = 0):")
        for group, effect in coeffs.get('beta_group', {}).items():
            print(f"    {group}: {effect:+.6f}")
        
        if mode == 'categorical':
            print("\n  Income Quintile Effects (Q1 is reference for categorical):")
            for q, effect in coeffs.get('beta_income_q', {}).items():
                print(f"    {q}: {effect:+.6f}")
        else:
            print(f"\n  Income Linear Effect: {coeffs.get('beta_income_linear', 0):.6f} per unit allowance")
        
        print("\n  Study Program Effects (reference: Grad2yr = 0):")
        for study, effect in coeffs.get('beta_study', {}).items():
            print(f"    {study}: {effect:+.6f}")
        
        print(f"\n  Honesty-Humility Effect: {coeffs.get('beta_hh', 0):.6f} per z-score")
    
    print("\n📋 OTHER PARAMETERS")
    print("-" * 40)
    print(f"  Anchor Weights: observed={donation_config['anchor_weights']['observed']}, predicted={donation_config['anchor_weights']['predicted']}")
    print(f"  Shift Value: {donation_config.get('adjustment', {}).get('shift_value', 0)}")
    print(f"  Scaling: observed [0,{OBS_MAX}] → [0,100], predicted [{PRED_MIN},{PRED_MAX}] → [0,100]")
    print(f"  HH Z-scoring: mean={HH_MEAN}, std={HH_STD}")


def analyze_actual_data():
    """Analyze the actual data to find agents with TWT+Sospeso = 0."""
    
    print("\n" + "="*80)
    print("ANALYZING ACTUAL DATA: Agents with TWT+Sospeso = 0")
    print("="*80)
    
    try:
        from src.validate_traits import merged
        from src.build_master_traits import get_master_trait_list
        
        traits = get_master_trait_list()
        data = merged[traits].dropna()
        
        # Find agents with TWT+Sospeso = 0 (or very close to 0)
        zero_threshold = 0.001
        zero_twt = data[data['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'] <= zero_threshold]
        
        print(f"\nFound {len(zero_twt)} agents with TWT+Sospeso ≤ {zero_threshold}")
        
        if len(zero_twt) > 0:
            print(f"\nTheir characteristics:")
            
            # Calculate donation rate for each
            results = []
            for idx, row in zero_twt.iterrows():
                result = calculate_donation_default_step_by_step(
                    twt_sospeso=row['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'],
                    group=row['Group_experiment'],
                    income_level=int(row['Assigned Allowance Level']),
                    study_program=row['Study Program'],
                    hh_score=row['Honesty_Humility'],
                    income_mode='categorical',
                    shift_value=-4.0,
                    verbose=False
                )
                results.append({
                    'Index': idx,
                    'TWT+Sospeso': row['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'],
                    'Group': row['Group_experiment'],
                    'Income': int(row['Assigned Allowance Level']),
                    'Study': row['Study Program'],
                    'HH': row['Honesty_Humility'],
                    'Predicted': result['predicted_raw'],
                    'Donation Rate': result['donation_rate']
                })
            
            df = pd.DataFrame(results)
            df = df.sort_values('Donation Rate', ascending=False)
            
            print(df.to_string(index=False))
            
            print(f"\n📊 Donation Rate Statistics for agents with TWT+Sospeso ≈ 0:")
            print(f"  Min:  {df['Donation Rate'].min():.4f} ({df['Donation Rate'].min()*100:.2f}%)")
            print(f"  Max:  {df['Donation Rate'].max():.4f} ({df['Donation Rate'].max()*100:.2f}%)")
            print(f"  Mean: {df['Donation Rate'].mean():.4f} ({df['Donation Rate'].mean()*100:.2f}%)")
            print(f"  Std:  {df['Donation Rate'].std():.4f} ({df['Donation Rate'].std()*100:.2f}%)")
            
            return df
        else:
            print("No agents found with TWT+Sospeso = 0 in the data.")
            return None
            
    except Exception as e:
        print(f"Could not load actual data: {e}")
        return None


if __name__ == '__main__':
    print("="*80)
    print("DONATION DEFAULT DECISION DIAGNOSTIC TOOL")
    print("="*80)
    
    # Show coefficient summary
    show_coefficient_summary()
    
    # Analyze zero TWT+Sospeso variation
    analyze_zero_twt_sospeso_variation()
    
    # Analyze actual data if available
    analyze_actual_data()
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
The variation in donation rates among agents with TWT+Sospeso = 0 is EXPECTED BEHAVIOR.

The anchor formula:
  anchor = 0.75 × observed + 0.25 × predicted

When observed = 0:
  anchor = 0.25 × predicted

The predicted value varies based on:
  - Group (NoSub reduces, MidSub increases)
  - Income (higher income increases predicted)
  - Study program (Incoming/Law/UG reduce, Graduate maintains)
  - Honesty-Humility (higher HH increases predicted)

This design allows the model to predict donation behavior even for agents
who showed zero prosocial behavior in the experiment, based on their
demographic and personality characteristics.

If the professor wants agents with TWT+Sospeso = 0 to always have donation_rate = 0,
the anchor weights would need to be changed to observed=1.0, predicted=0.0.
""")





