"""
Diagnostic script to verify agent mismatch bug in multi-config export.

This script checks if agents are properly aligned across different configurations.
"""
import pandas as pd
import streamlit as st

def diagnose_agent_alignment():
    """
    Check if agents in different configurations are actually the same people.
    """
    if not hasattr(st.session_state, 'simulation_results'):
        print("❌ No simulation results found in session state")
        return
    
    results_dict = st.session_state.simulation_results
    
    if not results_dict or len(results_dict) < 2:
        print("⚠️ Need at least 2 configurations to check alignment")
        return
    
    print("="*80)
    print("AGENT ALIGNMENT DIAGNOSTIC")
    print("="*80)
    
    # Get configuration names
    config_names = list(results_dict.keys())
    print(f"\nFound {len(config_names)} configurations:")
    for i, name in enumerate(config_names, 1):
        df = results_dict[name]
        print(f"  {i}. {name}: {len(df)} agents")
    
    # Check first configuration traits
    print("\n" + "="*80)
    print("TRAIT COMPARISON FOR FIRST 10 AGENTS")
    print("="*80)
    
    trait_columns = ['Honesty_Humility', 'Assigned Allowance Level', 'Study Program', 
                     'Group_experiment', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
    
    first_config_name = config_names[0]
    first_config_df = results_dict[first_config_name]
    
    for agent_idx in range(min(10, len(first_config_df))):
        print(f"\n{'='*80}")
        print(f"Agent ID {agent_idx + 1} (Row {agent_idx}):")
        print(f"{'='*80}")
        
        for config_name in config_names[:3]:  # Check first 3 configs
            df = results_dict[config_name]
            if agent_idx >= len(df):
                print(f"\n  ❌ {config_name}: AGENT DOESN'T EXIST (only {len(df)} agents)")
                continue
            
            print(f"\n  📊 {config_name}:")
            agent = df.iloc[agent_idx]
            
            # Check if agent_id column exists
            if 'agent_id' in agent:
                print(f"     agent_id: {agent['agent_id']}")
            
            # Print traits
            for trait in trait_columns:
                if trait in agent:
                    value = agent[trait]
                    if isinstance(value, float):
                        print(f"     {trait}: {value:.4f}")
                    else:
                        print(f"     {trait}: {value}")
            
            # Print donation_default if exists
            if 'donation_default' in agent:
                print(f"     donation_default: {agent['donation_default']:.6f}")
        
        # Check if traits match across configs
        print(f"\n  ✅ Traits Match Check:")
        all_match = True
        for trait in trait_columns:
            values = []
            for config_name in config_names:
                df = results_dict[config_name]
                if agent_idx < len(df) and trait in df.columns:
                    values.append(df.iloc[agent_idx][trait])
            
            if len(set([str(v) for v in values])) == 1:
                print(f"     ✅ {trait}: MATCH")
            else:
                print(f"     ❌ {trait}: MISMATCH - {values}")
                all_match = False
        
        if not all_match:
            print(f"\n  🚨 CRITICAL: Agent {agent_idx + 1} has DIFFERENT traits across configs!")
            print(f"     This proves agents are NOT aligned!")
    
    # Statistical summary
    print("\n" + "="*80)
    print("DONATION_DEFAULT CORRELATION CHECK")
    print("="*80)
    
    if len(config_names) >= 2:
        config1_name = config_names[0]
        config2_name = config_names[1]
        
        df1 = results_dict[config1_name]
        df2 = results_dict[config2_name]
        
        if 'donation_default' in df1.columns and 'donation_default' in df2.columns:
            min_len = min(len(df1), len(df2))
            
            vals1 = df1['donation_default'][:min_len].values
            vals2 = df2['donation_default'][:min_len].values
            
            # Calculate correlation
            import numpy as np
            correlation = np.corrcoef(vals1, vals2)[0, 1]
            
            print(f"\nCorrelation between {config1_name} and {config2_name}:")
            print(f"  Correlation coefficient: {correlation:.4f}")
            
            if correlation > 0.9:
                print(f"  ✅ STRONG correlation - agents likely aligned")
            elif correlation > 0.5:
                print(f"  ⚠️ MODERATE correlation - partial alignment?")
            else:
                print(f"  ❌ WEAK correlation - agents NOT aligned!")
            
            # Show some examples of large differences
            diff = np.abs(vals1 - vals2)
            large_diff_indices = np.where(diff > 0.3)[0]
            
            if len(large_diff_indices) > 0:
                print(f"\n  🚨 Found {len(large_diff_indices)} agents with difference > 0.3:")
                for idx in large_diff_indices[:5]:
                    print(f"     Agent {idx + 1}: {vals1[idx]:.4f} vs {vals2[idx]:.4f} (diff: {diff[idx]:.4f})")

if __name__ == "__main__":
    # Run in Streamlit context
    import sys
    if 'streamlit' in sys.modules:
        diagnose_agent_alignment()
    else:
        print("This script should be run in Streamlit context")
        print("Add this to your results page temporarily:")
        print("from diagnose_agent_mismatch import diagnose_agent_alignment")
        print("diagnose_agent_alignment()")

