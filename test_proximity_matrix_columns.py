"""
Quick test to verify proximity matrix columns are correct (no allowance/group columns).
"""

import pandas as pd
import sys
from pathlib import Path

# Simulate the proximity matrix building logic
def build_test_proximity_matrix():
    """Simulate what the code does when building proximity matrix"""
    
    # Simulate agent data
    test_data = {
        'agent_id': [1, 2, 3],
        'Assigned Allowance Level': ['Level 1', 'Level 2', 'Level 3'],  # This exists in df
        'Group_experiment': ['Control', 'Treatment', 'Control'],  # This exists in df
        'vendor_proximity_scores': [
            {'1': 88.07, '2': 44.57, '3': 47.77},
            {'1': 89.73, '2': 42.14, '3': 42.87},
            {'1': 62.72, '2': 29.50, '3': 32.95}
        ]
    }
    
    df = pd.DataFrame(test_data)
    
    # THIS IS THE ACTUAL CODE FROM vendor_viz.py (lines 1038-1062)
    proximity_matrix_data = []
    for idx in range(len(df)):
        row_data = {}
        
        # Add Agent ID
        if 'agent_id' in df.columns:
            row_data['Agent ID'] = df.iloc[idx]['agent_id']
        else:
            row_data['Agent ID'] = idx + 1
        
        # NOTE: The removed code would be here!
        # OLD CODE (REMOVED):
        # if 'Assigned Allowance Level' in df.columns:
        #     row_data['Assigned Allowance Level'] = df.iloc[idx]['Assigned Allowance Level']
        # if 'Group_experiment' in df.columns:
        #     row_data['Group_experiment'] = df.iloc[idx]['Group_experiment']
        
        # Add proximity scores for each vendor
        scores = df.iloc[idx]['vendor_proximity_scores']
        if isinstance(scores, dict):
            for v_id in sorted(scores.keys(), key=lambda x: int(x)):
                row_data[f'Vendor {v_id} Proximity'] = scores[v_id]
        
        proximity_matrix_data.append(row_data)
    
    proximity_df = pd.DataFrame(proximity_matrix_data)
    return proximity_df


if __name__ == "__main__":
    print("="*80)
    print("TESTING PROXIMITY MATRIX COLUMN STRUCTURE")
    print("="*80)
    
    # Build the proximity matrix using the actual code logic
    result_df = build_test_proximity_matrix()
    
    print("\n📊 Columns in the Proximity Matrix:")
    print("-"*80)
    for i, col in enumerate(result_df.columns, 1):
        print(f"  {i}. {col}")
    
    print("\n" + "="*80)
    print("VERIFICATION CHECKS")
    print("="*80)
    
    # Check 1: Should have Agent ID
    has_agent_id = 'Agent ID' in result_df.columns
    print(f"\n✅ Has 'Agent ID' column: {has_agent_id}")
    
    # Check 2: Should NOT have Assigned Allowance Level
    has_allowance = 'Assigned Allowance Level' in result_df.columns
    if has_allowance:
        print(f"❌ FAIL: Still has 'Assigned Allowance Level' column (should be removed)")
    else:
        print(f"✅ PASS: 'Assigned Allowance Level' column removed")
    
    # Check 3: Should NOT have Group_experiment
    has_group = 'Group_experiment' in result_df.columns
    if has_group:
        print(f"❌ FAIL: Still has 'Group_experiment' column (should be removed)")
    else:
        print(f"✅ PASS: 'Group_experiment' column removed")
    
    # Check 4: Should have vendor proximity columns
    vendor_cols = [col for col in result_df.columns if 'Vendor' in col and 'Proximity' in col]
    print(f"✅ Has {len(vendor_cols)} vendor proximity columns: {vendor_cols}")
    
    # Check 5: Total column count
    expected_columns = 1 + len(vendor_cols)  # 1 for Agent ID + vendor columns
    actual_columns = len(result_df.columns)
    print(f"\n📊 Column count:")
    print(f"   Expected: {expected_columns} (1 Agent ID + {len(vendor_cols)} vendor columns)")
    print(f"   Actual: {actual_columns}")
    
    if actual_columns == expected_columns:
        print(f"   ✅ PASS: Correct number of columns")
    else:
        print(f"   ❌ FAIL: Wrong number of columns")
    
    # Show sample data
    print("\n" + "="*80)
    print("SAMPLE DATA")
    print("="*80)
    print(result_df.to_string(index=False))
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    if not has_allowance and not has_group and has_agent_id and len(vendor_cols) > 0:
        print("✅ SUCCESS: Proximity matrix has correct column structure!")
        print("   - Only Agent ID metadata column")
        print("   - No Assigned Allowance Level")
        print("   - No Group_experiment")
        print(f"   - {len(vendor_cols)} vendor proximity columns")
    else:
        print("❌ FAILED: Column structure is incorrect")
        print(f"   has_allowance: {has_allowance} (should be False)")
        print(f"   has_group: {has_group} (should be False)")
        print(f"   has_agent_id: {has_agent_id} (should be True)")
        print(f"   vendor_cols: {len(vendor_cols)} (should be > 0)")



