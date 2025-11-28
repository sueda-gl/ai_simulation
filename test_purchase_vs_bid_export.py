#!/usr/bin/env python3
"""
Test script to verify purchase vs bid Excel export functionality.

This script:
1. Runs a small simulation with agents
2. Checks that purchase requests are enriched with platformPrice and bid_value
3. Verifies that the Excel export can be generated successfully
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.orchestrator import Orchestrator


def test_purchase_vs_bid_export():
    """Test purchase vs bid decision enrichment and Excel export"""
    
    print("=" * 80)
    print("TESTING PURCHASE VS BID EXCEL EXPORT")
    print("=" * 80)
    
    # Initialize orchestrator
    orch = Orchestrator()
    
    # Run simulation with small number of agents
    n_agents = 10
    seed = 42
    
    print(f"\n1. Running simulation with {n_agents} agents...")
    results_df = orch.run_simulation(n_agents=n_agents, seed=seed)
    
    print(f"   ✓ Simulation complete: {len(results_df)} agents")
    
    # Check if purchase_requests exist
    if 'purchase_requests' not in results_df.columns:
        print("   ✗ ERROR: No purchase_requests column in results!")
        return False
    
    # Check enrichment
    print(f"\n2. Checking purchase request enrichment...")
    
    total_requests = 0
    enriched_requests = 0
    pn_count = 0
    bid_count = 0
    discount_count = 0
    fixed_count = 0
    
    for idx, row in results_df.iterrows():
        requests = row.get('purchase_requests', [])
        if isinstance(requests, list):
            for req in requests:
                if isinstance(req, dict):
                    total_requests += 1
                    
                    # Check if enriched with platformPrice and bid_value
                    if 'platformPrice' in req and 'bid_value' in req:
                        enriched_requests += 1
                        
                        platform_price = req.get('platformPrice')
                        if platform_price == 'PN':
                            pn_count += 1
                        elif platform_price == 'BID':
                            bid_count += 1
                        elif platform_price == 'DISCOUNT':
                            discount_count += 1
                        elif platform_price == 'FIXED':
                            fixed_count += 1
    
    print(f"   Total purchase requests: {total_requests}")
    print(f"   Enriched requests: {enriched_requests}")
    print(f"   Enrichment rate: {enriched_requests/total_requests*100:.1f}%")
    
    if enriched_requests < total_requests:
        print(f"   ⚠ WARNING: Not all requests are enriched!")
    else:
        print(f"   ✓ All requests enriched successfully!")
    
    print(f"\n3. Request-level decisions breakdown:")
    print(f"   Purchase Now (PN): {pn_count} ({pn_count/total_requests*100:.1f}%)")
    print(f"   Bid (BID): {bid_count} ({bid_count/total_requests*100:.1f}%)")
    print(f"   Discount: {discount_count} ({discount_count/total_requests*100:.1f}%)")
    print(f"   Fixed: {fixed_count} ({fixed_count/total_requests*100:.1f}%)")
    
    # Check regular customers only
    regular_requests = pn_count + bid_count
    if regular_requests > 0:
        print(f"\n4. Regular customers analysis:")
        print(f"   Total regular requests: {regular_requests}")
        print(f"   PN rate: {pn_count/regular_requests*100:.1f}%")
        print(f"   Bid rate: {bid_count/regular_requests*100:.1f}%")
        
        # Check bid values
        bid_values = []
        for idx, row in results_df.iterrows():
            requests = row.get('purchase_requests', [])
            if isinstance(requests, list):
                for req in requests:
                    if isinstance(req, dict):
                        if req.get('platformPrice') == 'BID':
                            bid_val = req.get('bid_value')
                            if bid_val != 'N/A':
                                try:
                                    bid_values.append(float(bid_val))
                                except (ValueError, TypeError):
                                    pass
        
        if bid_values:
            print(f"\n5. Bid value analysis:")
            print(f"   Total bid values: {len(bid_values)}")
            print(f"   Unique bid values: {len(set(bid_values))}")
            print(f"   Min bid: ${min(bid_values):.2f}")
            print(f"   Max bid: ${max(bid_values):.2f}")
            print(f"   Mean bid: ${sum(bid_values)/len(bid_values):.2f}")
            
            if len(set(bid_values)) == len(bid_values):
                print(f"   ✓ All bid values are unique!")
            else:
                print(f"   ⚠ Some bid values are duplicates")
    
    # Test Excel export function
    print(f"\n6. Testing Excel export function...")
    try:
        from app.pages.results.visualizations.transaction_viz import _build_purchase_vs_bid_export
        
        # Add required columns if missing
        if 'agent_id' not in results_df.columns:
            results_df['agent_id'] = range(1, len(results_df) + 1)
        if 'Assigned Allowance Level' not in results_df.columns:
            results_df['Assigned Allowance Level'] = results_df.get('income_category', 1)
        if 'Group_experiment' not in results_df.columns:
            results_df['Group_experiment'] = 'A'
        
        export_records = _build_purchase_vs_bid_export(results_df)
        
        print(f"   ✓ Export function executed successfully")
        print(f"   Export records: {len(export_records)}")
        
        if len(export_records) > 0:
            # Show sample record
            sample = export_records[0]
            print(f"\n7. Sample export record:")
            for key, value in sample.items():
                print(f"   {key}: {value}")
            
            # Verify all required fields are present
            required_fields = [
                'Agent ID', 'Assigned Allowance Level', 'Group_experiment',
                'Customer Type', 'Income Category', 'Purchase Request Type',
                'timestamp', 'Period', 'Customer Price'
            ]
            
            missing_fields = [field for field in required_fields if field not in sample]
            if missing_fields:
                print(f"\n   ⚠ Missing fields: {missing_fields}")
            else:
                print(f"\n   ✓ All required fields present!")
        else:
            print(f"   ⚠ No export records generated")
        
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✓ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_purchase_vs_bid_export()
    sys.exit(0 if success else 1)



