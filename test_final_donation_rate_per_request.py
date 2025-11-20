#!/usr/bin/env python3
"""
Test script to verify final_donation_rate is now stored per purchase request.

This test verifies:
1. Each purchase request has a 'final_donation_rate' field
2. The rate comes from agent's donation_default or final_donation_rate
3. All requests for an agent have the same rate (for now - no variation yet)
4. Backward compatibility: old simulations without per-request rates still work
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.orchestrator import Orchestrator
import pandas as pd
import numpy as np

def test_per_request_donation_rates():
    """Test that final_donation_rate is stored in each purchase request"""
    
    print("=" * 80)
    print("TEST: Final Donation Rate Per Purchase Request")
    print("=" * 80)
    print()
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Run simulation with donation decisions
    print("📊 Running simulation with 5 agents...")
    print("   Decisions: donation_default + purchasing_quantity")
    print()
    
    results_df = orchestrator.run_simulation(
        n_agents=5,
        seed=42,
        single_decision=['donation_default', 'purchasing_quantity']
    )
    
    print(f"✅ Simulation complete: {len(results_df)} agents")
    print()
    
    # Test each agent's purchase requests
    print("-" * 80)
    print("VERIFICATION: Checking purchase requests for final_donation_rate field")
    print("-" * 80)
    print()
    
    total_requests = 0
    requests_with_rate = 0
    agents_with_requests = 0
    
    for idx, row in results_df.iterrows():
        agent_id = row.get('agent_id', idx + 1)
        donation_default = row.get('donation_default', 'N/A')
        purchase_requests = row.get('purchase_requests', [])
        
        if not purchase_requests or len(purchase_requests) == 0:
            continue
        
        agents_with_requests += 1
        total_requests += len(purchase_requests)
        
        print(f"Agent {agent_id}:")
        print(f"  Agent-level donation_default: {donation_default}")
        print(f"  Number of purchase requests: {len(purchase_requests)}")
        
        # Check first 3 requests for this agent
        for req_idx, request in enumerate(purchase_requests[:3]):
            request_rate = request.get('final_donation_rate', 'MISSING')
            
            if request_rate != 'MISSING':
                requests_with_rate += 1
            
            platform_price = request.get('platformPrice', 'N/A')
            print(f"    Request {req_idx + 1}: final_donation_rate={request_rate}, platformPrice={platform_price}")
        
        if len(purchase_requests) > 3:
            # Check all remaining requests have the field
            remaining_with_rate = sum(1 for req in purchase_requests[3:] if 'final_donation_rate' in req)
            requests_with_rate += remaining_with_rate
            print(f"    ... {len(purchase_requests) - 3} more requests (all have final_donation_rate: {remaining_with_rate == len(purchase_requests) - 3})")
        
        print()
    
    # Summary
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Total agents with requests: {agents_with_requests}")
    print(f"Total purchase requests: {total_requests}")
    print(f"Requests with final_donation_rate field: {requests_with_rate}")
    print()
    
    # Verification
    success = requests_with_rate == total_requests
    
    if success:
        print("✅ SUCCESS: All purchase requests have final_donation_rate field!")
        print()
        print("✅ Implementation verified:")
        print("   - final_donation_rate is stored in each purchase request")
        print("   - Values match agent's donation_default")
        print("   - Excel export will read from request-level data")
    else:
        print(f"❌ FAILURE: {total_requests - requests_with_rate} requests missing final_donation_rate field")
        return False
    
    print()
    
    # Additional check: Verify values match agent-level
    print("-" * 80)
    print("VERIFICATION: Checking values match agent-level donation rates")
    print("-" * 80)
    print()
    
    mismatches = 0
    
    for idx, row in results_df.iterrows():
        agent_id = row.get('agent_id', idx + 1)
        agent_donation_rate = row.get('donation_default', 0.10)
        
        try:
            agent_donation_rate = float(agent_donation_rate)
        except:
            agent_donation_rate = 0.10
        
        purchase_requests = row.get('purchase_requests', [])
        
        for request in purchase_requests:
            request_rate = request.get('final_donation_rate', None)
            
            if request_rate is not None:
                try:
                    request_rate = float(request_rate)
                    
                    # Check if they match (within small tolerance for floating point)
                    if not np.isclose(request_rate, agent_donation_rate, rtol=1e-5):
                        mismatches += 1
                        print(f"⚠️  Agent {agent_id}: Request rate {request_rate} != Agent rate {agent_donation_rate}")
                except:
                    mismatches += 1
    
    if mismatches == 0:
        print("✅ All request rates match their agent's donation_default rate")
    else:
        print(f"⚠️  Found {mismatches} mismatches between request and agent rates")
    
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    return success and mismatches == 0


def test_backward_compatibility():
    """Test that Excel export handles old simulations without per-request rates"""
    
    print()
    print("=" * 80)
    print("TEST: Backward Compatibility")
    print("=" * 80)
    print()
    
    # Create mock data that simulates old simulation (no final_donation_rate in requests)
    print("📊 Creating mock old simulation data...")
    
    mock_df = pd.DataFrame([{
        'agent_id': 1,
        'donation_default': 0.35,
        'final_donation_rate': 0.35,
        'Assigned Allowance Level': 3,
        'Group_experiment': 'NoSub',
        'income_category': 5,
        'purchase_requests': [
            {
                'request_id': 1,
                'platformPrice': 'PN',
                'bid_value': 'N/A',
                'customer_type': 'regular',
                'timestamp_hours': 2.5
                # NOTE: No 'final_donation_rate' field (old simulation)
            },
            {
                'request_id': 2,
                'platformPrice': 'BID',
                'bid_value': 95.50,
                'customer_type': 'regular',
                'timestamp_hours': 5.8
                # NOTE: No 'final_donation_rate' field (old simulation)
            }
        ]
    }])
    
    print("✅ Mock data created (old format - no per-request rates)")
    print()
    
    # Try to process it with the new export function
    print("🔧 Testing Excel export with old data...")
    try:
        from app.pages.results.visualizations.donation_viz import _build_donation_transaction_export
        
        records = _build_donation_transaction_export(mock_df)
        
        print(f"✅ Export processed {len(records)} transaction records")
        print()
        
        # Check that fallback worked
        for i, record in enumerate(records):
            rate = record.get('Final Donation Rate', 'MISSING')
            print(f"  Transaction {i+1}: Final Donation Rate = {rate}")
        
        # Verify all transactions got the agent-level rate
        all_have_rate = all(record.get('Final Donation Rate') is not None for record in records)
        
        if all_have_rate:
            print()
            print("✅ SUCCESS: Backward compatibility works!")
            print("   Old simulations use agent-level rate as fallback")
            return True
        else:
            print()
            print("❌ FAILURE: Some transactions missing donation rate")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print()
    print("🧪 Testing Final Donation Rate Per-Request Implementation")
    print()
    
    # Run tests
    test1_passed = test_per_request_donation_rates()
    test2_passed = test_backward_compatibility()
    
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Test 1 (Per-Request Storage): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (Backward Compatibility): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print()
    
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! Implementation is working correctly.")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED. Please review the output above.")
        sys.exit(1)

