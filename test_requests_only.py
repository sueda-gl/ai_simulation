#!/usr/bin/env python3
"""
Test script to verify simulation works without enrich_purchase_requests.

This tests that:
1. Purchase requests are still created with basic fields
2. Simulation completes successfully without Decision 6b
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import Orchestrator
import pandas as pd

def test_requests_only():
    print("=" * 80)
    print("Testing Purchase Requests Only Implementation")
    print("=" * 80)
    
    # Initialize orchestrator
    orch = Orchestrator()
    
    # Verify enrich_purchase_requests is NOT in decision order
    print("\n1. Checking decision order...")
    if 'enrich_purchase_requests' not in orch.decision_order:
        print("  ✓ enrich_purchase_requests correctly removed from decision order")
    else:
        print("  ✗ enrich_purchase_requests STILL in decision order!")
        return
    
    # Run simulation with small number of agents
    print("\n2. Running simulation with 5 agents...")
    try:
        results_df = orch.run_simulation(n_agents=5, seed=42)
        print(f"  ✓ Simulation complete: {len(results_df)} agents")
    except Exception as e:
        print(f"  ✗ Simulation FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Check columns
    print("\n3. Checking key columns exist...")
    required_cols = ['customer_type', 'purchase_requests', 'purchasing_quantity']
    for col in required_cols:
        if col in results_df.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ✗ {col} MISSING!")
    
    # Examine purchase requests for first agent
    print("\n4. Examining purchase requests for Agent 1...")
    agent1 = results_df.iloc[0]
    
    print(f"  Customer Type: {agent1.get('customer_type')}")
    print(f"  Consumption Quantity: {agent1.get('purchasing_quantity')}")
    
    purchase_requests = agent1.get('purchase_requests', [])
    print(f"  Number of Purchase Requests: {len(purchase_requests)}")
    
    if len(purchase_requests) > 0:
        print("\n  Sample Request #1:")
        req = purchase_requests[0]
        for k, v in req.items():
            print(f"    - {k}: {v}")
            
        # Verify transaction fields are None as expected
        if req.get('platformPrice') is None and req.get('bid_value') is None:
             print("\n  ✓ Transaction fields are None as expected")
        else:
             print("\n  ✗ Transaction fields should be None but have values!")

if __name__ == "__main__":
    test_requests_only()
