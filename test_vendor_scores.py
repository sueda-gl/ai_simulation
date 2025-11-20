#!/usr/bin/env python3
"""
Test vendor selection to verify scores are calculated correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import Orchestrator
import pandas as pd

def test_vendor_selection():
    print("=" * 80)
    print("Testing Vendor Selection Based on Scores")
    print("=" * 80)
    
    # Initialize orchestrator
    orch = Orchestrator()
    
    # Run simulation with small number of agents
    print("\n1. Running simulation with 3 agents...")
    results_df = orch.run_simulation(n_agents=3, seed=42)
    print(f"✓ Simulation complete: {len(results_df)} agents")
    
    # Examine vendor selection results
    print("\n2. Examining vendor selection for each agent...")
    for idx, agent in results_df.iterrows():
        agent_id = agent.get('agent_id', idx + 1)
        customer_type = agent.get('customer_type')
        vendor_selection = agent.get('vendor_selection')
        preferred_vendor = agent.get('preferred_vendor')
        got_preferred = agent.get('got_preferred')
        vendor_rank = agent.get('vendor_rank')
        
        print(f"\n  Agent {agent_id} ({customer_type}):")
        print(f"    Preferred Vendor (highest score): {preferred_vendor}")
        print(f"    Selected Vendor: {vendor_selection}")
        print(f"    Got Preferred: {got_preferred}")
        print(f"    Vendor Rank: {vendor_rank}")
        
        # Check purchase requests
        requests = agent.get('purchase_requests', [])
        if len(requests) > 0:
            req_vendors = set(req.get('vendorID') for req in requests)
            print(f"    VendorIDs in requests: {req_vendors}")
            print(f"    Total requests: {len(requests)}")

if __name__ == "__main__":
    test_vendor_selection()


