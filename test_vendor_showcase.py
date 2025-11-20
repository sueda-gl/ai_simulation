#!/usr/bin/env python3
"""
Test vendor selection to verify it showcases preferences correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import Orchestrator
import pandas as pd

def test_vendor_selection():
    print("=" * 80)
    print("Testing Vendor Selection - Preference Showcase")
    print("=" * 80)
    
    # Initialize orchestrator
    orch = Orchestrator()
    
    # Run simulation with small number of agents
    print("\n1. Running simulation with 3 agents...")
    results_df = orch.run_simulation(n_agents=3, seed=42)
    print(f"✓ Simulation complete: {len(results_df)} agents")
    
    # Examine vendor selection results
    print("\n2. Examining vendor preferences for each agent...")
    for idx, agent in results_df.iterrows():
        agent_id = agent.get('agent_id', idx + 1)
        customer_type = agent.get('customer_type')
        vendor_selection = agent.get('vendor_selection')
        preferred_vendor = agent.get('preferred_vendor')
        note = agent.get('note', 'N/A')
        
        print(f"\n  Agent {agent_id} ({customer_type}):")
        print(f"    Preferred Vendor: {preferred_vendor}")
        print(f"    Vendor Selection: {vendor_selection}")
        
        # Check purchase requests
        requests = agent.get('purchase_requests', [])
        if len(requests) > 0:
            req_vendors = set(req.get('vendorID') for req in requests)
            print(f"    VendorIDs in requests: {req_vendors}")
            print(f"    Total requests: {len(requests)}")
            
    print("\n" + "=" * 80)
    print("NOTE: Decision 8 currently showcases agent preferences only.")
    print("Vendor assignment algorithm will be implemented in the future.")
    print("=" * 80)

if __name__ == "__main__":
    test_vendor_selection()

