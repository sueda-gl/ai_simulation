"""
Reproduce the vendor 4 dominance issue with a proper simulation.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import Orchestrator

def test_vendor_assignment():
    print("=" * 80)
    print("VENDOR ASSIGNMENT TEST - Reproducing Vendor 4 Issue")
    print("=" * 80)
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Set simulation config with 4 vendors
    orchestrator.simulation_config['num_vendors'] = 4
    orchestrator.simulation_config['vendor_price_min'] = 5.0
    orchestrator.simulation_config['vendor_price_max'] = 10.0
    orchestrator.simulation_config['vendor_products_min'] = 5000  # High capacity to avoid sellouts
    orchestrator.simulation_config['vendor_products_max'] = 10000
    
    print("\n1. SIMULATION CONFIGURATION")
    print(f"   Number of agents: 100")
    print(f"   Number of vendors: 4")
    print(f"   Vendor price range: $5.00 - $10.00")
    print(f"   Random seed: 42")
    
    # Run simulation
    print("\n2. RUNNING SIMULATION...")
    df = orchestrator.run_simulation(n_agents=100, seed=42)
    print(f"   ✓ Simulation complete: {len(df)} agents processed")
    
    # Get vendors
    vendors = orchestrator.simulation_config.get('vendors', [])
    
    print(f"\n3. VENDOR ATTRIBUTES")
    for vendor in vendors:
        print(f"   Vendor {vendor['vendor_id']}: "
              f"price=${vendor['price']:.2f}, "
              f"quality={vendor['quality']}, "
              f"sustainability={vendor['sustainability']}, "
              f"capacity={vendor['quantity_offered']}")
    
    # Analyze purchase requests
    print(f"\n4. ANALYZING PURCHASE REQUESTS")
    
    vendor_counts = {}
    agent_vendor_map = {}
    total_requests = 0
    
    for idx, row in df.iterrows():
        agent_id = row['agent_id']
        requests = row.get('purchase_requests', [])
        
        if not isinstance(requests, list) or len(requests) == 0:
            continue
        
        # Count vendor IDs for this agent
        for req in requests:
            if isinstance(req, dict):
                vendor_id = req.get('vendorID')
                if vendor_id is not None:
                    vendor_counts[vendor_id] = vendor_counts.get(vendor_id, 0) + 1
                    total_requests += 1
                    
                    # Track which vendor this agent uses
                    if agent_id not in agent_vendor_map:
                        agent_vendor_map[agent_id] = vendor_id
    
    print(f"   Total purchase requests: {total_requests:,}")
    print(f"   Agents with requests: {len(agent_vendor_map)}")
    
    print(f"\n5. VENDOR DISTRIBUTION")
    for vendor_id in sorted(vendor_counts.keys()):
        count = vendor_counts[vendor_id]
        pct = count / total_requests * 100 if total_requests > 0 else 0
        
        # Count how many agents use this vendor
        agents_with_vendor = sum(1 for v in agent_vendor_map.values() if v == vendor_id)
        
        print(f"   Vendor {vendor_id}: {count:,} requests ({pct:.1f}%), {agents_with_vendor} agents")
    
    # Check if vendor 4 dominates
    if 4 in vendor_counts:
        vendor_4_pct = vendor_counts[4] / total_requests * 100
        if vendor_4_pct > 80:
            print(f"\n⚠️  WARNING: Vendor 4 dominates with {vendor_4_pct:.1f}% of requests!")
            print("   This matches the issue you reported.")
            
            # Explain why
            vendor_4 = next((v for v in vendors if v['vendor_id'] == 4), None)
            if vendor_4:
                print(f"\n6. WHY VENDOR 4 DOMINATES:")
                print(f"   Vendor 4 price: ${vendor_4['price']:.2f}")
                
                # Compare to other vendors
                all_prices = [v['price'] for v in vendors]
                min_price = min(all_prices)
                max_price = max(all_prices)
                
                if vendor_4['price'] == min_price:
                    print(f"   ✓ Vendor 4 has the LOWEST price (${min_price:.2f})")
                
                print(f"   Vendor 4 sustainability: {vendor_4['sustainability']}/5")
                
                if vendor_4['sustainability'] == max([v['sustainability'] for v in vendors]):
                    print(f"   ✓ Vendor 4 has the HIGHEST sustainability")
                
                print(f"\n   With equal weights (0.25 each), Vendor 4's price and sustainability")
                print(f"   advantages outweigh quality and proximity differences.")
        else:
            print(f"\n✓ Vendor distribution is reasonable (Vendor 4 has {vendor_4_pct:.1f}%)")
    
    # Show sample agents
    print(f"\n7. SAMPLE AGENTS (First 10):")
    for agent_id in sorted(list(agent_vendor_map.keys())[:10]):
        vendor_id = agent_vendor_map[agent_id]
        
        # Get agent's weights and proximity
        agent_row = df[df['agent_id'] == agent_id].iloc[0]
        weights = agent_row.get('vendor_choice_weights', {})
        proximity_scores = agent_row.get('vendor_proximity_scores', {})
        
        print(f"   Agent {agent_id}: Vendor {vendor_id}, "
              f"proximity to V{vendor_id}={proximity_scores.get(str(vendor_id), 'N/A')}")
    
    # Create test Excel export
    print(f"\n8. CREATING TEST EXCEL EXPORT...")
    
    purchase_records = []
    for idx, row in df.iterrows():
        agent_id = row['agent_id']
        requests = row.get('purchase_requests', [])
        
        for req in requests:
            if isinstance(req, dict):
                vendor_id = req.get('vendorID')
                if vendor_id is not None:
                    purchase_records.append({
                        'Agent ID': agent_id,
                        'Request ID': req.get('request_id'),
                        'Vendor ID': vendor_id,
                        'Timestamp': req.get('timestamp_hours')
                    })
    
    if purchase_records:
        pr_df = pd.DataFrame(purchase_records)
        output_file = Path("test_vendor_4_issue.xlsx")
        pr_df.to_excel(output_file, index=False, sheet_name='Purchase Requests')
        print(f"   ✓ Excel export saved: {output_file}")
        print(f"   ✓ Contains {len(pr_df)} purchase requests from {len(agent_vendor_map)} agents")
        
        # Show vendor ID distribution in Excel
        vendor_excel_counts = pr_df['Vendor ID'].value_counts().sort_index()
        print(f"\n   Vendor distribution in Excel:")
        for vendor_id, count in vendor_excel_counts.items():
            print(f"     Vendor {vendor_id}: {count} rows")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nIf you see Vendor 4 dominating (>80%), this confirms the issue.")
    print("The root cause is likely vendor attribute randomization creating")
    print("a single dominant vendor with optimal price+sustainability scores.")
    print("=" * 80)

if __name__ == "__main__":
    test_vendor_assignment()

