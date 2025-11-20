"""
Analyze the most recent simulation to understand vendor assignment patterns.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import ast

sys.path.insert(0, str(Path(__file__).parent))

def analyze_simulation():
    print("=" * 80)
    print("RECENT SIMULATION ANALYSIS")
    print("=" * 80)
    
    # Find most recent parquet file
    outputs_dir = Path("outputs")
    parquet_files = list(outputs_dir.glob("enhanced_simulation_*.parquet"))
    
    if not parquet_files:
        print("No simulation results found!")
        return
    
    # Get most recent file
    latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
    print(f"\nAnalyzing: {latest_file.name}")
    print(f"File size: {latest_file.stat().st_size / 1024:.1f} KB")
    
    # Load the simulation
    df = pd.read_parquet(latest_file)
    print(f"\nSimulation shape: {df.shape}")
    print(f"Number of agents: {len(df)}")
    
    # Check if vendors data is attached
    if hasattr(df, 'attrs') and 'vendors' in df.attrs:
        vendors = df.attrs['vendors']
        print(f"\n" + "=" * 80)
        print("VENDOR ATTRIBUTES")
        print("=" * 80)
        for vendor in vendors:
            print(f"\nVendor {vendor['vendor_id']}:")
            print(f"  Price: ${vendor['price']:.2f}")
            print(f"  Quality: {vendor['quality']}/5")
            print(f"  Sustainability: {vendor['sustainability']}/5")
            if 'quantity_offered' in vendor:
                print(f"  Capacity: {vendor['quantity_offered']} products")
            else:
                print(f"  Capacity: N/A")
    else:
        print("\n⚠️  No vendor data found in file attributes")
    
    # Check vendor_choice_weights
    if 'vendor_choice_weights' in df.columns:
        print(f"\n" + "=" * 80)
        print("VENDOR CHOICE WEIGHTS")
        print("=" * 80)
        
        # Get first agent's weights
        first_weights = df['vendor_choice_weights'].iloc[0]
        if isinstance(first_weights, dict):
            print(f"\nWeights (from first agent):")
            for key, value in first_weights.items():
                print(f"  {key}: {value}")
        else:
            print(f"  Weights: {first_weights}")
        
        # Check if all agents have same weights
        try:
            weights_str = df['vendor_choice_weights'].astype(str)
            unique_weights = weights_str.nunique()
            print(f"\nUnique weight configurations: {unique_weights}")
            if unique_weights > 1:
                print("  (Agents have different weights)")
            else:
                print("  (All agents have identical weights)")
        except:
            pass
    
    # Analyze purchase requests and vendor IDs
    if 'purchase_requests' in df.columns:
        print(f"\n" + "=" * 80)
        print("PURCHASE REQUEST ANALYSIS")
        print("=" * 80)
        
        vendor_id_counts = {}
        agents_with_requests = 0
        total_requests = 0
        
        for idx, row in df.iterrows():
            agent_id = row.get('agent_id', idx + 1)
            requests = row['purchase_requests']
            
            if pd.isna(requests):
                continue
            
            # Parse requests
            if isinstance(requests, str):
                try:
                    requests = ast.literal_eval(requests)
                except:
                    continue
            
            if not isinstance(requests, list) or len(requests) == 0:
                continue
            
            agents_with_requests += 1
            
            # Count vendor IDs
            for req in requests:
                if isinstance(req, dict):
                    vendor_id = req.get('vendorID', 'unknown')
                    vendor_id_counts[vendor_id] = vendor_id_counts.get(vendor_id, 0) + 1
                    total_requests += 1
        
        print(f"\nTotal purchase requests: {total_requests:,}")
        print(f"Agents with requests: {agents_with_requests:,} / {len(df):,}")
        
        print(f"\nVendor ID distribution across ALL requests:")
        for vendor_id in sorted(vendor_id_counts.keys()):
            count = vendor_id_counts[vendor_id]
            pct = count / total_requests * 100 if total_requests > 0 else 0
            print(f"  Vendor {vendor_id}: {count:,} requests ({pct:.1f}%)")
        
        # Sample 10 agents
        print(f"\nSample: First 10 agents with requests:")
        sample_count = 0
        for idx, row in df.iterrows():
            if sample_count >= 10:
                break
            
            agent_id = row.get('agent_id', idx + 1)
            requests = row['purchase_requests']
            
            if pd.isna(requests):
                continue
            
            # Parse requests
            if isinstance(requests, str):
                try:
                    requests = ast.literal_eval(requests)
                except:
                    continue
            
            if not isinstance(requests, list) or len(requests) == 0:
                continue
            
            # Get vendor IDs
            vendor_ids = set()
            for req in requests:
                if isinstance(req, dict):
                    vendor_ids.add(req.get('vendorID', 'unknown'))
            
            print(f"  Agent {agent_id}: {len(requests)} requests, vendors: {sorted(vendor_ids)}")
            sample_count += 1
    
    # Check vendor_selection column
    if 'vendor_selection' in df.columns:
        print(f"\n" + "=" * 80)
        print("VENDOR SELECTION DECISION")
        print("=" * 80)
        
        vendor_selection_counts = df['vendor_selection'].value_counts()
        print(f"\nVendor selection distribution (agent-level):")
        for vendor_id, count in vendor_selection_counts.items():
            pct = count / len(df) * 100
            print(f"  Vendor {vendor_id}: {count} agents ({pct:.1f}%)")
        
        # Check for NaN values
        nan_count = df['vendor_selection'].isna().sum()
        if nan_count > 0:
            print(f"\n  ⚠️  {nan_count} agents have NaN vendor_selection")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    analyze_simulation()

