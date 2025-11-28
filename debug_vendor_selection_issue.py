"""
Debug script to investigate why all agents have vendor 4 as their vendorID.

This script will:
1. Load the most recent simulation results
2. Check the vendor configuration
3. Analyze vendor selection patterns
4. Check if vendor scoring is working correctly
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def debug_vendor_selection():
    print("="*80)
    print("DEBUGGING VENDOR SELECTION ISSUE")
    print("="*80)
    
    # Find most recent simulation output
    outputs_dir = Path("/Users/suedagul/<sdg/outputs")
    if not outputs_dir.exists():
        print("❌ No outputs directory found")
        return
    
    # Find most recent Excel file
    excel_files = list(outputs_dir.glob("simulation_results_*.xlsx"))
    if not excel_files:
        print("❌ No simulation result files found")
        return
    
    most_recent = max(excel_files, key=lambda p: p.stat().st_mtime)
    print(f"\n📂 Loading: {most_recent.name}")
    
    # Load the results
    try:
        df = pd.read_excel(most_recent)
        print(f"✅ Loaded {len(df)} agents")
    except Exception as e:
        print(f"❌ Error loading Excel: {e}")
        return
    
    # Check vendor_selection column
    print("\n" + "="*80)
    print("CHECKING VENDOR SELECTION COLUMN")
    print("="*80)
    
    if 'vendor_selection' in df.columns:
        vendor_counts = df['vendor_selection'].value_counts().sort_index()
        print("\nVendor Selection Distribution:")
        for vendor_id, count in vendor_counts.items():
            if not pd.isna(vendor_id):
                print(f"  Vendor {int(vendor_id)}: {count} agents ({count/len(df)*100:.1f}%)")
        
        nan_count = df['vendor_selection'].isna().sum()
        if nan_count > 0:
            print(f"  No vendor (NaN): {nan_count} agents ({nan_count/len(df)*100:.1f}%)")
    else:
        print("❌ No 'vendor_selection' column found")
    
    # Check purchase_requests column for vendorID
    print("\n" + "="*80)
    print("CHECKING PURCHASE REQUESTS - VENDOR IDs")
    print("="*80)
    
    if 'purchase_requests' in df.columns:
        all_vendor_ids = []
        agents_with_requests = 0
        
        for idx, row in df.iterrows():
            agent_id = row.get('agent_id', idx + 1)
            purchase_requests = row['purchase_requests']
            
            if not isinstance(purchase_requests, list) or len(purchase_requests) == 0:
                continue
            
            agents_with_requests += 1
            
            # Extract vendor IDs from this agent's requests
            for req in purchase_requests:
                if isinstance(req, dict):
                    vendor_id = req.get('vendorID')
                    if vendor_id is not None:
                        all_vendor_ids.append(vendor_id)
        
        print(f"\nTotal agents with purchase requests: {agents_with_requests}")
        print(f"Total purchase requests: {len(all_vendor_ids)}")
        
        if all_vendor_ids:
            vendor_id_counts = pd.Series(all_vendor_ids).value_counts().sort_index()
            print("\nVendor ID Distribution in Purchase Requests:")
            for vendor_id, count in vendor_id_counts.items():
                print(f"  Vendor {int(vendor_id)}: {count} requests ({count/len(all_vendor_ids)*100:.1f}%)")
            
            # Check if all vendor IDs are the same
            unique_vendors = pd.Series(all_vendor_ids).nunique()
            if unique_vendors == 1:
                print(f"\n⚠️  WARNING: ALL requests have the same vendor ID: {all_vendor_ids[0]}")
            else:
                print(f"\n✅ Multiple vendors selected ({unique_vendors} different vendors)")
        else:
            print("❌ No vendor IDs found in purchase requests")
    else:
        print("❌ No 'purchase_requests' column found")
    
    # Check vendor configuration
    print("\n" + "="*80)
    print("CHECKING VENDOR CONFIGURATION")
    print("="*80)
    
    # Try to find the config file used
    config_files = list(Path("/Users/suedagul/<sdg/config").glob("*.yaml"))
    if config_files:
        print(f"\nFound {len(config_files)} config files:")
        for config_file in config_files:
            print(f"  - {config_file.name}")
        
        # Try to load default config
        default_config = Path("/Users/suedagul/<sdg/config/default_config.yaml")
        if default_config.exists():
            try:
                with open(default_config, 'r') as f:
                    config = yaml.safe_load(f)
                
                if 'vendors' in config:
                    vendors = config['vendors']
                    print(f"\n✅ Found {len(vendors)} vendors in config:")
                    for vendor in vendors:
                        print(f"  Vendor {vendor['vendor_id']}:")
                        print(f"    - Price: ${vendor['price']}")
                        print(f"    - Quality: {vendor['quality']}/5")
                        print(f"    - Sustainability: {vendor['sustainability']}/5")
                        print(f"    - Max Products: {vendor['max_products']}")
                else:
                    print("❌ No 'vendors' key in config")
            except Exception as e:
                print(f"❌ Error loading config: {e}")
    
    # Sample a few agents to check their vendor scoring
    print("\n" + "="*80)
    print("SAMPLING AGENT VENDOR PREFERENCES")
    print("="*80)
    
    if 'purchase_requests' in df.columns and 'vendor_choice_weights' in df.columns:
        sample_size = min(5, len(df))
        sample_agents = df.head(sample_size)
        
        for idx, row in sample_agents.iterrows():
            agent_id = row.get('agent_id', idx + 1)
            weights = row.get('vendor_choice_weights', {})
            purchase_requests = row.get('purchase_requests', [])
            
            if isinstance(purchase_requests, list) and len(purchase_requests) > 0:
                first_request = purchase_requests[0]
                vendor_id = first_request.get('vendorID', 'N/A')
                
                print(f"\nAgent {agent_id}:")
                print(f"  Weights: {weights}")
                print(f"  Selected Vendor: {vendor_id}")
                print(f"  Number of requests: {len(purchase_requests)}")

if __name__ == "__main__":
    debug_vendor_selection()



