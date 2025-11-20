"""
Find simulations with multiple vendors to analyze vendor 4 issue.
"""

import pandas as pd
from pathlib import Path
import numpy as np

def check_simulations():
    print("=" * 80)
    print("SEARCHING FOR MULTI-VENDOR SIMULATIONS")
    print("=" * 80)
    
    outputs_dir = Path("outputs")
    parquet_files = sorted(outputs_dir.glob("enhanced_simulation_*.parquet"), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
    
    print(f"\nChecking {min(20, len(parquet_files))} most recent simulation files...\n")
    
    found_multi_vendor = []
    
    for i, file_path in enumerate(parquet_files[:20]):
        try:
            df = pd.read_parquet(file_path)
            
            # Check if vendors data is attached
            num_vendors = 0
            if hasattr(df, 'attrs') and 'vendors' in df.attrs:
                vendors = df.attrs['vendors']
                num_vendors = len(vendors)
            
            # Check vendor IDs in purchase requests
            vendor_ids_in_requests = set()
            if 'purchase_requests' in df.columns:
                import ast
                for requests in df['purchase_requests'].head(100):
                    if pd.notna(requests):
                        if isinstance(requests, str):
                            try:
                                requests = ast.literal_eval(requests)
                            except:
                                continue
                        if isinstance(requests, list):
                            for req in requests:
                                if isinstance(req, dict):
                                    vendor_id = req.get('vendorID')
                                    if vendor_id is not None:
                                        vendor_ids_in_requests.add(vendor_id)
            
            print(f"{i+1}. {file_path.name}")
            print(f"   Vendors in config: {num_vendors}")
            print(f"   Vendor IDs in requests: {sorted(vendor_ids_in_requests)}")
            
            if num_vendors >= 4 and 4 in vendor_ids_in_requests:
                found_multi_vendor.append(file_path)
                print(f"   ⭐ Found simulation with Vendor 4!")
            
            print()
            
        except Exception as e:
            print(f"{i+1}. {file_path.name} - Error: {e}\n")
    
    return found_multi_vendor

if __name__ == "__main__":
    multi_vendor_files = check_simulations()
    
    if multi_vendor_files:
        print("\n" + "=" * 80)
        print(f"ANALYZING FIRST FILE WITH VENDOR 4")
        print("=" * 80)
        
        file_path = multi_vendor_files[0]
        print(f"\nFile: {file_path.name}")
        
        df = pd.read_parquet(file_path)
        vendors = df.attrs.get('vendors', [])
        
        print(f"\n{len(vendors)} vendors configured:")
        for vendor in vendors:
            print(f"  Vendor {vendor['vendor_id']}: "
                  f"price=${vendor['price']:.2f}, "
                  f"quality={vendor['quality']}, "
                  f"sustainability={vendor['sustainability']}")
        
        # Analyze vendor distribution
        import ast
        vendor_counts = {}
        total_requests = 0
        
        for requests in df['purchase_requests']:
            if pd.notna(requests):
                if isinstance(requests, str):
                    try:
                        requests = ast.literal_eval(requests)
                    except:
                        continue
                if isinstance(requests, list):
                    for req in requests:
                        if isinstance(req, dict):
                            vendor_id = req.get('vendorID')
                            if vendor_id is not None:
                                vendor_counts[vendor_id] = vendor_counts.get(vendor_id, 0) + 1
                                total_requests += 1
        
        print(f"\nVendor distribution across {total_requests:,} requests:")
        for vendor_id in sorted(vendor_counts.keys()):
            count = vendor_counts[vendor_id]
            pct = count / total_requests * 100 if total_requests > 0 else 0
            print(f"  Vendor {vendor_id}: {count:,} ({pct:.1f}%)")
    else:
        print("\n⚠️  No simulations found with Vendor 4")

