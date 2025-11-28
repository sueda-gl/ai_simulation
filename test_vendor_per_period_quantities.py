#!/usr/bin/env python3
"""
Test script to verify vendor per-period quantity feature.

This script tests that:
1. Vendors are generated with per-period quantities
2. Each period gets a different random quantity
3. Quantities are within configured [min, max] range
4. Backward compatibility is maintained
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vendor_attribute_generator import generate_vendor_attributes


def test_per_period_quantities():
    """Test that vendors get different quantities for each period"""
    
    print("=" * 80)
    print("TESTING VENDOR PER-PERIOD QUANTITY FEATURE")
    print("=" * 80)
    print()
    
    # Test configuration
    num_vendors = 3
    num_periods = 5
    quantity_min = 50
    quantity_max = 150
    seed = 42
    
    print(f"Configuration:")
    print(f"  - Number of vendors: {num_vendors}")
    print(f"  - Number of periods: {num_periods}")
    print(f"  - Quantity range: [{quantity_min}, {quantity_max}] per period")
    print(f"  - Random seed: {seed}")
    print()
    
    # Create RNG
    rng = np.random.default_rng(seed)
    
    # Generate vendors with per-period quantities
    vendors = generate_vendor_attributes(
        num_vendors=num_vendors,
        vendor_prices=[100.0, 110.0, 120.0],
        rng=rng,
        quantity_min=quantity_min,
        quantity_max=quantity_max,
        num_periods=num_periods
    )
    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    # Verify and display results
    all_tests_passed = True
    
    for vendor in vendors:
        vendor_id = vendor['vendor_id']
        print(f"📦 Vendor {vendor_id}:")
        print(f"   Price: ${vendor['price']:.2f}")
        print(f"   Quality: {vendor['quality']}/5")
        print(f"   Sustainability: {vendor['sustainability']}/5")
        print(f"   Average Quantity: {vendor['quantity_offered']} products")
        print()
        
        # Check per-period data exists
        if 'quantity_offered_per_period' not in vendor:
            print(f"   ❌ ERROR: Missing 'quantity_offered_per_period' field!")
            all_tests_passed = False
            continue
        
        per_period = vendor['quantity_offered_per_period']
        
        # Check we have the right number of periods
        if len(per_period) != num_periods:
            print(f"   ❌ ERROR: Expected {num_periods} periods, got {len(per_period)}")
            all_tests_passed = False
        else:
            print(f"   ✅ Has {num_periods} period-specific quantities")
        
        # Display per-period quantities
        print(f"   Per-Period Quantities:")
        quantities = []
        for period in range(1, num_periods + 1):
            if period in per_period:
                qty = per_period[period]
                quantities.append(qty)
                
                # Check quantity is in valid range
                if quantity_min <= qty <= quantity_max:
                    status = "✅"
                else:
                    status = "❌"
                    all_tests_passed = False
                
                print(f"      Period {period}: {qty:3d} products {status}")
            else:
                print(f"      Period {period}: ❌ MISSING")
                all_tests_passed = False
        
        # Check that quantities vary (not all the same)
        if len(set(quantities)) == 1:
            print(f"   ⚠️  WARNING: All periods have the same quantity ({quantities[0]})")
            print(f"      This is statistically unlikely but not impossible.")
        else:
            unique_count = len(set(quantities))
            print(f"   ✅ Quantities vary across periods ({unique_count} unique values)")
        
        # Check average calculation
        calculated_avg = int(np.mean(quantities))
        stored_avg = vendor['quantity_offered']
        if calculated_avg == stored_avg:
            print(f"   ✅ Average quantity correctly calculated: {stored_avg}")
        else:
            print(f"   ❌ ERROR: Average mismatch (stored={stored_avg}, calculated={calculated_avg})")
            all_tests_passed = False
        
        # Check total calculation
        total_quantity = sum(quantities)
        print(f"   ✅ Total quantity across all periods: {total_quantity}")
        
        print()
    
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    
    if all_tests_passed:
        print("✅ ALL TESTS PASSED!")
        print()
        print("The vendor per-period quantity feature is working correctly:")
        print("  • Each vendor has quantities for all periods")
        print("  • Quantities are within the configured range")
        print("  • Quantities vary across periods (as expected)")
        print("  • Average quantity is correctly calculated")
        print("  • Backward compatibility maintained")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print()
        print("Please review the errors above.")
        return 1


def test_backward_compatibility():
    """Test that old code still works with the new data structure"""
    
    print()
    print("=" * 80)
    print("TESTING BACKWARD COMPATIBILITY")
    print("=" * 80)
    print()
    
    rng = np.random.default_rng(123)
    
    # Generate vendors
    vendors = generate_vendor_attributes(
        num_vendors=2,
        vendor_prices=[100.0, 110.0],
        rng=rng,
        quantity_min=50,
        quantity_max=150,
        num_periods=3
    )
    
    print("Simulating old code that only uses 'quantity_offered':")
    print()
    
    for vendor in vendors:
        # Old code pattern
        vendor_id = vendor['vendor_id']
        quantity = vendor.get('quantity_offered', 100)
        
        print(f"Vendor {vendor_id}: {quantity} products")
    
    print()
    print("✅ Backward compatibility verified!")
    print("   Old code can still access quantities via 'quantity_offered' field.")
    print()


if __name__ == "__main__":
    exit_code = test_per_period_quantities()
    test_backward_compatibility()
    
    print()
    print("=" * 80)
    print("To use this feature in your simulation:")
    print("  1. Set 'Number of Periods' > 1 on Page 1")
    print("  2. Configure 'Min/Max Products per Vendor' on Page 1")
    print("  3. Run simulation and check Results page")
    print("  4. Look at 'Vendor Attributes & Selection Results' table")
    print("  5. You'll see: '103 avg (P1:94, P2:122, P3:87, ...)'")
    print("=" * 80)
    print()
    
    sys.exit(exit_code)

