#!/usr/bin/env python3
"""
Test script to verify the donation Excel export has:
1. Separate 'Purchase Date' and 'Purchase Time' columns
2. Separate 'Period' column
3. Records sorted by timestamp
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.orchestrator import Orchestrator
import pandas as pd
from datetime import datetime

def test_donation_excel_columns():
    """Test that Excel export has correct column structure and sorting"""
    
    print("=" * 80)
    print("TEST: Donation Excel Export - Column Structure and Sorting")
    print("=" * 80)
    print()
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Run simulation
    print("📊 Running simulation with 3 agents...")
    print("   Decisions: donation_default + purchasing_quantity + final_donation_rate")
    print()
    
    results_df = orchestrator.run_simulation(
        n_agents=3,
        seed=42,
        single_decision=['donation_default', 'purchasing_quantity', 'final_donation_rate']
    )
    
    print(f"✅ Simulation complete: {len(results_df)} agents")
    print()
    
    # Import the export function
    # Mock streamlit session state
    class MockSessionState:
        def __init__(self):
            self.simulation_params = {
                'simulation': {
                    'periods': 2,
                    'duration_hours': 12.0,
                    'market_price': 100.0,
                    'platform_markup': 0.1
                }
            }
    
    import streamlit as st
    st.session_state = MockSessionState()
    
    from app.pages.results.visualizations.donation_viz import _build_donation_transaction_export
    
    # Build transaction records
    print("🔧 Building transaction export...")
    transaction_records = _build_donation_transaction_export(results_df)
    
    if not transaction_records:
        print("❌ No transaction records generated")
        return False
    
    print(f"✅ Generated {len(transaction_records)} transaction records")
    print()
    
    # Convert to DataFrame
    transactions_df = pd.DataFrame(transaction_records)
    
    # Check columns
    print("-" * 80)
    print("VERIFICATION: Column Structure")
    print("-" * 80)
    print()
    
    expected_columns = [
        'Agent ID',
        'Assigned Allowance Level',
        'Group_experiment',
        'Customer Type',
        'Income Category',
        'Purchase Request Type',
        'Purchase Date',  # NEW!
        'Purchase Time',  # NEW!
        'Period',
        'Customer Price',
        'Transaction Completed',
        'Default Donation Rate',
        'Final Donation Rate',
        'Donation Paid',
        'Total Paid by Customer'
    ]
    
    print("Expected columns:")
    for col in expected_columns:
        exists = col in transactions_df.columns
        status = "✅" if exists else "❌"
        print(f"  {status} {col}")
    
    missing = set(expected_columns) - set(transactions_df.columns)
    extra = set(transactions_df.columns) - set(expected_columns)
    
    if missing:
        print(f"\n❌ Missing columns: {missing}")
    if extra:
        print(f"\n⚠️  Extra columns: {extra}")
    
    print()
    
    # Check for old combined column
    if 'Date/Time of Purchase Request' in transactions_df.columns:
        print("❌ Old column 'Date/Time of Purchase Request' still exists - should be removed!")
        columns_ok = False
    else:
        print("✅ Old combined column removed successfully")
        columns_ok = True
    
    print()
    
    # Check data types
    print("-" * 80)
    print("VERIFICATION: Data Types")
    print("-" * 80)
    print()
    
    print(f"Purchase Date type: {transactions_df['Purchase Date'].dtype}")
    print(f"Purchase Time type: {transactions_df['Purchase Time'].dtype}")
    print(f"Period type: {transactions_df['Period'].dtype}")
    print()
    
    # Show sample data
    print("-" * 80)
    print("VERIFICATION: Sample Data (First 5 Rows)")
    print("-" * 80)
    print()
    
    sample_cols = ['Agent ID', 'Purchase Date', 'Purchase Time', 'Period', 'Final Donation Rate']
    print(transactions_df[sample_cols].head(5).to_string(index=False))
    print()
    
    # Check sorting
    print("-" * 80)
    print("VERIFICATION: Chronological Sorting")
    print("-" * 80)
    print()
    
    # Combine date and time to check sorting
    transactions_df['_check_datetime'] = pd.to_datetime(
        transactions_df['Purchase Date'].astype(str) + ' ' + 
        transactions_df['Purchase Time'].astype(str)
    )
    
    is_sorted = transactions_df['_check_datetime'].is_monotonic_increasing
    
    if is_sorted:
        print("✅ Records are sorted chronologically!")
    else:
        print("❌ Records are NOT sorted chronologically")
        print("\nFirst 10 timestamps:")
        print(transactions_df[['Agent ID', '_check_datetime']].head(10).to_string(index=False))
    
    print()
    
    # Summary
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Column structure: {'✅ PASSED' if not missing else '❌ FAILED'}")
    print(f"Old column removed: {'✅ PASSED' if columns_ok else '❌ FAILED'}")
    print(f"Chronological sorting: {'✅ PASSED' if is_sorted else '❌ FAILED'}")
    print()
    
    all_passed = not missing and columns_ok and is_sorted
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    
    return all_passed


if __name__ == "__main__":
    print()
    print("🧪 Testing Donation Excel Column Structure")
    print()
    
    try:
        success = test_donation_excel_columns()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



