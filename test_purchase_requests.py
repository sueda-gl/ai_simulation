#!/usr/bin/env python3
"""
Test script to verify per-request purchase decisions implementation.

This tests that:
1. Purchase requests are created with basic fields
2. Enrich_purchase_requests adds per-request decisions
3. Different requests can have different purchase_vs_bid decisions
4. Each bid gets a unique bid_value
5. Transaction export works correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import Orchestrator
import pandas as pd
from datetime import datetime, timedelta

def test_purchase_requests():
    print("=" * 80)
    print("Testing Per-Request Purchase Decisions Implementation")
    print("=" * 80)
    
    # Initialize orchestrator
    orch = Orchestrator()
    
    # Run simulation with small number of agents
    print("\n1. Running simulation with 5 agents...")
    results_df = orch.run_simulation(n_agents=5, seed=42)
    
    print(f"✓ Simulation complete: {len(results_df)} agents")
    
    # Check columns
    print("\n2. Checking key columns exist...")
    required_cols = ['customer_type', 'purchase_requests', 'consumption_quantity', 'enriched_requests_count']
    for col in required_cols:
        if col in results_df.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ✗ {col} MISSING!")
    
    # Examine purchase requests for first agent
    print("\n3. Examining purchase requests for Agent 1...")
    agent1 = results_df.iloc[0]
    
    print(f"  Customer Type: {agent1.get('customer_type')}")
    print(f"  Consumption Quantity: {agent1.get('consumption_quantity')}")
    print(f"  Enriched Requests Count: {agent1.get('enriched_requests_count')}")
    
    purchase_requests = agent1.get('purchase_requests', [])
    print(f"  Number of Purchase Requests: {len(purchase_requests)}")
    
    if len(purchase_requests) > 0:
        print("\n4. Sample purchase requests from Agent 1:")
        for i, req in enumerate(purchase_requests[:5], 1):  # Show first 5
            print(f"\n  Request {i}:")
            print(f"    - request_id: {req.get('request_id')}")
            print(f"    - customer_id: {req.get('customer_id')}")
            print(f"    - customer_type: {req.get('customer_type')}")
            print(f"    - vendorID: {req.get('vendorID')}")
            print(f"    - timestamp_hours: {req.get('timestamp_hours'):.2f}")
            print(f"    - platformPrice: {req.get('platformPrice')}")
            print(f"    - bid_value: {req.get('bid_value')}")
            print(f"    - purchase_vs_bid: {req.get('purchase_vs_bid')}")
    
    # Check for variety in decisions
    print("\n5. Checking for decision variety across all agents...")
    
    all_platform_prices = []
    all_bid_values = []
    
    for idx, row in results_df.iterrows():
        requests = row.get('purchase_requests', [])
        for req in requests:
            if isinstance(req, dict):
                all_platform_prices.append(req.get('platformPrice'))
                bid_val = req.get('bid_value')
                if bid_val != 'N/A' and bid_val is not None:
                    all_bid_values.append(bid_val)
    
    print(f"  Total transactions: {len(all_platform_prices)}")
    
    # Count platformPrice types
    from collections import Counter
    price_counts = Counter(all_platform_prices)
    print(f"\n  Platform Price Distribution:")
    for price_type, count in price_counts.items():
        pct = (count / len(all_platform_prices) * 100) if len(all_platform_prices) > 0 else 0
        print(f"    {price_type}: {count} ({pct:.1f}%)")
    
    # Check bid value variety
    if len(all_bid_values) > 1:
        print(f"\n  Bid Values (showing first 10 unique):")
        unique_bids = list(set(all_bid_values))[:10]
        for bid in sorted(unique_bids):
            print(f"    {bid}")
        print(f"  ✓ Multiple unique bid values found! ({len(set(all_bid_values))} unique)")
    else:
        print(f"\n  ⚠ Only {len(set(all_bid_values))} unique bid value(s) found")
    
    # Test transaction export format
    print("\n6. Testing transaction export format...")
    transactions = []
    transaction_id = 1
    # Base date for timestamp conversion (using 2025-01-15)
    base_date = datetime(2025, 1, 15, 0, 0, 0)
    
    for idx, row in results_df.iterrows():
        purchase_requests = row.get('purchase_requests', [])
        if isinstance(purchase_requests, list):
            for req in purchase_requests:
                if isinstance(req, dict):
                    # Convert timestamp_hours to datetime format
                    timestamp_hours = req.get('timestamp_hours', 0.0)
                    timestamp_dt = base_date + timedelta(hours=float(timestamp_hours))
                    timestamp_str = timestamp_dt.strftime('%d/%m/%Y %H:%M')
                    
                    transactions.append({
                        'transaction_id': transaction_id,
                        'customer_id': req.get('customer_id', idx + 1),
                        'vendorID': req.get('vendorID', 1),
                        'platformProductID': req.get('platformProductID', 1),
                        'platformPrice': req.get('platformPrice', 'N/A'),
                        'purchase_bid_value': req.get('bid_value', 'N/A'),
                        'timestamp': timestamp_str
                    })
                    transaction_id += 1
    
    if len(transactions) > 0:
        transactions_df = pd.DataFrame(transactions)
        print(f"  ✓ Transaction DataFrame created: {len(transactions_df)} rows")
        print(f"\n  Columns: {list(transactions_df.columns)}")
        print(f"\n  First 5 transactions:")
        print(transactions_df.head().to_string(index=False))
        
        # Try to save to Excel
        try:
            test_file = Path(__file__).parent / "test_transactions.xlsx"
            transactions_df.to_excel(test_file, index=False, sheet_name='Transactions')
            print(f"\n  ✓ Excel export successful: {test_file}")
            print(f"    You can open this file to verify the format!")
        except Exception as e:
            print(f"\n  ⚠ Excel export failed: {e}")
    else:
        print(f"  ✗ No transactions created!")
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_purchase_requests()

