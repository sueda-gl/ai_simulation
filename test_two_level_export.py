#!/usr/bin/env python3
"""
Test script for two-level Excel export functionality.

This script creates mock simulation data and tests the export functions
to verify that both agent-level and transaction-level DataFrames are
generated correctly.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the export functions
from app.pages.results.components.export_section import (
    _build_agent_level_dataframe,
    _build_transaction_level_dataframe
)


def create_mock_agent_data(n_agents=5, n_vendors=3):
    """
    Create mock agent data that simulates actual simulation results.
    
    Args:
        n_agents: Number of agents to create
        n_vendors: Number of vendors in simulation
        
    Returns:
        pd.DataFrame: Mock agent data with all required fields
    """
    print(f"Creating mock data for {n_agents} agents and {n_vendors} vendors...")
    
    agents = []
    
    for agent_id in range(1, n_agents + 1):
        # Generate purchase requests for this agent
        n_requests = np.random.randint(5, 15)
        purchase_requests = []
        
        for req_id in range(1, n_requests + 1):
            # Determine platform price (regular customer has PN/BID choices)
            platform_price = np.random.choice(['PN', 'BID'], p=[0.6, 0.4])
            
            request = {
                'request_id': req_id,
                'quantity': 1,
                'timestamp_hours': np.random.uniform(0, 6.0),  # 3 periods × 2 hours
                'customer_id': agent_id,
                'customer_type': 'regular',
                'vendorID': np.random.randint(1, n_vendors + 1),
                'platformPrice': platform_price,
                'bid_value': np.random.uniform(82.5, 137.5) if platform_price == 'BID' else 'N/A',
                'final_donation_rate': 0.30 + np.random.uniform(-0.1, 0.1)  # ~30% ± 10%
            }
            
            purchase_requests.append(request)
        
        # Create vendor proximity scores
        proximity_scores = {str(v_id): np.random.uniform(10, 100) for v_id in range(1, n_vendors + 1)}
        
        # Create agent data
        agent = {
            'agent_id': agent_id,
            'Honesty_Humility': np.random.uniform(2.0, 4.5),
            'Assigned Allowance Level': np.random.randint(1, 6),
            'Study Program': np.random.choice(['CLEAM', 'BESS', 'CLEF', 'Law']),
            'Group_experiment': np.random.choice(['HighSub', 'MidSub', 'NoSub']),
            'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': np.random.uniform(0, 50),
            
            # Decisions
            'disclose_income': np.random.choice(['Y', 'N']),
            'disclose_documents': 'NA',  # Regular customer
            'customer_type': 'regular',
            'donation_default': 0.30 + np.random.uniform(-0.1, 0.1),
            'rejected_transaction_defaults': 'wait',
            'vendor_choice_weights': {
                'price': 0.25,
                'quality': 0.25,
                'proximity': 0.25,
                'sustainability': 0.25
            },
            'income': np.random.lognormal(10, 0.5),
            'income_category': np.random.randint(1, 11),
            'purchasing_quantity': n_requests,
            'purchasing_frequency': n_requests / 3.0,  # 3 periods
            'preferred_vendor': np.random.randint(1, n_vendors + 1),
            'vendor_proximity_scores': proximity_scores,
            'purchase_vs_bid': 'Purchase Now',  # Legacy field
            'rejected_transaction_option': 'try_another',
            'final_donation_rate': 0.30 + np.random.uniform(-0.1, 0.1),
            
            # Purchase requests
            'purchase_requests': purchase_requests
        }
        
        agents.append(agent)
    
    df = pd.DataFrame(agents)
    print(f"✓ Created mock data with {len(df)} agents")
    return df


def create_mock_vendors(n_vendors=3):
    """
    Create mock vendor data.
    
    Args:
        n_vendors: Number of vendors
        
    Returns:
        list: List of vendor dictionaries
    """
    print(f"Creating {n_vendors} mock vendors...")
    
    vendors = []
    for v_id in range(1, n_vendors + 1):
        vendor = {
            'vendor_id': v_id,
            'price': 90 + np.random.uniform(0, 20),
            'quality': np.random.randint(1, 6),
            'sustainability': np.random.randint(1, 6)
        }
        vendors.append(vendor)
    
    print(f"✓ Created {len(vendors)} vendors")
    return vendors


def verify_agent_level_dataframe(agent_df, expected_agents=5):
    """Verify agent-level DataFrame structure and content."""
    print("\n" + "="*60)
    print("VERIFYING AGENT-LEVEL DATAFRAME")
    print("="*60)
    
    # Check shape
    print(f"\n1. Shape: {agent_df.shape}")
    assert agent_df.shape[0] == expected_agents, f"Expected {expected_agents} agents, got {agent_df.shape[0]}"
    print(f"   ✓ Correct number of agents: {expected_agents}")
    
    # Check required columns
    print(f"\n2. Columns: {agent_df.shape[1]} total")
    
    required_columns = [
        'Agent ID',
        'Honesty_Humility',
        'Assigned Allowance Level',
        'disclose_income',
        'customer_type',
        'donation_default',
        'weight_price',
        'weight_quality',
        'weight_proximity',
        'weight_sustainability',
        'purchasing_quantity',
        'preferred_vendor',
        'total_purchase_requests',
        'pn_requests_count',
        'bid_requests_count',
        'final_donation_rate'
    ]
    
    for col in required_columns:
        assert col in agent_df.columns, f"Missing required column: {col}"
        print(f"   ✓ {col}")
    
    # Check proximity columns
    proximity_cols = [col for col in agent_df.columns if col.startswith('proximity_v')]
    print(f"\n3. Proximity columns: {len(proximity_cols)}")
    for col in proximity_cols:
        print(f"   ✓ {col}")
    
    # Check data integrity
    print(f"\n4. Data Integrity:")
    print(f"   - Agent IDs: {agent_df['Agent ID'].min()} to {agent_df['Agent ID'].max()}")
    print(f"   - Donation rates: {agent_df['donation_default'].min():.3f} to {agent_df['donation_default'].max():.3f}")
    print(f"   - Avg requests per agent: {agent_df['purchasing_quantity'].mean():.1f}")
    print(f"   - PN requests: {agent_df['pn_requests_count'].sum()}")
    print(f"   - BID requests: {agent_df['bid_requests_count'].sum()}")
    
    # Show sample data
    print(f"\n5. Sample Data (first 3 agents):")
    display_cols = ['Agent ID', 'customer_type', 'donation_default', 'purchasing_quantity', 'pn_requests_count', 'bid_requests_count']
    print(agent_df[display_cols].head(3).to_string(index=False))
    
    print(f"\n✅ Agent-level DataFrame verification PASSED")
    return True


def verify_transaction_level_dataframe(transaction_df, agent_df):
    """Verify transaction-level DataFrame structure and content."""
    print("\n" + "="*60)
    print("VERIFYING TRANSACTION-LEVEL DATAFRAME")
    print("="*60)
    
    # Check shape
    print(f"\n1. Shape: {transaction_df.shape}")
    expected_transactions = agent_df['purchasing_quantity'].sum()
    assert transaction_df.shape[0] == expected_transactions, \
        f"Expected {expected_transactions} transactions, got {transaction_df.shape[0]}"
    print(f"   ✓ Correct number of transactions: {expected_transactions}")
    
    # Check required columns
    print(f"\n2. Columns: {transaction_df.shape[1]} total")
    
    required_columns = [
        'Transaction ID',
        'Agent ID',
        'Request ID',
        'Customer Type',
        'Timestamp (hours)',
        'Period',
        'Vendor ID',
        'Platform Price',
        'Purchase Request Type',
        'Customer Price',
        'Final Donation Rate',
        'Donation Paid',
        'Total Paid'
    ]
    
    for col in required_columns:
        assert col in transaction_df.columns, f"Missing required column: {col}"
        print(f"   ✓ {col}")
    
    # Check transaction IDs are unique
    print(f"\n3. Transaction ID Uniqueness:")
    n_unique = transaction_df['Transaction ID'].nunique()
    assert n_unique == len(transaction_df), "Transaction IDs not unique!"
    print(f"   ✓ All {n_unique} transaction IDs are unique")
    
    # Check Agent ID linkage
    print(f"\n4. Agent ID Linkage:")
    agent_ids_transactions = set(transaction_df['Agent ID'].unique())
    agent_ids_agents = set(agent_df['Agent ID'].unique())
    assert agent_ids_transactions == agent_ids_agents, "Agent ID mismatch between sheets!"
    print(f"   ✓ All agent IDs match between sheets")
    
    # Check purchase types
    print(f"\n5. Purchase Types:")
    purchase_types = transaction_df['Purchase Request Type'].value_counts()
    for ptype, count in purchase_types.items():
        print(f"   - {ptype}: {count}")
    
    # Check pricing
    print(f"\n6. Pricing Summary:")
    print(f"   - Customer Price range: ${transaction_df['Customer Price'].min():.2f} - ${transaction_df['Customer Price'].max():.2f}")
    print(f"   - Avg donation rate: {transaction_df['Final Donation Rate'].mean():.3f}")
    print(f"   - Total donations: ${transaction_df['Donation Paid'].sum():.2f}")
    print(f"   - Total paid: ${transaction_df['Total Paid'].sum():.2f}")
    
    # Check bid values
    bid_transactions = transaction_df[transaction_df['Purchase Request Type'] == 'Bid']
    if len(bid_transactions) > 0:
        print(f"\n7. Bid Values:")
        print(f"   - Number of bids: {len(bid_transactions)}")
        bid_values = bid_transactions['Bid Value']
        # Convert to numeric (some might be 'N/A')
        numeric_bids = pd.to_numeric(bid_values, errors='coerce').dropna()
        if len(numeric_bids) > 0:
            print(f"   - Bid range: ${numeric_bids.min():.2f} - ${numeric_bids.max():.2f}")
            print(f"   - Avg bid: ${numeric_bids.mean():.2f}")
    
    # Show sample data
    print(f"\n8. Sample Data (first 5 transactions):")
    display_cols = ['Transaction ID', 'Agent ID', 'Period', 'Purchase Request Type', 'Customer Price', 'Donation Paid']
    print(transaction_df[display_cols].head(5).to_string(index=False))
    
    print(f"\n✅ Transaction-level DataFrame verification PASSED")
    return True


def verify_cross_level_consistency(agent_df, transaction_df):
    """Verify consistency between agent and transaction levels."""
    print("\n" + "="*60)
    print("VERIFYING CROSS-LEVEL CONSISTENCY")
    print("="*60)
    
    print("\n1. Checking purchase request counts...")
    for _, agent in agent_df.iterrows():
        agent_id = agent['Agent ID']
        expected_count = agent['purchasing_quantity']
        actual_count = len(transaction_df[transaction_df['Agent ID'] == agent_id])
        assert expected_count == actual_count, \
            f"Agent {agent_id}: expected {expected_count} transactions, got {actual_count}"
    print(f"   ✓ All agents have correct transaction counts")
    
    print("\n2. Checking PN/BID counts...")
    for _, agent in agent_df.iterrows():
        agent_id = agent['Agent ID']
        agent_transactions = transaction_df[transaction_df['Agent ID'] == agent_id]
        
        expected_pn = agent['pn_requests_count']
        actual_pn = len(agent_transactions[agent_transactions['Purchase Request Type'] == 'Purchase Now'])
        assert expected_pn == actual_pn, \
            f"Agent {agent_id}: expected {expected_pn} PN requests, got {actual_pn}"
        
        expected_bid = agent['bid_requests_count']
        actual_bid = len(agent_transactions[agent_transactions['Purchase Request Type'] == 'Bid'])
        assert expected_bid == actual_bid, \
            f"Agent {agent_id}: expected {expected_bid} BID requests, got {actual_bid}"
    print(f"   ✓ PN/BID counts match for all agents")
    
    print("\n3. Checking donation rates...")
    for _, agent in agent_df.iterrows():
        agent_id = agent['Agent ID']
        agent_donation_rate = agent['donation_default']
        agent_transactions = transaction_df[transaction_df['Agent ID'] == agent_id]
        
        # Transaction donation rates should be close to agent rate (allowing for small variations)
        for _, trans in agent_transactions.iterrows():
            trans_rate = trans['Final Donation Rate']
            # Allow 20% deviation (since we added random variation in mock data)
            assert abs(trans_rate - agent_donation_rate) < 0.20, \
                f"Agent {agent_id}: transaction donation rate {trans_rate:.3f} too far from agent rate {agent_donation_rate:.3f}"
    print(f"   ✓ Donation rates consistent across levels")
    
    print(f"\n✅ Cross-level consistency verification PASSED")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("TWO-LEVEL EXCEL EXPORT TEST")
    print("="*60)
    
    try:
        # Create mock data
        n_agents = 10
        n_vendors = 4
        
        df = create_mock_agent_data(n_agents=n_agents, n_vendors=n_vendors)
        vendors = create_mock_vendors(n_vendors=n_vendors)
        
        # Build DataFrames
        print("\n" + "-"*60)
        print("Building DataFrames...")
        print("-"*60)
        
        agent_df = _build_agent_level_dataframe(df, vendors_data=vendors)
        print(f"✓ Agent-level DataFrame built: {agent_df.shape}")
        
        transaction_df = _build_transaction_level_dataframe(df, vendors_data=vendors)
        print(f"✓ Transaction-level DataFrame built: {transaction_df.shape}")
        
        # Verify DataFrames
        verify_agent_level_dataframe(agent_df, expected_agents=n_agents)
        verify_transaction_level_dataframe(transaction_df, agent_df)
        verify_cross_level_consistency(agent_df, transaction_df)
        
        # Test Excel export
        print("\n" + "="*60)
        print("TESTING EXCEL EXPORT")
        print("="*60)
        
        from io import BytesIO
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            agent_df.to_excel(writer, index=False, sheet_name='Agent Level')
            transaction_df.to_excel(writer, index=False, sheet_name='Transaction Level')
        
        excel_size = len(buffer.getvalue())
        print(f"\n✓ Excel file created successfully")
        print(f"  - File size: {excel_size:,} bytes ({excel_size/1024:.1f} KB)")
        
        # Optionally save to file
        test_filename = f"test_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        with open(test_filename, 'wb') as f:
            f.write(buffer.getvalue())
        print(f"  - Saved to: {test_filename}")
        
        # Final summary
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print(f"\nSummary:")
        print(f"  - Agents: {len(agent_df)}")
        print(f"  - Transactions: {len(transaction_df)}")
        print(f"  - Agent-level columns: {len(agent_df.columns)}")
        print(f"  - Transaction-level columns: {len(transaction_df.columns)}")
        print(f"  - Excel file: {test_filename}")
        print(f"\nThe two-level Excel export system is working correctly! ✨")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

