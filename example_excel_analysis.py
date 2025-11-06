"""
Example: How to Extract Per-Request Data from Simulation Results Excel
Author: AI Agent Simulation Platform
Date: October 31, 2025

This script shows how to:
1. Load the Excel file
2. Extract per-request purchase data
3. Analyze decisions at the request level (not just agent level)
4. Separate data by time period if needed
"""

import pandas as pd
import json

# ============================================================================
# 1. LOAD THE EXCEL FILE
# ============================================================================

# Replace with your actual filename
excel_file = "enhanced_simulation_results_20251031_120000.xlsx"

# Read the Excel file
df = pd.read_excel(excel_file, sheet_name='Results')

print(f"Loaded {len(df)} agents")
print(f"Columns: {list(df.columns)}\n")

# ============================================================================
# 2. EXTRACT PER-REQUEST DATA
# ============================================================================

def extract_purchase_requests(df):
    """
    Converts nested purchase_requests column into a flat DataFrame
    where each row = one purchase request
    """
    all_requests = []
    
    for idx, row in df.iterrows():
        agent_id = row.get('Agent ID', idx + 1)
        
        # Get purchase requests (might be string or list)
        purchase_requests = row.get('purchase_requests', [])
        
        # If it's a string (JSON), parse it
        if isinstance(purchase_requests, str):
            try:
                purchase_requests = json.loads(purchase_requests)
            except:
                purchase_requests = []
        
        # If no requests, skip
        if not purchase_requests or not isinstance(purchase_requests, list):
            continue
        
        # Add agent-level data to each request
        for request in purchase_requests:
            request_data = {
                'agent_id': agent_id,
                'agent_honesty': row.get('Honesty_Humility'),
                'agent_allowance': row.get('Assigned Allowance Level'),
                'agent_donation_rate': row.get('donation_default'),
                **request  # Unpack request fields
            }
            all_requests.append(request_data)
    
    return pd.DataFrame(all_requests)

# Create per-request DataFrame
requests_df = extract_purchase_requests(df)

print(f"\nExtracted {len(requests_df)} purchase requests from {len(df)} agents")
print(f"Request columns: {list(requests_df.columns)}\n")

# ============================================================================
# 3. ANALYZE AT REQUEST LEVEL
# ============================================================================

# Example: Count purchase vs bid decisions across all requests
if 'platformPrice' in requests_df.columns:
    print("Platform Price Distribution (across all requests):")
    print(requests_df['platformPrice'].value_counts())
    print()

# Example: Average bid values (excluding N/A)
if 'bid_value' in requests_df.columns:
    bid_values = pd.to_numeric(requests_df['bid_value'], errors='coerce')
    valid_bids = bid_values.dropna()
    if len(valid_bids) > 0:
        print(f"Average bid value: ${valid_bids.mean():.2f}")
        print(f"Bid range: ${valid_bids.min():.2f} - ${valid_bids.max():.2f}\n")

# Example: Vendor selection distribution
if 'vendorID' in requests_df.columns:
    print("Vendor Selection Distribution:")
    print(requests_df['vendorID'].value_counts())
    print()

# ============================================================================
# 4. SEPARATE BY TIME PERIOD
# ============================================================================

def assign_period(timestamp_hours, duration_per_period):
    """
    Assigns a period number to each request based on timestamp
    
    Args:
        timestamp_hours: Time of request in hours (e.g., 2.34)
        duration_per_period: Length of each period in hours (e.g., 2.0)
    
    Returns:
        Period number (1, 2, 3, ...)
    """
    if pd.isna(timestamp_hours):
        return None
    return int(timestamp_hours // duration_per_period) + 1

# CONFIGURE YOUR SIMULATION PARAMETERS
NUM_PERIODS = 3          # From Page 1 of your simulation
DURATION_PER_PERIOD = 2.0  # Hours per period from Page 1

# Add period column
if 'timestamp_hours' in requests_df.columns:
    requests_df['period'] = requests_df['timestamp_hours'].apply(
        lambda t: assign_period(t, DURATION_PER_PERIOD)
    )
    
    print("Requests by Period:")
    print(requests_df['period'].value_counts().sort_index())
    print()
    
    # Example: Purchase vs Bid decisions by period
    if 'platformPrice' in requests_df.columns:
        print("Platform Price by Period:")
        period_breakdown = pd.crosstab(
            requests_df['period'], 
            requests_df['platformPrice']
        )
        print(period_breakdown)
        print()

# ============================================================================
# 5. AGENT-LEVEL SUMMARY STATISTICS
# ============================================================================

# Aggregate request-level data back to agent level
if len(requests_df) > 0:
    agent_summary = requests_df.groupby('agent_id').agg({
        'request_id': 'count',  # Total requests per agent
        'timestamp_hours': ['min', 'max'],  # Time range
        'bid_value': lambda x: pd.to_numeric(x, errors='coerce').mean()  # Avg bid
    })
    
    agent_summary.columns = ['total_requests', 'first_request_time', 
                              'last_request_time', 'avg_bid_value']
    
    print("Agent-Level Summary (first 5 agents):")
    print(agent_summary.head())
    print()

# ============================================================================
# 6. EXPORT PROCESSED DATA
# ============================================================================

# Option 1: Save per-request data to CSV
requests_df.to_csv('purchase_requests_detailed.csv', index=False)
print("✅ Saved detailed requests to: purchase_requests_detailed.csv")

# Option 2: Save period-separated sheets to Excel
if 'period' in requests_df.columns:
    with pd.ExcelWriter('requests_by_period.xlsx') as writer:
        for period in sorted(requests_df['period'].dropna().unique()):
            period_data = requests_df[requests_df['period'] == period]
            period_data.to_excel(writer, sheet_name=f'Period_{int(period)}', index=False)
    print("✅ Saved period-separated data to: requests_by_period.xlsx")

# Option 3: Save agent summary
if len(requests_df) > 0:
    agent_summary.to_csv('agent_summary_stats.csv')
    print("✅ Saved agent summary to: agent_summary_stats.csv")

# ============================================================================
# 7. MERGE WITH AGENT TRAITS FOR ANALYSIS
# ============================================================================

# Merge request-level data with full agent traits
requests_with_traits = requests_df.merge(
    df[['Agent ID', 'Honesty_Humility', 'Study Program', 'Group_experiment', 
        'donation_default', 'vendor_selection']],
    left_on='agent_id',
    right_on='Agent ID',
    how='left'
)

print("\n✅ Created merged dataset with traits + requests")
print(f"Shape: {requests_with_traits.shape}")
print(f"Sample columns: {list(requests_with_traits.columns)[:10]}")

# Save merged dataset
requests_with_traits.to_csv('requests_with_full_agent_data.csv', index=False)
print("✅ Saved to: requests_with_full_agent_data.csv")

print("\n" + "="*70)
print("COMPLETE! You now have:")
print("1. purchase_requests_detailed.csv - All requests in flat format")
print("2. requests_by_period.xlsx - Requests separated by period")
print("3. agent_summary_stats.csv - Aggregated agent statistics")
print("4. requests_with_full_agent_data.csv - Requests + agent traits")
print("="*70)


