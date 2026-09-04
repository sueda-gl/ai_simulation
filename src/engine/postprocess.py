# src/engine/postprocess.py
"""
Post-processing applied to a finished results frame.

Moved verbatim from app/simulation.py:30-72 (``_assign_global_transaction_ids``)
so the engine no longer depends on the Streamlit app layer.
"""

import pandas as pd


def assign_global_transaction_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign unique, chronologically ordered transaction IDs to all purchase requests across all agents.
    
    This ensures that Transaction IDs are consistent across different exports (Decision 6, Decision 9).
    
    Logic:
    1. Collect all requests from all agents.
    2. Sort globally by timestamp_hours.
    3. Assign sequential IDs (1, 2, 3...).
    4. Write IDs back to the agent's purchase_requests structure.
    """
    if 'purchase_requests' not in df.columns:
        return df
    
    # 1. Collect all requests with metadata to trace back
    # We need a flat list of (timestamp, agent_idx, req_idx, request_obj)
    all_requests = []
    
    for idx, row in df.iterrows():
        requests = row.get('purchase_requests', [])
        if isinstance(requests, list):
            for req_idx, req in enumerate(requests):
                if isinstance(req, dict):
                    # Store reference to the mutable dict object so we can update it in place
                    all_requests.append({
                        'timestamp': req.get('timestamp_hours', 0),
                        'request_obj': req
                    })
    
    # 2. Sort globally by timestamp
    # Use a stable sort to ensure determinism for identical timestamps
    all_requests.sort(key=lambda x: x['timestamp'])
    
    # 3. Assign sequential IDs
    for i, item in enumerate(all_requests):
        transaction_id = i + 1
        
        # Update the request object IN PLACE
        # This updates the dict inside the list inside the DataFrame
        item['request_obj']['transaction_id'] = transaction_id
        
    return df
