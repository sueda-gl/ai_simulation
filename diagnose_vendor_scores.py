"""
Diagnostic script to investigate vendor selection issue.

This script will:
1. Run a small simulation (10 agents)
2. Generate a detailed vendor scores table showing how each agent scores each vendor
3. Generate a proximity matrix showing agent-vendor proximity scores
4. Export both tables to Excel for investigation
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import Orchestrator
from src.vendor_attribute_generator import (
    generate_proximity_scores,
    calculate_vendor_composite_score
)
import yaml


def diagnose_vendor_scores(num_agents=10):
    """
    Run a small simulation and generate diagnostic tables for vendor scoring.
    """
    
    print("="*80)
    print("VENDOR SCORING DIAGNOSTICS")
    print("="*80)
    
    # Load configuration
    config_path = Path("/Users/suedagul/<sdg/config/simulation.yaml")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # IMPORTANT: Override num_vendors to test with multiple vendors
    # The default config has num_vendors=1, which is why everyone selects the same vendor!
    if 'simulation' not in config:
        config['simulation'] = {}
    
    config['simulation']['num_vendors'] = 5  # Override to 5 vendors for testing
    config['simulation']['num_agents'] = num_agents
    
    print(f"⚠️  NOTE: Default config has num_vendors=1 (that's the root cause!)")
    print(f"   Overriding to 5 vendors for diagnostic purposes...")
    
    # Create orchestrator and run simulation
    print(f"\n🔄 Running simulation with {num_agents} agents and 5 vendors...")
    orchestrator = Orchestrator()
    orchestrator.simulation_config = config
    
    # Run simulation
    results_df = orchestrator.run_simulation(n_agents=num_agents, seed=42)
    
    print(f"✅ Simulation complete: {len(results_df)} agents")
    
    # Get vendors from simulation config (they were generated during initialization)
    vendors = orchestrator.simulation_config.get('vendors', [])
    if not vendors:
        print("❌ No vendors found after simulation!")
        return
    
    print(f"✅ Generated {len(vendors)} vendors:")
    for vendor in vendors:
        print(f"  Vendor {vendor['vendor_id']}: Price=${vendor['price']:.2f}, "
              f"Quality={vendor['quality']}, Sustainability={vendor['sustainability']}")
    
    # =========================================================================
    # TABLE 1: DETAILED VENDOR SCORES BY AGENT
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING VENDOR SCORES TABLE")
    print("="*80)
    
    vendor_scores_records = []
    
    for idx, row in results_df.iterrows():
        agent_id = row.get('agent_id', idx + 1)
        
        # Get agent's vendor choice weights
        weights = row.get('vendor_choice_weights', {
            'price': 0.25,
            'quality': 0.25,
            'proximity': 0.25,
            'sustainability': 0.25
        })
        
        # Get agent's proximity scores
        proximity_scores = row.get('vendor_proximity_scores', {})
        
        # Get agent's selected vendor
        selected_vendor = row.get('vendor_selection', np.nan)
        
        # Get purchase requests to check vendorID
        purchase_requests = row.get('purchase_requests', [])
        vendor_in_requests = None
        if isinstance(purchase_requests, list) and len(purchase_requests) > 0:
            first_request = purchase_requests[0]
            if isinstance(first_request, dict):
                vendor_in_requests = first_request.get('vendorID', np.nan)
        
        # Calculate score for each vendor
        for vendor in vendors:
            vendor_id = vendor['vendor_id']
            
            # Get proximity for this agent-vendor pair
            proximity = proximity_scores.get(str(vendor_id), 50.0)
            
            # Calculate composite score
            composite_score = calculate_vendor_composite_score(
                vendor, weights, proximity, vendors
            )
            
            # Calculate normalized components for transparency
            # Price normalization (inverted)
            all_prices = [v['price'] for v in vendors]
            min_price = min(all_prices)
            max_price = max(all_prices)
            if max_price > min_price:
                norm_price = 1.0 - (vendor['price'] - min_price) / (max_price - min_price)
            else:
                norm_price = 1.0
            
            # Quality normalization
            norm_quality = (vendor['quality'] - 1) / 4.0
            
            # Sustainability normalization
            norm_sustainability = (vendor['sustainability'] - 1) / 4.0
            
            # Proximity normalization
            norm_proximity = proximity / 100.0
            
            # Weighted components
            weighted_price = weights.get('price', 0.0) * norm_price
            weighted_quality = weights.get('quality', 0.0) * norm_quality
            weighted_proximity = weights.get('proximity', 0.0) * norm_proximity
            weighted_sustainability = weights.get('sustainability', 0.0) * norm_sustainability
            
            # Is this the selected vendor?
            is_selected = (vendor_id == selected_vendor) or (vendor_id == vendor_in_requests)
            
            vendor_scores_records.append({
                'Agent_ID': int(agent_id),
                'Vendor_ID': int(vendor_id),
                'Weight_Price': weights.get('price', 0.25),
                'Weight_Quality': weights.get('quality', 0.25),
                'Weight_Proximity': weights.get('proximity', 0.25),
                'Weight_Sustainability': weights.get('sustainability', 0.25),
                'Raw_Price': vendor['price'],
                'Raw_Quality': vendor['quality'],
                'Raw_Sustainability': vendor['sustainability'],
                'Raw_Proximity': proximity,
                'Norm_Price': norm_price,
                'Norm_Quality': norm_quality,
                'Norm_Sustainability': norm_sustainability,
                'Norm_Proximity': norm_proximity,
                'Weighted_Price': weighted_price,
                'Weighted_Quality': weighted_quality,
                'Weighted_Proximity': weighted_proximity,
                'Weighted_Sustainability': weighted_sustainability,
                'Composite_Score': composite_score,
                'Is_Selected': '✓' if is_selected else '',
                'Selected_Vendor_ID': selected_vendor if not pd.isna(selected_vendor) else vendor_in_requests
            })
    
    vendor_scores_df = pd.DataFrame(vendor_scores_records)
    
    print(f"✅ Generated {len(vendor_scores_df)} vendor score records")
    print(f"   ({num_agents} agents × {len(vendors)} vendors)")
    
    # =========================================================================
    # TABLE 2: PROXIMITY MATRIX
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING PROXIMITY MATRIX")
    print("="*80)
    
    proximity_matrix_records = []
    
    for idx, row in results_df.iterrows():
        agent_id = row.get('agent_id', idx + 1)
        proximity_scores = row.get('vendor_proximity_scores', {})
        
        record = {'Agent_ID': int(agent_id)}
        for vendor in vendors:
            vendor_id = vendor['vendor_id']
            proximity = proximity_scores.get(str(vendor_id), 50.0)
            record[f'Vendor_{vendor_id}_Proximity'] = proximity
        
        proximity_matrix_records.append(record)
    
    proximity_matrix_df = pd.DataFrame(proximity_matrix_records)
    
    print(f"✅ Generated proximity matrix for {len(proximity_matrix_df)} agents")
    
    # =========================================================================
    # ANALYZE VENDOR SELECTION DISTRIBUTION
    # =========================================================================
    print("\n" + "="*80)
    print("VENDOR SELECTION DISTRIBUTION")
    print("="*80)
    
    # Check vendor_selection column
    if 'vendor_selection' in results_df.columns:
        print("\nFrom vendor_selection column:")
        vendor_counts = results_df['vendor_selection'].value_counts().sort_index()
        for vendor_id, count in vendor_counts.items():
            if not pd.isna(vendor_id):
                print(f"  Vendor {int(vendor_id)}: {count} agents ({count/num_agents*100:.1f}%)")
    
    # Check vendorID in purchase_requests
    print("\nFrom purchase_requests vendorID:")
    vendor_ids_in_requests = []
    for idx, row in results_df.iterrows():
        purchase_requests = row.get('purchase_requests', [])
        if isinstance(purchase_requests, list) and len(purchase_requests) > 0:
            first_request = purchase_requests[0]
            if isinstance(first_request, dict):
                vendor_id = first_request.get('vendorID')
                if vendor_id is not None and not pd.isna(vendor_id):
                    vendor_ids_in_requests.append(vendor_id)
    
    if vendor_ids_in_requests:
        vendor_request_counts = pd.Series(vendor_ids_in_requests).value_counts().sort_index()
        for vendor_id, count in vendor_request_counts.items():
            print(f"  Vendor {int(vendor_id)}: {count} agents ({count/len(vendor_ids_in_requests)*100:.1f}%)")
        
        unique_vendors = pd.Series(vendor_ids_in_requests).nunique()
        if unique_vendors == 1:
            print(f"\n⚠️  WARNING: ALL agents selected vendor {vendor_ids_in_requests[0]}")
        else:
            print(f"\n✅ Multiple vendors selected ({unique_vendors} different vendors)")
    
    # =========================================================================
    # EXPORT TO EXCEL
    # =========================================================================
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)
    
    output_file = Path("/Users/suedagul/<sdg/vendor_score_diagnostics.xlsx")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: Vendor Scores (sorted by Agent_ID, then Composite_Score descending)
        vendor_scores_df_sorted = vendor_scores_df.sort_values(
            ['Agent_ID', 'Composite_Score'], 
            ascending=[True, False]
        )
        vendor_scores_df_sorted.to_excel(writer, sheet_name='Vendor Scores', index=False)
        
        # Sheet 2: Proximity Matrix
        proximity_matrix_df.to_excel(writer, sheet_name='Proximity Matrix', index=False)
        
        # Sheet 3: Summary Statistics
        summary_records = []
        
        # For each agent, find the best vendor
        for agent_id in vendor_scores_df['Agent_ID'].unique():
            agent_scores = vendor_scores_df[vendor_scores_df['Agent_ID'] == agent_id]
            best_vendor_row = agent_scores.loc[agent_scores['Composite_Score'].idxmax()]
            
            summary_records.append({
                'Agent_ID': int(agent_id),
                'Best_Vendor': int(best_vendor_row['Vendor_ID']),
                'Best_Score': best_vendor_row['Composite_Score'],
                'Selected_Vendor': best_vendor_row['Selected_Vendor_ID'],
                'Match': '✓' if best_vendor_row['Vendor_ID'] == best_vendor_row['Selected_Vendor_ID'] else '✗'
            })
        
        summary_df = pd.DataFrame(summary_records)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✅ Exported diagnostics to: {output_file}")
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total agents: {num_agents}")
    print(f"Total vendors: {len(vendors)}")
    print(f"Vendor scores calculated: {len(vendor_scores_df)}")
    
    # Check if all agents selected the same vendor
    selected_vendors = vendor_scores_df[vendor_scores_df['Is_Selected'] == '✓']['Vendor_ID'].unique()
    if len(selected_vendors) == 1:
        print(f"\n⚠️  ISSUE CONFIRMED: All agents selected Vendor {int(selected_vendors[0])}")
    else:
        print(f"\n✅ Agents selected {len(selected_vendors)} different vendors")
    
    return vendor_scores_df, proximity_matrix_df


if __name__ == "__main__":
    # Run with 10 agents for detailed analysis
    vendor_scores_df, proximity_matrix_df = diagnose_vendor_scores(num_agents=10)
    
    print("\n" + "="*80)
    print("SAMPLE DATA (First Agent)")
    print("="*80)
    
    # Show first agent's scores
    first_agent_scores = vendor_scores_df[vendor_scores_df['Agent_ID'] == 1]
    print("\nVendor Scores for Agent 1:")
    print(first_agent_scores[['Vendor_ID', 'Composite_Score', 'Raw_Proximity', 
                               'Norm_Price', 'Norm_Quality', 'Norm_Sustainability', 
                               'Norm_Proximity', 'Is_Selected']].to_string(index=False))
    
    print("\n✅ Check vendor_score_diagnostics.xlsx for complete analysis")

