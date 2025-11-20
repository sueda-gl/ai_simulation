"""
Debug script to investigate vendor ID assignment issue.
User reports all agents have vendor 4 in purchase requests Excel file.
"""

import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import Orchestrator
from src.vendor_attribute_generator import generate_vendor_attributes, generate_proximity_scores, calculate_vendor_composite_score
from src.decisions.purchasing_quantity import _calculate_preferred_vendor

def debug_vendor_assignment():
    print("=" * 80)
    print("VENDOR ASSIGNMENT DEBUG")
    print("=" * 80)
    
    # Load a simulation configuration
    config_path = Path("config/simulation.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return
    
    with open(config_path) as f:
        simulation_config = yaml.safe_load(f)
    
    print(f"\n1. SIMULATION CONFIGURATION")
    print(f"   Number of agents: {simulation_config.get('num_agents', 'N/A')}")
    print(f"   Number of vendors: {simulation_config.get('num_vendors', 'N/A')}")
    print(f"   Random seed: {simulation_config.get('random_seed', 'N/A')}")
    
    # Generate vendors
    num_vendors = simulation_config.get('num_vendors', 4)
    vendor_price_min = simulation_config.get('vendor_price_min', 5.0)
    vendor_price_max = simulation_config.get('vendor_price_max', 10.0)
    vendor_products_min = simulation_config.get('vendor_products_min', 50)
    vendor_products_max = simulation_config.get('vendor_products_max', 150)
    
    rng = np.random.default_rng(42)
    vendors = generate_vendor_attributes(
        num_vendors=num_vendors,
        vendor_prices=[],  # Empty list since we're using price range
        rng=rng,
        price_min=vendor_price_min,
        price_max=vendor_price_max,
        quantity_min=vendor_products_min,
        quantity_max=vendor_products_max
    )
    
    print(f"\n2. GENERATED VENDORS")
    for vendor in vendors:
        print(f"   Vendor {vendor['vendor_id']}: "
              f"price=${vendor['price']:.2f}, "
              f"quality={vendor['quality']}, "
              f"sustainability={vendor['sustainability']}, "
              f"products={vendor['quantity_offered']}")
    
    # Add vendors to config
    simulation_config['vendors'] = vendors
    
    # Test vendor assignment for first 10 agents
    print(f"\n3. TESTING VENDOR ASSIGNMENT FOR FIRST 10 AGENTS")
    print(f"   Using default weights: price=0.25, quality=0.25, proximity=0.25, sustainability=0.25")
    
    vendor_assignments = []
    
    for agent_id in range(1, 11):
        # Create a simple agent state
        agent_state = {
            'agent_id': agent_id,
            'index': agent_id - 1,
            'vendor_choice_weights': {
                'price': 0.25,
                'quality': 0.25,
                'proximity': 0.25,
                'sustainability': 0.25
            }
        }
        
        # Calculate preferred vendor
        preferred_vendor_id = _calculate_preferred_vendor(agent_state, simulation_config, rng)
        
        # Get proximity scores for display
        proximity_scores = agent_state.get('vendor_proximity_scores', {})
        
        vendor_assignments.append({
            'agent_id': agent_id,
            'preferred_vendor': preferred_vendor_id,
            'proximity_scores': proximity_scores
        })
        
        print(f"\n   Agent {agent_id}:")
        print(f"     Preferred vendor: {preferred_vendor_id}")
        print(f"     Proximity scores: {proximity_scores}")
    
    # Count vendor assignments
    print(f"\n4. VENDOR ASSIGNMENT SUMMARY")
    vendor_counts = {}
    for assignment in vendor_assignments:
        vendor_id = assignment['preferred_vendor']
        vendor_counts[vendor_id] = vendor_counts.get(vendor_id, 0) + 1
    
    for vendor_id in sorted(vendor_counts.keys()):
        count = vendor_counts[vendor_id]
        print(f"   Vendor {vendor_id}: {count} agents ({count/len(vendor_assignments)*100:.1f}%)")
    
    # Test with different weights
    print(f"\n5. TESTING WITH DIFFERENT WEIGHT CONFIGURATIONS")
    
    weight_configs = [
        {'name': 'Price Only', 'weights': {'price': 1.0, 'quality': 0.0, 'proximity': 0.0, 'sustainability': 0.0}},
        {'name': 'Quality Only', 'weights': {'price': 0.0, 'quality': 1.0, 'proximity': 0.0, 'sustainability': 0.0}},
        {'name': 'Proximity Only', 'weights': {'price': 0.0, 'quality': 0.0, 'proximity': 1.0, 'sustainability': 0.0}},
        {'name': 'Sustainability Only', 'weights': {'price': 0.0, 'quality': 0.0, 'proximity': 0.0, 'sustainability': 1.0}},
    ]
    
    for config in weight_configs:
        print(f"\n   {config['name']}:")
        
        # Test for agent 1
        agent_state = {
            'agent_id': 1,
            'index': 0,
            'vendor_choice_weights': config['weights']
        }
        
        preferred_vendor_id = _calculate_preferred_vendor(agent_state, simulation_config, rng)
        print(f"     Agent 1 prefers: Vendor {preferred_vendor_id}")
        
        # Show why this vendor was chosen
        proximity_scores = agent_state.get('vendor_proximity_scores', {})
        
        print(f"     Vendor scores:")
        for vendor in vendors:
            vendor_id = vendor['vendor_id']
            proximity = proximity_scores.get(str(vendor_id), 50.0)
            score = calculate_vendor_composite_score(vendor, config['weights'], proximity, vendors)
            print(f"       Vendor {vendor_id}: score={score:.4f}")
    
    # Check if there's a bug in the code
    print(f"\n6. CHECKING FOR BUGS")
    
    # Test: Are all agents getting the same RNG state?
    print(f"\n   Test 1: RNG state for multiple agents")
    test_rng = np.random.default_rng(42)
    proximities_agent1 = generate_proximity_scores(1, num_vendors, test_rng)
    test_rng = np.random.default_rng(42)  # Reset to same seed
    proximities_agent2 = generate_proximity_scores(2, num_vendors, test_rng)
    
    if proximities_agent1 == proximities_agent2:
        print(f"   ⚠️  WARNING: Agents 1 and 2 have IDENTICAL proximity scores!")
        print(f"       This could cause all agents to select the same vendor!")
        print(f"       Agent 1 proximities: {proximities_agent1}")
        print(f"       Agent 2 proximities: {proximities_agent2}")
    else:
        print(f"   ✓ Agents have different proximity scores (expected)")
        print(f"       Agent 1: {proximities_agent1}")
        print(f"       Agent 2: {proximities_agent2}")
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    debug_vendor_assignment()

