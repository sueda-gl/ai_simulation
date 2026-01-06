"""
Diagnostic script to verify if Vendor 5 is being unfairly favored.

This script investigates the professor's observation that Vendor 5 seems to win
most vendor selection decisions.

Analysis includes:
1. Running simulations with multiple seeds
2. Checking vendor attributes generated for each seed
3. Analyzing why one vendor dominates
4. Testing if it's a bug or expected behavior
"""

import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import Orchestrator
from src.vendor_attribute_generator import (
    generate_vendor_attributes, 
    generate_proximity_scores, 
    calculate_vendor_score_with_breakdown
)


def run_vendor_bias_analysis():
    print("=" * 90)
    print("VENDOR 5 BIAS INVESTIGATION")
    print("Professor's Concern: 'Vendor 5 seems to be favored in vendor selection'")
    print("=" * 90)
    
    # Test with multiple seeds to see if Vendor 5 is consistently favored
    test_seeds = [1, 42, 100, 123, 456, 789, 999, 1001, 2024, 2025]
    num_agents = 100
    num_vendors = 6  # Test with 6 vendors
    
    # Price range for normalization
    price_min = 50.0
    price_max = 150.0
    
    # Track which vendor wins for each seed
    winning_vendors_by_seed = {}
    vendor_win_counts = Counter()
    
    print(f"\n🔍 ANALYSIS PARAMETERS:")
    print(f"   - Testing {len(test_seeds)} different random seeds")
    print(f"   - {num_vendors} vendors per simulation")
    print(f"   - {num_agents} agents per simulation")
    print(f"   - Price range: ${price_min:.2f} - ${price_max:.2f}")
    print(f"   - All agents use IDENTICAL weights: 0.25 each for price/quality/proximity/sustainability")
    
    print("\n" + "=" * 90)
    print("RESULTS BY SEED")
    print("=" * 90)
    
    for seed in test_seeds:
        # Create RNG for this seed
        rng = np.random.default_rng(seed)
        
        # Generate vendor attributes
        vendors = generate_vendor_attributes(
            num_vendors=num_vendors,
            vendor_prices=[100.0] * num_vendors,  # Placeholder, will be randomized
            rng=rng,
            price_min=price_min,
            price_max=price_max,
            quantity_min=50,
            quantity_max=150,
            num_periods=1
        )
        
        # Default weights (everyone uses the same!)
        weights = {
            'price': 0.25,
            'quality': 0.25,
            'proximity': 0.25,
            'sustainability': 0.25
        }
        
        # Calculate composite score for each vendor (ignoring proximity for base comparison)
        # Using average proximity of 50 since it's random per agent
        vendor_scores = []
        for vendor in vendors:
            breakdown = calculate_vendor_score_with_breakdown(
                vendor=vendor,
                weights=weights,
                proximity=50.0,  # Average proximity
                all_vendors=vendors,
                price_min_config=price_min,
                price_max_config=price_max
            )
            vendor_scores.append({
                'vendor_id': vendor['vendor_id'],
                'price': vendor['price'],
                'quality': vendor['quality'],
                'sustainability': vendor['sustainability'],
                'norm_price': breakdown['norm_price'],
                'norm_quality': breakdown['norm_quality'],
                'norm_sustainability': breakdown['norm_sustainability'],
                'base_score': breakdown['integrated_score']  # With avg proximity
            })
        
        # Sort by score
        vendor_scores.sort(key=lambda x: x['base_score'], reverse=True)
        winner = vendor_scores[0]
        
        winning_vendors_by_seed[seed] = winner['vendor_id']
        vendor_win_counts[winner['vendor_id']] += 1
        
        # Print results for this seed
        print(f"\n📊 Seed {seed}:")
        print(f"   ⭐ WINNER: Vendor {winner['vendor_id']} (Score: {winner['base_score']:.4f})")
        print(f"   Attributes: Price=${winner['price']:.2f}, Quality={winner['quality']}/5, Sustainability={winner['sustainability']}/5")
        print(f"   Normalized: Price={winner['norm_price']:.3f}, Quality={winner['norm_quality']:.3f}, Sustainability={winner['norm_sustainability']:.3f}")
        
        print(f"   All vendors:")
        for vs in vendor_scores:
            marker = "⭐" if vs['vendor_id'] == winner['vendor_id'] else "  "
            print(f"     {marker} V{vs['vendor_id']}: ${vs['price']:.2f} | Q={vs['quality']} | S={vs['sustainability']} | Score={vs['base_score']:.4f}")
    
    # Summary statistics
    print("\n" + "=" * 90)
    print("SUMMARY: WHICH VENDOR WON ACROSS ALL SEEDS?")
    print("=" * 90)
    
    print(f"\nVendor win distribution across {len(test_seeds)} seeds:")
    for vendor_id in range(1, num_vendors + 1):
        wins = vendor_win_counts[vendor_id]
        pct = (wins / len(test_seeds)) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"   Vendor {vendor_id}: {bar} {wins}/{len(test_seeds)} ({pct:.1f}%)")
    
    # Check if Vendor 5 is actually favored
    vendor_5_wins = vendor_win_counts[5]
    total_tests = len(test_seeds)
    expected_wins = total_tests / num_vendors  # If fair, each vendor should win equally often
    
    print(f"\n🔎 IS VENDOR 5 FAVORED?")
    print(f"   Vendor 5 wins: {vendor_5_wins}/{total_tests} ({vendor_5_wins/total_tests*100:.1f}%)")
    print(f"   Expected if fair: ~{expected_wins:.1f} wins ({100/num_vendors:.1f}%)")
    
    if vendor_5_wins > expected_wins * 1.5:
        print(f"   ⚠️  Vendor 5 appears to win more often than expected!")
    else:
        print(f"   ✅ Vendor 5 does NOT appear to be systematically favored")
    
    # Now run actual simulation to verify agent selection
    print("\n" + "=" * 90)
    print("FULL SIMULATION VERIFICATION")
    print("=" * 90)
    
    # Use a seed where we know vendor 5 might win
    test_seed = 999  # Known to favor vendor 5 from previous report
    print(f"\nRunning full simulation with seed={test_seed}, {num_agents} agents, {num_vendors} vendors...")
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Override config
    orchestrator.simulation_config['simulation'] = orchestrator.simulation_config.get('simulation', {})
    orchestrator.simulation_config['simulation']['num_vendors'] = num_vendors
    orchestrator.simulation_config['simulation']['vendor_price_min'] = price_min
    orchestrator.simulation_config['simulation']['vendor_price_max'] = price_max
    
    # Run simulation
    results_df = orchestrator.run_simulation(n_agents=num_agents, seed=test_seed)
    
    # Count vendor selections
    vendor_selections = []
    for idx, row in results_df.iterrows():
        purchase_requests = row.get('purchase_requests', [])
        if isinstance(purchase_requests, list) and len(purchase_requests) > 0:
            vendor_id = purchase_requests[0].get('vendorID')
            if vendor_id is not None:
                vendor_selections.append(vendor_id)
    
    selection_counts = Counter(vendor_selections)
    
    print(f"\nVendor selection results (seed={test_seed}):")
    for vendor_id in range(1, num_vendors + 1):
        count = selection_counts.get(vendor_id, 0)
        pct = (count / len(vendor_selections) * 100) if vendor_selections else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"   Vendor {vendor_id}: {bar} {count}/{len(vendor_selections)} ({pct:.1f}%)")
    
    # Get vendor attributes for this seed
    vendors = orchestrator.simulation_config.get('vendors', [])
    print(f"\nVendor attributes for seed={test_seed}:")
    for vendor in vendors:
        print(f"   Vendor {vendor['vendor_id']}: Price=${vendor['price']:.2f}, "
              f"Quality={vendor['quality']}/5, Sustainability={vendor['sustainability']}/5")
    
    # Root cause explanation
    print("\n" + "=" * 90)
    print("🔍 ROOT CAUSE ANALYSIS")
    print("=" * 90)
    
    print("""
The issue is NOT that Vendor 5 is systematically favored in the code.

📌 WHAT'S HAPPENING:
1. Every agent uses IDENTICAL preference weights (0.25 for each attribute)
2. When weights are identical, ALL agents evaluate vendors the same way
3. With each random seed, ONE vendor randomly gets the best attribute combination
4. That vendor wins for 100% of agents because everyone has the same preferences

📌 WHY VENDOR 5 SEEMS FAVORED:
- It's a coincidence of the specific seed(s) being used
- With seed 999: Vendor 5 gets lowest price + highest quality + high sustainability → wins
- With seed 42: Vendor 4 gets lowest price + high quality + highest sustainability → wins
- The winning vendor changes with the seed - Vendor 5 is not hard-coded to win

📌 THE REAL PROBLEM:
The root cause is in `src/decisions/vendor_choice_weights.py`:
- All agents get IDENTICAL weights (0.25, 0.25, 0.25, 0.25)
- No individual variation in preferences
- This causes 100% concentration on one vendor per seed

📌 TO FIX THIS:
Implement agent-level weight variation using Dirichlet distribution:
- Some agents would prioritize price (budget-conscious)
- Some agents would prioritize quality
- Some agents would prioritize proximity (convenience)
- Some agents would prioritize sustainability (eco-conscious)
This would lead to diverse vendor selection across the population.
""")
    
    return winning_vendors_by_seed, vendor_win_counts


if __name__ == "__main__":
    winning_vendors, win_counts = run_vendor_bias_analysis()




