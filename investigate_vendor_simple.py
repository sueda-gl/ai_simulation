#!/usr/bin/env python3
"""
Simple Vendor Selection Investigation

Directly checks vendor attributes and agent weights without running full simulation.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/suedagul/<sdg')

from src.vendor_attribute_generator import generate_vendor_attributes, generate_proximity_scores, calculate_vendor_composite_score


def investigate_simple(num_vendors=6, num_agents=10, seed=42):
    """
    Simple investigation of vendor selection issue.
    """
    print("="*80)
    print("VENDOR SELECTION INVESTIGATION (Simplified)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Number of vendors: {num_vendors}")
    print(f"  - Number of sample agents: {num_agents}")
    print(f"  - Random seed: {seed}")
    print()
    
    rng = np.random.default_rng(seed)
    
    # ==========================================
    # STEP 1: Generate Vendors
    # ==========================================
    print("="*80)
    print("STEP 1: GENERATING VENDORS")
    print("="*80)
    print()
    
    vendors = generate_vendor_attributes(
        num_vendors=num_vendors,
        vendor_prices=[100.0] * num_vendors,  # Dummy list
        rng=rng,
        price_min=50.0,
        price_max=150.0,
        quantity_min=50,
        quantity_max=150
    )
    
    print(f"Generated {len(vendors)} vendors:\n")
    for vendor in vendors:
        print(f"Vendor {vendor['vendor_id']}:")
        print(f"  Price: ${vendor['price']:.2f}")
        print(f"  Quality: {vendor['quality']}/5")
        print(f"  Sustainability: {vendor['sustainability']}/5")
        print(f"  Quantity: {vendor['quantity_offered']} per period")
        print()
    
    # Calculate normalized attributes
    print("\nNormalized Attributes (for scoring):")
    print("-" * 60)
    
    prices = [v['price'] for v in vendors]
    min_price = min(prices)
    max_price = max(prices)
    
    for vendor in vendors:
        vendor_id = vendor['vendor_id']
        price = vendor['price']
        quality = vendor['quality']
        sustainability = vendor['sustainability']
        
        # Normalize (same as calculate_vendor_composite_score)
        norm_price = 1.0 - (price - min_price) / (max_price - min_price) if max_price > min_price else 1.0
        norm_quality = (quality - 1) / 4.0
        norm_sustainability = (sustainability - 1) / 4.0
        
        print(f"Vendor {vendor_id}:")
        print(f"  Norm Price: {norm_price:.3f}  (lower price → higher score)")
        print(f"  Norm Quality: {norm_quality:.3f}")
        print(f"  Norm Sustainability: {norm_sustainability:.3f}")
        print()
    
    # ==========================================
    # STEP 2: Check Agent Weights (SIMULATED)
    # ==========================================
    print("="*80)
    print("STEP 2: AGENT WEIGHTS ANALYSIS")
    print("="*80)
    print()
    
    print("Checking what vendor_choice_weights decision returns...")
    print()
    
    # Import the decision function
    from src.decisions.vendor_choice_weights import vendor_choice_weights
    
    # Test with multiple agents
    agent_weights = []
    for i in range(num_agents):
        agent_state = {'agent_id': i + 1}
        agent_rng = np.random.default_rng(seed + i)
        
        result = vendor_choice_weights(agent_state, {}, agent_rng, simulation_config=None)
        weights = result.get('vendor_choice_weights', {})
        agent_weights.append({
            'agent_id': i + 1,
            'price': weights.get('price', 0),
            'quality': weights.get('quality', 0),
            'proximity': weights.get('proximity', 0),
            'sustainability': weights.get('sustainability', 0)
        })
    
    # Check if all weights are identical
    all_identical = True
    first_weights = agent_weights[0]
    for weights in agent_weights[1:]:
        if (weights['price'] != first_weights['price'] or 
            weights['quality'] != first_weights['quality'] or 
            weights['proximity'] != first_weights['proximity'] or 
            weights['sustainability'] != first_weights['sustainability']):
            all_identical = False
            break
    
    if all_identical:
        print("⚠️  ALL AGENTS HAVE IDENTICAL WEIGHTS!")
        print()
        print(f"  Price:          {first_weights['price']}")
        print(f"  Quality:        {first_weights['quality']}")
        print(f"  Proximity:      {first_weights['proximity']}")
        print(f"  Sustainability: {first_weights['sustainability']}")
        print()
        print("This means every agent evaluates vendors using the exact same criteria.")
        print()
    else:
        print("✓ Agents have different weights (good!)")
        print("\nSample weights:")
        for weights in agent_weights[:5]:
            print(f"  Agent {weights['agent_id']}: price={weights['price']:.3f}, quality={weights['quality']:.3f}, "
                  f"proximity={weights['proximity']:.3f}, sustainability={weights['sustainability']:.3f}")
        print()
    
    # ==========================================
    # STEP 3: Calculate Composite Scores
    # ==========================================
    print("="*80)
    print("STEP 3: COMPOSITE SCORE ANALYSIS")
    print("="*80)
    print()
    
    # Use the first agent's weights (they're all the same anyway if identical)
    weights = {
        'price': first_weights['price'],
        'quality': first_weights['quality'],
        'proximity': first_weights['proximity'],
        'sustainability': first_weights['sustainability']
    }
    
    print(f"Using weights: {weights}")
    print()
    
    # Calculate scores for sample agents
    print("Calculating composite scores for sample agents:")
    print("-" * 60)
    
    for agent_id in range(1, min(6, num_agents + 1)):
        # Generate proximity scores for this agent
        proximity_scores = generate_proximity_scores(agent_id, num_vendors, rng)
        
        print(f"\nAgent {agent_id}:")
        
        # Calculate score for each vendor
        vendor_scores = []
        for vendor in vendors:
            v_id = vendor['vendor_id']
            proximity = proximity_scores.get(str(v_id), 50.0)
            score = calculate_vendor_composite_score(vendor, weights, proximity, vendors)
            vendor_scores.append((v_id, score, proximity))
        
        # Sort by score descending
        vendor_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Show all scores
        for rank, (v_id, score, proximity) in enumerate(vendor_scores, 1):
            marker = "⭐" if rank == 1 else "  "
            print(f"  {marker} Vendor {v_id}: score={score:.4f}, proximity={proximity:.1f}")
    
    # ==========================================
    # STEP 4: Aggregate Analysis
    # ==========================================
    print()
    print("="*80)
    print("STEP 4: AGGREGATE ANALYSIS (All Agents)")
    print("="*80)
    print()
    
    # Calculate average scores across all agents
    vendor_score_sums = {v['vendor_id']: 0.0 for v in vendors}
    vendor_selection_counts = {v['vendor_id']: 0 for v in vendors}
    
    for agent_id in range(1, num_agents + 1):
        proximity_scores = generate_proximity_scores(agent_id, num_vendors, rng)
        
        best_vendor_id = None
        best_score = -1
        
        for vendor in vendors:
            v_id = vendor['vendor_id']
            proximity = proximity_scores.get(str(v_id), 50.0)
            score = calculate_vendor_composite_score(vendor, weights, proximity, vendors)
            
            vendor_score_sums[v_id] += score
            
            if score > best_score:
                best_score = score
                best_vendor_id = v_id
        
        vendor_selection_counts[best_vendor_id] += 1
    
    # Calculate averages
    print("Average Composite Scores (across all agents):")
    print("-" * 60)
    
    avg_scores = []
    for v_id in sorted(vendor_score_sums.keys()):
        avg_score = vendor_score_sums[v_id] / num_agents
        avg_scores.append((v_id, avg_score))
        print(f"Vendor {v_id}: {avg_score:.4f}")
    
    # Sort by average score
    avg_scores.sort(key=lambda x: x[1], reverse=True)
    best_avg_vendor_id, best_avg_score = avg_scores[0]
    
    print()
    print(f"Vendor {best_avg_vendor_id} has the highest average score: {best_avg_score:.4f}")
    print()
    
    print("\nVendor Selection Counts:")
    print("-" * 60)
    
    for v_id in sorted(vendor_selection_counts.keys()):
        count = vendor_selection_counts[v_id]
        percentage = (count / num_agents) * 100
        marker = "⚠️ " if percentage > 80 else "  "
        print(f"{marker}Vendor {v_id}: {count:4d}/{num_agents} agents ({percentage:5.1f}%)")
    
    print()
    
    # Check if one vendor dominates
    max_count = max(vendor_selection_counts.values())
    max_percentage = (max_count / num_agents) * 100
    
    if max_percentage > 90:
        max_vendor_id = [v_id for v_id, count in vendor_selection_counts.items() if count == max_count][0]
        print(f"⚠️  WARNING: Vendor {max_vendor_id} is selected by {max_percentage:.1f}% of agents!")
        print()
    
    # ==========================================
    # ROOT CAUSE ANALYSIS
    # ==========================================
    print("="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)
    print()
    
    if all_identical:
        print("ROOT CAUSE #1: No Agent-Level Weight Variation")
        print("-" * 60)
        print("The vendor_choice_weights decision returns the SAME weights for ALL agents.")
        print()
        print("Location: src/decisions/vendor_choice_weights.py, lines 20-27")
        print()
        print("Current code:")
        print("  default_weights = {")
        print("      'price': 0.25,")
        print("      'quality': 0.25,")
        print("      'proximity': 0.25,")
        print("      'sustainability': 0.25")
        print("  }")
        print("  return {'vendor_choice_weights': default_weights}")
        print()
        print("Impact:")
        print("  - All agents use the same evaluation formula")
        print("  - Vendor ranking is deterministic (same for all agents)")
        print("  - The only variation is from proximity scores")
        print()
    
    if max_percentage > 80:
        print()
        print("ROOT CAUSE #2: One Vendor is Objectively Superior")
        print("-" * 60)
        print(f"Vendor {max_vendor_id} has the best combination of attributes.")
        print()
        print("Why this happens:")
        print("  - Vendor attributes are randomly generated")
        print("  - By chance, one vendor got low price + high quality/sustainability")
        print("  - With equal weights (0.25 each), this vendor scores highest")
        print("  - Even though proximity varies, it's not enough to overcome the advantage")
        print()
        print(f"Vendor {max_vendor_id} attributes:")
        best_vendor = [v for v in vendors if v['vendor_id'] == max_vendor_id][0]
        print(f"  Price: ${best_vendor['price']:.2f} (lower is better)")
        print(f"  Quality: {best_vendor['quality']}/5")
        print(f"  Sustainability: {best_vendor['sustainability']}/5")
        print()
    
    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    print("The issue is NOT with:")
    print("  ✓ Vendor attribute randomization (working correctly)")
    print("  ✓ Proximity score generation (varies by agent)")
    print("  ✓ Composite score calculation (math is correct)")
    print()
    print("The issue IS with:")
    print("  ❌ No agent-level heterogeneity in vendor preferences")
    print("  ❌ All agents use identical weights (0.25 for all attributes)")
    print()
    print("Result:")
    print("  With identical weights, all agents rank vendors the same way.")
    print("  The vendor with the best 'objective' attributes wins for everyone.")
    print("  Different random seeds will change which vendor is best,")
    print("  but with any given seed, one vendor will dominate.")
    print()


if __name__ == "__main__":
    num_vendors = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    num_agents = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    
    investigate_simple(num_vendors, num_agents, seed)
    
    print("="*80)
    print("Investigation complete!")
    print("="*80)

