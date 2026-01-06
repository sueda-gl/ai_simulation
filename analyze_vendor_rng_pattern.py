"""
Analyze the RNG pattern to understand why certain vendors 
tend to get better attributes across different seeds.

This script runs 100 simulations with different seeds and
analyzes the statistical pattern of vendor attributes.
"""

import numpy as np
from collections import Counter

def analyze_vendor_attributes():
    print("=" * 90)
    print("VENDOR ATTRIBUTE RNG PATTERN ANALYSIS")
    print("=" * 90)
    print("\nAnalyzing vendor attribute distribution across 100 different seeds...")
    print("(Each seed creates 6 vendors with random quality and sustainability)\n")
    
    # Test 100 seeds
    seeds = range(1, 101)
    num_vendors = 6
    
    # Track quality and sustainability by vendor position
    quality_by_vendor = {i: [] for i in range(1, num_vendors + 1)}
    sustainability_by_vendor = {i: [] for i in range(1, num_vendors + 1)}
    
    # Track winners
    winners = []
    
    for seed in seeds:
        rng = np.random.default_rng(seed)
        
        vendor_scores = []
        for vendor_id in range(1, num_vendors + 1):
            # Simulate vendor attribute generation (matching vendor_attribute_generator.py)
            price = float(rng.uniform(50.0, 150.0))
            quality = int(rng.integers(1, 6))
            sustainability = int(rng.integers(1, 6))
            
            # Track for statistics
            quality_by_vendor[vendor_id].append(quality)
            sustainability_by_vendor[vendor_id].append(sustainability)
            
            # Calculate score with equal weights (0.25 each)
            norm_price = 1.0 - (price - 50.0) / 100.0
            norm_quality = (quality - 1) / 4.0
            norm_sustainability = (sustainability - 1) / 4.0
            norm_proximity = 0.5  # Average proximity
            
            score = 0.25 * norm_price + 0.25 * norm_quality + 0.25 * norm_proximity + 0.25 * norm_sustainability
            vendor_scores.append((vendor_id, score))
        
        # Find winner
        vendor_scores.sort(key=lambda x: x[1], reverse=True)
        winners.append(vendor_scores[0][0])
    
    # Statistics
    print("=" * 90)
    print("QUALITY DISTRIBUTION BY VENDOR POSITION")
    print("=" * 90)
    print("\nVendor | Mean Quality | Std Dev | Min | Max | % with Q=5")
    print("-" * 60)
    
    for vendor_id in range(1, num_vendors + 1):
        qualities = quality_by_vendor[vendor_id]
        q5_pct = (sum(1 for q in qualities if q == 5) / len(qualities)) * 100
        print(f"  V{vendor_id}   |    {np.mean(qualities):.2f}     | {np.std(qualities):.2f}    |  {min(qualities)}  |  {max(qualities)}  |  {q5_pct:.1f}%")
    
    print("\n" + "=" * 90)
    print("SUSTAINABILITY DISTRIBUTION BY VENDOR POSITION")
    print("=" * 90)
    print("\nVendor | Mean Sust. | Std Dev | Min | Max | % with S=5")
    print("-" * 60)
    
    for vendor_id in range(1, num_vendors + 1):
        sustains = sustainability_by_vendor[vendor_id]
        s5_pct = (sum(1 for s in sustains if s == 5) / len(sustains)) * 100
        print(f"  V{vendor_id}   |    {np.mean(sustains):.2f}     | {np.std(sustains):.2f}    |  {min(sustains)}  |  {max(sustains)}  |  {s5_pct:.1f}%")
    
    print("\n" + "=" * 90)
    print("WINNER DISTRIBUTION (100 SEEDS)")
    print("=" * 90)
    
    winner_counts = Counter(winners)
    expected = 100 / num_vendors
    
    print(f"\nExpected wins per vendor (if fair): {expected:.1f}")
    print("\nActual wins:")
    for vendor_id in range(1, num_vendors + 1):
        wins = winner_counts.get(vendor_id, 0)
        bar = "█" * (wins // 2) + "░" * (25 - wins // 2)
        deviation = ((wins - expected) / expected) * 100 if expected > 0 else 0
        flag = "⚠️" if abs(deviation) > 50 else ""
        print(f"  V{vendor_id}: {bar} {wins:3d} wins ({deviation:+.1f}% from expected) {flag}")
    
    print("\n" + "=" * 90)
    print("STATISTICAL CONCLUSION")
    print("=" * 90)
    
    # Chi-square test approximation
    chi_sq = sum((winner_counts.get(v, 0) - expected)**2 / expected for v in range(1, num_vendors + 1))
    
    print(f"""
📊 Chi-Square Statistic: {chi_sq:.2f}
   (Critical value at p=0.05 with df=5 is ~11.07)

📌 INTERPRETATION:
   - If Chi-Square < 11.07: Vendor wins are statistically fair (within random variance)
   - If Chi-Square > 11.07: There may be a pattern worth investigating

🔍 KEY INSIGHT:
   The winner distribution depends on the RNG sequence, but with a large enough
   sample size (100 seeds), we should see roughly equal wins IF:
   1. The RNG is truly random
   2. There's no systematic bias in how vendor attributes are generated

   Note: Even if distribution looks uneven, it could be normal statistical variance.
   The REAL issue is that all agents have identical weights (0.25 each).
""")
    
    # Show which seeds favor which vendor
    print("\n" + "=" * 90)
    print("SEEDS WHERE VENDOR 5 WINS")
    print("=" * 90)
    
    v5_winning_seeds = [seed for seed, winner in zip(seeds, winners) if winner == 5]
    print(f"\nVendor 5 wins with seeds: {v5_winning_seeds[:20]}..." if len(v5_winning_seeds) > 20 else f"\nVendor 5 wins with seeds: {v5_winning_seeds}")
    
    return winner_counts, quality_by_vendor, sustainability_by_vendor


if __name__ == "__main__":
    winner_counts, _, _ = analyze_vendor_attributes()




