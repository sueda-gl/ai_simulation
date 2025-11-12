"""
Diagnostic script to understand why only 7 income categories are populated instead of 10.

This analyzes:
1. How incomes are generated from 5 "Assigned Allowance Levels" (quintiles)
2. How those incomes map to N income categories
3. Which categories end up empty and why
"""

import numpy as np
from scipy import stats
import pandas as pd

# Simulation parameters (matching your configuration)
LOGNORMAL_MU = 10.0
LOGNORMAL_SIGMA = 0.5
LOGNORMAL_MIN = 0.0
LOGNORMAL_MAX = None  # No max clipping

NUM_FIXED_CATEGORIES = 10  # Your NFIC setting
NUM_AGENTS = 1000  # Sample size for diagnosis

# Define the 5 Assigned Allowance Levels (quintiles)
ALLOWANCE_LEVELS = {
    1: (0.00, 0.20),  # Bottom 20%
    2: (0.20, 0.40),  # 20-40%
    3: (0.40, 0.60),  # 40-60%
    4: (0.60, 0.80),  # 60-80%
    5: (0.80, 1.00),  # Top 20%
}

def generate_income_for_level(level, num_agents, rng):
    """Generate incomes for agents in a specific allowance level."""
    percentile_low, percentile_high = ALLOWANCE_LEVELS[level]
    
    # Create lognormal distribution
    dist = stats.lognorm(s=LOGNORMAL_SIGMA, scale=np.exp(LOGNORMAL_MU), loc=LOGNORMAL_MIN)
    
    # Sample random percentiles within the level's range
    random_percentiles = rng.uniform(percentile_low, percentile_high, num_agents)
    
    # Convert to dollar incomes
    incomes = dist.ppf(random_percentiles)
    
    return incomes

def assign_income_category(income, min_income, max_income, num_categories):
    """Assign an income to a category (1 to N) based on equal intervals."""
    if income <= min_income:
        return 1
    elif income >= max_income:
        return num_categories
    else:
        income_range = max_income - min_income
        position = (income - min_income) / income_range
        category_index = int(np.floor(position * num_categories))
        category = category_index + 1
        category = max(1, min(category, num_categories))
        return int(category)

def main():
    print("=" * 80)
    print("INCOME CATEGORY DIAGNOSTIC REPORT")
    print("=" * 80)
    print()
    
    rng = np.random.default_rng(42)
    
    # Step 1: Generate incomes for all 5 allowance levels
    print("STEP 1: Generating incomes from 5 Assigned Allowance Levels")
    print("-" * 80)
    
    # Create distribution to get PPF values
    dist = stats.lognorm(s=LOGNORMAL_SIGMA, scale=np.exp(LOGNORMAL_MU), loc=LOGNORMAL_MIN)
    
    all_incomes = []
    all_levels = []
    
    agents_per_level = NUM_AGENTS // 5
    
    for level in range(1, 6):
        incomes = generate_income_for_level(level, agents_per_level, rng)
        all_incomes.extend(incomes)
        all_levels.extend([level] * agents_per_level)
        
        percentile_low, percentile_high = ALLOWANCE_LEVELS[level]
        min_income_level = dist.ppf(percentile_low)
        max_income_level = dist.ppf(percentile_high)
        
        print(f"Level {level}: Percentiles {percentile_low:.2f}-{percentile_high:.2f}")
        print(f"  Income range: ${min_income_level:,.2f} - ${max_income_level:,.2f}")
        print(f"  Actual incomes: ${min(incomes):,.2f} - ${max(incomes):,.2f}")
        print(f"  Mean: ${np.mean(incomes):,.2f}")
        print()
    
    all_incomes = np.array(all_incomes)
    all_levels = np.array(all_levels)
    
    # Step 2: Determine the actual income range
    print("\nSTEP 2: Determining Income Range for Category Assignment")
    print("-" * 80)
    
    min_income = dist.ppf(0.0001)  # PPF(~0.00)
    max_income = dist.ppf(0.9999)  # PPF(~1.00)
    
    print(f"Min income (PPF 0.0001): ${min_income:,.2f}")
    print(f"Max income (PPF 0.9999): ${max_income:,.2f}")
    print(f"Income range: ${max_income - min_income:,.2f}")
    print(f"Interval width (for {NUM_FIXED_CATEGORIES} categories): ${(max_income - min_income) / NUM_FIXED_CATEGORIES:,.2f}")
    print()
    
    # Step 3: Assign categories
    print("\nSTEP 3: Assigning Incomes to Income Categories")
    print("-" * 80)
    
    interval_width = (max_income - min_income) / NUM_FIXED_CATEGORIES
    
    # Print category boundaries
    print("Category boundaries:")
    for i in range(1, NUM_FIXED_CATEGORIES + 1):
        cat_min = min_income + (i - 1) * interval_width
        cat_max = min_income + i * interval_width
        print(f"  Category {i:2d}: [${cat_min:>10,.2f} - ${cat_max:>10,.2f})")
    print()
    
    # Assign categories to all agents
    categories = [assign_income_category(inc, min_income, max_income, NUM_FIXED_CATEGORIES) 
                  for inc in all_incomes]
    
    # Step 4: Analyze category distribution
    print("\nSTEP 4: Category Distribution Analysis")
    print("-" * 80)
    
    df = pd.DataFrame({
        'allowance_level': all_levels,
        'income': all_incomes,
        'income_category': categories
    })
    
    category_stats = df.groupby('income_category').agg({
        'income': ['count', 'min', 'max', 'mean'],
        'allowance_level': lambda x: list(x.value_counts().to_dict().items())
    }).reset_index()
    
    print(f"{'Category':<10} {'Count':<8} {'Min Income':<15} {'Max Income':<15} {'Mean Income':<15} {'Allowance Levels'}")
    print("-" * 100)
    
    populated_categories = []
    empty_categories = []
    
    for cat in range(1, NUM_FIXED_CATEGORIES + 1):
        cat_data = df[df['income_category'] == cat]
        if len(cat_data) > 0:
            populated_categories.append(cat)
            count = len(cat_data)
            min_inc = cat_data['income'].min()
            max_inc = cat_data['income'].max()
            mean_inc = cat_data['income'].mean()
            
            # Get allowance level distribution
            level_dist = cat_data['allowance_level'].value_counts().to_dict()
            level_str = ", ".join([f"L{k}:{v}" for k, v in sorted(level_dist.items())])
            
            print(f"{cat:<10} {count:<8} ${min_inc:<14,.2f} ${max_inc:<14,.2f} ${mean_inc:<14,.2f} {level_str}")
        else:
            empty_categories.append(cat)
            print(f"{cat:<10} {0:<8} {'N/A':<15} {'N/A':<15} {'N/A':<15} (empty)")
    
    print()
    print(f"✅ Populated categories: {len(populated_categories)}/{NUM_FIXED_CATEGORIES}")
    print(f"❌ Empty categories: {empty_categories}")
    
    # Step 5: Root cause analysis
    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)
    print()
    
    print("The issue occurs because:")
    print()
    print("1. **Income Generation from 5 Quintiles:**")
    print("   - All agents have 'Assigned Allowance Level' from 1-5 (from copula)")
    print("   - Incomes are sampled from these 5 percentile buckets (quintiles)")
    print("   - Level 1: 0-20%, Level 2: 20-40%, Level 3: 40-60%, Level 4: 60-80%, Level 5: 80-100%")
    print()
    
    print("2. **Lognormal Distribution Characteristics:**")
    print("   - With μ=10.0, σ=0.5, the distribution is right-skewed")
    print("   - Lower percentiles cover a SMALL income range")
    print("   - Higher percentiles cover a LARGE income range")
    print()
    
    print("3. **Category Mapping Problem:**")
    print(f"   - {NUM_FIXED_CATEGORIES} equal-width income categories are created")
    print(f"   - Categories divide the full range (${min_income:,.2f} - ${max_income:,.2f}) equally")
    print(f"   - But most agents (Levels 1-4) cluster in the LOWER part of the income range")
    print(f"   - Higher categories ({max(populated_categories)+1}-{NUM_FIXED_CATEGORIES}) span income ranges that no agents reach")
    print()
    
    # Calculate which categories fall outside the actual generated income range
    actual_min = all_incomes.min()
    actual_max = all_incomes.max()
    
    print("4. **Actual vs. Theoretical Income Ranges:**")
    print(f"   - Theoretical range (PPF 0.00-1.00): ${min_income:,.2f} - ${max_income:,.2f}")
    print(f"   - Actual generated range: ${actual_min:,.2f} - ${actual_max:,.2f}")
    print(f"   - Gap: Categories {max(populated_categories)+1}-{NUM_FIXED_CATEGORIES} cover incomes above ${actual_max:,.2f}")
    print()
    
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("Option 1: Use fewer income categories that match the 5 allowance levels")
    print(f"  - Set NFIC to 5 or 6 instead of {NUM_FIXED_CATEGORIES}")
    print(f"  - This ensures all categories are populated")
    print()
    print("Option 2: Change income generation to not use quintile buckets")
    print("  - Sample incomes uniformly from the full distribution (0-100%ile)")
    print("  - This would break the Category-First architecture")
    print()
    print("Option 3: Accept that some high-income categories may be empty")
    print(f"  - This is expected behavior when using {NUM_FIXED_CATEGORIES} categories with 5 allowance levels")
    print(f"  - The empty categories represent income ranges no agents reach")
    print()
    
    # Show mapping table
    print("=" * 80)
    print("ALLOWANCE LEVEL → INCOME CATEGORY MAPPING")
    print("=" * 80)
    pivot = df.groupby(['allowance_level', 'income_category']).size().unstack(fill_value=0)
    print(pivot)
    print()

if __name__ == "__main__":
    main()

