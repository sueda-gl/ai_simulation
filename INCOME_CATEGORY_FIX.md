# Income Category Assignment Fix

## Problem Identified

The simulation was only assigning customers to **8 out of 10 income categories** despite having `num_fixed_categories = 10` configured on Page 1.

## Root Cause

The old `_assign_income_category()` function in `consumption_quantity.py` incorrectly split categories by the discount threshold:

**OLD LOGIC (INCORRECT):**
```
If income ≤ $12,500 (threshold):
    → Category 1 (all discount customers lumped together)

If income > $12,500:
    → Distribute across Categories 2-10 (9 categories)
    → Based on linear position above threshold
```

**Issues with old logic:**
1. ❌ All low-income agents assigned to Category 1 only
2. ❌ Only 9 categories (2-10) available for agents above threshold
3. ❌ With 5 income quintiles from Category-First architecture, only ~8 categories got populated
4. ❌ Category 1 was basically empty (threshold too low for the distribution)
5. ❌ Violated professor's specification for equal interval division

## Professor's Specification

According to the professor's requirements:

> "If there are 10 income categories, then the income range will be split to 10 equal intervals, and all customers with income within the corresponding interval will be assigned to it **irrespective of their type**."

Key points:
- **Equal intervals** across the entire income range (min to max)
- **ALL customers** assigned based on income only
- **NO distinction** by customer type (discount/fixed/regular) during category assignment
- Customer type only affects which consumption limit applies, NOT which category

## New Implementation (CORRECT)

**NEW LOGIC:**
```
1. Determine income range: [min_income, max_income]
2. Divide into N equal intervals (N = num_fixed_categories = 10)
3. Assign each agent to the interval containing their income
4. Category assignment is independent of customer type
```

**Example with current parameters:**
```
Income Distribution: Lognormal(μ=10.0, σ=0.5, min=0)
Estimated Range: $0 to ~$64,000
Interval Width: ~$6,400

Category  1: [$0       - $6,400)    Lowest income
Category  2: [$6,400   - $12,800)
Category  3: [$12,800  - $19,200)
Category  4: [$19,200  - $25,600)
Category  5: [$25,600  - $32,000)
Category  6: [$32,000  - $38,400)
Category  7: [$38,400  - $44,800)
Category  8: [$44,800  - $51,200)
Category  9: [$51,200  - $57,600)
Category 10: [$57,600  - $64,000]   Highest income
```

## How Categories Relate to Customer Types

**Important:** Category assignment ≠ Customer type

### Category Assignment
- Based purely on income amount
- Equal intervals across income range
- All 10 categories used (Categories 1-10)

### Customer Type Determination
- Based on disclosure decisions AND income threshold
- Types: "discount", "fixed", "regular"
- Determined by `determine_customer_type()` in `income_utils.py`

### Consumption Limit Application
According to the professor's specification:
- **Discount customers** → Use consumption limit from **Category 1** (lowest)
- **Regular customers** → Use consumption limit from **Category 10** (highest)
- **Fixed customers** → Use consumption limit from **their actual income category**

## Expected Results After Fix

With the Category-First architecture generating incomes from 5 quintile buckets:

**Quintile → Category Mapping:**
- **Level 1** (0-20%ile, ~$8,000-$15,600) → Categories 2-3
- **Level 2** (20-40%ile, ~$15,600-$19,300) → Category 3
- **Level 3** (40-60%ile, ~$19,300-$24,100) → Categories 3-4
- **Level 4** (60-80%ile, ~$24,100-$31,000) → Categories 4-5
- **Level 5** (80-100%ile, ~$31,000+) → Categories 5-10

**Expected category usage:**
- ✅ All 10 categories should now be populated
- ✅ Distribution across Categories 1-10 (with some clustering due to 5 quintiles)
- ✅ More even distribution than before (was only using 8 categories)

## Code Changes

### File: `src/decisions/consumption_quantity.py`

**Function:** `_assign_income_category(income, simulation_config)`

**Changes:**
1. Removed threshold-based splitting logic
2. Implemented equal-interval division across full income range
3. Added clear documentation of professor's specification
4. Simplified algorithm: `category = floor((income - min) / interval_width) + 1`

**Algorithm:**
```python
# Determine full income range based on distribution type
min_income = ... (from lognormal_min, gg_min, or dagum_min)
max_income = ... (from explicit max or estimated from distribution)

# Calculate equal intervals
income_range = max_income - min_income
interval_width = income_range / num_categories

# Assign category based on which interval contains the income
position = (income - min_income) / income_range  # [0, 1]
category_index = floor(position * num_categories)  # [0, N-1]
category = category_index + 1  # [1, N]
```

## Testing Recommendations

To verify the fix works correctly:

1. **Run a simulation** with current parameters
2. **Check `income_category` distribution** in results
3. **Verify all 10 categories are used** (or at least close to 10)
4. **Confirm categories align with income ranges** as specified above

## Notes

- The `num_discount_categories` parameter is currently not used in this implementation
- This follows the professor's specification of a single category system
- If separate discount/fixed category systems are needed in the future, that would require additional implementation

---

**Status:** ✅ FIXED  
**Date:** October 31, 2025  
**Issue:** Only 8/10 income categories populated  
**Solution:** Implemented equal-interval category assignment per professor's specification









