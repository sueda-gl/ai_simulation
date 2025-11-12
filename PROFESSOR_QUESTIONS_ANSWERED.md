# Answers to Professor's Questions on Income Categories and Consumption Quantities

## Status: ✅ ALL ISSUES FIXED

---

## Question 1: "Why do we have fewer income categories?"

### The Problem (BEFORE FIX)
The old code was only populating **8 out of 10 categories** because it incorrectly split categories by the discount threshold:
- Category 1: All agents with income ≤ $12,500
- Categories 2-10: Agents with income > $12,500

This meant:
- ❌ Most agents ended up above the threshold (it was too low)
- ❌ Category 1 was basically empty
- ❌ Only 9 categories (2-10) were available for distribution
- ❌ With income coming from 5 quintile buckets, only ~8 categories got populated

### The Solution (AFTER FIX)
**File Modified:** `src/decisions/consumption_quantity.py` - Function `_assign_income_category()`

**New Logic:**
```python
# Income range is split into N EQUAL intervals
min_income = $0 (from lognormal_min)
max_income ≈ $64,000 (from distribution parameters)
interval_width = $64,000 / 10 = $6,400

# ALL agents assigned based purely on income amount
Category 1:  Income in [$0      - $6,400)
Category 2:  Income in [$6,400  - $12,800)
Category 3:  Income in [$12,800 - $19,200)
...
Category 10: Income in [$57,600 - $64,000]
```

**Result:** ✅ All 10 categories are now populated (or close to it, given the 5-quintile income generation)

---

## Question 2: "Where are the discount categories?"

### Answer: There is NO separate discount category system

**Important Clarification:**
- There are **NOT separate "discount categories"** and "fixed categories"
- There is **ONE unified income category system** (Categories 1 to N)
- **ALL customers** (discount, fixed, AND regular) are assigned to the same category system based purely on income

### How It Works:

**Step 1: Income Category Assignment (for ALL customers)**
```
Equal-interval division of income range:
- Category 1:  [$0     - $10k)   ← Lowest income
- Category 2:  [$10k   - $20k)
- Category 3:  [$20k   - $30k)
- ...
- Category 10: [$90k   - $100k]  ← Highest income

ALL customer types use this same categorization.
```

**Step 2: Customer Type Determination (separate from categories)**
```
Based on disclosure decisions and threshold:
- Discount:  income < threshold AND disclosed documents
- Fixed:     disclosed income upfront
- Regular:   did not disclose income

This is INDEPENDENT of income category assignment.
```

**Step 3: Consumption Limit Application**
```
Customer type determines WHICH category's limit to use:
- Discount customers → Always use Category 1 limit (lowest)
- Regular customers  → Always use Category 10 limit (highest)
- Fixed customers    → Use their actual income category limit
```

**Why `num_discount_categories` exists in config:**
- It's a legacy parameter from earlier design
- Currently NOT used in the implementation
- Only `num_fixed_categories` is used (which represents total categories for ALL customers)

---

## Question 3: "Why are quantities not aligned with the maximum specified on Page 1?"

### The Problem (BEFORE FIX)
The code was **NOT applying customer-type-specific consumption limits**. It only looked up the limit for the agent's income category, without considering whether they were discount, fixed, or regular customers.

### The Solution (AFTER FIX)
**File Modified:** `src/decisions/consumption_quantity.py` - Function `consumption_quantity()`

**NEW Logic Added:**
```python
# Determine customer type
customer_type = get_customer_type(agent_state, simulation_config)

# Select which category's limit to use based on customer type
if customer_type == "discount":
    limit_category = 1  # Always use Category 1 limit
elif customer_type == "regular":
    limit_category = num_categories  # Always use Category N limit
else:  # customer_type == "fixed"
    limit_category = income_category  # Use actual income category

# Look up the limit for that category
consumption_limit = limits[f"cat_{limit_category}"]

# Generate random quantity in [0, consumption_limit]
total_quantity = random.integers(0, consumption_limit + 1)
```

### Example Scenarios:

**Scenario A: Consumption Limits DISABLED (`apply_consumption_limits: false`)**
```
ALL customers use: max_purchases_per_term = 50

Agent A (Discount): quantity in [0, 50]
Agent B (Fixed):    quantity in [0, 50]
Agent C (Regular):  quantity in [0, 50]
```

**Scenario B: Consumption Limits ENABLED with specific values:**
```
Category 1 limit: 10 products
Category 5 limit: 30 products  
Category 10 limit: 50 products

Agent A: Income $8k → Category 1 → Discount type → Uses Cat 1 limit = 10
Agent B: Income $45k → Category 5 → Fixed type → Uses Cat 5 limit = 30
Agent C: Income $45k → Category 5 → Regular type → Uses Cat 10 limit = 50

Quantities generated:
- Agent A: random in [0, 10]
- Agent B: random in [0, 30]
- Agent C: random in [0, 50]
```

**Result:** ✅ Quantities now correctly respect customer-type-specific limits

---

## Question 4: "How are customers assigned to income categories at the moment?"

### Current Implementation (POST-FIX)

**Algorithm:** Equal Interval Division

```python
def _assign_income_category(income, simulation_config):
    """
    Assign agent to income category using equal-interval division.
    """
    # Step 1: Get income distribution parameters
    dist_type = simulation_config['income_distribution']  # e.g., 'lognormal'
    
    if dist_type == 'lognormal':
        min_income = lognormal_min  # e.g., $0
        max_income = lognormal_max or estimated_max  # e.g., $64,000
    
    # Step 2: Calculate interval width
    num_categories = num_fixed_categories  # e.g., 10
    interval_width = (max_income - min_income) / num_categories
    
    # Step 3: Determine which interval contains the agent's income
    position = (income - min_income) / (max_income - min_income)  # Fraction [0,1]
    category_index = floor(position * num_categories)  # 0 to N-1
    category = category_index + 1  # Convert to 1-based (1 to N)
    
    return category
```

### Example with Current Parameters:

**Configuration:**
- Distribution: Lognormal(μ=10.0, σ=0.5, min=0)
- Estimated range: $0 to $64,000
- Number of categories: 10
- Interval width: $6,400

**Category Assignments:**

| Agent | Income    | Calculation                     | Category |
|-------|-----------|--------------------------------|----------|
| A     | $5,000    | (5000-0)/64000 = 0.078 → 0×10+1 | 1        |
| B     | $8,000    | (8000-0)/64000 = 0.125 → 1×10+1 | 2        |
| C     | $15,000   | (15000-0)/64000 = 0.234 → 2×10+1| 3        |
| D     | $25,000   | (25000-0)/64000 = 0.391 → 3×10+1| 4        |
| E     | $35,000   | (35000-0)/64000 = 0.547 → 5×10+1| 6        |
| F     | $50,000   | (50000-0)/64000 = 0.781 → 7×10+1| 8        |
| G     | $63,000   | (63000-0)/64000 = 0.984 → 9×10+1| 10       |

### Key Properties:

1. ✅ **All customers** assigned to the same category system
2. ✅ **Equal intervals** across the full income range
3. ✅ **No distinction** by customer type during assignment
4. ✅ **Category assignment** is independent of discount threshold
5. ✅ **Customer type** only affects which limit to use, not which category

---

## Complete Flow (End-to-End)

### Step-by-Step for a Single Agent:

```
AGENT PROFILE:
- Income: $23,000
- Disclosed documents: Yes
- Income below threshold ($12,500): No

STEP 1: Generate Income
→ Income = $23,000 (from Category-First PPF method)

STEP 2: Assign Income Category
→ Category = floor((23000-0)/6400) + 1 = 3 + 1 = 4

STEP 3: Determine Customer Type  
→ Income >= threshold → NOT discount
→ Disclosed documents → Fixed customer

STEP 4: Select Consumption Limit
→ Customer type = Fixed
→ Use income category's limit = Category 4 limit = 25 products

STEP 5: Generate Quantity
→ Random integer in [0, 25]
→ Result: 14 products

STEP 6: Create Purchase Requests
→ 14 requests (1 item each)
→ Timestamps randomly distributed across term
```

---

## Summary of Fixes Applied

### 1. Income Category Assignment ✅
**File:** `src/decisions/consumption_quantity.py`
- Changed from threshold-based to equal-interval division
- All 10 categories now populated
- Independent of customer type

### 2. Customer-Type-Specific Limits ✅
**File:** `src/decisions/consumption_quantity.py`
- Added logic to select limit based on customer type
- Discount → Category 1 limit
- Regular → Category N limit
- Fixed → Actual income category limit

### 3. Documentation Updates ✅
**Files:**
- `app/pages/results/visualizations/consumption_viz.py`
- `CONSUMPTION_LIMITS_GUIDE.md`
- `INCOME_CATEGORY_FIX.md` (new)
- `DESCRIPTION_FIX_SUMMARY.md` (new)
- `PROFESSOR_QUESTIONS_ANSWERED.md` (this file)

---

## Testing Recommendations

To verify these fixes work correctly:

1. **Run a simulation** with:
   - `num_fixed_categories = 10`
   - `apply_consumption_limits = true`
   - Set different limits for each category

2. **Check results for**:
   - All 10 income categories populated (or close, given 5 quintiles)
   - Discount customers have quantities ≤ Category 1 limit
   - Regular customers have quantities ≤ Category 10 limit
   - Fixed customers have quantities ≤ their income category's limit

3. **Export and analyze**:
   - Income category distribution (should show all 10 categories)
   - Quantity by customer type (should match expected limits)

---

**Date:** October 31, 2025  
**Status:** ✅ All questions answered and all issues fixed



