# Discount Qualification vs Discount Customers - Clarification Fix

**Date:** November 27, 2025  
**Status:** ✅ Complete

---

## The Issue

There was confusion about two different percentages related to discount eligibility:

1. **~11%** - "Discount Qualification" shown on Page 1 (Common Parameters)
2. **~7.5%** - "Discount Customers" shown on Results Page

### Root Cause

These two numbers measure different things:

- **11% = Potential Eligibility** (income-based only)
  - Calculated from income distribution alone
  - Counts: `agents with income ≤ threshold`
  - Does NOT account for agent decisions
  
- **7.5% = Actual Discount Customers** (after all decisions)
  - Calculated from simulation results
  - Counts: `agents with income ≤ threshold AND disclose_income = "Y" AND disclose_documents = "Y"`
  - Accounts for THREE conditions that must ALL be met

---

## Why 7.5% < 11%?

The actual discount customer rate is lower because agents must pass through multiple decision gates:

### Decision Flow:

```
100 agents with income ≤ threshold (11% of total)
    ↓
Decision 1: Disclose Income?
    ↓ ~60% say "Yes"
60 agents continue
    ↓
Decision 2: Disclose Documents?
    ↓ ~50% say "Yes"  
30 agents become discount customers
    ↓
Result: 30 / 400 total = 7.5% actual discount customers
```

### Mathematical Relationship:

```
Actual Discount % = Potential Eligibility % 
                    × P(disclose_income = "Y" | low income)
                    × P(disclose_documents = "Y" | low income, disclosed income)
```

---

## Code Changes Made

### 1. **Page 1 - Income Distribution Preview** (`app/components.py`)

**Location:** Line 540-541

**Before:**
```python
st.metric("Discount Qualification", f"{discount_rate:.1%}",
         help="Estimated percentage - actual results may vary slightly due to random seed differences")
```

**After:**
```python
st.metric("Discount Qualification", f"{discount_rate:.1%}",
         help="Percentage of agents with income ≤ threshold (potential eligibility based on income alone). Actual discount customers will be lower, as they must also choose to disclose income AND documents in their decisions.")
```

**Change:** Clarified that this is potential eligibility, not actual discount rate.

---

### 2. **Page 1 - Discount Threshold Input** (`app/pages/page1_common_params.py`)

**Location:** Line 1246

**Before:**
```python
help="Income threshold below which agents qualify for discounts (pending document disclosure)"
```

**After:**
```python
help="Income threshold for potential discount eligibility. Agents with income ≤ this value can become discount customers if they also choose to disclose income AND documents."
```

**Change:** Made explicit that income threshold is just the first condition, and agents must also make disclosure choices.

---

### 3. **Results Page - Disclose Documents Metrics** (`app/pages/results/visualizations/disclosure_viz.py`)

**Location:** Line 86-87

**Before:**
```python
st.metric("Qualified for Discount", f"{qualified_agents:,}", 
          help="Agents with income < discount threshold")
```

**After:**
```python
st.metric("Eligible to Disclose Documents", f"{qualified_agents:,}", 
          help="Agents with income < threshold who disclosed income. These agents are asked if they want to disclose documents for discount eligibility.")
```

**Changes:** 
- Renamed metric from "Qualified for Discount" to "Eligible to Disclose Documents" (more accurate)
- Clarified that these agents have BOTH low income AND disclosed income
- Explained that these are the agents who get asked about documents

---

## Three Distinct Metrics

| Metric | Meaning | Where Shown | Calculation |
|--------|---------|-------------|-------------|
| **Discount Qualification (11%)** | Potential eligibility based on income alone | Page 1 - Distribution Preview | `income ≤ threshold` |
| **Eligible to Disclose Documents** | Agents asked about document disclosure | Results - Disclose Documents | `income ≤ threshold AND disclose_income = "Y"` |
| **Discount Customers (7.5%)** | Actual discount customers | Results - Customer Types | `income ≤ threshold AND disclose_income = "Y" AND disclose_documents = "Y"` |

---

## Customer Type Logic Reference

From `src/decisions/income_utils.py`:

```python
def determine_customer_type(agent_state: dict, simulation_config: dict) -> str:
    """
    Customer Type Logic:
    1. DISCOUNT: disclose_income = "Y" AND income < threshold AND disclose_documents = "Y"
    2. FIXED: disclose_income = "Y" (and not discount)
    3. REGULAR: disclose_income = "N"
    """
    
    income = agent_state.get('income', 0)
    disclose_income = agent_state.get('disclose_income', 'N')
    disclose_documents = agent_state.get('disclose_documents', 'NA')
    threshold = get_simulation_param(simulation_config, 'discount_income_threshold', 12500.0)
    
    # Priority 1: Check for DISCOUNT customer
    # Must have disclosed income AND low income AND submitted documents
    if disclose_income == "Y" and income <= threshold and disclose_documents == "Y":
        return "discount"
    
    # Priority 2: Check for FIXED customer
    # Disclosed income upfront (regardless of income level)
    elif disclose_income == "Y":
        return "fixed"
    
    # Priority 3: REGULAR customer (default)
    # Did not disclose income
    else:
        return "regular"
```

---

## Files Modified

1. ✅ `app/components.py` (line 540-541)
2. ✅ `app/pages/page1_common_params.py` (line 1246)
3. ✅ `app/pages/results/visualizations/disclosure_viz.py` (line 86-87)

---

## Verification

The tooltips now correctly explain:

✅ **Page 1 "Discount Qualification"**: Clearly states this is potential eligibility based on income distribution, and actual discount customers will be lower due to disclosure decisions.

✅ **Page 1 "Threshold Input"**: Explains that income threshold is just one condition, and agents must also choose to disclose.

✅ **Results Page**: Renamed metric to be more accurate and explains that these are agents eligible to be ASKED about documents (not yet discount customers).

---

## Related Documentation

- `CUSTOMER_TYPE_LOGIC_BUG_FIX.md` - Customer type determination logic
- `DISCLOSED_DOCUMENTS_NA_EXPLANATION.md` - Why most document disclosure values are NA
- `PROFESSOR_QUESTIONS_ANSWERED.md` - Income categories and discount logic

---

## Summary

The discrepancy between 11% and 7.5% is **intentional and correct**:

- **11%** = Income-based potential (what the distribution predicts)
- **7.5%** = Behavior-based actual (what agents actually choose)

The difference represents the impact of agent decision-making on final outcomes. The tooltips now clearly explain this distinction to avoid confusion.









