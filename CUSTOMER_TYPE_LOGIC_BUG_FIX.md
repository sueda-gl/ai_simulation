# Customer Type Classification Logic Bug Fix

## Issue Reported

In the `agent_disclosure_customer_types` Excel export, agents who **did not disclose income** (`Disclosed income = 0`) were incorrectly being classified as **Discount customers** and showing `Disclosed documents = 1`.

This is logically impossible because:
- Only agents who disclose their income can be asked to disclose documents
- If an agent said "No" to disclosing income, they should never be asked about documents
- Therefore, `disclose_income = "N"` should always result in `disclose_documents = "NA"` and `customer_type = "regular"`

## Root Causes

### Bug #1: `disclose_documents()` function didn't check income disclosure status

**Location:** `src/decisions/disclose_documents.py`

**Problem:** The function only checked if `income < threshold` but didn't verify that the agent had disclosed income first.

**Before:**
```python
def disclose_documents(agent_state: dict, params: dict, rng, simulation_config: dict = None):
    # Get income
    income = get_agent_income(agent_state, simulation_config, rng)
    
    # Only checked income threshold, NOT disclose_income status
    if agent_income >= threshold:
        agent_state['disclose_documents'] = "NA"
        ...
```

**After:**
```python
def disclose_documents(agent_state: dict, params: dict, rng, simulation_config: dict = None):
    # STEP 1: Check if agent disclosed income first (NEW CHECK)
    disclose_income = agent_state.get('disclose_income', 'N')
    
    if disclose_income != "Y":
        # Agent did not disclose income, cannot be asked for documents
        agent_state['disclose_documents'] = "NA"
        return {"disclose_documents": "NA", "customer_type": customer_type}
    
    # STEP 2: Get income
    income = get_agent_income(agent_state, simulation_config, rng)
    
    # STEP 3: Check income threshold
    if agent_income >= threshold:
        agent_state['disclose_documents'] = "NA"
        ...
```

### Bug #2: `determine_customer_type()` didn't enforce income disclosure requirement for discount

**Location:** `src/decisions/income_utils.py`

**Problem:** The function checked for discount status without verifying that the agent disclosed income.

**Before:**
```python
# Priority 1: Check for DISCOUNT customer
if income <= threshold and disclose_documents == "Y":
    return "discount"
```

This allowed agents with `disclose_income = "N"` to be classified as "discount" if they somehow had `disclose_documents = "Y"` (which should never happen, but was possible due to Bug #1).

**After:**
```python
# Priority 1: Check for DISCOUNT customer
# Must have disclosed income AND low income AND submitted documents
if disclose_income == "Y" and income <= threshold and disclose_documents == "Y":
    return "discount"
```

## Correct Logic Flow

### Decision Sequence

1. **Decision 1: Disclose Income?**
   - Agent chooses: "Y" or "N"
   
2. **Decision 2: Disclose Documents?** (conditional)
   - If `disclose_income = "N"` → **Skip this decision**, set `disclose_documents = "NA"`
   - If `disclose_income = "Y"` AND `income >= threshold` → **Skip this decision**, set `disclose_documents = "NA"`
   - If `disclose_income = "Y"` AND `income < threshold` → **Ask agent**, they choose "Y" or "N"

### Customer Type Classification

| Scenario | disclose_income | income vs threshold | disclose_documents | Customer Type |
|----------|----------------|--------------------|--------------------|---------------|
| 1 | N | (any) | NA | **Regular** |
| 2 | Y | >= threshold | NA | **Fixed** |
| 3 | Y | < threshold | N | **Fixed** |
| 4 | Y | < threshold | Y | **Discount** |

### Simplified Rules

- **Regular**: `disclose_income = "N"` → Never asked about documents
- **Fixed**: `disclose_income = "Y"` AND (not discount) → Includes those above threshold OR below threshold who didn't submit docs
- **Discount**: `disclose_income = "Y"` AND `income < threshold` AND `disclose_documents = "Y"` → All three conditions required

## Impact

This fix ensures:

1. ✅ Agents who don't disclose income are NEVER asked to disclose documents
2. ✅ Agents who don't disclose income are ALWAYS classified as "Regular" customers
3. ✅ The `agent_disclosure_customer_types` Excel export will no longer show impossible combinations
4. ✅ The `Group_experiment` field and other derived fields will have consistent values
5. ✅ Customer type classification matches the logical decision flow

## Files Modified

1. **`src/decisions/disclose_documents.py`**
   - Added check for `disclose_income = "Y"` at the start of the function
   - Returns `"NA"` immediately if agent didn't disclose income
   
2. **`src/decisions/income_utils.py`**
   - Updated `determine_customer_type()` to require `disclose_income = "Y"` for discount classification
   - Updated docstring to clarify the complete logic

## Testing

To verify the fix works:

1. Run a simulation with agents
2. Export the `agent_disclosure_customer_types.xlsx` file
3. Check that:
   - All rows with `Disclosed income = 0` have `Disclosed documents = (empty)`
   - All rows with `Disclosed income = 0` have `Regular = 1` and `Fixed = 0` and `Discount = 0`
   - All rows with `Discount = 1` have `Disclosed income = 1` and `Disclosed documents = 1`
   - No impossible combinations exist

---

**Date:** November 7, 2025  
**Status:** ✅ Fixed


