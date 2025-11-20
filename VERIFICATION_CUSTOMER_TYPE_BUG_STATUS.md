# Verification: Customer Type Bug Status

**Date:** November 12, 2025  
**Status:** ✅ **BUG IS FIXED IN CURRENT CODE**  
**Issue:** Excel file contains data from OLD simulation run

---

## Summary

The bug you identified in the `agent_disclosure_customer_types` Excel file **has already been fixed** in the current codebase. However, the Excel file you're viewing contains **data from a simulation that was run BEFORE the fix was applied**.

---

## Evidence

### Issue in Your Excel File (Row 9, Agent ID 8):
- **Disclosed income:** 0 (did NOT disclose)
- **Disclosed documents:** 1 (disclosed - IMPOSSIBLE!)
- **Customer Type:** Discount (IMPOSSIBLE!)
- **Regular:** 0, **Fixed:** 0, **Discount:** 1

This is logically impossible because agents who don't disclose income should NEVER be asked to disclose documents.

### Current Code Behavior (Tested):

I ran the exact scenario from your Excel file through the current code:

```
Input state:
  Assigned Allowance Level: 1
  Income: 9605.29
  Disclosed income: N (did NOT disclose)

CURRENT CODE OUTPUT:
  Disclosed documents: (empty/NA)
  Customer type: regular
  Regular: 1
  Fixed: 0
  Discount: 0
```

**Result:** ✅ The current code produces the CORRECT output!

---

## Fix Details

The bug was fixed in two locations:

### 1. `src/decisions/disclose_documents.py` (Lines 22-33)

The function now checks if the agent disclosed income BEFORE proceeding:

```python
# STEP 1: Check if agent disclosed income first
disclose_income = agent_state.get('disclose_income', 'N')

if disclose_income != "Y":
    # Agent did not disclose income, cannot be asked for documents
    agent_state['disclose_documents'] = "NA"
    customer_type = get_customer_type(agent_state, simulation_config)
    return {
        "disclose_documents": "NA",
        "customer_type": customer_type
    }
```

### 2. `src/decisions/income_utils.py` (Line 526)

The `determine_customer_type()` function now requires income disclosure for discount classification:

```python
# Priority 1: Check for DISCOUNT customer
# Must have disclosed income AND low income AND submitted documents
if disclose_income == "Y" and income <= threshold and disclose_documents == "Y":
    return "discount"
```

---

## Test Results

### ✅ Test 1: Agent who did NOT disclose income
```
Input: disclose_income = N
Output: disclose_documents = NA
Output: customer_type = regular
✅ TEST PASSED: Fix is working correctly
```

### ✅ Test 2: Agent who SHOULD be discount customer
```
Input: disclose_income = Y, income = 9000 (< threshold)
Output: disclose_documents = Y
Output: customer_type = discount
✅ TEST PASSED: Discount customer logic still works
```

### ✅ Test 3: Exact scenario from your Excel (Agent ID 8)
```
Input: disclose_income = N, income = 9605.29
CURRENT CODE: disclose_documents = NA, customer_type = regular
YOUR EXCEL: disclose_documents = 1, customer_type = discount
✅ CONFIRMED: Current code is CORRECT
```

---

## Root Cause Analysis

### Why Your Excel File Shows Wrong Data

1. **When Fix Was Applied:** The files were last modified on November 12, 2025 at 02:00
2. **Your Data Source:** The Excel file was downloaded from a Streamlit session
3. **Problem:** The simulation was likely run BEFORE the fix was applied, OR the Streamlit session is using cached/stale data

### Possible Scenarios

1. **Scenario A:** The simulation was run before November 12, 2025 02:00
2. **Scenario B:** The Streamlit app is running old code and needs to be restarted
3. **Scenario C:** The session state contains cached results from an old simulation
4. **Scenario D:** The Excel file was saved from a previous session and you're reviewing old data

---

## Resolution Steps

To get the corrected data, you need to:

### 1. **Restart the Streamlit App**
```bash
# Stop the current Streamlit session (Ctrl+C)
# Restart it
cd /Users/suedagul/<sdg
source prosocial_analysis_env/bin/activate
streamlit run app_enhanced_new.py
```

### 2. **Run a NEW Simulation**
- Configure your parameters
- Click "🚀 Run Simulation"
- This will generate fresh data using the fixed code

### 3. **Download New Excel File**
- Go to Results page
- Navigate to "Disclose Documents" decision visualization
- Click "📥 Download Agent Disclosure Data (Excel)"

### 4. **Verify the Fix**
Check that in the new Excel file:
- All rows with `Disclosed income = 0` have `Disclosed documents = (empty)`
- All rows with `Disclosed income = 0` have `Regular = 1` and `Discount = 0`
- All rows with `Discount = 1` have both `Disclosed income = 1` AND `Disclosed documents = 1`
- No impossible combinations exist

---

## About the Missing Group_experiment Field

The issue you mentioned about `Group_experiment` being empty is separate from the customer type bug. This occurs when the `group` or `group_experiment` column is not present in the simulation results dataframe.

**Location:** `app/pages/results/visualizations/disclosure_viz.py` (Lines 317-323)

```python
# Group_experiment
if 'group' in df.columns:
    export_df['Group_experiment'] = df['group']
elif 'group_experiment' in df.columns:
    export_df['Group_experiment'] = df['group_experiment']
else:
    export_df['Group_experiment'] = ''  # Empty if not found
```

This is expected behavior when the simulation doesn't include experimental group assignments.

---

## Conclusion

**✅ The customer type classification bug HAS BEEN FIXED in the current code.**

**❌ Your Excel file contains OLD DATA from before the fix.**

**Action Required:** Run a NEW simulation to generate corrected data.

---

**Verification Date:** November 12, 2025  
**Verified By:** AI Code Analysis + Unit Tests  
**Files Tested:**
- `src/decisions/disclose_documents.py`
- `src/decisions/income_utils.py`
- `app/pages/results/visualizations/disclosure_viz.py`




