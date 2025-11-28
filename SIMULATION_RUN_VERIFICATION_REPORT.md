# Simulation Run - Customer Type Bug Verification Report

**Date:** November 12, 2025  
**Virtual Environment:** `.venv`  
**Simulation:** 100 agents, seed 42  
**Status:** ✅ **BUG FIX VERIFIED - WORKING CORRECTLY**

---

## Executive Summary

I successfully ran a **fresh simulation** using the `.venv` virtual environment to verify the customer type bug fix. The results confirm:

✅ **The bug fix IS working correctly in the current code**  
✅ **Fresh simulations produce correct data with NO impossible combinations**  
❌ **Your Excel file contains stale data from BEFORE the fix was applied**

---

## Simulation Results

### Run Details
- **Agents:** 100
- **Seed:** 42 (for reproducibility)
- **Date/Time:** November 12, 2025 at 07:26:00
- **Output File:** `outputs/simulation_seed42_agents100_all_20251112_072600.csv`

### Customer Type Distribution

| Customer Type | Count | Percentage | Logic |
|--------------|-------|------------|-------|
| **Regular** | 41 | 41.0% | Did NOT disclose income |
| **Fixed** | 54 | 54.0% | Disclosed income but not documents OR above threshold |
| **Discount** | 5 | 5.0% | Disclosed income AND documents AND below threshold |
| **Total** | 100 | 100.0% | |

---

## Bug Verification Results

### ✅ Test 1: Agents Who Did NOT Disclose Income
- **Count:** 41 agents with `disclose_income = N`
- **Expected:** All should have `disclose_documents = NA` and `customer_type = regular`
- **Result:** ✅ **PASS**
  - All 41 have `disclose_documents = NA` (shown as empty in Excel)
  - All 41 classified as `regular` customers
  - **ZERO** classified as `fixed` or `discount` (would be impossible)

### ✅ Test 2: Discount Customers
- **Count:** 5 discount customers
- **Expected:** ALL must have disclosed income AND documents
- **Result:** ✅ **PASS**
  - All 5 have `disclose_income = Y`
  - All 5 have `disclose_documents = Y`
  - All 5 have income below threshold (< $12,500)

### ✅ Test 3: Impossible Combinations
- **Agents with `disclose_income=0` AND `disclose_documents=1`:** 0 ✅
- **Agents with `disclose_income=0` AND `Discount=1`:** 0 ✅
- **Discount customers missing required disclosures:** 0 ✅

---

## Comparison: Old Excel vs. New Simulation

### Your Old Excel File (Agent ID 8) - INCORRECT ❌

```
Agent ID: 8
Assigned Allowance Level: 1
Assigned income: 9605.29
Disclosed income: 0 (did NOT disclose)
Disclosed documents: 1 (disclosed) ← IMPOSSIBLE!
Regular: 0
Fixed: 0
Discount: 1 ← IMPOSSIBLE!
```

**Problems:**
1. Agent did NOT disclose income but somehow disclosed documents (impossible)
2. Agent classified as discount customer without income disclosure (impossible)

### New Simulation - Discount Customers - CORRECT ✅

All 5 discount customers follow the correct logic:

```
Agent 20: income=$11,343.55, disclose_income=Y, disclose_documents=Y ✓
Agent 33: income=$9,246.26,  disclose_income=Y, disclose_documents=Y ✓
Agent 40: income=$10,574.27, disclose_income=Y, disclose_documents=Y ✓
Agent 68: income=$12,438.79, disclose_income=Y, disclose_documents=Y ✓
Agent 69: income=$11,697.43, disclose_income=Y, disclose_documents=Y ✓
```

All have:
- ✅ Disclosed income (Y)
- ✅ Disclosed documents (Y)
- ✅ Income below threshold ($12,500)

### New Simulation - Regular Customers - CORRECT ✅

Sample of agents who did NOT disclose income:

```
Agent 1: income=$29,667.28, disclose_income=N, disclose_documents=NA ✓
Agent 2: income=$13,079.63, disclose_income=N, disclose_documents=NA ✓
Agent 3: income=$35,578.57, disclose_income=N, disclose_documents=NA ✓
Agent 5: income=$23,459.14, disclose_income=N, disclose_documents=NA ✓
Agent 6: income=$19,326.78, disclose_income=N, disclose_documents=NA ✓
```

All have:
- ✅ Did NOT disclose income (N)
- ✅ Not asked to disclose documents (NA)
- ✅ Classified as regular customers

---

## Excel Export Verification

I also generated the Excel export format that matches `agent_disclosure_customer_types.xlsx`:

**Sample rows from NEW export:**

| Agent ID | Assigned Allowance Level | Income | Disclosed income | Disclosed documents | Regular | Fixed | Discount |
|----------|-------------------------|---------|------------------|--------------------|---------| ------|----------|
| 1 | 2 | 29667.28 | 0 | (empty) | 1 | 0 | 0 |
| 2 | 1 | 13079.63 | 0 | (empty) | 1 | 0 | 0 |
| 4 | 1 | 13889.76 | 1 | 0 | 0 | 1 | 0 |
| 20 | 1 | 11343.55 | 1 | 1 | 0 | 0 | 1 |

**Key observations:**
- ✅ All rows with `Disclosed income = 0` have `Disclosed documents = (empty)`
- ✅ All rows with `Disclosed income = 0` have `Regular = 1, Fixed = 0, Discount = 0`
- ✅ All rows with `Discount = 1` have `Disclosed income = 1, Disclosed documents = 1`
- ✅ NO impossible combinations exist

---

## Files Generated

### 1. CSV Results
**File:** `outputs/simulation_seed42_agents100_all_20251112_072600.csv`  
**Size:** 100 agents × 31 columns  
**Contains:** Complete simulation results including all 13 decisions

### 2. Excel Export
**File:** `agent_disclosure_customer_types_FIXED.xlsx`  
**Format:** Matches the `agent_disclosure_customer_types.xlsx` export format  
**Purpose:** Direct comparison with your old file

### 3. Verification Report
**File:** `VERIFICATION_CUSTOMER_TYPE_BUG_STATUS.md`  
**Contains:** Detailed technical analysis of the bug fix

---

## Conclusion

### ✅ Verified: Bug Fix is Working

The customer type classification bug has been **successfully fixed** in the current codebase:

1. **Fixed in code:** The logic in `disclose_documents.py` and `income_utils.py` is correct
2. **Fixed in simulations:** New simulations produce correct data
3. **Fixed in exports:** Excel exports contain no impossible combinations

### ❌ Your Data is Stale

Your Excel file (`agent_disclosure_customer_types.xlsx`) contains data from an **old simulation** that was run **before the bug fix was applied**.

---

## Recommended Actions

### Option 1: Use the New Excel File (Immediate)
The file `agent_disclosure_customer_types_FIXED.xlsx` contains correct data from today's simulation.

### Option 2: Run New Simulation in Streamlit (Best Practice)
1. Start the Streamlit app:
   ```bash
   cd /Users/suedagul/<sdg
   source .venv/bin/activate
   streamlit run app_enhanced_new.py
   ```

2. Configure simulation parameters

3. Click **"🚀 Run Simulation"**

4. Go to **Results** page → **Disclose Documents** tab

5. Click **"📥 Download Agent Disclosure Data (Excel)"**

6. Verify the new file has:
   - ✅ All `Disclosed income = 0` rows have `Disclosed documents = (empty)`
   - ✅ All `Disclosed income = 0` rows have `Regular = 1`
   - ✅ All `Discount = 1` rows have `Disclosed income = 1` AND `Disclosed documents = 1`

---

## Technical Details

### Fix Applied In:

**File 1: `src/decisions/disclose_documents.py` (Lines 22-33)**
- Added check for `disclose_income = "Y"` before asking about documents
- Returns `NA` immediately if agent didn't disclose income

**File 2: `src/decisions/income_utils.py` (Line 526)**
- Updated `determine_customer_type()` to require income disclosure for discount
- Fixed condition: `if disclose_income == "Y" and income <= threshold and disclose_documents == "Y"`

### Simulation Configuration:

```yaml
Population mode: copula
Income specification: categorical
Number of agents: 100
Random seed: 42
Anchor weights: 0.75 observed | 0.25 predicted
Income distribution: lognormal
Discount threshold: $12,500
```

---

**Report Generated:** November 12, 2025  
**Verification Status:** ✅ COMPLETE  
**Next Steps:** Use new data from fresh simulations







