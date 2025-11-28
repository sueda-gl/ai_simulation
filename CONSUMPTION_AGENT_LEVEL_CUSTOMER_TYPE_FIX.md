# Consumption Agent-Level Customer Type Fix

**Date:** November 12, 2025  
**Issue:** Agents with 0 purchases had blank Customer Type in the agent-level Excel export  
**Status:** ✅ **FIXED**

---

## Problem Description

In the agent-level purchases Excel export (`agent_level_purchases_YYYYMMDD_HHMMSS.xlsx`), some agents were showing blank values in the "Customer Type" column. This occurred specifically for agents who made **0 purchases** during the simulation.

### Example from Screenshot
- **Agent 8:** 0 purchases → blank Customer Type
- **Agent 25:** 0 purchases → blank Customer Type

### Root Cause

The code in `consumption_viz.py` was only extracting `customer_type` from the `purchase_requests` list:

```python
# OLD CODE (lines 651-660)
purchase_requests = row.get('purchase_requests', [])
customer_type = ''

if isinstance(purchase_requests, list) and len(purchase_requests) > 0:
    first_req = purchase_requests[0]
    if isinstance(first_req, dict):
        customer_type = first_req.get('customer_type', '')
```

**Problem:** If an agent has 0 purchases, `purchase_requests` is an empty list, so `customer_type` remained blank.

**Why this is wrong:** Every agent DOES have a `customer_type` (Regular, Fixed, or Discount) determined during the simulation based on their disclosure decisions and income. This information is stored in the dataframe's `customer_type` column, but the code wasn't checking there first.

---

## Solution

Modified the code to use a **two-tier fallback strategy**:

1. **Priority 1:** Check if `customer_type` exists directly in the dataframe row (agent-level)
2. **Priority 2:** If not available, extract from the first `purchase_request` (backward compatibility)

### Code Changes

**File:** `app/pages/results/visualizations/consumption_viz.py`

#### Change 1: Agent-Level Export (Lines 651-668)

```python
# NEW CODE
# Get customer type - try direct column first, then purchase_requests as fallback
customer_type = ''

# Priority 1: Check if customer_type is directly available in the dataframe
if 'customer_type' in row and pd.notna(row['customer_type']) and str(row['customer_type']).strip():
    customer_type = str(row['customer_type']).capitalize()
else:
    # Priority 2: Extract from purchase_requests if available
    purchase_requests = row.get('purchase_requests', [])
    if isinstance(purchase_requests, list) and len(purchase_requests) > 0:
        first_req = purchase_requests[0]
        if isinstance(first_req, dict):
            customer_type = first_req.get('customer_type', '')
            if isinstance(customer_type, str):
                customer_type = customer_type.capitalize()

# Get purchase_requests for counting
purchase_requests = row.get('purchase_requests', [])
```

#### Change 2: Customer Type Analysis Helper (Lines 173-200)

Fixed the `get_quantities_by_customer_type` helper function that was excluding agents with 0 purchases from customer type analysis:

```python
# NEW CODE
def get_quantities_by_customer_type(df, target_type):
    """Extract consumption quantities for agents of a specific customer type"""
    quantities = []
    for idx, row in df.iterrows():
        # Get customer type - try direct column first, then purchase_requests as fallback
        customer_type = ''
        
        # Priority 1: Check if customer_type is directly available in the dataframe
        if 'customer_type' in row and pd.notna(row['customer_type']) and str(row['customer_type']).strip():
            customer_type = str(row['customer_type']).capitalize()
        else:
            # Priority 2: Extract from purchase_requests if available
            purchase_requests = row.get('purchase_requests', [])
            if isinstance(purchase_requests, list) and len(purchase_requests) > 0:
                first_req = purchase_requests[0]
                if isinstance(first_req, dict):
                    customer_type = first_req.get('customer_type', 'regular')
                    if isinstance(customer_type, str):
                        customer_type = customer_type.capitalize()
        
        # If this agent matches the target customer type, include their quantity
        if customer_type == target_type:
            qty = row.get('consumption_quantity', 0)
            quantities.append(qty)
    
    return pd.Series(quantities) if quantities else pd.Series([0])
```

---

## Impact

### ✅ Fixed Issues

1. **Agent-level Excel export** now correctly shows Customer Type for ALL agents, including those with 0 purchases
2. **Customer type analysis** now correctly includes agents with 0 purchases in the consumption quantity distributions

### 🔍 What Was NOT Changed

The following code was **intentionally left unchanged** because it correctly extracts `customer_type` from individual purchase request objects (not agent rows):

1. **Lines 101-107:** Pie chart counting purchases by customer type (iterates through requests)
2. **Lines 439-442:** Consumption frequency timing analysis (iterates through requests)
3. **vendor_viz.py line 87:** Vendor selection export (works with request objects)
4. **donation_viz.py line 85:** Donation transaction export (works with request objects)

These locations work with **request-level data** where the customer type is correctly stored in each request.

---

## Testing Recommendations

To verify the fix:

1. **Run a simulation** with agents making different numbers of purchases (including some with 0)
2. **Download the agent-level Excel** from the Consumption Quantity results page
3. **Verify** that ALL agents have a Customer Type value (Regular, Fixed, or Discount)
4. **Check agents with 0 purchases** specifically - they should now show their customer type

### Expected Customer Type Logic

- **Regular:** Agents who did NOT disclose income (`disclose_income = "N"`)
- **Fixed:** Agents who disclosed income (`disclose_income = "Y"`) but either:
  - Did not submit documents, OR
  - Have income above the discount threshold
- **Discount:** Agents who disclosed income AND submitted documents AND have income ≤ threshold

---

## Related Files

- **Fixed:** `app/pages/results/visualizations/consumption_viz.py` (lines 173-200, 651-668)
- **Related:** `src/decisions/income_utils.py` (determines customer type during simulation)
- **Related:** `CUSTOMER_TYPE_LOGIC_BUG_FIX.md` (previous customer type logic fix)

---

## Summary

This fix ensures that the agent-level export correctly identifies every agent's customer type, even when they made 0 purchases. The solution uses a robust two-tier fallback that prioritizes the agent-level `customer_type` column (which is always present in simulation results) before falling back to extracting from purchase requests.







