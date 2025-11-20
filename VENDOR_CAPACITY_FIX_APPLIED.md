# Vendor Capacity Fix - Applied November 12, 2025

## 🎯 **Problem Identified**

**User Configuration (Page 1):**
- Min Products per Vendor/**Period**: 50
- Max Products per Vendor/**Period**: 150
- Average Products per Vendor/**Period**: 100
- **Number of Periods:** 5

**The Bug:**
Vendor capacity was being calculated as if products were for the ENTIRE simulation, but the configuration clearly states "per period". This meant vendors had only 1/5th the capacity they should have!

**Example:**
- Vendor offers 100 products **per period**
- Simulation has 5 periods
- **Should have:** 100 × 5 = **500 total capacity**
- **Actually had:** **100 total capacity** ❌

This caused 90%+ of agents to fail allocation and keep default `vendorID = 1`.

---

## ✅ **Fixes Applied**

### **Fix #1: Correct Capacity Calculation**

**File:** `src/decisions/vendor_selection.py` (Lines 96-109)

**Before:**
```python
if 'vendor_remaining_capacity' not in simulation_config:
    simulation_config['vendor_remaining_capacity'] = {}
    for vendor in vendors:
        vendor_id = vendor['vendor_id']
        capacity = vendor.get('quantity_offered', 100)  # ❌ Used per-period value as total
        simulation_config['vendor_remaining_capacity'][vendor_id] = capacity
```

**After:**
```python
if 'vendor_remaining_capacity' not in simulation_config:
    # IMPORTANT: quantity_offered is PER PERIOD, so multiply by number of periods
    from src.decisions.income_utils import get_simulation_param
    num_periods = get_simulation_param(simulation_config, 'periods', 1)
    
    simulation_config['vendor_remaining_capacity'] = {}
    for vendor in vendors:
        vendor_id = vendor['vendor_id']
        capacity_per_period = vendor.get('quantity_offered', 100)
        # Total capacity = capacity per period × number of periods
        total_capacity = capacity_per_period * num_periods  # ✅ Now respects configuration
        simulation_config['vendor_remaining_capacity'][vendor_id] = total_capacity
```

**Impact:**
- Vendors now have 5x more capacity (for 5 periods)
- More agents should successfully get assigned to vendors
- Distribution should be more even across all vendors

---

### **Fix #2: Mark Failed Allocations Properly**

**File:** `src/decisions/vendor_selection.py` (Lines 173-191)

**Before:**
```python
else:
    # ALL VENDORS SOLD OUT
    return {
        "vendor_selection": np.nan,
        "purchase_requests": purchase_requests,  # ❌ Still has default vendorID=1
        "allocation_failed": True
    }
```

**After:**
```python
else:
    # ALL VENDORS SOLD OUT - agent cannot be allocated
    # Mark all purchase requests as failed by setting vendorID to NaN
    for request in purchase_requests:
        if isinstance(request, dict):
            request['vendorID'] = np.nan  # ✅ Mark as failed allocation
            request['allocation_failed'] = True
    
    # Update agent_state with modified purchase_requests
    agent_state['purchase_requests'] = purchase_requests
    
    return {
        "vendor_selection": np.nan,
        "purchase_requests": purchase_requests,
        "allocation_failed": True
    }
```

**Impact:**
- Failed allocations now have `vendorID = NaN` instead of default=1
- Vendor 1 won't show inflated numbers from failed allocations
- Can distinguish between real assignments and failures
- More honest reporting in visualizations and exports

---

## 📊 **Expected Results After Re-running**

### **Before Fix:**

**Vendor Capacity Example:**
- Vendor 1: 93 products total capacity
- Vendor 1: 1,629 purchase requests (17.5x over!)
- Most agents: Failed allocation with vendorID=1

**Distribution:**
- Vendor 1: 734 agents (96.1%) ← Mostly failures
- Vendor 2: 11 agents (1.4%)
- Vendor 3: 19 agents (2.5%)
- Other vendors: < 3% each

### **After Fix (Expected):**

**Vendor Capacity Example:**
- Vendor 1: 93 × 5 = **465 products total capacity** ✅
- Vendor 1: ~465 purchase requests (at capacity)
- Distribution: Much more even across all 10 vendors

**Distribution (Expected):**
- Each vendor: ~8-12% of agents (more balanced)
- Failed allocations: Much fewer (only after ALL vendors exhausted)
- If failures occur: Show as NaN, not inflated Vendor 1 count

---

## 🔬 **How Capacity Now Works**

### **Configuration Flow:**

1. **Page 1:** User sets:
   - Min Products/Period: 50
   - Max Products/Period: 150
   - Number of Periods: 5

2. **Vendor Generation** (`orchestrator.py`):
   ```python
   quantity_min = sim_config.get('vendor_products_min', 50)
   quantity_max = sim_config.get('vendor_products_max', 150)
   
   # Each vendor gets random quantity in range [50, 150]
   vendors = generate_vendor_attributes(
       quantity_min=quantity_min,
       quantity_max=quantity_max
   )
   # Result: Vendor has 'quantity_offered' between 50-150
   ```

3. **Capacity Initialization** (`vendor_selection.py`):
   ```python
   num_periods = 5  # From configuration
   capacity_per_period = 93  # From vendor['quantity_offered']
   total_capacity = 93 × 5 = 465  # ✅ Correct calculation
   ```

4. **Agent Processing:**
   - Agent 1: Wants 30 items → Vendor 1 (465 - 30 = 435 left)
   - Agent 2: Wants 25 items → Vendor 4 (470 - 25 = 445 left)
   - Agent 3: Wants 28 items → Vendor 1 (435 - 28 = 407 left)
   - ... continues until capacity exhausted

5. **When Capacity Exhausted:**
   - Agent X: Wants 30 items → **No vendor has capacity**
   - ✅ All requests get `vendorID = NaN`
   - ✅ `allocation_failed = True`
   - ✅ Won't inflate Vendor 1 count

---

## 🧮 **Capacity Math Verification**

### **Your Configuration:**
- **10 vendors**
- **Products per vendor:** 50-150 (avg ~100)
- **Periods:** 5
- **Total vendor capacity:** 10 vendors × 100 avg × 5 periods = **5,000 products**

### **Your Demand:**
- **1,000 agents**
- **Consumption limit:** 50 (disabled, fallback)
- **Random quantity:** 0-50 per agent
- **Average:** ~25 per agent
- **Total demand:** 1,000 × 25 = **25,000 products**

### **Analysis:**
- **Supply:** 5,000 products
- **Demand:** 25,000 products
- **Shortfall:** 20,000 products (80% demand unmet!)

**This means:**
- Only ~200 agents (20%) will get fully served
- ~800 agents (80%) will get `vendorID = NaN`

**To Fix:**
- **Option A:** Increase vendor capacity (500-1500 per period)
- **Option B:** Reduce consumption limits (10-20 per agent)
- **Option C:** Add more vendors (20-30 vendors)

---

## ✅ **Testing Checklist**

After re-running simulation, verify:

1. ✅ Vendor capacity is 5x higher (check vendor attributes table)
2. ✅ Distribution more even across all 10 vendors
3. ✅ No single vendor dominates with 90%+
4. ✅ Failed allocations show NaN, not Vendor 1
5. ✅ All 5 periods appear in breakdown
6. ✅ Total capacity = quantity_offered × 5 periods

---

## 📋 **Remaining Issues**

1. **Supply/Demand Imbalance:** Total supply (5k) < total demand (25k)
   - Will still have many failed allocations (now correctly marked as NaN)
   - Consider adjusting Page 1 settings

2. **Customer Paid Price:** Still needs implementation (separate fix)

3. **Partial Fulfillment:** Currently assigns ALL agent requests to vendor even if partially filled
   - May need refinement for more realistic behavior

---

**Status:** ✅ APPLIED  
**Date:** November 12, 2025  
**Files Modified:** `src/decisions/vendor_selection.py`  
**Test Required:** Re-run simulation to verify capacity fix





