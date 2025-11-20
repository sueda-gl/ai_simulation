# Final Donation Rate Per-Request Implementation

**Date:** November 20, 2025  
**Status:** ✅ Implemented and Tested

---

## 🎯 Summary

Successfully implemented per-request storage of `final_donation_rate` in purchase requests, enabling Excel export at the purchase-request level instead of agent-level only.

---

## 📊 What Changed

### **Change 1: Modified `_enrich_purchase_requests()` Function**

**File:** `src/decisions/purchasing_quantity.py`  
**Lines:** 84-196

**Added:**
- New parameter: `agent_state` - provides access to agent's donation decisions
- Logic to extract agent's baseline donation rate from `final_donation_rate` or `donation_default`
- Adds `final_donation_rate` field to each purchase request dictionary

**How it works:**
```python
# Get agent's baseline rate (priority order)
agent_baseline_rate = agent_state.get('final_donation_rate',  # Decision 13
                                      agent_state.get('donation_default', 0.10))  # Decision 3

# Add to each request
for request in requests:
    enriched_request['final_donation_rate'] = agent_baseline_rate
```

---

### **Change 2: Updated Function Call in `purchasing_quantity()`**

**File:** `src/decisions/purchasing_quantity.py`  
**Lines:** 457-466

**Changed:**
```python
# Before
purchase_requests = _enrich_purchase_requests(purchase_requests, customer_type, rng, simulation_config)

# After
purchase_requests = _enrich_purchase_requests(
    requests=purchase_requests,
    customer_type=customer_type,
    rng=rng,
    simulation_config=simulation_config,
    agent_state=agent_state  # NEW: Pass agent_state for donation rates
)
```

---

### **Change 3: Updated Excel Export to Read Per-Request Rates**

**File:** `app/pages/results/visualizations/donation_viz.py`  
**Function:** `_build_donation_transaction_export()`

**Modified Lines 59-72:** Renamed agent-level variables for clarity
```python
# Now called agent_default_rate and agent_final_rate (for fallback)
agent_default_rate = row.get('donation_default', np.nan)
agent_final_rate = row.get('final_donation_rate', agent_default_rate)
```

**Added Lines 139-156:** Request-level donation rate logic
```python
# Check if request has its own donation rate
request_donation_rate = request.get('final_donation_rate', None)

# Use request-level if available, otherwise fall back to agent-level
if request_donation_rate is not None:
    final_donation_rate = float(request_donation_rate)
else:
    final_donation_rate = agent_final_rate  # Backward compatibility
```

---

## ✅ Test Results

### **Test 1: Per-Request Storage** ✅ **PASSED**

```
Total agents with requests: 5
Total purchase requests: 89
Requests with final_donation_rate field: 89 ✅

✅ SUCCESS: All purchase requests have final_donation_rate field!
✅ All request rates match their agent's donation_default rate
```

**Example Output:**
```
Agent 1:
  Agent-level donation_default: 0.17256452206342718
  Number of purchase requests: 48
    Request 1: final_donation_rate=0.17256452206342718, platformPrice=BID
    Request 2: final_donation_rate=0.17256452206342718, platformPrice=PN
    Request 3: final_donation_rate=0.17256452206342718, platformPrice=BID
    ... 45 more requests (all have final_donation_rate: True)
```

---

## 📊 How It Works Now

### **Data Flow:**

```
1. Agent Decisions Execute (Orchestrator)
   └─ Decision 3: donation_default
      └─ agent_state['donation_default'] = 0.423
   └─ Decision 13: final_donation_rate  
      └─ agent_state['final_donation_rate'] = 0.423

2. Purchase Requests Created (Decision 6: purchasing_quantity)
   └─ Creates basic request dictionaries with timestamps, vendorID, etc.

3. Requests Enriched (_enrich_purchase_requests)
   └─ Adds platformPrice (PN/BID/FIXED/DISCOUNT)
   └─ Adds bid_value (if BID)
   └─ NEW: Adds final_donation_rate ✅
   
4. Excel Export Reads Per-Request Data
   └─ For each request:
      ├─ Reads request['final_donation_rate']
      ├─ Falls back to agent-level if not present (backward compatibility)
      └─ Calculates: donation_paid = customer_price × final_donation_rate
```

---

## 🔄 Data Structure

### **Before (Agent-Level Only):**
```python
{
    'agent_id': 1,
    'donation_default': 0.423,
    'final_donation_rate': 0.423,
    'purchase_requests': [
        {'request_id': 1, 'platformPrice': 'PN'},  # No donation rate
        {'request_id': 2, 'platformPrice': 'BID'}  # No donation rate
    ]
}
```

### **After (Per-Request):**
```python
{
    'agent_id': 1,
    'donation_default': 0.423,
    'final_donation_rate': 0.423,
    'purchase_requests': [
        {
            'request_id': 1,
            'platformPrice': 'PN',
            'final_donation_rate': 0.423  # ✅ NOW IN EACH REQUEST!
        },
        {
            'request_id': 2,
            'platformPrice': 'BID',
            'bid_value': 95.50,
            'final_donation_rate': 0.423  # ✅ NOW IN EACH REQUEST!
        }
    ]
}
```

---

## 📥 Excel Export Structure

The Excel export now has **one row per purchase request** with donation rates:

| Column | Source | Notes |
|--------|--------|-------|
| Agent ID | Agent-level | Same for all requests |
| Assigned Allowance Level | Agent-level | Same for all requests |
| Customer Type | Request-level | Per request |
| Period | Request-level | Calculated from timestamp |
| Customer Price | Request-level | Varies by platformPrice |
| **Default Donation Rate** | Agent-level | Reference value |
| **Final Donation Rate** | **Request-level** | ✅ **NOW PER REQUEST!** |
| **Donation Paid** | Calculated | = Customer Price × Final Donation Rate |
| **Total Paid** | Calculated | = Customer Price + Donation Paid |

### **Example Excel Output:**

```excel
| Agent ID | Customer Price | Final Donation Rate | Donation Paid | Total Paid |
|----------|----------------|---------------------|---------------|------------|
| 1        | 110.00         | 0.423               | 46.53         | 156.53     |
| 1        | 95.50          | 0.423               | 40.40         | 135.90     |
| 1        | 110.00         | 0.423               | 46.53         | 156.53     |
| 2        | 110.00         | 0.100               | 11.00         | 121.00     |
| 2        | 70.00          | 0.100               | 7.00          | 77.00      |
```

---

## ✅ Key Features

### **1. Infrastructure Ready for Variation**

Currently, all requests for an agent use the same rate (agent's baseline). However, the infrastructure is now in place to easily add variation:

**Option A: Random Variation**
```python
# In _enrich_purchase_requests():
variation_factor = rng.normal(1.0, 0.15)  # Mean=1.0, SD=15%
enriched_request['final_donation_rate'] = np.clip(
    agent_baseline_rate * variation_factor, 
    0.0, 1.0
)
```

**Option B: Price-Based**
```python
# Higher prices → higher donation rates
if customer_price > 100:
    enriched_request['final_donation_rate'] = agent_baseline_rate * 1.2
else:
    enriched_request['final_donation_rate'] = agent_baseline_rate
```

**Option C: Time-Based**
```python
# Later periods → different rates
period = int(timestamp_hours // duration_hours) + 1
multiplier = 1.0 + (period - 1) * 0.05  # 5% increase per period
enriched_request['final_donation_rate'] = agent_baseline_rate * multiplier
```

---

### **2. Backward Compatibility** ✅

Old simulations (run before this change) will still work:

- If `request['final_donation_rate']` doesn't exist → Falls back to agent-level rate
- Excel export handles both old and new data formats
- No breaking changes

---

### **3. Consistent with Other Per-Request Decisions**

This implementation follows the same pattern as:
- `platformPrice` - per request
- `bid_value` - per request  
- `final_donation_rate` - **now per request** ✅

All three are added in `_enrich_purchase_requests()` and stored in each request dictionary.

---

## 🧪 How to Test

### **Run Test Script:**
```bash
cd /Users/suedagul/<sdg
source .venv/bin/activate
python test_final_donation_rate_per_request.py
```

### **Expected Output:**
```
✅ SUCCESS: All purchase requests have final_donation_rate field!
✅ All request rates match their agent's donation_default rate
```

### **Manual Verification in UI:**
1. Run simulation with `donation_default` + `purchasing_quantity` decisions
2. Go to Results → Final Donation Rate section
3. Click "📥 Download Transaction-Level Excel"
4. Open Excel file
5. Verify "Final Donation Rate" column exists and has values for each row

---

## 📝 Files Modified

| File | Lines Changed | Description |
|------|---------------|-------------|
| `src/decisions/purchasing_quantity.py` | 84-196 | Modified `_enrich_purchase_requests()` to add `agent_state` param and donation rate logic |
| `src/decisions/purchasing_quantity.py` | 457-466 | Updated function call to pass `agent_state` |
| `app/pages/results/visualizations/donation_viz.py` | 59-72 | Renamed agent-level variables for clarity |
| `app/pages/results/visualizations/donation_viz.py` | 139-172 | Added request-level donation rate reading with fallback |
| `test_final_donation_rate_per_request.py` | NEW | Test script to verify implementation |

---

## 🎯 Benefits

1. ✅ **Purchase-Request Basis**: Excel export now truly per-request, not per-agent
2. ✅ **Accurate Calculations**: Each transaction has its own donation rate
3. ✅ **Flexible Architecture**: Easy to add variation later
4. ✅ **Consistent Pattern**: Matches how `bid_value` and `platformPrice` work
5. ✅ **Backward Compatible**: Old simulations still work
6. ✅ **Well Tested**: Verified with 89 purchase requests across 5 agents

---

## 🚀 Next Steps (Optional Enhancements)

If you want to add variation to donation rates in the future:

1. **Open:** `src/decisions/purchasing_quantity.py`
2. **Find:** Line ~189 where `enriched_request['final_donation_rate'] = agent_baseline_rate`
3. **Replace with:** One of the variation options shown above
4. **Test:** Run simulation and verify rates vary appropriately

---

## 📌 Summary

✅ **Implemented:** `final_donation_rate` now stored in each purchase request  
✅ **Tested:** All 89 requests across 5 agents have the field  
✅ **Excel Export:** Reads per-request values with agent-level fallback  
✅ **Infrastructure:** Ready for future variation if needed  
✅ **Backward Compatible:** Old simulations still work  

**Key Takeaway:** Donation rates are now truly purchase-request-based, enabling accurate per-transaction analysis in Excel exports.

---

*Implementation Date: November 20, 2025*  
*Modified Files:*
- `src/decisions/purchasing_quantity.py` 
- `app/pages/results/visualizations/donation_viz.py`
- `test_final_donation_rate_per_request.py` (new test file)

