# Per-Request Purchase Decisions Implementation

**Date:** October 15, 2025  
**Status:** ❌ DEPRECATED - Removed November 20, 2025

**This feature has been removed. The system now only tracks purchase REQUESTS, not actual transaction details.**

---

## 🎯 Overview

Implemented professor's requirement that purchase decisions are made **PER REQUEST**, not per agent.

### Previous Behavior (Agent-Level):
```
Agent 1 (7 purchases):
  - purchase_vs_bid: "bid"  ← ONE decision for all 7 purchases
  - bid_value: 51           ← ONE value for all 7 purchases
```

### New Behavior (Request-Level):
```
Agent 1 (7 purchases):
  - Request 1: PN, bid_value: N/A
  - Request 2: BID, bid_value: 51
  - Request 3: PN, bid_value: N/A
  - Request 4: BID, bid_value: 52
  - Request 5: BID, bid_value: 53
  - Request 6: PN, bid_value: N/A
  - Request 7: PN, bid_value: N/A
```

---

## 📋 Files Modified

### 1. `src/decisions/purchase_vs_bid.py` ✅
- **Added:** `purchase_vs_bid_single()` function for core decision logic
- **Modified:** Main function to support both agent-level and request-level calls
- **Supports:** Being called multiple times per agent with different results

### 2. `src/decisions/bid_value.py` ✅
- **Added:** `generate_single_bid_value()` function for core bid generation
- **Modified:** Main function to support `request_context` parameter
- **Key Feature:** Each call generates a UNIQUE random bid value

### 3. `src/decisions/consumption_quantity.py` ✅
- **Added:** Transaction fields to each purchase request:
  - `customer_id` (agent ID)
  - `customer_type` (discount/fixed/regular)
  - `vendorID` (default 1)
  - `platformPrice` (to be filled by Decision 6b)
  - `bid_value` (to be filled by Decision 6b)

### 4. `src/decisions/enrich_purchase_requests.py` ✅ **NEW FILE!**
- **Purpose:** Decision 6b - Enriches each purchase request with transaction decisions
- **Process:**
  1. Loops through each agent's purchase_requests
  2. For REGULAR customers: Makes new purchase_vs_bid decision per request
  3. For BID requests: Generates unique bid_value
  4. Sets platformPrice based on customer type and decision
  5. Updates purchase_requests in place

### 5. `src/orchestrator.py` ✅
- **Added:** `enrich_purchase_requests` to decision order (Decision 6b)
- **Placed:** After `consumption_quantity`, before `consumption_frequency`
- **Note:** Decisions 9 and 10 are now deprecated but kept for backward compatibility

### 6. `app/simulation.py` ✅
- **Fixed:** Parquet saving issue with complex nested structures
- **Solution:** Convert `purchase_requests` to JSON strings before saving
- **Impact:** Simulations can now be saved without errors

### 7. `app/pages/results/details.py` ✅
- **Added:** Transaction-level Excel export button
- **Features:**
  - Flattens purchase_requests into transaction rows
  - One row per purchase request
  - Downloads as separate Excel file
  - Shows preview of first 50 transactions

### 8. `src/decisions/transaction_expander.py` ❌ **DELETED**
- **Reason:** No longer needed - purchase requests are enriched during simulation

---

## 📊 Transaction Export Format

Excel export includes the following columns (matching your image):

| transaction_id | customer_id | vendorID | platformPrice | purchase_bid_value | timestamp |
|----------------|-------------|----------|---------------|-------------------|-----------|
| 1              | 1           | 1        | PN            | N/A               | 2.4       |
| 2              | 2           | 1        | FIXED         | N/A               | 5.8       |
| 3              | 3           | 1        | DISCOUNT      | N/A               | 11.3      |
| 4              | 4           | 1        | BID           | 51                | 16.7      |
| 5              | 5           | 1        | PN            | N/A               | 20.1      |

---

## ✅ Test Results

**Test Command:**
```bash
source .venv/bin/activate
python test_purchase_requests.py
```

**Results:**
- ✅ 5 agents simulated
- ✅ 134 total transactions
- ✅ Platform Price Distribution:
  - FIXED: 52 (38.8%)
  - BID: 43 (32.1%)
  - PN: 39 (29.1%)
- ✅ **43 unique bid values** (each bid is different!)
- ✅ Excel export successful: `test_transactions.xlsx`

---

## 🔧 Decision Flow

### New Decision Order:
```
1. disclose_income
2. disclose_documents (sets customer_type)
3. donation_default
4. rejected_transaction_defaults
5. vendor_choice_weights
6. consumption_quantity (creates basic purchase requests)
6b. enrich_purchase_requests ← NEW! (adds per-request decisions)
7. consumption_frequency
8. vendor_selection
9. purchase_vs_bid (deprecated - now handled by 6b)
10. bid_value (deprecated - now handled by 6b)
11. rejected_transaction_option
12. rejected_bid_value
13. final_donation_rate
```

---

## 🎓 How It Works

### Step-by-Step Process:

**Step 1:** Agent completes Decisions 1-2
- Gets `customer_type`: "discount", "fixed", or "regular"

**Step 2:** Decision 6 (consumption_quantity)
- Generates 7 purchase requests with basic fields:
  - `request_id`, `quantity`, `timestamp_hours`
  - `customer_id`, `customer_type`, `vendorID`
  - `platformPrice: None`, `bid_value: None`

**Step 3:** Decision 6b (enrich_purchase_requests) ← **THE KEY!**
- Loops through all 7 requests
- For each request:
  - If DISCOUNT → platformPrice="DISCOUNT", bid_value="N/A"
  - If FIXED → platformPrice="FIXED", bid_value="N/A"
  - If REGULAR:
    - Call `purchase_vs_bid_single()` → get "PN" or "BID"
    - If "BID" → call `generate_single_bid_value()` → get unique bid
    - If "PN" → bid_value="N/A"

**Step 4:** Continue with remaining decisions

**Step 5:** Export
- Flatten purchase_requests to transaction-level DataFrame
- One row per purchase request
- Download as Excel

---

## 📝 Usage in UI

### Running Simulation:
1. Configure simulation parameters in Page 1
2. Select decisions in Page 2
3. Click "🚀 Run Simulation"
4. Go to Results page

### Exporting Transactions:
1. In Results page, scroll to "Export Data" section
2. Click "📋 Download Transactions" button
3. Excel file downloads with transaction-level data
4. Open in Excel to view format matching your image

---

## ⚠️ Important Notes

1. **Backward Compatibility:** Decisions 9 and 10 are kept for legacy code but are now deprecated in favor of Decision 6b

2. **Vendor Selection:** Currently defaults to vendorID=1. This can be updated by vendor_selection decision if implemented

3. **Parquet Saving:** purchase_requests are automatically converted to JSON strings when saving to parquet

4. **Performance:** Each agent's RNG is used for all their requests, maintaining reproducibility

---

## 🧪 How to Test

```bash
# Activate virtual environment
source .venv/bin/activate

# Run test script
python test_purchase_requests.py

# Check generated Excel file
open test_transactions.xlsx
```

---

## ✨ Key Achievement

**Professor's Requirement:** Agent 1 with 7 purchase requests can choose BID for 3 requests and PN for 4 requests, each with different bid values.

**Implementation Result:** ✅ **43 unique bid values across all requests!** Each bid is independently generated with its own random value.

---

**Implementation Complete!** 🎉

