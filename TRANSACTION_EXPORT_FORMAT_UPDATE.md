# Transaction Export Format Update

## 📋 Changes Made

### Overview
Updated transaction-level Excel exports across all visualizations to properly handle:
1. **Separate Period and Hour columns**
2. **Timestamp with date/time formatting**
3. **Transaction Completed set to N/A** (since `enrich_requests` is not currently used)

---

## 🔧 Files Modified

### 1. **`app/pages/results/visualizations/purchasing_viz.py`** (Lines 581-617)

**Changes:**
- ✅ Added **Period** calculation from `timestamp_hours`
- ✅ Added **Hour** (hour within period) as separate column
- ✅ Kept **timestamp** in date/time format (`DD/MM/YYYY HH:MM`)
- ✅ Added **transaction_completed** column with value `'N/A'`
- ✅ Maintained sorting by transaction time (already implemented)

**New Transaction Structure:**
```python
{
    'customer_id': ...,
    'vendorID': ...,
    'platformProductID': ...,
    'purchase type': ...,
    'purchase_bid_value': ...,
    'timestamp': '20/11/2025 14:30',  # Date/time formatted
    'period': 2,                       # NEW: Which period
    'hour': 14.5,                      # NEW: Hour within period
    'transaction_completed': 'N/A',   # NEW: Not tracked (no enrich_requests)
    'timestamp_hours': ...             # Internal - used for sorting, then dropped
}
```

**Calculation Logic:**
```python
# Get simulation parameters
periods = sim_params.get('periods', 1)
duration_hours = sim_params.get('duration_hours', 1.0)

# Calculate period and hour
period = int(timestamp_hours // duration_hours) + 1
hour_in_period = timestamp_hours % duration_hours
```

---

### 2. **`app/pages/results/visualizations/donation_viz.py`** (Lines 135-140)

**Before:**
```python
# Transaction Completed (0/1)
transaction_completed = request.get('transaction_completed', 
                                   request.get('transactionCompleted', 1))
# Ensure it's 0 or 1
if transaction_completed not in [0, 1]:
    transaction_completed = 1 if transaction_completed else 0
```

**After:**
```python
# Transaction Completed - Not available (enrich_requests not currently used)
# Decision 6 and 7 only track purchase REQUESTS, not completed transactions
transaction_completed = 'N/A'
```

---

### 3. **`app/pages/results/visualizations/transaction_viz.py`** (Lines 117-122)

**Same change as donation_viz.py:**
```python
# Transaction Completed - Not available (enrich_requests not currently used)
# Decision 6 and 7 only track purchase REQUESTS, not completed transactions
transaction_completed = 'N/A'
```

---

### 4. **`app/pages/results/visualizations/vendor_viz.py`** (Lines 128-137)

**Before:**
```python
# Get transaction outcome
# Check multiple possible field names
transaction_completed = request.get('transactionCompleted', 
                                   request.get('completed', 
                                   request.get('transaction_completed', np.nan)))
if isinstance(transaction_completed, bool):
    transaction_completed = 1 if transaction_completed else 0
elif pd.isna(transaction_completed):
    # If not tracked, assume completed for now (can be updated later)
    transaction_completed = np.nan
```

**After:**
```python
# Transaction Completed - Not available (enrich_requests not currently used)
# Decision 6 and 7 only track purchase REQUESTS, not completed transactions
transaction_completed = 'N/A'
```

---

## 📊 Excel Export Structure

### Purchasing Transactions Export

**File:** `purchasing_transactions_YYYYMMDD_HHMMSS.xlsx`

**Columns (in order):**
1. `transaction_id` - Sequential ID (1, 2, 3, ...) after sorting by time
2. `customer_id` - Agent ID
3. `vendorID` - Selected vendor
4. `platformProductID` - Product ID
5. `purchase type` - DISCOUNT / FIXED / PN / BID / N/A
6. `purchase_bid_value` - Bid value if applicable, else N/A
7. `timestamp` - Date and time formatted as `DD/MM/YYYY HH:MM`
8. `period` - Which period (1, 2, 3, ...)

**Sorting:** All transactions sorted by `timestamp_hours` (chronological order across ALL customers)

---

### Donation Transactions Export

**File:** `donation_transactions_YYYYMMDD_HHMMSS.xlsx`

**Includes:**
- All columns from purchasing export
- Plus: Donation-specific fields (donation rate, donation paid, total paid)

---

### Vendor Selection Export

**File:** Downloaded from Decision 7 visualization

**Includes:**
- Vendor-specific information

---

## 🎯 Rationale

### Why Period Column?

**User Requirements:**
1. **Analysis by Period**: Allows filtering/grouping by specific periods
2. **Full Timestamp**: Provides exact date/time for precise ordering

**Example:**
```
Period 1 → Transactions in first period
Period 2 → Transactions in second period
```

---

## ✅ Testing Recommendations

### 1. Run a simulation and download purchasing transactions
- Verify columns appear in correct order
- Verify `period` matches expected values (1 to `periods`)
- Verify `hour` is within [0, `duration_hours`)
- Verify `transaction_completed` shows "N/A"

### 2. Check sorting
- Verify transactions are sorted by timestamp across ALL agents
- First transaction should be earliest timestamp
- Last transaction should be latest timestamp

### 3. Verify Period Calculation
**Example:**
- Periods: 3
- Duration: 10 hours each
- Timestamp 0.0 → Period 1, Hour 0.0
- Timestamp 9.9 → Period 1, Hour 9.9
- Timestamp 10.0 → Period 2, Hour 0.0
- Timestamp 25.5 → Period 3, Hour 5.5

---

## 📅 Date Updated

November 20, 2025

---

## 🔗 Related Issues Fixed

1. ✅ Period and hour were combined - now separated
2. ✅ Transaction completed was incorrectly defaulting to 1 - now N/A
3. ✅ Timestamps already sorted (confirmed working)
4. ✅ `disclose_income` missing from main export (fixed in separate update)

