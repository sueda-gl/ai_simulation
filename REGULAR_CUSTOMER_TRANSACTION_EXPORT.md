# Regular Customer Transaction Export Feature

**Date:** November 6, 2025  
**Status:** ✅ Implemented and Ready for Testing

---

## 🎯 Overview

Added a new Excel export feature for transaction-level data specifically for **Regular Customers**. The export provides detailed transaction records organized by period.

---

## 📋 Export Fields

The Excel file contains the following fields at the **Transaction Level**:

1. **Agent ID** - Unique identifier for each agent
2. **Assigned Allowance Level** - Income category (1-5)
3. **Group_experiment** - Experimental group assignment
4. **Customer Type** - Always "Regular" (filtered for regular customers only)
5. **Income Category** - Numeric income category from consumption_quantity decision
6. **Purchase Request Type** - "PN" or "Bid"
7. **Date/Time of Purchase Request** - Format: "Period X, Hour Y.Y"
8. **Period** - Numeric period (1, 2, 3, etc.)
9. **Customer Price** - Actual price paid (PN price or bid value)
10. **Transaction Completed** - 0 (failed) or 1 (completed)

---

## 📊 Excel File Structure

The exported Excel file contains **multiple sheets**:

### 1. **Total Sheet**
- Contains all transactions from all periods combined
- Sorted by Period and Agent ID

### 2. **Period Sheets**
- One sheet per period (e.g., "Period 1", "Period 2", etc.)
- Contains only transactions from that specific period
- Same column structure as Total sheet

**Example Structure:**
```
regular_customers_transactions_20251106_143522.xlsx
├── Total (all periods)
├── Period 1
├── Period 2
└── Period 3
```

---

## 🎨 User Interface

The export feature appears in the **Results Page** under the **Export Results** section:

### Display Elements:
1. **Section Header**: "📋 Transaction-Level Export (Regular Customers)"
2. **Summary Metrics** (shown before download):
   - Total Transactions count
   - Number of Regular Customers
   - Number of Periods
3. **Download Button**: "📥 Download Regular Customer Transactions Excel"

### User Experience:
- If Regular customer transactions exist: Shows metrics and download button
- If no Regular transactions: Shows info message "ℹ️ No Regular customer transactions found in this simulation"
- If error occurs: Shows error message with details

---

## 🔧 Implementation Details

### File Modified:
**`app/pages/results/components/export_section.py`**

### Key Functions:

#### 1. `_build_regular_customer_transaction_export(df)`
- Extracts transaction-level data from purchase_requests
- **Filters**: Only includes customer_type == 'Regular'
- Processes each request to extract all required fields
- Handles multiple timestamp formats
- Determines PN vs Bid from platformPrice and bid_value fields
- Returns list of transaction records

#### 2. Updated `render_export_section()`
- Added new transaction export section
- Creates multi-sheet Excel with openpyxl
- Provides summary metrics before download
- Error handling for missing data or export failures

---

## 📝 Data Extraction Logic

### Purchase Request Type Determination:
```python
if platformPrice == 'PN' or (platformPrice != 'BID' and bid_value == 'N/A'):
    → Purchase Request Type = 'PN'
elif platformPrice == 'BID' or (bid_value != 'N/A' and bid_value is not None):
    → Purchase Request Type = 'Bid'
else:
    → Purchase Request Type = 'Unknown'
```

### Customer Price Extraction:
1. First tries: `price_paid` or `customer_paid_price` or `price`
2. If not found and type is Bid: Uses `bid_value`
3. Otherwise: Uses `vendor_price` or `vendorPrice`

### Period Calculation:
- From `timestamp_hours`: `period = (timestamp_hours // 24) + 1`
- Date/Time format: `"Period {period}, Hour {hour_in_period}"`

---

## ✅ Testing Checklist

When testing this feature:

1. **Run a simulation** with Regular customers
2. **Navigate to Results page**
3. **Scroll to Export section**
4. **Verify metrics** show:
   - ✓ Total transaction count
   - ✓ Number of regular customers
   - ✓ Number of periods
5. **Download Excel** and verify:
   - ✓ "Total" sheet contains all transactions
   - ✓ One sheet per period exists
   - ✓ All 10 fields are present
   - ✓ Only Regular customers included
   - ✓ Data is sorted by Period and Agent ID
   - ✓ Purchase Request Type correctly shows PN/Bid
   - ✓ Customer Price matches PN price or bid value
   - ✓ Transaction Completed is 0 or 1

---

## 🔍 Filter Behavior

**Important:** This export **ONLY includes Regular customers**

- Fixed customers → **EXCLUDED**
- Discount customers → **EXCLUDED**
- Regular customers → **INCLUDED**

This is enforced at line 53 of export_section.py:
```python
if customer_type != 'Regular':
    continue
```

---

## 📂 File Location

**Modified File:**
```
app/pages/results/components/export_section.py
```

**Lines Added:** 
- `_build_regular_customer_transaction_export()`: Lines 9-121
- Transaction export UI section: Lines 233-286

---

## 🎯 Next Steps

1. Run a simulation to test the feature
2. Verify all fields are correctly populated
3. Check that period sheets are created correctly
4. Confirm Regular customer filter works as expected
5. Test with different numbers of periods

---

## 💡 Notes

- Uses existing `purchase_requests` data structure from simulation results
- Compatible with request-level purchase decisions (Decision 9 & 10)
- Leverages income_category from consumption_quantity decision
- Excel filename includes timestamp for uniqueness
- Error handling prevents UI crashes if data is missing

