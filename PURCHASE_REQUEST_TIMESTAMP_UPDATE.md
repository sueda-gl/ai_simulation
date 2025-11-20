# Purchase Request Excel Timestamp Update

**Date:** November 20, 2025  
**Status:** ✅ Implemented

---

## 🎯 Changes Made

Updated the Purchase Request Excel export to use actual timestamps instead of formatted strings, with proper chronological sorting.

---

## 📋 What Changed

### **File Modified:**
`app/pages/results/visualizations/vendor_viz.py` - Function `_build_purchase_request_export()`

### **Changes:**

1. **Request Date & Time Column** (Line 93)
   - **Before:** Formatted string like `"Period 1, Hour 2.4"`
   - **After:** Actual datetime timestamp (current time + elapsed hours)
   - **Implementation:** Uses `datetime.now()` as simulation start time, then adds `timestamp_hours` as a timedelta

2. **Period Column** (Line 89)
   - Already existed as separate column
   - No changes needed - already properly separated from timestamp

3. **Sorting** (Lines 166-168)
   - **New:** All records now sorted chronologically by timestamp before export
   - **Implementation:** Sorts by 'Request Date & Time' column in ascending order

---

## 📊 Column Structure (Unchanged)

The Excel file still contains the same 15 columns:

1. **Transaction ID**
2. **Agent ID**
3. **Assigned Allowance Level**
4. **Group_experiment**
5. **Customer Type**
6. **Request Date & Time** ← **NOW ACTUAL TIMESTAMP, SORTED**
7. **Period** ← **SEPARATE COLUMN** (already was)
8. **Selected Vendor**
9. **Vendor Price**
10. **Quality**
11. **Sustainability**
12. **Proximity**
13. **Vendor Integrated Score**
14. **Transaction Completed**
15. **Customer Paid Price**

---

## 🔧 Technical Implementation

### Timestamp Calculation:
```python
from datetime import datetime, timedelta

# Get simulation start time (current time when export is generated)
simulation_start_time = datetime.now()

# For each request, add elapsed hours to get actual timestamp
request_datetime = simulation_start_time + timedelta(hours=float(timestamp_hours))
```

### Sorting:
```python
# Sort all records by timestamp chronologically
purchase_request_records.sort(
    key=lambda x: x['Request Date & Time'] 
    if isinstance(x['Request Date & Time'], datetime) 
    else datetime.min
)
```

### Period Calculation (unchanged):
```python
# Period still calculated the same way
period = int(timestamp_hours // duration_hours) + 1
```

---

## 📝 Example Data

### Before:
| Request Date & Time | Period |
|---------------------|--------|
| Period 1, Hour 0.5  | 1      |
| Period 1, Hour 1.2  | 1      |
| Period 2, Hour 0.3  | 2      |

### After:
| Request Date & Time          | Period |
|------------------------------|--------|
| 2025-11-20 10:30:30.000000  | 1      |
| 2025-11-20 11:12:42.000000  | 1      |
| 2025-11-20 12:18:36.000000  | 2      |

*(Actual timestamps will vary based on when the export is run)*

---

## 📂 Files Updated

1. **app/pages/results/visualizations/vendor_viz.py**
   - Added datetime imports (line 25)
   - Added simulation_start_time calculation (line 34)
   - Changed timestamp conversion logic (lines 79-97)
   - Added sorting logic (lines 166-168)

2. **VENDOR_SELECTION_EXPORT_IMPLEMENTATION.md**
   - Updated column 6 description
   - Updated "Key Features" section to reflect timestamp changes

---

## ℹ️ Additional Note

There is another similar export feature for "Regular Customer Transactions" in:
- **File:** `app/pages/results/components/export_section.py`
- **Column:** "Date/Time of Purchase Request"
- **Current Format:** Still uses `"Period X, Hour Y.Y"` format

If you want this export to also use actual timestamps with sorting, let me know and I can update that as well.

---

## ✅ Benefits

1. **Real Timestamps:** Export now contains actual date/time values that can be used for time-based analysis
2. **Chronological Order:** All requests are sorted by when they occurred in the simulation
3. **Excel Compatible:** Datetime objects work with Excel's date/time formatting and sorting features
4. **Separate Period Column:** Period remains as a separate column for easy filtering and grouping
5. **Backward Compatible:** Still calculates periods the same way using duration_hours from config

---

## 🧪 Testing

To verify the changes work correctly:

1. Run a simulation with multiple periods
2. Navigate to Results → Vendor Selection
3. Click "📥 Download Purchase Requests Excel"
4. Open the Excel file and verify:
   - ✅ Request Date & Time shows actual timestamps
   - ✅ Period column shows correct period numbers
   - ✅ Rows are sorted chronologically by timestamp
   - ✅ All periods included in correct sheets
   - ✅ Excel recognizes timestamps as date/time format

---

