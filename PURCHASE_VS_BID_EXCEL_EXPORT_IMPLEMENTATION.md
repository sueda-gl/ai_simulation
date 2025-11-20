# Purchase vs Bid Excel Export Implementation

**Date:** November 20, 2025  
**Status:** ✅ COMPLETED

## 🎯 Overview

Implemented Excel export functionality for the **Purchase vs Bid decision (Decision 9)**, providing detailed request-level data for all Regular Customers. This export allows analysis of individual purchase decisions, pricing, and transaction outcomes.

## 📋 What Was Implemented

### 1. Purchase Request Enrichment

**File:** `src/decisions/purchasing_quantity.py`

**New Function:** `_enrich_purchase_requests()`
- Enriches each purchase request with per-request decisions
- Populates `platformPrice` and `bid_value` fields based on customer type
- Makes unique purchase_vs_bid decision for each request
- Generates unique bid values for each bid request

**Logic:**
```python
For each purchase request:
  - If DISCOUNT customer:
      platformPrice = "DISCOUNT"
      bid_value = "N/A"
  
  - If FIXED customer:
      platformPrice = "FIXED"
      bid_value = "N/A"
  
  - If REGULAR customer:
      Make purchase_vs_bid decision:
        If "bid":
          platformPrice = "BID"
          bid_value = <unique random bid value>
        If "Purchase Now":
          platformPrice = "PN"
          bid_value = "N/A"
```

### 2. Excel Export Functionality

**File:** `app/pages/results/visualizations/transaction_viz.py`

**New Function:** `_build_purchase_vs_bid_export()`
- Builds transaction-level export data for regular customers
- Extracts all required fields from simulation results
- Calculates customer price based on purchase type (PN vs Bid)
- Filters to include only regular customers with PN or BID decisions

**Modified Function:** `render_purchase_vs_bid()`
- Added Excel export section at the end of the visualization
- Creates multi-sheet Excel file with:
  - **Total sheet**: All transactions across all periods
  - **Period sheets**: One sheet per period (Period 1, Period 2, etc.)
- Provides download button with timestamp in filename

## 📊 Excel Export Fields

The Excel export includes the following fields for each purchase request:

| Field | Description | Example |
|-------|-------------|---------|
| **Agent ID** | Unique agent identifier | 1, 2, 3, ... |
| **Assigned Allowance Level** | Income category/allowance level | 1-10 |
| **Group_experiment** | Experimental group assignment | A, B, C |
| **Customer Type** | Customer classification | Regular, Fixed, Discount |
| **Income Category** | Income bracket assignment | 1-10 |
| **Purchase Request Type** | Purchase method chosen | PN, Bid |
| **Date/Time of Purchase Request** | Timestamp of request | "Period 1, Hour 0.5" |
| **Period** | Period number | 1, 2, 3, ... |
| **Customer Price** | Price paid by customer | $82.50, $95.00, etc. |
| **Transaction Completed** | Completion status | 0 (failed) or 1 (completed) |

## 💰 Pricing Logic

Customer prices are calculated as follows:

1. **PN (Purchase Now)**:
   - Price = Baseline Price = (1 + platform_markup) × market_price
   - Example: If market_price = $100, platform_markup = 0.1
   - PN Price = $110

2. **Bid**:
   - Price = Unique random bid value from bidding range
   - Range: [(1 - price_range) × baseline_price, (1 + price_range) × baseline_price]
   - Example: If baseline = $110, price_range = 0.25
   - Bid Range = [$82.50, $137.50]

## 📁 Excel File Structure

The exported Excel file contains multiple sheets:

```
purchase_vs_bid_decisions_20251120_143052.xlsx
├── Total (all requests across all periods)
├── Period 1 (requests from period 1)
├── Period 2 (requests from period 2)
├── Period 3 (requests from period 3)
└── ... (one sheet per period)
```

Each sheet contains the same columns, filtered by period.

## 🎯 Key Features

### 1. Request-Level Decisions
- Each purchase request gets a unique decision
- A single agent can choose differently for each purchase
- Example: Agent 1 might have 7 requests: 3 PN, 4 Bid

### 2. Unique Bid Values
- Each bid request generates a NEW random bid value
- No two bids are identical (unless by random chance)
- Ensures realistic bidding behavior

### 3. Customer Type Filtering
- Export includes ONLY Regular Customers
- Fixed and Discount customers are excluded
- Only PN and BID requests are included

### 4. Multi-Sheet Organization
- Total sheet for overall analysis
- Period sheets for temporal analysis
- Easy comparison across time periods

## 🔧 How to Use

### In the Streamlit App:

1. **Run a Simulation**:
   - Configure parameters in Page 1
   - Customize decisions in Page 2 (or use defaults)
   - Click "Run Simulation"

2. **View Results**:
   - Go to Results page
   - Scroll to "Decision 9: Purchase vs Bid"
   - Review visualizations and statistics

3. **Download Excel**:
   - Scroll to "Export Purchase vs Bid Decision Data" section
   - Click "📊 Download Purchase vs Bid Excel"
   - File downloads with timestamp: `purchase_vs_bid_decisions_YYYYMMDD_HHMMSS.xlsx`

4. **Analyze Data**:
   - Open Excel file
   - Use "Total" sheet for overall analysis
   - Use period sheets for temporal analysis
   - Apply filters, pivot tables, charts as needed

## 📝 Code Changes Summary

### Files Modified:

1. **`src/decisions/purchasing_quantity.py`**
   - Added `_enrich_purchase_requests()` function
   - Integrated enrichment into main `purchasing_quantity()` function
   - Ensures all purchase requests have platformPrice and bid_value

2. **`app/pages/results/visualizations/transaction_viz.py`**
   - Added `_build_purchase_vs_bid_export()` helper function
   - Modified `render_purchase_vs_bid()` to include Excel export section
   - Added import for `numpy as np` for NaN handling

### Dependencies:

- `openpyxl` (required for Excel export)
- Already listed in `requirements.txt`

## ✅ Testing

To test the implementation:

```bash
# Activate virtual environment
source prosocial_analysis_env/bin/activate

# Run test script
python3 test_purchase_vs_bid_export.py
```

The test script verifies:
1. Purchase requests are enriched with platformPrice and bid_value
2. Regular customers have PN or BID decisions
3. Bid values are unique
4. Excel export function works correctly
5. All required fields are present in export

## 📊 Example Output

### Sample Export Record:

```python
{
    'Agent ID': 5,
    'Assigned Allowance Level': 3,
    'Group_experiment': 'A',
    'Customer Type': 'Regular',
    'Income Category': 3,
    'Purchase Request Type': 'Bid',
    'Date/Time of Purchase Request': 'Period 1, Hour 0.8',
    'Period': 1,
    'Customer Price': 95.67,
    'Transaction Completed': 1
}
```

### Sample Statistics:

```
Total regular requests: 450
PN requests: 225 (50.0%)
Bid requests: 225 (50.0%)

Bid value analysis:
  Total bid values: 225
  Unique bid values: 225
  Min bid: $82.50
  Max bid: $137.45
  Mean bid: $110.12
```

## 🚀 Future Enhancements

Potential improvements for future versions:

1. **Additional Filters**:
   - Filter by income category
   - Filter by agent ID range
   - Filter by price range

2. **Additional Metrics**:
   - Vendor information per request
   - Proximity scores
   - Quality ratings

3. **Visualization Enhancements**:
   - Time series charts per period
   - Price distribution histograms
   - Agent-level decision patterns

4. **Export Formats**:
   - CSV export option
   - JSON export option
   - Combined export with other decisions

## 🐛 Known Limitations

1. **Regular Customers Only**:
   - Export includes only Regular customers (by design)
   - Fixed and Discount customers have separate pricing models
   - For all customer types, use the donation decision export

2. **Pricing Parameters**:
   - Uses market_price and platform_markup from Page 1
   - Falls back to defaults if not available
   - Ensure parameters are set correctly for accurate prices

3. **Transaction Completion**:
   - Currently all transactions show as completed (1)
   - Rejection logic not yet implemented
   - Will be updated when rejection logic is added

## 📚 Related Documentation

- `PER_REQUEST_PURCHASE_DECISIONS_IMPLEMENTATION.md` - Original per-request decisions design
- `BID_VALUE_IMPLEMENTATION.md` - Bid value calculation details
- `EXCEL_EXPORT_EXPLANATION.md` - General Excel export architecture
- `TECHNICAL_DOCUMENTATION.md` - Overall system architecture

## 🎓 Technical Notes

### Why Enrichment in purchasing_quantity?

- Purchase requests are created in `purchasing_quantity()`
- Enrichment happens immediately after creation
- Ensures all requests have platformPrice and bid_value
- Simplifies downstream processing

### Why Separate Export Function?

- `_build_purchase_vs_bid_export()` is reusable
- Can be called from other modules if needed
- Separates data processing from visualization
- Easier to test and maintain

### Why Multi-Sheet Excel?

- Total sheet for cross-period analysis
- Period sheets for temporal analysis
- Standard format matching other exports
- Easy to use in Excel/spreadsheet software

## ✅ Verification Checklist

- [x] Purchase requests are enriched with platformPrice
- [x] Purchase requests are enriched with bid_value
- [x] Each bid request gets unique bid value
- [x] Regular customers have PN or BID decisions
- [x] Excel export includes all required fields
- [x] Multi-sheet structure (Total + Periods)
- [x] Customer price calculated correctly
- [x] Download button functional
- [x] No linter errors
- [x] Documentation complete

## 🎉 Conclusion

The Purchase vs Bid Excel export is now fully functional and integrated into the simulation platform. Users can download detailed request-level data for Regular Customers, including purchase decisions, pricing, and transaction outcomes. The export provides valuable insights for analysis and research purposes.

