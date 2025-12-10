# Vendor Breakdown Excel Export

## Summary
Added an Excel download button to the **Vendor Selection Breakdown by Period** section.

## Changes Implemented
**Date:** November 28, 2025
**File:** `app/pages/results/visualizations/vendor_viz.py`

### Functionality
- **Button Label**: "📥 Download Period Breakdown Excel"
- **Location**: Immediately below the consolidated "Breakdown by Period" table
- **Content**: Exports the exact data displayed in the table (Period, Vendor, Agents, %, Requests, %, Transactions, Completion Rate)
- **Format**: Excel (.xlsx)

### Technical Details
- Uses `pandas.ExcelWriter` with `openpyxl` engine
- Creates a single sheet named "Vendor Breakdown"
- Generates timestamped filename: `vendor_selection_breakdown_YYYYMMDD_HHMMSS.xlsx`
- Includes error handling for missing dependencies or write errors

## User Benefit
Users can now easily export the consolidated multi-period vendor performance data for offline analysis or reporting.







