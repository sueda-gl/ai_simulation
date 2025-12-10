# Vendor Selection Breakdown by Period - UI Update

## Summary

Updated the **Vendor Selection Breakdown by Period** section in the Results page to display all periods in a single consolidated table instead of requiring users to click through period tabs.

## Changes Made

### File Modified
- `app/pages/results/visualizations/vendor_viz.py` (Lines 587-698)

### Key Improvements

#### 1. **Single Consolidated Table**
- **Before**: Separate tabs for each period requiring users to click through
- **After**: One table showing all periods at once

#### 2. **Period Column Added**
- New "Period" column as the first column in the table
- Shows which period each row belongs to
- Sorted by period number (ascending)

#### 3. **Summary Metrics Updated**
- **Before**: Metrics showed totals for the selected period only
- **After**: Metrics show totals across ALL periods:
  - **Total Agents**: Unique count of agents across all periods
  - **Total Purchase Requests**: Sum of all purchase requests across all periods
  - **Total Transactions**: Sum of all completed transactions across all periods

#### 4. **Removed UI Elements**
- Deleted period selection tabs/buttons
- Streamlined interface with direct data access

### Table Structure

The new combined table includes the following columns:

| Column | Description |
|--------|-------------|
| **Period** | Period number (1, 2, 3, ...) |
| **Vendor** | Vendor ID (e.g., "Vendor 1", "Vendor 2") |
| **Agents** | Number of unique agents who selected this vendor in this period |
| **% Agents** | Percentage of agents (relative to period total) |
| **Purchase Requests** | Number of purchase requests for this vendor in this period |
| **% Requests** | Percentage of purchase requests (relative to period total) |
| **Transactions** | Number of completed transactions |
| **% Completed** | Completion rate (transactions ÷ requests × 100%) |

### Data Flow

```
For each period:
  For each vendor active in that period:
    - Add row with Period number
    - Calculate metrics for that vendor-period combination
    - Include percentage calculations relative to period totals
```

### Example Output

**Summary Metrics (Above Table):**
```
Total Agents: 1,000 | Total Purchase Requests: 25,598 | Total Transactions: 25,598
```

**Table (All Periods Combined):**
```
Period | Vendor    | Agents | % Agents | Purchase Requests | % Requests | Transactions | % Completed
-------|-----------|--------|----------|-------------------|------------|--------------|------------
1      | Vendor 1  | 150    | 15.0%    | 750               | 14.8%      | 750          | 100.0%
1      | Vendor 2  | 200    | 20.0%    | 1000              | 19.7%      | 1000         | 100.0%
1      | Vendor 3  | 650    | 65.0%    | 3325              | 65.5%      | 3325         | 100.0%
2      | Vendor 1  | 148    | 14.8%    | 740               | 14.6%      | 740          | 100.0%
2      | Vendor 2  | 202    | 20.2%    | 1010              | 19.9%      | 1010         | 100.0%
...
```

## Benefits

✅ **Better UX**: Users see all data at once without clicking through tabs  
✅ **Easier Comparison**: Can compare vendor performance across periods directly  
✅ **Accurate Totals**: Summary metrics now reflect true totals across all periods  
✅ **Cleaner Interface**: Removed unnecessary UI elements (period buttons)  
✅ **Exportable**: Can easily copy/export the complete dataset  

## Technical Notes

- Percentage calculations (% Agents, % Requests) are **period-relative** (not global)
  - This allows users to see how each vendor performed relative to others in that specific period
- Summary metrics at the top are **global totals** across all periods
- Agent counts use `set()` to ensure unique agents per vendor-period combination
- Table is sorted by Period (ascending), then by Vendor ID (ascending)

## Testing Recommendations

1. Run a simulation with multiple periods (e.g., 5 periods)
2. Navigate to Results → Decision 8 (Vendor Selection)
3. Scroll to "Vendor Selection Breakdown by Period"
4. Verify:
   - All periods appear in a single table
   - Period column shows correctly (1, 2, 3, ...)
   - Summary metrics show totals across all periods
   - No period selection tabs/buttons present
   - Data is sorted by period then vendor
   - Percentages are calculated relative to each period

## Location in UI

**Path**: Results Page → Decision 8: Vendor Selection → Vendor Selection Breakdown by Period (Third section)

**Position**: Below the overall "Selection Breakdown" table and above the "Purchase Request Level Data Export" section







