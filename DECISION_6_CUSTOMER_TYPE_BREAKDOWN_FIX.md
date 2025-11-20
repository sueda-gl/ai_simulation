# Decision 6: Customer Type Breakdown Fix

## Problem

In the Results page for Decision 6 (Purchasing Quantity), the "Purchase Decisions per Request" section was showing:
- **Total Requests**: 25,598 ✓ (correct)
- **Discount Requests**: 0 ✗ (incorrect - should show actual count)
- **Fixed Requests**: 0 ✗ (incorrect - should show actual count)
- **Regular Requests**: 0 ✗ (incorrect - should show actual count)

And the same issue for "Completed Transactions" section.

## Root Cause

The bug had two interconnected causes:

### 1. Missing `platformPrice` Field in Purchase Requests

The file `src/decisions/enrich_purchase_requests.py` was **deleted** (visible in git status), which was responsible for adding the `platformPrice` field to each purchase request. This field should contain values like:
- `'DISCOUNT'` - for discount customers
- `'FIXED'` - for fixed customers
- `'PN'` - for regular customers choosing Purchase Now
- `'BID'` - for regular customers choosing to Bid

### 2. Visualization Looking for Wrong Field

The visualization code in `render_purchasing_frequency()` (in `app/pages/results/visualizations/purchasing_viz.py`) was trying to count customer types by looking for `platformPrice` values:

```python
# OLD CODE (BROKEN)
platform_price = req.get('platformPrice')
all_platform_prices.append(platform_price)

# Then counting:
discount_count = all_counts.get('DISCOUNT', 0)
fixed_count = all_counts.get('FIXED', 0)
```

However, this field was never being populated because the enrichment step was deleted!

## Solution

Instead of relying on the missing `platformPrice` field, the visualization now reads the `customer_type` field directly from purchase requests, which **is** properly populated by Decision 6 (`purchasing_quantity`):

### Changes Made

**File**: `app/pages/results/visualizations/purchasing_viz.py`

1. **Lines 818-826**: Updated to count by `customer_type` field instead of `platformPrice`:
   ```python
   # NEW CODE (FIXED)
   customer_type_counts = Counter()
   
   for idx, row in df.iterrows():
       requests = row.get('purchase_requests', [])
       if isinstance(requests, list):
           for req in requests:
               if isinstance(req, dict):
                   customer_type = req.get('customer_type', 'regular')
                   if isinstance(customer_type, str):
                       customer_type = customer_type.lower()
                       customer_type_counts[customer_type] += 1
   ```

2. **Lines 830-875**: Updated metrics to use counts from `customer_type_counts`:
   ```python
   discount_count = customer_type_counts.get('discount', 0)
   fixed_count = customer_type_counts.get('fixed', 0)
   regular_count = customer_type_counts.get('regular', 0)
   ```

3. **Lines 795-807**: Removed obsolete code that was extracting `platformPrice` data

4. **Line 442**: Updated comment to clarify we're using `customer_type` field directly

## Data Flow

### Before Fix (Broken)
```
Decision 6 (purchasing_quantity)
    ↓ creates purchase_requests with customer_type field
    ↓ (lowercase: "discount", "fixed", "regular")
    ↓
[MISSING: enrich_purchase_requests - DELETED!]
    ↓ should add platformPrice field
    ↓ (values: 'DISCOUNT', 'FIXED', 'PN', 'BID')
    ↓
Visualization
    ↓ looks for platformPrice
    ✗ finds nothing → shows zeros
```

### After Fix (Working)
```
Decision 6 (purchasing_quantity)
    ↓ creates purchase_requests with customer_type field
    ↓ (lowercase: "discount", "fixed", "regular")
    ↓
Visualization
    ↓ looks for customer_type field directly
    ✓ finds correct data → shows actual counts
```

## Testing

After applying this fix, the Results page should now correctly show:
- **Total Requests**: Total count across all customer types
- **Discount Requests**: Count + percentage of discount customers
- **Fixed Requests**: Count + percentage of fixed customers
- **Regular Requests**: Count + percentage of regular customers

And the same breakdown for completed transactions.

## Notes

- The `customer_type` field is populated in `src/decisions/purchasing_quantity.py` at line 348
- Values are lowercase strings: `"discount"`, `"fixed"`, or `"regular"`
- The visualization now normalizes these to lowercase for counting and title case for display
- This fix applies to both the "Purchase Decisions per Request" and "Completed Transactions" sections

## Related Files

- **Fixed**: `app/pages/results/visualizations/purchasing_viz.py`
- **Reference**: `src/decisions/purchasing_quantity.py` (line 348 - where customer_type is set)
- **Deleted**: `src/decisions/enrich_purchase_requests.py` (the root cause)

