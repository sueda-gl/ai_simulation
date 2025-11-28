# Income Category Export Bug Fix

**Date:** November 12, 2025  
**Status:** ✅ Fixed  
**File Modified:** `app/pages/results/components/export_section.py`

---

## 🐛 Problem

The Regular Customer Transaction Export feature was incomplete - specifically, the **Income Category** column (Column E) in the exported Excel file was completely empty for all transactions.

### Example of the Issue:
Looking at the exported Excel file, all fields were populated correctly EXCEPT:
- ✅ Agent ID - populated
- ✅ Assigned Allowance Level - populated  
- ✅ Group_experiment - populated
- ✅ Customer Type - populated
- ❌ **Income Category - EMPTY** 
- ✅ Purchase Request Type - populated
- ✅ Date/Time of Purchase Request - populated
- ✅ Period - populated
- ✅ Customer Price - populated
- ✅ Transaction Completed - populated

---

## 🔍 Root Cause Analysis

The bug was in the column filtering logic at the top of the `render_export_section()` function.

### The Problematic Code (Lines 127-130):

```python
columns_to_exclude = ['raw', 'index', 'purchasing_frequency', 'actual_allowance', 'income', 'customer_type', 'enriched_requests_count']

if df is not None:
    df = df[[col for col in df.columns if not any(excl in col.lower() for excl in columns_to_exclude)]]
```

### The Issue:

The filtering logic uses **substring matching**: `excl in col.lower()`

When checking if `'income'` (from the exclusion list) is in column names:
- ✅ Correctly excludes: `'income'` 
- ❌ **Incorrectly excludes: `'income_category'`** ← THE BUG!

Since `'income'` is a substring of `'income_category'`, the filter removed the `income_category` column before the export function could read it.

### Why This Happened:

The original intent was to exclude the raw `'income'` field (dollar amount) from exports, since it's derived data. However, the substring matching approach accidentally caught `'income_category'` as well, which IS needed for the transaction export.

---

## ✅ The Fix

Changed the filtering logic to distinguish between:
1. **Substring matching** - for most exclusions (e.g., anything with 'raw', 'index', etc.)
2. **Exact matching** - for 'income' only (to preserve 'income_category')

### New Code (Lines 128-132):

```python
columns_to_exclude = ['raw', 'index', 'purchasing_frequency', 'actual_allowance', 'customer_type', 'enriched_requests_count']
exact_columns_to_exclude = ['income']  # Exclude exact column name only, not substrings

if df is not None:
    df = df[[col for col in df.columns if not any(excl in col.lower() for excl in columns_to_exclude) and col not in exact_columns_to_exclude]]
```

### What Changed:

1. Removed `'income'` from `columns_to_exclude` list
2. Created new list: `exact_columns_to_exclude = ['income']`
3. Updated filter to check both:
   - Substring exclusions: `not any(excl in col.lower() for excl in columns_to_exclude)`
   - Exact exclusions: `and col not in exact_columns_to_exclude`

### Result:

- ✅ `'income'` is excluded (exact match)
- ✅ `'income_category'` is preserved (not an exact match)

Applied the same fix to the `results_dict` filtering on line 135.

---

## 🧪 How to Verify the Fix

1. **Run a simulation** with the purchasing_quantity decision enabled
2. **Navigate to Results page** → Export section
3. **Download Regular Customer Transactions Excel**
4. **Open the Excel file** and verify:
   - ✅ Column E "Income Category" is now populated with values (1-N)
   - ✅ All other columns remain correctly populated
   - ✅ Values match the agent's income category from the purchasing_quantity decision

### Expected Values:

For a simulation with `num_fixed_categories = 10`:
- Income Category should show values: 1, 2, 3, 4, 5, 6, 7, 8, 9, or 10
- Not all categories may appear (depends on income distribution - see `INCOME_CATEGORY_FIX.md`)
- Each agent has ONE income category (same for all their transactions)

---

## 📋 Technical Context

### Where `income_category` Comes From:

1. **Source**: `src/decisions/purchasing_quantity.py`
2. **Calculation**: `_assign_income_category()` function
3. **Storage**: Returned in decision output dict, merged into agent_state by orchestrator
4. **Return**: Line 303 of `purchasing_quantity.py`:
   ```python
   return {
       "purchasing_quantity": int(total_quantity),
       "purchase_requests": purchase_requests,
       "income_category": int(income_category),  # ← This value
       "income": float(income)
   }
   ```

### How Export Function Uses It:

From `_build_regular_customer_transaction_export()` (line 35):
```python
income_category = row.get('income_category', np.nan)
```

This reads the `income_category` value from each agent's row in the dataframe. With the bug, this column was filtered out BEFORE this line executed, so it always returned `np.nan`.

---

## 🎯 Impact

### Before Fix:
- ❌ Income Category column in Excel export was empty
- ❌ Unable to analyze transactions by income category
- ❌ Export was "incomplete" as reported by user

### After Fix:
- ✅ Income Category column properly populated
- ✅ Can analyze Regular customer transactions by income category
- ✅ Export is complete with all 10 specified fields

---

## 📝 Related Files

- `app/pages/results/components/export_section.py` - Fixed file
- `src/decisions/purchasing_quantity.py` - Where income_category is calculated
- `REGULAR_CUSTOMER_TRANSACTION_EXPORT.md` - Feature documentation
- `INCOME_CATEGORY_FIX.md` - How income categories are assigned

---

## 💡 Lessons Learned

1. **Be careful with substring matching** when filtering column names
2. **Document exclusion intent** - why each field is excluded
3. **Test exports with sample data** to verify all fields populate
4. Consider using **explicit inclusion lists** instead of exclusion lists when possible

---

## ✅ Verification Status

- [x] Bug identified (overly broad substring matching)
- [x] Root cause confirmed (income_category caught by 'income' filter)
- [x] Fix implemented (separate exact vs. substring exclusions)
- [x] No linter errors
- [x] Documentation created

**Next Step for User:** Test the export with a new simulation run to confirm the Income Category column is now populated correctly.







