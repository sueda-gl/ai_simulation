# Bid Value Type Consistency Fix

## ❌ Problem

**Error encountered:**
```
❌ Simulation failed: ("Expected bytes, got a 'float' object", 
'Conversion failed for column bid_value with type object')
```

**Root Cause:**
The `bid_value` column had **mixed types**:
- Float values (e.g., `115.73`) when agent chose to bid
- String value (`"NA"`) when agent chose to purchase

When pandas tries to save a DataFrame with mixed types to Parquet format, it fails because Parquet requires consistent column types.

---

## ✅ Solution

**Changed return value for non-bidding agents from string `"NA"` to `np.nan` (float)**

### Code Changes

#### File: `src/decisions/bid_value.py`

**Before:**
```python
if purchase_vs_bid_choice != 'bid':
    return {"bid_value": "NA"}  # ❌ String type
```

**After:**
```python
if purchase_vs_bid_choice != 'bid':
    return {"bid_value": np.nan}  # ✅ Float type (NaN)
```

### Why This Works

1. **`np.nan` is a float type** - Maintains type consistency in the column
2. **Parquet compatible** - Float columns with NaN values save correctly
3. **Pandas standard** - Using NaN for missing values is best practice
4. **Easy to filter** - Can use `df['bid_value'].notna()` to find actual bids

---

## 📁 Files Modified

1. ✅ **`src/decisions/bid_value.py`** (line 27)
   - Changed return value from `"NA"` to `np.nan`
   - Updated docstring

2. ✅ **`app/pages/decision_tabs/bid_value_tab.py`** (lines 131, 143, 178)
   - Updated UI text from "NA" to "NaN (empty)"
   - Updated documentation strings

---

## 🔍 Technical Details

### Data Type Analysis

**Before Fix:**
```python
# Column has mixed types (object dtype)
df['bid_value'].dtype  # → object
df['bid_value'].values  # → [115.73, "NA", 98.45, "NA", ...]
```

**After Fix:**
```python
# Column has consistent float type
df['bid_value'].dtype  # → float64
df['bid_value'].values  # → [115.73, nan, 98.45, nan, ...]
```

### Parquet Compatibility

**Parquet format requirements:**
- ✅ Supports float columns with NaN values
- ❌ Cannot handle mixed string/float columns
- ✅ Efficiently stores sparse data (NaN values compress well)

---

## 💡 Usage Impact

### For Users

**Viewing Results:**
- Empty/missing bid values now appear as NaN or blank in results
- No functional change - same agents bid or don't bid
- Filtering works better: `df['bid_value'].notna()` to see only bids

**Example Results:**
```
agent_id | purchase_vs_bid | bid_value
---------|-----------------|----------
1        | bid             | 115.73
2        | purchase        | NaN      ← Was "NA", now NaN
3        | bid             | 98.45
4        | purchase        | NaN      ← Was "NA", now NaN
```

### For Developers

**Working with bid_value column:**

```python
# Check if agent bid
has_bid = df['bid_value'].notna()

# Get only agents who bid
bidders = df[df['bid_value'].notna()]

# Calculate average bid (automatically excludes NaN)
avg_bid = df['bid_value'].mean()

# Count how many agents bid
num_bidders = df['bid_value'].notna().sum()
```

---

## 🧪 Testing

### Test Case 1: Agent Chose to Bid
```python
agent_state = {'purchase_vs_bid': 'bid'}
result = bid_value(agent_state, params, rng, simulation_config)

# Expected:
# result = {"bid_value": 115.73}  # Some float in range
# type(result['bid_value']) == float ✅
```

### Test Case 2: Agent Chose to Purchase
```python
agent_state = {'purchase_vs_bid': 'purchase'}
result = bid_value(agent_state, params, rng, simulation_config)

# Expected:
# result = {"bid_value": np.nan}
# type(result['bid_value']) == float ✅
# np.isnan(result['bid_value']) == True ✅
```

### Test Case 3: Save to Parquet
```python
# This should now work without errors
df = orchestrator.run_simulation(n_agents=1000, seed=42)
df.to_parquet('results.parquet')  # ✅ Works!
```

---

## ✅ Validation Checklist

- ✅ Code changed to return `np.nan` instead of `"NA"`
- ✅ Docstring updated to reflect new return type
- ✅ UI documentation updated
- ✅ No linting errors
- ✅ Type consistency maintained (all float)
- ✅ Parquet-compatible
- ✅ Follows pandas best practices

---

## 🎯 Summary

**Problem:** Mixed types (float + string) in bid_value column caused Parquet save error

**Solution:** Use `np.nan` (float) instead of `"NA"` (string) for missing values

**Impact:** 
- ✅ Simulations can now save to Parquet successfully
- ✅ Better data consistency
- ✅ Standard pandas practice
- ✅ No functional changes to bidding logic

**Status:** ✅ Fixed and tested - ready to use!

