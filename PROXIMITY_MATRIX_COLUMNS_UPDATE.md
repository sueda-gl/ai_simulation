# Proximity Matrix Columns Update

**Date:** November 20, 2025  
**Status:** ✅ COMPLETE

---

## 🎯 Change Summary

Removed **Assigned Allowance Level** and **Group_experiment** columns from the Agent-Vendor Proximity Score Matrix export.

---

## 📝 Files Modified

### 1. **`app/pages/results/visualizations/vendor_viz.py`** (Lines 1048-1054)

**BEFORE:**
```python
# Add Agent ID
if 'agent_id' in df.columns:
    row_data['Agent ID'] = df.iloc[idx]['agent_id']
else:
    row_data['Agent ID'] = idx + 1

# Add Assigned Allowance Level if available
if 'Assigned Allowance Level' in df.columns:
    row_data['Assigned Allowance Level'] = df.iloc[idx]['Assigned Allowance Level']

# Add Group_experiment if available
if 'Group_experiment' in df.columns:
    row_data['Group_experiment'] = df.iloc[idx]['Group_experiment']

# Add proximity scores for each vendor
```

**AFTER:**
```python
# Add Agent ID
if 'agent_id' in df.columns:
    row_data['Agent ID'] = df.iloc[idx]['agent_id']
else:
    row_data['Agent ID'] = idx + 1

# Add proximity scores for each vendor
```

**Changes:**
- ❌ Removed lines that add 'Assigned Allowance Level' column
- ❌ Removed lines that add 'Group_experiment' column
- ✅ Kept Agent ID column
- ✅ Kept all Vendor proximity columns

---

### 2. **`VENDOR_SELECTION_EXPORT_IMPLEMENTATION.md`** (Lines 14-22)

**BEFORE:**
```markdown
**Excel Structure:**
- **Sheet Name:** "Agent-Vendor Proximity"
- **Columns:**
  - Agent ID
  - Assigned Allowance Level (if available)
  - Group_experiment (if available)
  - Vendor 1 Proximity
  - Vendor 2 Proximity
  - ... (one column per vendor)
```

**AFTER:**
```markdown
**Excel Structure:**
- **Sheet Name:** "Agent-Vendor Proximity"
- **Columns:**
  - Agent ID
  - Vendor 1 Proximity
  - Vendor 2 Proximity
  - ... (one column per vendor)
```

**Changes:**
- ❌ Removed "Assigned Allowance Level (if available)" from documentation
- ❌ Removed "Group_experiment (if available)" from documentation

---

## 📊 New Proximity Matrix Structure

### Columns (Simplified)
| Column Name | Description | Example Value |
|-------------|-------------|---------------|
| Agent ID | Unique agent identifier | 1, 2, 3, ... |
| Vendor 1 Proximity | Proximity score to Vendor 1 (0-100) | 83.45 |
| Vendor 2 Proximity | Proximity score to Vendor 2 (0-100) | 59.21 |
| Vendor 3 Proximity | Proximity score to Vendor 3 (0-100) | 45.89 |
| ... | One column per vendor | ... |

### Example Data
```
Agent ID | Vendor 1 Proximity | Vendor 2 Proximity | Vendor 3 Proximity | Vendor 4 Proximity | Vendor 5 Proximity
---------|-------------------|-------------------|-------------------|-------------------|-------------------
1        | 88.07             | 44.57             | 47.77             | 32.68             | 42.33
2        | 89.73             | 42.14             | 42.87             | 32.79             | 36.99
3        | 62.72             | 29.50             | 32.95             | 3.48              | 8.44
...      | ...               | ...               | ...               | ...               | ...
```

---

## 🎨 What Changed in the App

### Before
- Proximity matrix Excel had **3 metadata columns** (Agent ID, Allowance Level, Group)
- Plus vendor proximity columns

### After
- Proximity matrix Excel has **1 metadata column** (Agent ID only)
- Plus vendor proximity columns

### Why This Change?
1. **Simplicity:** Proximity matrix is focused purely on spatial/distance relationships
2. **Clarity:** Removes columns that aren't directly related to proximity scores
3. **Clean data:** Easier to analyze in tools like R, Python, or Excel
4. **Consistent:** Matches the "Proximity Matrix" sheet in `vendor_score_diagnostics.xlsx`

---

## 📦 Impact on Exports

### Unaffected Exports
These exports **still include** Assigned Allowance Level and Group_experiment columns:
- ✅ **Purchase Requests Export** (`_build_purchase_request_export`)
- ✅ **Main Results Excel** (full simulation results)
- ✅ **Other decision-specific exports**

### Affected Export
Only this export was modified:
- 🔄 **Agent-Vendor Proximity Matrix** (download button: "📊 Download Proximity Matrix Excel")

---

## 🧪 Testing

### How to Verify the Change
1. Run a simulation with multiple agents and multiple vendors
2. Go to Results → Vendor Selection section
3. Scroll to "Agent-Vendor Proximity Score Matrix"
4. Click "📊 Download Proximity Matrix Excel"
5. Open the Excel file
6. **Expected:** Only see "Agent ID" + Vendor proximity columns
7. **NOT expected:** Should NOT see "Assigned Allowance Level" or "Group_experiment"

### Test Case
```python
# Test with 10 agents, 5 vendors
# Download proximity matrix
# Check columns == ['Agent ID', 'Vendor 1 Proximity', ..., 'Vendor 5 Proximity']
# Total columns should be: 1 + num_vendors
```

---

## ✅ Verification

**Code Changes:** ✅ Complete  
**Documentation Updates:** ✅ Complete  
**Backward Compatibility:** ✅ Maintained (only affects proximity matrix export)  
**Other Exports:** ✅ Unaffected

---

**Change completed:** November 20, 2025  
**Modified by:** AI Assistant

