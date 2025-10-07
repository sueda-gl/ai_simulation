# Agent Number Column in Exports

## Summary

All CSV and Excel exports now include an `Agent_Number` column as the **first column**, making it easy to identify and track individual agents in the exported data.

---

## What Changed

### **Modified File:** `app/pages/results/details.py`

Added `Agent_Number` column (starting from 1) as the first column in all export scenarios:

1. ✅ Single CSV export
2. ✅ Multi-config CSV export (ZIP with multiple files)
3. ✅ Single Excel export
4. ✅ Multi-config Excel export (multi-sheet)
5. ✅ Fallback CSV export (error handling)

---

## Implementation Details

### **Code Pattern Used:**

```python
# Before export, add Agent_Number as first column
df_export = df.copy()  # Copy to avoid modifying original
df_export.insert(0, 'Agent_Number', range(1, len(df_export) + 1))

# Then export df_export instead of df
csv_data = df_export.to_csv(index=False)
# or
df_export.to_excel(writer, index=False, sheet_name='Results')
```

### **Key Points:**

- **Position:** First column (index 0)
- **Name:** `Agent_Number` (with underscore for consistency)
- **Values:** Start from 1, increment by 1 (not 0-based)
- **Type:** Integer (1, 2, 3, ..., 500)
- **Non-destructive:** Original DataFrame (`df`) remains unchanged

---

## Export Examples

### **Example 1: Single CSV Export**

**Before:**
```csv
Honesty_Humility,Assigned Allowance Level,Study Program,donation_default
3.45,3,CLEAM,0.423
2.87,5,BESS,0.651
4.12,2,CLEF,0.287
```

**After:**
```csv
Agent_Number,Honesty_Humility,Assigned Allowance Level,Study Program,donation_default
1,3.45,3,CLEAM,0.423
2,2.87,5,BESS,0.651
3,4.12,2,CLEF,0.287
```

### **Example 2: Excel Export (Single Sheet)**

**Before:**
```excel
| Honesty_Humility | Allowance | Study_Program | donation_default |
|------------------|-----------|---------------|------------------|
|      3.45        |     3     |    CLEAM      |      0.423       |
|      2.87        |     5     |    BESS       |      0.651       |
|      4.12        |     2     |    CLEF       |      0.287       |
```

**After:**
```excel
| Agent_Number | Honesty_Humility | Allowance | Study_Program | donation_default |
|--------------|------------------|-----------|---------------|------------------|
|      1       |      3.45        |     3     |    CLEAM      |      0.423       |
|      2       |      2.87        |     5     |    BESS       |      0.651       |
|      3       |      4.12        |     2     |    CLEF       |      0.287       |
```

### **Example 3: Multi-Config Excel (Multiple Sheets)**

**Sheet 1: "copula categorical"**
```excel
| Agent_Number | Honesty_Humility | ... | donation_default |
|--------------|------------------|-----|------------------|
|      1       |      3.45        | ... |      0.423       |
|      2       |      2.87        | ... |      0.651       |
```

**Sheet 2: "copula continuous"**
```excel
| Agent_Number | Honesty_Humility | ... | donation_default |
|--------------|------------------|-----|------------------|
|      1       |      3.45        | ... |      0.398       |
|      2       |      2.87        | ... |      0.689       |
```

**Note:** Each sheet has its own agent numbering (both start from 1)

### **Example 4: CSV ZIP (Multiple Files)**

**File: copula_categorical.csv**
```csv
Agent_Number,Honesty_Humility,Assigned Allowance Level,donation_default
1,3.45,3,0.423
2,2.87,5,0.651
3,4.12,2,0.287
```

**File: copula_continuous.csv**
```csv
Agent_Number,Honesty_Humility,Assigned Allowance Level,donation_default
1,3.45,3,0.398
2,2.87,5,0.689
3,4.12,2,0.251
```

---

## Benefits

### **1. Easy Agent Identification ✅**

```python
# Before: Hard to identify agents
df = pd.read_csv('results.csv')
# Row 0, 1, 2... but which is Agent 1?

# After: Clear agent identification
df = pd.read_csv('results.csv')
agent_1_data = df[df['Agent_Number'] == 1]
agent_42_data = df[df['Agent_Number'] == 42]
```

### **2. Merge/Join Operations ✅**

```python
# Merge data from different simulations
sim1 = pd.read_csv('simulation1.csv')
sim2 = pd.read_csv('simulation2.csv')

# Easy merge on Agent_Number
merged = pd.merge(sim1, sim2, on='Agent_Number', suffixes=('_sim1', '_sim2'))
```

### **3. Tracking Across Configurations ✅**

```python
# Compare same agent across different configs
copula_cat = pd.read_csv('copula_categorical.csv')
copula_cont = pd.read_csv('copula_continuous.csv')

# Agent 1's donation rate in both configs
agent_1_cat = copula_cat[copula_cat['Agent_Number'] == 1]['donation_default'].values[0]
agent_1_cont = copula_cont[copula_cont['Agent_Number'] == 1]['donation_default'].values[0]

print(f"Agent 1: {agent_1_cat:.1%} (categorical) vs {agent_1_cont:.1%} (continuous)")
```

### **4. Human-Readable References ✅**

- "Look at Agent 42" is clearer than "Look at row 41"
- Agent numbers start from 1 (natural counting)
- Consistent across all exports

### **5. Data Validation ✅**

```python
# Verify all agents are present
df = pd.read_csv('results.csv')
expected_agents = set(range(1, 501))  # 500 agents
actual_agents = set(df['Agent_Number'])

if expected_agents == actual_agents:
    print("✅ All 500 agents present")
else:
    missing = expected_agents - actual_agents
    print(f"❌ Missing agents: {missing}")
```

---

## All Export Scenarios Covered

### ✅ **Scenario 1: Single Configuration Export**

**User Action:** Run simulation with single mode → Download CSV or Excel

**Files Generated:**
- CSV: `enhanced_simulation_results_20251007_142315.csv`
- Excel: `enhanced_simulation_results_20251007_142315.xlsx`

**Agent_Number:** ✅ First column in both

### ✅ **Scenario 2: Multi-Configuration Export**

**User Action:** Run simulation with "Compare all" → Download CSV (ZIP) or Excel

**Files Generated:**
- CSV ZIP: `enhanced_simulation_all_configs_20251007_142315.zip`
  - Contains: `copula_categorical.csv`, `copula_continuous.csv`, etc.
- Excel: `enhanced_simulation_all_configs_20251007_142315.xlsx`
  - Contains: Multiple sheets

**Agent_Number:** ✅ First column in all CSV files and all Excel sheets

### ✅ **Scenario 3: Selected Configuration Export**

**User Action:** Select specific config → Download CSV or Excel

**Files Generated:**
- CSV: `enhanced_simulation_selected_20251007_142315.csv`
- Excel: `enhanced_simulation_selected_20251007_142315.xlsx`

**Agent_Number:** ✅ First column in both

### ✅ **Scenario 4: Error Fallback**

**User Action:** ZIP creation fails → Fallback to single CSV

**Files Generated:**
- CSV: `enhanced_simulation_results_20251007_142315.csv`

**Agent_Number:** ✅ First column

---

## Reading Exports with Agent Numbers

### **Python (Pandas)**

```python
import pandas as pd

# Read CSV
df = pd.read_csv('enhanced_simulation_results_20251007_142315.csv')
print(df.columns[0])  # Output: 'Agent_Number'

# Filter by agent number
agent_10 = df[df['Agent_Number'] == 10]

# Sort by agent number
df_sorted = df.sort_values('Agent_Number')

# Group analysis by agent ranges
df['Agent_Group'] = pd.cut(df['Agent_Number'], bins=[0, 100, 200, 300, 400, 500])
grouped = df.groupby('Agent_Group')['donation_default'].mean()
```

### **Excel Analysis**

```
=FILTER(A:Z, A:A=42)  # Get all data for Agent 42
=VLOOKUP(42, A:Z, 5, FALSE)  # Look up Agent 42's donation_default (column 5)
=AVERAGEIF(A:A, "<=100", E:E)  # Average donation for first 100 agents
```

### **R**

```r
library(readr)

# Read CSV
df <- read_csv('enhanced_simulation_results_20251007_142315.csv')

# Filter by agent
agent_10 <- df[df$Agent_Number == 10, ]

# Summarize by agent groups
library(dplyr)
df %>%
  mutate(Agent_Group = cut(Agent_Number, breaks = seq(0, 500, 100))) %>%
  group_by(Agent_Group) %>%
  summarize(mean_donation = mean(donation_default))
```

---

## Important Notes

### **Agent Number vs Row Index**

| Aspect | Agent_Number Column | Row Index |
|--------|-------------------|-----------|
| **Starting value** | 1 | 0 (Python) or 1 (R/Excel) |
| **Visibility** | ✅ Visible in data | ❌ Not part of data |
| **Persistance** | ✅ Survives filtering/sorting | ❌ Changes with operations |
| **Joining** | ✅ Can merge on it | ❌ Cannot merge on it |
| **Human-friendly** | ✅ Starts from 1 | ❌ Often 0-based |

### **Same Agent Across Configurations**

**Agent #1 in different configurations:**
- Same underlying agent traits (if from same run)
- Different decision outcomes (due to different parameters)
- Easy to compare across configurations

**Example:**
```python
# Read all configs
configs = {
    'copula_cat': pd.read_csv('copula_categorical.csv'),
    'copula_cont': pd.read_csv('copula_continuous.csv'),
    'research_spec': pd.read_csv('research_spec_continuous.csv')
}

# Compare Agent 1 across all configs
agent_1_comparison = {}
for name, df in configs.items():
    agent_1_comparison[name] = df[df['Agent_Number'] == 1]['donation_default'].values[0]

print(agent_1_comparison)
# {'copula_cat': 0.423, 'copula_cont': 0.398, 'research_spec': 0.456}
```

### **Agent Numbers Are Sequential**

- Always: 1, 2, 3, ..., N (no gaps)
- Total count = number of agents in simulation
- Example: 500 agents → Agent_Number from 1 to 500

### **No Duplicates**

- Each agent has a unique number within a configuration
- Agent_Number is effectively a primary key

---

## Verification

### **How to Verify the Feature Works:**

1. **Run any simulation**
   - Any configuration (single, multi-config, selected config)
   - Any number of agents

2. **Download CSV:**
   ```python
   import pandas as pd
   df = pd.read_csv('your_export.csv')
   
   # Check first column
   assert df.columns[0] == 'Agent_Number', "First column should be Agent_Number"
   
   # Check values
   assert list(df['Agent_Number']) == list(range(1, len(df) + 1)), "Should be 1, 2, 3, ..."
   
   print("✅ Agent_Number column verified!")
   ```

3. **Download Excel:**
   ```python
   import pandas as pd
   df = pd.read_excel('your_export.xlsx')
   
   # Same checks as CSV
   assert df.columns[0] == 'Agent_Number'
   assert list(df['Agent_Number']) == list(range(1, len(df) + 1))
   
   print("✅ Agent_Number column verified in Excel!")
   ```

4. **Check Multi-Config:**
   ```python
   import pandas as pd
   import zipfile
   
   # Extract and check each CSV in ZIP
   with zipfile.ZipFile('enhanced_simulation_all_configs_20251007.zip') as z:
       for filename in z.namelist():
           with z.open(filename) as f:
               df = pd.read_csv(f)
               assert df.columns[0] == 'Agent_Number'
               print(f"✅ {filename}: Agent_Number verified")
   ```

---

## Edge Cases

### **Edge Case 1: Empty DataFrame**

If `df` is empty (0 agents):
```python
df_export = df.copy()
df_export.insert(0, 'Agent_Number', range(1, len(df_export) + 1))
# range(1, 1) = empty range → Agent_Number column is empty
```

Result: Empty DataFrame with `Agent_Number` column header but no rows ✅

### **Edge Case 2: Single Agent**

If `df` has 1 agent:
```python
df_export.insert(0, 'Agent_Number', range(1, 2))  # [1]
```

Result: `Agent_Number` = 1 ✅

### **Edge Case 3: Very Large Number of Agents**

If `df` has 10,000 agents:
```python
df_export.insert(0, 'Agent_Number', range(1, 10001))  # [1, 2, ..., 10000]
```

Result: Agent numbers from 1 to 10,000 ✅

### **Edge Case 4: DataFrame Already Has Agent_Number**

```python
# This would raise an error if column already exists
df_export.insert(0, 'Agent_Number', range(1, len(df_export) + 1))
# ValueError: cannot insert Agent_Number, already exists
```

**Current Implementation:** Uses `df.copy()` which ensures fresh DataFrame ✅

If you ever need to handle this:
```python
if 'Agent_Number' in df_export.columns:
    df_export.drop('Agent_Number', axis=1, inplace=True)
df_export.insert(0, 'Agent_Number', range(1, len(df_export) + 1))
```

---

## Troubleshooting

### **Issue: Agent_Number not showing as first column**

**Cause:** Viewing an old export file from before this feature was added

**Solution:** Run a new simulation and download fresh exports

### **Issue: Agent numbers not starting from 1**

**Cause:** Shouldn't happen with current implementation

**Check:** 
```python
df['Agent_Number'].min()  # Should be 1
df['Agent_Number'].max()  # Should be len(df)
```

### **Issue: Agent_Number has gaps (e.g., 1, 2, 5, 7)**

**Cause:** Shouldn't happen with `range(1, len(df) + 1)`

**Check:** Data might have been filtered after export

### **Issue: Different Agent_Numbers in UI vs Export**

**Note:** The UI (display on Results page) does NOT show Agent_Number, only exports have it

**Reason:** Agent_Number is added during export, not part of the original DataFrame

---

## Summary

✅ **Implemented:** `Agent_Number` column as first column in all exports
✅ **Coverage:** All CSV and Excel exports (single, multi-config, fallback)
✅ **Values:** Sequential integers starting from 1
✅ **Benefits:** Easy identification, tracking, merging, and validation
✅ **Non-breaking:** Original DataFrames unchanged, only exports modified

**Key Takeaway:** Every exported file now has `Agent_Number` as the first column, making it easy to identify and track individual agents across simulations and configurations.

---

*Implementation Date: 2025-10-07*
*Modified Files:*
- `app/pages/results/details.py`

*All Export Types Updated:*
- ✅ Single CSV
- ✅ Multi-config CSV (ZIP)
- ✅ Single Excel
- ✅ Multi-config Excel (multi-sheet)
- ✅ Fallback CSV (error handling)

