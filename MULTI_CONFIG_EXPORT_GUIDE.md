# Multi-Configuration Export Guide

## Overview

The Excel export feature now intelligently handles multiple simulation configurations. When you run simulations in **comparison modes** (e.g., "Compare all" or "Compare both income"), you can now export **ALL configurations at once** as separate sheets in a single Excel file.

---

## What Changed?

### **Before (Old Behavior)**

❌ Only exported the **first DataFrame** from comparison results  
❌ Lost data from other 5 configurations (in "Compare all" + "Compare both" mode)  
❌ No way to get all results without re-running simulations

### **After (New Behavior)**

✅ Exports **ALL configurations** as separate sheets in one Excel file  
✅ Automatically detects when multiple configs are available  
✅ Clear labeling: Button shows "📊 Download Excel (All 6 Configs)"  
✅ Intelligent: Only exports all when NO config is selected  
✅ User-friendly: Shows preview of what will be exported

---

## How It Works

### **Scenario 1: Comparison Mode WITHOUT Selected Config**

**Setup:**
- Population Mode: "Compare all"
- Income Mode: "Compare both"
- No configuration selected

**Result:** Generates **6 configurations**:
1. copula_categorical
2. copula_continuous
3. research_spec_categorical
4. research_spec_continuous
5. research_baseline_categorical
6. research_baseline_continuous

**Export Behavior:**

📋 **Info Banner Shows:**
```
📋 Multi-Configuration Export Available: 6 configurations will be exported as separate sheets
```

📊 **Expandable Preview:**
```
View Configurations to be Exported
1. Copula Categorical         | Agents: 500 | Mean donation: 42.3%
2. Copula Continuous          | Agents: 500 | Mean donation: 39.8%
3. Research Spec Categorical   | Agents: 500 | Mean donation: 41.5%
4. Research Spec Continuous    | Agents: 500 | Mean donation: 38.7%
5. Research Baseline Categorical | Agents: 500 | Mean donation: 44.2%
6. Research Baseline Continuous  | Agents: 500 | Mean donation: 41.9%
```

📥 **Download Buttons:**
- **CSV Button:** "📥 Download CSV (All 6 Configs)" - Downloads all 6 as separate CSV files in a ZIP
  - Tooltip: "Downloads all 6 configurations as separate CSV files in a ZIP archive"
- **Excel Button:** "📊 Download Excel (All 6 Configs)" - Downloads all 6 in ONE sheet with separate columns
  - Tooltip: "Downloads all 6 configurations in one sheet with separate columns for easy comparison"

📦 **CSV ZIP File Structure:**
```
enhanced_simulation_all_configs_20251007_142315.zip
├── copula_categorical.csv                     (500 rows × N columns)
├── copula_continuous.csv                      (500 rows × N columns)
├── research_spec_categorical.csv              (500 rows × N columns)
├── research_spec_continuous.csv               (500 rows × N columns)
├── research_baseline_categorical.csv          (500 rows × N columns)
└── research_baseline_continuous.csv           (500 rows × N columns)
```

📊 **Excel File Structure:**
```
enhanced_simulation_all_configs_20251007_142315.xlsx
└── Sheet: "All Configurations"  (500 rows × many columns)
    
    Columns structure:
    - Agent_Number
    - Honesty_Humility
    - Assigned Allowance Level
    - Study Program
    - Group_experiment
    - TWT+Sospeso [=AW2+AX2]{Periods 1+2}
    - donation_default_Copula_Categorical
    - donation_default_Copula_Continuous
    - donation_default_Research_Spec_Categorical
    - donation_default_Research_Spec_Continuous
    - donation_default_Research_Baseline_Categorical
    - donation_default_Research_Baseline_Continuous
    ... (other decision columns with config suffixes)
```

**Note:** All configurations are in a SINGLE sheet with separate columns for each config, making comparisons much easier. CSV files in the ZIP remain as separate files.

### **Scenario 2: Comparison Mode WITH Selected Config**

**Setup:**
- Population Mode: "Compare all"
- Income Mode: "Compare both"
- ✅ Configuration selected: "Research Spec + Continuous"

**Result:** Generates **1 configuration** (only the selected one)

**Export Behavior:**

🎯 **Info Banner Shows:**
```
🎯 Results using selected donation configuration: Research Specification + continuous only
```

📥 **Download Buttons:**
- **CSV Button:** "📥 Download CSV" - Downloads the single configuration
- **Excel Button:** "📊 Download Excel" - Downloads single configuration as one sheet

📊 **Excel File Structure:**
```
enhanced_simulation_results_20251007_142315.xlsx
└── Sheet: "Results"  (500 rows × N columns)
```

### **Scenario 3: Single Mode (No Comparison)**

**Setup:**
- Population Mode: "Copula (synthetic)"
- Income Mode: "categorical only"

**Result:** Generates **1 configuration**

**Export Behavior:**

📥 **Download Buttons:**
- **CSV Button:** "📥 Download CSV" - Downloads the single configuration
- **Excel Button:** "📊 Download Excel" - Downloads single configuration as one sheet

📊 **Excel File Structure:**
```
enhanced_simulation_results_20251007_142315.xlsx
└── Sheet: "Results"  (500 rows × N columns)
```

---

## Technical Implementation

### **Detection Logic**

The system determines whether to export all configurations based on three conditions:

```python
export_all_configs = (
    results_dict is not None and        # Results dictionary exists
    len(results_dict) > 1 and          # Multiple configurations available
    not using_selected_config          # No configuration was selected
)
```

### **Multi-Sheet Export Code**

```python
if export_all_configs:
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        for config_key, config_df in results_dict.items():
            if not config_df.empty:
                # Clean sheet name (max 31 chars, no special chars)
                sheet_name = config_key.replace('_', ' ')[:31]
                config_df.to_excel(writer, index=False, sheet_name=sheet_name)
```

### **Sheet Naming**

Sheet names are derived from the configuration keys:

| Configuration Key | Sheet Name | Notes |
|-------------------|------------|-------|
| `copula_categorical` | "copula categorical" | Underscores → spaces |
| `research_spec_continuous` | "research spec continuous" | Within 31 char limit |
| `research_baseline_categorical` | "research baseline categori" | Truncated to 31 chars |

### **File Naming**

| Scenario | Format | Filename Pattern | Example |
|----------|--------|------------------|---------|
| Single config | CSV | `enhanced_simulation_results_[timestamp].csv` | `enhanced_simulation_results_20251007_142315.csv` |
| Single config | Excel | `enhanced_simulation_results_[timestamp].xlsx` | `enhanced_simulation_results_20251007_142315.xlsx` |
| Multiple configs | CSV | `enhanced_simulation_all_configs_[timestamp].zip` | `enhanced_simulation_all_configs_20251007_142315.zip` |
| Multiple configs | Excel | `enhanced_simulation_all_configs_[timestamp].xlsx` | `enhanced_simulation_all_configs_20251007_142315.xlsx` |

---

## What Each Sheet Contains

Each sheet in the multi-config Excel file contains:

### **Columns**

1. **Trait Columns (5):**
   - `Honesty_Humility`
   - `Assigned Allowance Level`
   - `Study Program`
   - `Group_experiment`
   - `TWT+Sospeso [=AW2+AX2]{Periods 1+2}`

2. **Decision Columns (varies):**
   - `donation_default` (primary donation rate 0-1)
   - `donation_default_raw` (optional, pre-truncation)
   - `donation_default_raw_pos` (optional, floored 0-100)
   - Other decision outputs if multiple decisions were run

### **Rows**

- One row per agent
- Typical: 500 rows (or whatever `n_agents` was set to)
- Each configuration has the **same number of agents**

### **Example Sheet Content**

```
| Honesty_Humility | Allowance | Study_Program | Group | TWT+Sospeso | donation_default | ... |
|------------------|-----------|---------------|-------|-------------|------------------|-----|
|      3.45        |     3     |    CLEAM      | NoSub |    56.0     |      0.387       | ... |
|      2.87        |     5     |    BESS       | MidSub|    72.0     |      0.651       | ... |
|      4.12        |     2     |    CLEF       | FullSub|   28.0     |      0.245       | ... |
|      ...         |    ...    |      ...      |  ...  |    ...      |       ...        | ... |
```

---

## Benefits

### **1. Complete Data Preservation**

✅ No data loss from comparison modes  
✅ All configurations preserved in one file  
✅ Easy to compare across configurations

### **2. Convenience**

✅ One download gets everything  
✅ No need to re-run simulations  
✅ All data organized in logical sheets

### **3. Analysis-Friendly**

✅ Easy to compare sheets in Excel  
✅ Can create cross-sheet formulas  
✅ Pivot tables can reference multiple sheets  
✅ Compatible with data analysis tools (Pandas, R, etc.)

### **4. Clear Communication**

✅ Sheet names clearly identify each configuration  
✅ Preview shows what will be exported  
✅ File name indicates "all configs" vs single config

### **5. Flexibility**

✅ Can still get single config via selected config feature  
✅ CSV option available for simple exports  
✅ Automatic detection - no extra configuration needed

---

## Use Cases

### **Use Case 1: Comprehensive Analysis**

**Goal:** Compare donation behavior across all population modes and income specifications

**Workflow:**
1. Set "Compare all" + "Compare both"
2. Run simulation
3. Download "Excel (All 6 Configs)"
4. Open in Excel/Python/R
5. Create comparison charts across sheets
6. Identify which configuration is most realistic

### **Use Case 2: Documentation**

**Goal:** Document all simulation results for research paper

**Workflow:**
1. Run comparison simulation
2. Download "Excel (All 6 Configs)"
3. Each sheet becomes a supplementary table
4. Include filename and timestamp in paper
5. Readers can verify all configurations

### **Use Case 3: Sensitivity Analysis**

**Goal:** Understand how population mode affects donation rates

**Workflow:**
1. Download all configs
2. Extract `donation_default` from each sheet
3. Compare distributions (mean, std, quartiles)
4. Visualize sensitivity to population mode choice

### **Use Case 4: Model Selection**

**Goal:** Choose best configuration for subsequent simulations

**Workflow:**
1. Download all configs
2. Compare each against known benchmark data
3. Calculate fit metrics (RMSE, correlation, etc.)
4. Select best-fitting configuration
5. Use "Use This Config" to lock it in

---

## Reading Multi-Sheet Excel Files Programmatically

### **Python (Pandas)**

```python
import pandas as pd

# Read all sheets at once
all_configs = pd.read_excel(
    'enhanced_simulation_all_configs_20251007_142315.xlsx',
    sheet_name=None  # None = read all sheets
)

# Result is a dictionary of DataFrames
print(f"Loaded {len(all_configs)} configurations")

# Access specific sheet
copula_cat = all_configs['copula categorical']
print(f"Copula categorical mean: {copula_cat['donation_default'].mean():.1%}")

# Compare across all sheets
for sheet_name, df in all_configs.items():
    mean_donation = df['donation_default'].mean()
    print(f"{sheet_name:30s}: {mean_donation:.1%}")
```

### **Python (openpyxl)**

```python
from openpyxl import load_workbook
import pandas as pd

# Load workbook
wb = load_workbook('enhanced_simulation_all_configs_20251007_142315.xlsx')

# List all sheet names
print(f"Sheets: {wb.sheetnames}")

# Read specific sheet
ws = wb['copula categorical']
data = []
for row in ws.iter_rows(min_row=2, values_only=True):  # Skip header
    data.append(row)

# Convert to DataFrame
columns = [cell.value for cell in ws[1]]
df = pd.DataFrame(data, columns=columns)
```

### **R**

```r
library(readxl)

# Read all sheets
all_configs <- lapply(excel_sheets('enhanced_simulation_all_configs_20251007_142315.xlsx'),
                     function(sheet) {
                       read_excel('enhanced_simulation_all_configs_20251007_142315.xlsx',
                                 sheet = sheet)
                     })

# Name the list elements
names(all_configs) <- excel_sheets('enhanced_simulation_all_configs_20251007_142315.xlsx')

# Access specific config
copula_cat <- all_configs[['copula categorical']]
mean_donation <- mean(copula_cat$donation_default)
cat(sprintf("Copula categorical mean: %.1f%%\n", mean_donation * 100))
```

---

## Comparison: Old vs New Behavior

| Aspect | Old Behavior | New Behavior |
|--------|-------------|--------------|
| **Configs exported** | Only first one | All configurations |
| **File structure** | Single sheet | Multiple sheets (one per config) |
| **Data loss** | Yes (5 out of 6 configs lost) | No data loss |
| **User awareness** | Not shown | Info banner + preview |
| **Button label** | "Download Excel" | "Download Excel (All 6 Configs)" |
| **CSV behavior** | Same as Excel | Only first config + tooltip |
| **File naming** | Generic timestamp | Includes "all_configs" |
| **Works with selected config** | N/A | Automatically reverts to single sheet |

---

## Edge Cases & Special Scenarios

### **1. Single Configuration Result**

**Scenario:** Only 1 configuration generated (single mode or selected config)

**Behavior:** 
- No multi-config banner shown
- Excel exports single sheet
- Standard "Download Excel" button label

### **2. Empty DataFrames**

**Scenario:** Some configurations have empty DataFrames

**Behavior:**
- Empty DataFrames are **skipped** in export
- Only non-empty configurations create sheets
- Button shows actual count: "Download Excel (All 4 Configs)" if 2 are empty

### **3. Very Long Configuration Names**

**Scenario:** Configuration key > 31 characters

**Behavior:**
- Sheet name truncated to 31 chars
- Example: `research_baseline_categorical` → `"research baseline categori"`
- Still unique and identifiable

### **4. Special Characters in Names**

**Scenario:** Configuration keys with special characters

**Behavior:**
- Underscores replaced with spaces
- Other special characters preserved (Excel handles them)
- Example: `copula_v2.1` → `"copula v2.1"`

### **5. Selected Config + Comparison Mode**

**Scenario:** User selected a config, then ran comparison simulation

**Behavior:**
- `using_selected_config=True` detected
- Only selected config is generated and exported
- Multi-config export is **disabled**
- Shows: "🎯 Results using selected donation configuration: ..."

---

## Troubleshooting

### **Issue: Button still says "Download Excel" instead of "Download Excel (All X Configs)"**

**Cause:** One of these conditions is not met:
- `results_dict is None` (results not passed correctly)
- `len(results_dict) <= 1` (only one config generated)
- `using_selected_config is True` (config was selected)

**Solution:**
- Check that you ran in comparison mode ("Compare all" or "Compare both")
- Verify no configuration is selected (check Page 2)
- Check debug panel on Results page for `simulation_results` structure

### **Issue: Some sheets are missing**

**Cause:** Those DataFrames were empty

**Solution:**
- Check simulation log for errors during those configurations
- Verify all modes completed successfully
- Some modes might legitimately produce no data (e.g., depvar mode)

### **Issue: Sheet names are truncated weirdly**

**Cause:** Excel 31-character limit for sheet names

**Solution:**
- This is expected behavior
- Sheet names are still unique and identifiable
- Check configuration key in the preview expander to see full name

### **Issue: Excel file is very large**

**Cause:** Multiple configurations × many agents × many columns

**Example:**
- 6 configs × 500 agents × 20 columns = 60,000 cells
- With actual data, can be 5-10 MB

**Solution:**
- This is normal for comprehensive exports
- Consider reducing `n_agents` if file size is an issue
- Use parquet format for more efficient storage (future enhancement)

---

## Future Enhancements

Potential improvements to consider:

### **1. Separate File Option**

Allow users to choose between:
- **Option A:** One Excel file with multiple sheets (current)
- **Option B:** Multiple Excel files (one per config)

```python
if st.checkbox("Export as separate files", value=False):
    # Generate one file per config
    for config_key, config_df in results_dict.items():
        # Create individual download button
        ...
```

### **2. ZIP Archive Export**

For very large datasets:

```python
st.download_button(
    label="📦 Download ZIP (All Configs)",
    data=create_zip_archive(results_dict),
    file_name=f"simulation_configs_{timestamp}.zip",
    mime="application/zip"
)
```

### **3. Selective Export**

Allow users to choose which configs to export:

```python
selected_configs = st.multiselect(
    "Select configurations to export",
    options=list(results_dict.keys()),
    default=list(results_dict.keys())
)
```

### **4. Metadata Sheet**

Add a "Metadata" sheet with simulation parameters:

```
Sheet: "Metadata"
| Parameter              | Value                      |
|------------------------|----------------------------|
| Timestamp              | 2025-10-07 14:23:15       |
| Population Mode        | Compare all                |
| Income Mode            | Compare both               |
| Number of Agents       | 500                        |
| Seed                   | 42                         |
| Configurations Exported| 6                          |
| ...                    | ...                        |
```

### **5. Summary Statistics Sheet**

Add a sheet comparing all configurations:

```
Sheet: "Summary"
| Configuration              | Mean Donation | Std Dev | Median | Min  | Max  |
|----------------------------|---------------|---------|--------|------|------|
| Copula Categorical         | 42.3%         | 12.1%   | 40.5%  | 0%   | 98%  |
| Copula Continuous          | 39.8%         | 10.5%   | 38.2%  | 0%   | 95%  |
| ...                        | ...           | ...     | ...    | ...  | ...  |
```

---

## Summary

The multi-configuration export feature:

✅ **Automatically detects** when multiple configurations are available  
✅ **Exports all configurations** as separate sheets in one Excel file  
✅ **Clearly communicates** what will be exported with info banners and previews  
✅ **Intelligently adapts** based on whether a config is selected  
✅ **Preserves all data** - no more losing 5 out of 6 configurations  
✅ **User-friendly** with clear labeling and helpful tooltips  

**Usage:**
- Run simulations in comparison mode
- Check the preview to see what will be exported
- Click "📊 Download Excel (All X Configs)"
- Open the multi-sheet Excel file and analyze all configurations

**Key Files Modified:**
- `app/pages/results/details.py` - Added multi-config export logic
- `app/pages/results/main_results.py` - Pass results_dict to export function

---

*Document created: 2025-10-07*
*Last updated: 2025-10-07*

