# Single-Sheet Comparison Export

## Overview

Multi-configuration Excel exports now combine all configurations into a **single sheet** with separate columns for each configuration, making side-by-side comparisons much easier.

---

## What Changed

### **Old Behavior (Separate Sheets)**
```
Excel File with 6 sheets:
├── Sheet 1: "copula categorical"
├── Sheet 2: "copula continuous"
├── Sheet 3: "research spec categorical"
├── Sheet 4: "research spec continuous"
├── Sheet 5: "research baseline categorical"
└── Sheet 6: "research baseline continuous"
```

**Problem:** Had to copy-paste between sheets to compare configurations

### **New Behavior (Single Sheet)**
```
Excel File with 1 sheet:
└── Sheet: "All Configurations"
    All configs side-by-side in the same sheet!
```

**Benefit:** Easy comparison with Excel formulas, sorting, filtering, and analysis

---

## Excel File Structure

### **Example: 6 Configurations Export**

**Sheet Name:** "All Configurations"

**Column Structure:**

```
| Agent_Number | Honesty_Humility | Allowance | ... | donation_default_Copula_Categorical | donation_default_Copula_Continuous | donation_default_Research_Spec_Categorical | ... |
|--------------|------------------|-----------|-----|-------------------------------------|------------------------------------|--------------------------------------------|-----|
|      1       |      3.45        |     3     | ... |               0.423                 |              0.398                 |                  0.456                     | ... |
|      2       |      2.87        |     5     | ... |               0.651                 |              0.689                 |                  0.623                     | ... |
|      3       |      4.12        |     2     | ... |               0.287                 |              0.251                 |                  0.312                     | ... |
```

**Column Groups:**

1. **Agent_Number** (1 column)
   - Agent identifier: 1, 2, 3, ..., N

2. **Trait Columns** (5 columns, shared across all configs)
   - Honesty_Humility
   - Assigned Allowance Level
   - Study Program
   - Group_experiment
   - TWT+Sospeso [=AW2+AX2]{Periods 1+2}

3. **Decision Columns per Configuration** (varies by selected decisions)
   - Format: `{decision_name}_{Config_Name}`
   - Examples:
     - `donation_default_Copula_Categorical`
     - `donation_default_Copula_Continuous`
     - `donation_default_Research_Spec_Categorical`
     - `donation_default_Research_Spec_Continuous`
     - `donation_default_Research_Baseline_Categorical`
     - `donation_default_Research_Baseline_Continuous`

---

## Column Naming Convention

**Format:** `{decision_name}_{Configuration_Name}`

**Examples:**

| Configuration Key | Column Suffix | Example Column Name |
|-------------------|---------------|---------------------|
| `copula_categorical` | `Copula_Categorical` | `donation_default_Copula_Categorical` |
| `copula_continuous` | `Copula_Continuous` | `donation_default_Copula_Continuous` |
| `research_spec_categorical` | `Research_Spec_Categorical` | `donation_default_Research_Spec_Categorical` |
| `research_spec_continuous` | `Research_Spec_Continuous` | `donation_default_Research_Spec_Continuous` |
| `research_baseline_categorical` | `Research_Baseline_Categorical` | `donation_default_Research_Baseline_Categorical` |
| `research_baseline_continuous` | `Research_Baseline_Continuous` | `donation_default_Research_Baseline_Continuous` |

**Transformation:**
```python
config_key = "research_spec_continuous"
# Step 1: Replace underscores with spaces
"research spec continuous"
# Step 2: Title case
"Research Spec Continuous"
# Step 3: Replace spaces with underscores
"Research_Spec_Continuous"
```

---

## Benefits

### **1. Easy Side-by-Side Comparison ✅**

**Excel Formula Example:**
```excel
# Compare Agent 1's donation rate across all configs
=MAX(E1, F1, G1, H1, I1, J1)  # Find max donation across configs
=MIN(E1, F1, G1, H1, I1, J1)  # Find min donation across configs
=AVERAGE(E1:J1)                # Average across configs
```

### **2. Conditional Formatting ✅**

Highlight which configuration gives highest/lowest values:
```excel
# Select all donation columns, apply conditional formatting:
# Green = highest value per row
# Red = lowest value per row
```

### **3. Easy Sorting ✅**

```excel
# Sort by Copula Categorical donation, then by Research Spec Continuous
Data → Sort → Add Level
```

### **4. Quick Filtering ✅**

```excel
# Filter to agents where Copula > Research Spec
Filter: donation_default_Copula_Categorical > donation_default_Research_Spec_Categorical
```

### **5. Pivot Tables ✅**

```excel
# Create pivot table to compare averages
Rows: Study Program
Values: Average of all donation_default columns
```

### **6. Charts ✅**

```excel
# Create comparison chart for first 10 agents
Select columns: Agent_Number + all donation_default columns
Insert → Line Chart
```

---

## Comparison Examples

### **Example 1: Find Which Config Has Highest Donation**

Add a helper column:
```excel
# Column K: "Highest Config"
=INDEX({"Copula Cat","Copula Cont","Research Spec Cat","Research Spec Cont","Baseline Cat","Baseline Cont"},
       MATCH(MAX(E2:J2),E2:J2,0))
```

### **Example 2: Calculate Difference Between Configs**

```excel
# Column K: "Copula Cat vs Research Spec Cont"
=E2-H2

# Column L: "Percent Difference"
=(E2-H2)/E2
```

### **Example 3: Count Configs Above Threshold**

```excel
# Column K: "Configs Above 40%"
=COUNTIF(E2:J2,">0.4")
```

### **Example 4: Find Agent with Biggest Config Variance**

```excel
# Column K: "Variance Across Configs"
=STDEV(E2:J2)

# Then sort by Column K to find agents with most variation
```

---

## Python Analysis Examples

### **Read and Compare**

```python
import pandas as pd

# Read the single-sheet Excel file
df = pd.read_excel('enhanced_simulation_all_configs_20251007.xlsx', sheet_name='All Configurations')

# Get all donation_default columns
donation_cols = [col for col in df.columns if 'donation_default' in col]

print(f"Found {len(donation_cols)} configurations")
# Output: Found 6 configurations

# Compare mean donation rates across configs
for col in donation_cols:
    mean_val = df[col].mean()
    config_name = col.replace('donation_default_', '').replace('_', ' ')
    print(f"{config_name:30s}: {mean_val:.1%}")
```

### **Find Best Configuration Per Agent**

```python
# For each agent, find which config gives highest donation
donation_cols = [col for col in df.columns if 'donation_default' in col]

df['Best_Config'] = df[donation_cols].idxmax(axis=1)
df['Best_Config'] = df['Best_Config'].str.replace('donation_default_', '').str.replace('_', ' ')

print(df[['Agent_Number', 'Best_Config', 'Honesty_Humility']].head(10))
```

### **Statistical Comparison**

```python
# Compare distributions
donation_cols = [col for col in df.columns if 'donation_default' in col]

comparison = df[donation_cols].describe()
print(comparison)

# Output:
#        donation_default_Copula_Categorical  donation_default_Copula_Continuous  ...
# count                              500.000                         500.000      ...
# mean                                 0.423                           0.398      ...
# std                                  0.121                           0.105      ...
# min                                  0.000                           0.000      ...
# 25%                                  0.345                           0.321      ...
# 50%                                  0.405                           0.382      ...
# 75%                                  0.489                           0.467      ...
# max                                  0.980                           0.950      ...
```

### **Correlation Analysis**

```python
# How correlated are the configurations?
donation_cols = [col for col in df.columns if 'donation_default' in col]
correlation_matrix = df[donation_cols].corr()

print(correlation_matrix)

# Visualize
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Between Configurations')
plt.tight_layout()
plt.savefig('config_correlation.png')
```

### **Agent-Level Variance**

```python
# Which agents have the most variance across configs?
donation_cols = [col for col in df.columns if 'donation_default' in col]

df['Config_Variance'] = df[donation_cols].var(axis=1)
df['Config_Range'] = df[donation_cols].max(axis=1) - df[donation_cols].min(axis=1)

# Top 10 agents with most variance
top_variance = df.nlargest(10, 'Config_Variance')[['Agent_Number', 'Config_Variance', 'Config_Range', 'Honesty_Humility']]
print(top_variance)
```

---

## Use Cases

### **Use Case 1: Model Selection**

**Goal:** Choose the best configuration based on empirical data

**Workflow:**
1. Download Excel with all configurations
2. Add empirical data as a new column
3. Calculate error for each configuration: `=(E2-empirical_data)^2`
4. Sum squared errors for each config
5. Select config with lowest error

**Excel:**
```excel
# Column Z: "Empirical_Donation"
# (manually entered or imported)

# Column AA: "Error_Copula_Cat"
=POWER(E2-$Z2, 2)

# Row 502: "Sum of Squared Errors"
=SUM(AA2:AA501)

# Compare sums across all config error columns
```

### **Use Case 2: Sensitivity Analysis**

**Goal:** Understand how sensitive results are to population/income specification

**Workflow:**
1. Download Excel with all configurations
2. Calculate coefficient of variation per agent: `=STDEV(E2:J2)/AVERAGE(E2:J2)`
3. Sort by CV to find most/least sensitive agents
4. Analyze traits of high-sensitivity agents

### **Use Case 3: Documentation**

**Goal:** Create comparison table for research paper

**Workflow:**
1. Download Excel with all configurations
2. Calculate summary statistics for each config
3. Create formatted table with means, std devs, confidence intervals
4. Copy to Word/LaTeX

### **Use Case 4: Configuration Blending**

**Goal:** Create weighted average of multiple configs

**Workflow:**
1. Download Excel with all configurations
2. Create new column: `=0.5*E2 + 0.3*F2 + 0.2*G2`
3. Adjust weights based on model confidence
4. Use blended values for final analysis

---

## CSV vs Excel Trade-offs

### **CSV (ZIP of Separate Files)**

✅ **Pros:**
- Each config is a clean, independent file
- Easy to load in memory-constrained environments
- Can process configs in parallel
- Smaller individual file sizes

❌ **Cons:**
- Need to open multiple files to compare
- Manual merging required in code
- Harder to spot-check comparisons

### **Excel (Single Sheet)**

✅ **Pros:**
- All data in one place
- Easy Excel formulas for comparison
- Visual comparison without code
- Can add analysis columns alongside data
- Conditional formatting shows patterns

❌ **Cons:**
- Wider spreadsheet (horizontal scrolling)
- More columns to manage
- Single file can be larger

---

## Technical Details

### **Implementation**

```python
# Start with traits from first config
combined_df = first_config_df[trait_columns].copy()
combined_df.insert(0, 'Agent_Number', range(1, len(combined_df) + 1))

# Add decision columns from each config with suffix
for config_key, config_df in results_dict.items():
    decision_cols = [col for col in config_df.columns if col not in trait_columns]
    
    for col in decision_cols:
        config_suffix = config_key.replace('_', ' ').title().replace(' ', '_')
        new_col_name = f"{col}_{config_suffix}"
        combined_df[new_col_name] = config_df[col].values

# Export as single sheet
combined_df.to_excel(writer, index=False, sheet_name='All Configurations')
```

### **Column Count**

**Formula:** `Total Columns = 1 (Agent_Number) + 5 (Traits) + (# Decisions × # Configs)`

**Example:**
- 1 decision (`donation_default`)
- 6 configurations
- Total: 1 + 5 + (1 × 6) = **12 columns**

**Example with multiple decisions:**
- 3 decisions (`donation_default`, `disclose_income`, `disclose_documents`)
- 6 configurations
- Total: 1 + 5 + (3 × 6) = **24 columns**

### **Row Count**

Same as number of agents (e.g., 500 agents = 500 rows)

### **File Size Estimate**

- 500 agents × 12 columns × ~20 bytes/cell ≈ **120 KB**
- 500 agents × 24 columns × ~20 bytes/cell ≈ **240 KB**
- Plus Excel overhead ≈ **50-100 KB**

**Total:** Usually 200-400 KB for typical simulations

---

## Migration Guide

### **If You Were Using Multi-Sheet Excel**

**Old code:**
```python
# Read specific sheet
df_copula = pd.read_excel('file.xlsx', sheet_name='copula categorical')
df_research = pd.read_excel('file.xlsx', sheet_name='research spec continuous')

# Merge for comparison
merged = pd.merge(df_copula, df_research, 
                  left_index=True, right_index=True, 
                  suffixes=('_copula', '_research'))
```

**New code:**
```python
# Read single sheet - already merged!
df = pd.read_excel('file.xlsx', sheet_name='All Configurations')

# Columns are already named with suffixes
df['donation_default_Copula_Categorical']
df['donation_default_Research_Spec_Continuous']

# No merging needed!
```

---

## Summary

✅ **Changed:** Multi-config Excel exports now use single sheet with separate columns
✅ **Benefit:** Much easier side-by-side comparison
✅ **Column Format:** `{decision_name}_{Config_Name}`
✅ **Sheet Name:** "All Configurations"
✅ **CSV Export:** Still separate files in ZIP (unchanged)

**Key Takeaway:** All configurations are now in one Excel sheet, making comparisons as simple as looking left/right instead of switching between sheets!

---

*Implementation Date: 2025-10-07*
*Modified Files:*
- `app/pages/results/details.py`
- `MULTI_CONFIG_EXPORT_GUIDE.md`

