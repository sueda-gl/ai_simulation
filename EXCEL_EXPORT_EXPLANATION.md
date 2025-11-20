# Excel Export Implementation Explanation

## 📁 File Location

**Main Export Function:**
- **File:** `app/pages/results/components/export_section.py`
- **Function:** `render_export_section(df, results_dict=None, using_selected_config=False)`
- **Lines:** 1-131

**Called From:**
- `app/pages/results/main_results.py` (line 325, 328, 331)
- Imported via `app/pages/results/details.py` (line 6)

---

## 🔄 Execution Flow

```
Results Page
    ↓
render_single_run_results()
    ↓ (line 305-331)
render_export_section(df, results_dict, using_selected_config)
    ↓
Create Excel file in BytesIO buffer
    ↓
Streamlit download button
    ↓
User downloads file
```

---

## 🧠 What the Code Does (Step-by-Step)

### **Phase 1: Data Cleaning (Lines 11-19)**

```python
columns_to_exclude = ['raw', 'index', 'consumption_frequency', 
                      'actual_allowance', 'income', 'customer_type', 
                      'enriched_requests_count']

# Remove these columns from df AND results_dict
```

**Why?** These columns are:
- Internal/non-display fields (`raw`, `index`, `enriched_requests_count`)
- Derived calculated fields (`consumption_frequency`)
- Intermediate calculations (`actual_allowance`, `income`)
- User-facing fields should show `customer_type` but it's hidden

---

### **Phase 2: Donation-Only Detection (Lines 26-40)**

```python
is_donation_only_run = (
    custom_decisions == ['donation_default'] AND
    len(default_decisions) == 0
)
```

**If TRUE:** Only keep columns with:
- `'donation'` in the column name
- Agent traits (5 columns)
- `agent_id`

**Why?** When testing donation decisions, other columns add noise.

---

### **Phase 3: Multi-Config Detection (Line 44)**

```python
export_all_configs = (
    results_dict is not None AND
    len(results_dict) > 1 AND
    not using_selected_config
)
```

**Decision Point:**
- **TRUE** → Export ALL configurations in ONE sheet (comparison mode)
- **FALSE** → Export single configuration (single run or selected config)

---

### **Phase 4A: Multi-Config Export (Lines 46-91)**

**When:** Comparison mode with multiple configs

**Process:**

#### **Step 1: Combine Configurations (Lines 47-61)**
```python
# Start with traits from first config
combined_df = first_config_df[trait_columns].copy()

# Add Agent ID
combined_df['Agent ID'] = first_config_df['agent_id'].values

# Add non-donation decisions from first config (don't duplicate)
for col in decision_cols_first:
    if 'donation_default' not in col:
        combined_df[col] = first_config_df[col].values
```

#### **Step 2: Add Donation Columns per Config (Lines 63-71)**
```python
for config_key, config_df in results_dict.items():
    for col in decision_cols:
        if 'donation_default' in col:
            # Create unique column name with config suffix
            config_suffix = config_key.replace('_', ' ').title().replace(' ', '_')
            new_col_name = f"{col}_{config_suffix}"
            combined_df[new_col_name] = config_df[col].values
            green_columns.append(new_col_name)
```

**Example Result:**
```
Agent ID | donation_default_Copula_Categorical | donation_default_Copula_Continuous
--------|-------------------------------------|-----------------------------------
   1    |             0.42                    |             0.38
   2    |             0.51                    |             0.47
```

#### **Step 3: Column Ordering (Lines 73-76)**
```python
cols = ['Agent ID'] + [col for col in combined_df.columns if col != 'Agent ID']
combined_df = combined_df[cols]
```

**Why?** Agent ID should be the first column.

#### **Step 4: Create Excel with Formatting (Lines 78-88)**
```python
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    combined_df.to_excel(writer, index=False, sheet_name='All Configurations')
    
    # Apply green highlighting to donation config columns
    worksheet = writer.sheets['All Configurations']
    green_fill = PatternFill(start_color='90EE90', ...)  # Light green
    
    for col_name in green_columns:
        col_idx = header_row.index(col_name) + 1
        # Fill entire column (header + data) with green
        for row_idx in range(1, len(combined_df) + 2):
            worksheet.cell(row=row_idx, column=col_idx).fill = green_fill
```

**Visual Result:**
```
| Agent ID | trait1 | trait2 | donation_default_Copula_Categorical | ... |
| (white)  |(white) |(white) |         (GREEN BACKGROUND)          | ... |
```

#### **Step 5: Create Download Button (Lines 90-92, 108-113)**
```python
excel_label = f"📊 Download Excel (All {len(results_dict)} Configs)"
excel_filename = f"enhanced_simulation_all_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

st.download_button(
    label=excel_label,
    data=buffer.getvalue(),
    file_name=excel_filename,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
```

---

### **Phase 4B: Single-Config Export (Lines 92-106)**

**When:** Single run or selected config

**Process:**

#### **Step 1: Simple Copy (Lines 93-96)**
```python
df_export = df.copy()

# Rename agent_id to 'Agent ID' for clarity
if 'agent_id' in df_export.columns:
    df_export = df_export.rename(columns={'agent_id': 'Agent ID'})
```

#### **Step 2: Reorder Columns (Lines 98-101)**
```python
if 'Agent ID' in df_export.columns:
    cols = ['Agent ID'] + [col for col in df_export.columns if col != 'Agent ID']
    df_export = df_export[cols]
```

#### **Step 3: Create Excel (Lines 103-104)**
```python
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    df_export.to_excel(writer, index=False, sheet_name='Results')
```

**No special formatting** - just clean data.

#### **Step 4: Create Download Button (Lines 105-106, 108-113)**
```python
excel_label = "📊 Download Excel"
excel_filename = f"enhanced_simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
```

---

### **Phase 5: Clear Results Button (Lines 117-130)**

```python
if st.button("🔄 Clear Results"):
    # Delete ALL session state
    for key in st.session_state.keys():
        del st.session_state[key]
    
    # Reinitialize with defaults
    initialize_session_state()
    
    # Stay on results page
    st.session_state.page = 'results'
    st.rerun()
```

**Why?** Reset the app to start fresh.

---

## 📊 Output File Structures

### **Single Configuration Export**

**File:** `enhanced_simulation_results_20251031_143022.xlsx`  
**Sheet:** `Results`

**Columns:**
```
Agent ID | Honesty_Humility | Assigned Allowance Level | Study Program | 
Group_experiment | TWT+Sospeso [...] | donation_default | disclose_income | 
disclose_documents | vendor_choice_weights | consumption_quantity | 
purchase_requests | vendor_selection | purchase_vs_bid | bid_value | 
rejected_transaction_option | rejected_bid_value | final_donation_rate
```

**Rows:** One per agent (e.g., 500 rows)

---

### **Multi-Configuration Export**

**File:** `enhanced_simulation_all_configs_20251031_143022.xlsx`  
**Sheet:** `All Configurations`

**Structure:**
```
Agent ID | [Traits...] | [Non-donation decisions once] | 
donation_default_Copula_Categorical | donation_default_Copula_Continuous | 
donation_default_Research_Spec_Categorical | donation_default_Research_Spec_Continuous | 
donation_default_Research_Baseline_Categorical | donation_default_Research_Baseline_Continuous
```

**Rows:** One per agent (same 500 agents across all configs)

**Formatting:**
- Donation columns: **Light green background (#90EE90)**
- Other columns: White background

**Why Green?** Highlights columns that vary across configurations for easy comparison.

---

## 🔑 Key Design Decisions

### **1. BytesIO Buffer (Line 43)**
```python
buffer = BytesIO()
```
**Why?** Excel is created in memory; Streamlit downloads bytes without touching disk.

### **2. openpyxl Engine (Line 78, 103)**
```python
pd.ExcelWriter(buffer, engine='openpyxl')
```
**Why?**
- Required for formatting (green cells)
- Avoids temporary `.xlsx` writes
- In-memory with BytesIO

### **3. Column Exclusion vs Column Selection**

**Exclusion approach:**
```python
df[[col for col in df.columns if not any(excl in col.lower() for excl in columns_to_exclude)]]
```

**Why?** Flexible—decision outputs can change.

### **4. Case-Insensitive Matching**
```python
if not any(excl in col.lower() for excl in columns_to_exclude)
```

**Why?** Handles variations like `raw_data`, `Raw_Data`, etc.

### **5. Timestamp in Filename**
```python
datetime.now().strftime('%Y%m%d_%H%M%S')
```

**Why?**
- Unique files across runs
- Chronological sorting
- Avoids overwrites

---

## 🎯 Special Features

### **Feature 1: Multi-Config Comparison**

**Scenario:** User runs "Compare all" + "Compare both income"

**Result:** 
- 6 configurations generated
- Merged into 1 Excel sheet
- Shared traits and decisions shown once
- Donation results in separate columns

**Use Case:** Compare population and income methods side-by-side.

---

### **Feature 2: Green Highlighting**

**Why Implemented?**
- Find per-config differences quickly
- Donation columns are variant
- Shared columns stay white

**Example:**
```
Agent ID | donation_Copula | donation_Research | vendor_selection
   1     |  0.42 [GREEN]  |   0.38 [GREEN]   |   3 [white]
   2     |  0.51 [GREEN]  |   0.47 [GREEN]   |   3 [white]
```

---

### **Feature 3: Donation-Only Mode**

**Scenario:** Testing `donation_default` alone

**Behavior:**
- Drop non-donation outputs
- Keep traits and donation fields

**Why?** 
- Smaller exports
- Focused testing
- Less noise

---

### **Feature 4: Selected Configuration Override**

**Scenario:** User picks a specific config from comparison

**Behavior:**
- Export only that config
- Ignore multi-config export
- No green highlighting

**Why?**
- Promotes one variant as the main choice
- Simpler export for that variant

---

## 🐛 Error Handling

### **Missing openpyxl (Lines 114-115)**
```python
except ImportError:
    st.caption("⚠️ Excel export requires openpyxl")
```

**Fallback:** Show a warning; no Excel button.

**Solution:** `pip install openpyxl`

---

### **Empty DataFrames (Line 64)**
```python
if not config_df.empty:
```

**Why?** Skips empty configs in multi-export without errors.

---

## 🔗 Dependencies

**Python Packages:**
- `pandas`
- `openpyxl`
- `streamlit`
- `datetime`
- `BytesIO`

**Session State:**
- `st.session_state.simulation_results`
- `st.session_state.custom_decisions`
- `st.session_state.default_decisions`
- `st.session_state.selected_donation_config` (optional)
- `st.session_state._using_selected_config` (optional)

---

## 📝 Summary

**Excel export in `/app/pages/results/components/export_section.py`:**

1. Clean data by removing internal fields
2. Detect single vs multi-config
3. Merge comparisons into one sheet with highlights
4. Format and rename columns
5. Write Excel via openpyxl
6. Expose a Streamlit download button

**Output:**
- One row per agent
- All decisions and traits
- Downloadable from the Results page






