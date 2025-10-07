# Donation Default Decision & Excel Export Flow

## Overview
This document traces the complete flow of the donation default decision from implementation through to Excel file export.

---

## 1. Donation Default Decision Implementation

### 1.1 Decision Module Files

There are **two versions** of the donation default decision:

#### **A. `src/decisions/donation_default.py`** (Main version)
- Used by Orchestrator (copula mode) and OrchestratorBaseline
- Implements the 6-step process:
  1. Compute predicted prosocial from regression
  2. Scale both observed and predicted to 0-100
  3. Compute anchor (weighted average of observed + predicted)
  4. Optionally add stochastic component: Draw from Normal(anchor, sigma)
  5. Floor negative values at 0
  6. Rescale to [0,1] range

**Key Features:**
- Supports both categorical and continuous income modes
- Conditional stochastic component based on `pop_context` parameter
- Can use simple default values when decision is unselected
- Returns `donation_default` (final rate), optionally `donation_default_raw` and `donation_default_raw_pos`

**Stochastic Logic:**
```python
use_stochastic = (
    (params['stochastic'].get('in_copula', False) and pop_context == 'copula') or
    pop_context == 'documentation'
)
```

#### **B. `src/decisions/donation_default_stochastic.py`** (Documentation mode)
- Nearly identical to the main version
- Used specifically for documentation mode (research specification)
- Same 6-step process with slightly different stochastic handling

### 1.2 Decision Parameters

From `config/simulation.yaml`, the decision uses these parameters:

**Regression Coefficients:**
- `intercept`: Base regression intercept
- `beta_hh`: Honesty-Humility coefficient
- `beta_group`: Group experiment effects (MidSub, NoSub, FullSub)
- `beta_income_q`: Income quintile effects (Q1-Q5) - for categorical mode
- `beta_income_linear`: Linear income effect - for continuous mode
- `beta_study`: Study program effects (Incoming, Law5yr, UG3yr, Grad2yr)

**Anchor Weights:**
- `observed`: Weight for observed prosocial behavior (default: 0.75)
- `predicted`: Weight for predicted prosocial behavior (default: 0.25)

**Stochastic Parameters:**
- `sigma_value`: Standard deviation for normal draw (default: 9.8995)
- `sigma_strategy`: Strategy for sigma calculation
- `in_copula`: Whether to use stochastic in copula mode
- `raw_output`: Whether to include raw (pre-truncation) values

**Adjustment Parameters:**
- `shift_value`: Value to shift the anchor distribution (default: 0.0)

### 1.3 Decision Execution Flow

The decision is called by orchestrators during simulation:

```python
# From src/orchestrator.py (line 115-122)
if decision_name == 'donation_default':
    decision_output = self.decision_modules[decision_name](
        agent_state, params, agent_rng, 
        pop_context=self.pop_context, 
        simulation_config=self.simulation_config
    )
```

**Agent State Inputs:**
- `Honesty_Humility`: Trait score
- `Assigned Allowance Level`: Income level (1-5)
- `Study Program`: Study program category
- `Group_experiment`: Experimental group
- `TWT+Sospeso [=AW2+AX2]{Periods 1+2}`: Observed prosocial behavior

**Decision Output:**
Returns a dictionary with:
- `donation_default`: Final donation rate (0-1 scale)
- `donation_default_raw` (optional): Raw draw before truncation (can be negative)
- `donation_default_raw_pos` (optional): Non-negative draw on 0-100 scale

---

## 2. Simulation Results Assembly

### 2.1 Orchestrator Execution

All orchestrators follow a similar pattern:

1. **Load/Sample Agents** (lines 74-108 in orchestrator.py)
   - Copula mode: Sample from trait engine
   - Documentation mode: Load original 280 participants
   - Baseline mode: Load original 280 participants
   - DepVar mode: Resample only donation_default values

2. **Process Each Agent** (lines 99-129 in orchestrator.py)
   ```python
   for idx, row in agents_df.iterrows():
       agent_state = row.to_dict()  # Start with traits
       agent_rng = np.random.default_rng(rng_global.integers(1e9))
       
       # Execute decisions in order
       for decision_name in decisions_to_run:
           decision_output = self.decision_modules[decision_name](
               agent_state, params, agent_rng, ...
           )
           agent_state.update(decision_output)  # Add decision results
       
       results.append(agent_state)
   ```

3. **Return DataFrame** (line 131 in orchestrator.py)
   ```python
   return pd.DataFrame(results)
   ```

### 2.2 Results Dictionary Structure

From `app/simulation.py` (lines 366-396):

The simulation creates a **results dictionary** where:
- **Keys**: Mode identifiers (e.g., "copula_categorical", "research_spec_continuous")
- **Values**: DataFrames with agent traits + decision outputs

**Example Structure:**
```python
results = {
    "copula_categorical": DataFrame with 500 rows × (5 traits + N decisions),
    "copula_continuous": DataFrame with 500 rows × (5 traits + N decisions),
    ...
}
```

**DataFrame Columns:**
- Trait columns (5): `Honesty_Humility`, `Assigned Allowance Level`, `Study Program`, `Group_experiment`, `TWT+Sospeso [=AW2+AX2]{Periods 1+2}`
- Decision columns (varies): `donation_default`, `donation_default_raw`, `donation_default_raw_pos`, etc.

### 2.3 Storing Results

From `app/simulation.py` (line 418):
```python
st.session_state.simulation_results = results
```

This stores the results dictionary in Streamlit's session state, making it available to the Results page.

---

## 3. Excel File Download Implementation

### 3.1 Export Function Location

**File:** `app/pages/results/details.py`
**Function:** `render_export_section(df)` (lines 144-220)

### 3.2 Excel Export Code

The Excel export happens in lines 198-215:

```python
with col2:
    # Excel export
    try:
        from io import BytesIO
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        
        excel_data = buffer.getvalue()
        st.download_button(
            label="📊 Download Excel",
            data=excel_data,
            file_name=f"enhanced_simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except ImportError:
        st.caption("⚠️ Excel export requires openpyxl")
        st.caption("Install with: pip install openpyxl")
```

### 3.3 Where is `render_export_section` Called?

**File:** `app/pages/results/main_results.py`
**Function:** `render_single_run_results()` (line 338)

```python
# Raw data download
if not df.empty:
    render_export_section(df)
```

### 3.4 Which DataFrame is Exported?

From `app/pages/results/main_results.py` (lines 321-330):

The exported DataFrame is selected based on the comparison mode:

```python
if st.session_state.population_mode == "Compare all":
    if st.session_state.income_spec_mode == "Compare both":
        df = next((results_dict[k] for k in ["copula_categorical", "research_spec_categorical", ...] if k in results_dict), pd.DataFrame())
    else:
        income_type = "continuous" if st.session_state.income_spec_mode == "continuous only" else "categorical"
        df = next((results_dict[k] for k in [f"copula_{income_type}", ...] if k in results_dict), pd.DataFrame())
elif st.session_state.income_spec_mode == "Compare both":
    df = next((results_dict[k] for k in ["categorical", "continuous"] if k in results_dict), pd.DataFrame())
else:
    df = next(iter(results_dict.values()))  # First DataFrame in results_dict
```

**Important:** In comparison modes, only **ONE** DataFrame is selected for export (the first matching one). All other modes are not exported.

---

## 4. Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER CONFIGURES SIMULATION                                   │
│    - Selects population mode (copula/research/baseline/depvar)  │
│    - Selects income spec mode (categorical/continuous/both)     │
│    - Configures donation_default parameters                     │
│    - Sets number of agents, seed, etc.                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. STREAMLIT TRIGGERS SIMULATION                                │
│    File: app/simulation.py                                      │
│    Function: run_simulation_from_sidebar()                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. ORCHESTRATOR INITIALIZATION                                  │
│    Files: src/orchestrator*.py                                  │
│    - Copula: Orchestrator()                                     │
│    - Research Spec: OrchestratorDocMode()                       │
│    - Research Baseline: OrchestratorBaseline()                  │
│    - Dep Var: OrchestratorDepVar()                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. AGENT GENERATION                                             │
│    For each agent:                                              │
│    - Copula: Sample from TraitEngine                           │
│    - Research/Baseline: Load from original 280 participants     │
│    - DepVar: Bootstrap from empirical distribution             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. DECISION EXECUTION (For each agent)                          │
│    File: src/decisions/donation_default.py                      │
│    Function: donation_default(agent_state, params, rng, ...)   │
│                                                                 │
│    Input: agent_state dict with traits                          │
│    ├─ Honesty_Humility                                          │
│    ├─ Assigned Allowance Level                                  │
│    ├─ Study Program                                             │
│    ├─ Group_experiment                                          │
│    └─ TWT+Sospeso [=AW2+AX2]{Periods 1+2}                      │
│                                                                 │
│    Process:                                                     │
│    1. Compute predicted prosocial (regression)                  │
│    2. Scale observed & predicted to 0-100                       │
│    3. Compute anchor (weighted average)                         │
│    4. Optional: Add stochastic Normal(anchor, σ)               │
│    5. Floor at 0                                                │
│    6. Rescale to [0, 1]                                         │
│                                                                 │
│    Output: dict with decision results                           │
│    ├─ donation_default: final rate [0-1]                        │
│    ├─ donation_default_raw: raw draw (optional, can be < 0)    │
│    └─ donation_default_raw_pos: floored draw on 0-100 (optional)│
│                                                                 │
│    agent_state.update(decision_output)  # Merge into state     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. RESULTS ASSEMBLY                                             │
│    File: src/orchestrator.py                                    │
│    Line: return pd.DataFrame(results)                           │
│                                                                 │
│    DataFrame structure:                                         │
│    Columns = [Trait1, Trait2, ..., Decision1, Decision2, ...]  │
│    Rows = One per agent                                         │
│                                                                 │
│    Example:                                                     │
│    | Honesty_Humility | Allowance | ... | donation_default |   │
│    |      3.45        |     3     | ... |      0.423       |   │
│    |      2.87        |     5     | ... |      0.651       |   │
│    |       ...        |    ...    | ... |       ...        |   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. RESULTS STORAGE                                              │
│    File: app/simulation.py                                      │
│    Line: st.session_state.simulation_results = results          │
│                                                                 │
│    Storage structure (dict):                                    │
│    {                                                            │
│      "copula_categorical": DataFrame,                           │
│      "copula_continuous": DataFrame,                            │
│      "research_spec_categorical": DataFrame,                    │
│      ...                                                        │
│    }                                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8. RESULTS PAGE DISPLAY                                         │
│    File: app/pages/results/main_results.py                      │
│    Function: render_single_run_results()                        │
│                                                                 │
│    - Shows decision results                                     │
│    - Shows comparison grids (if comparison mode)                │
│    - Shows simulation overview                                  │
│    - Shows individual agent details                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 9. SELECT DATAFRAME FOR EXPORT                                  │
│    File: app/pages/results/main_results.py                      │
│    Lines: 321-330                                               │
│                                                                 │
│    Logic:                                                       │
│    - If Compare all: Select first matching mode DataFrame      │
│    - If Compare both income: Select first income mode          │
│    - Else: Select first DataFrame from results_dict            │
│                                                                 │
│    ⚠️ IMPORTANT: Only ONE DataFrame is selected!               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 10. EXCEL EXPORT                                                │
│    File: app/pages/results/details.py                           │
│    Function: render_export_section(df)                          │
│    Lines: 198-215                                               │
│                                                                 │
│    Process:                                                     │
│    1. Create BytesIO buffer                                     │
│    2. Use pd.ExcelWriter with openpyxl engine                  │
│    3. Write df to buffer (sheet: "Results")                     │
│    4. Get buffer value as bytes                                 │
│    5. Create Streamlit download button                          │
│                                                                 │
│    Output file:                                                 │
│    enhanced_simulation_results_YYYYMMDD_HHMMSS.xlsx            │
│                                                                 │
│    Excel structure:                                             │
│    Sheet "Results": All columns from selected DataFrame         │
│    - Trait columns (5)                                          │
│    - Decision columns (varies)                                  │
│    - donation_default column (final rate 0-1)                   │
│    - Optional: donation_default_raw, donation_default_raw_pos   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Excel File Contents

### 5.1 What's in the Excel File?

The Excel file contains **exactly one sheet** named "Results" with:

**Columns:**
1. **Trait columns** (5 columns, always present):
   - `Honesty_Humility`: Float value (e.g., 3.45)
   - `Assigned Allowance Level`: Integer 1-5
   - `Study Program`: String (e.g., "CLEAM", "BESS")
   - `Group_experiment`: String ("NoSub", "MidSub", "FullSub", "HighSub")
   - `TWT+Sospeso [=AW2+AX2]{Periods 1+2}`: Float (observed prosocial behavior)

2. **Decision columns** (varies based on selected decisions):
   - `donation_default`: **PRIMARY COLUMN** - Final donation rate [0-1]
   - `donation_default_raw`: (Optional) Raw draw before truncation (can be negative)
   - `donation_default_raw_pos`: (Optional) Non-negative draw on 0-100 scale
   - Other decision columns if multiple decisions were selected

**Rows:**
- One row per agent (e.g., 500 rows for 500 agents)
- No index column

### 5.2 Understanding the donation_default Values

The `donation_default` column values are:
- **Scale:** 0.0 to 1.0 (proportion)
- **Interpretation:** % of income to donate
- **Example:** 0.423 means 42.3% donation rate

**Important Notes:**
1. ✅ **`donation_default`** is the **final, processed value** - use this!
2. ⚠️ `donation_default_raw` is pre-truncation (can be negative) - for debugging only
3. ⚠️ `donation_default_raw_pos` is on 0-100 scale - for debugging only

### 5.3 Which Simulation Configuration?

In **comparison modes** (Compare all, Compare both income), the Excel export contains:
- **Only ONE of the comparison configurations**
- The first matching DataFrame from the results dictionary
- Other configurations are **NOT included** in the export

To export a specific configuration:
1. Run simulation in single mode (not comparison mode)
2. OR select the configuration you want from the Results page
3. OR save results programmatically (see section 6)

---

## 6. Programmatic Access to Results

If you need to access all simulation results (not just the exported one):

```python
# In Streamlit app
results_dict = st.session_state.simulation_results

# results_dict is a dictionary:
# {
#   "copula_categorical": DataFrame,
#   "copula_continuous": DataFrame,
#   "research_spec_categorical": DataFrame,
#   ...
# }

# Access specific configuration
copula_cat_df = results_dict.get("copula_categorical")
research_spec_cont_df = results_dict.get("research_spec_continuous")

# Export all configurations
for mode_name, df in results_dict.items():
    df.to_excel(f"results_{mode_name}.xlsx", index=False)
```

---

## 7. Key Implementation Files Summary

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **Decision Implementation** | `src/decisions/donation_default.py` | 1-188 | Main donation decision logic |
| **Decision (Stochastic)** | `src/decisions/donation_default_stochastic.py` | 1-179 | Documentation mode version |
| **Orchestrator (Copula)** | `src/orchestrator.py` | 63-131 | Run simulation with copula sampling |
| **Orchestrator (Research)** | `src/orchestrator_doc_mode.py` | 71-150 | Run simulation with original participants |
| **Orchestrator (Baseline)** | `src/orchestrator_baseline.py` | 73-136 | Run simulation with no stochastic |
| **Orchestrator (DepVar)** | `src/orchestrator_depvar.py` | 114-157 | Run simulation with resampling |
| **Simulation Execution** | `app/simulation.py` | 188-441 | Coordinate simulation runs |
| **Results Display** | `app/pages/results/main_results.py` | 102-342 | Display results page |
| **Excel Export** | `app/pages/results/details.py` | 144-220 | Handle CSV/Excel export |

---

## 8. Troubleshooting

### Issue: Excel file doesn't contain all comparison modes

**Reason:** Only one DataFrame is selected for export in comparison modes.

**Solution:** 
- Run simulation in single mode, or
- Use programmatic access to export all modes separately

### Issue: donation_default values look wrong

**Check:**
1. Are you looking at `donation_default` (final) or `donation_default_raw` (pre-truncation)?
2. What are the session state parameters? (Check debug section on Results page)
3. What is the sigma value and stochastic settings?

### Issue: Excel export button missing or not working

**Check:**
1. Is `openpyxl` installed? (`pip install openpyxl`)
2. Is `df` empty? (Export section only shows if `not df.empty`)
3. Check browser console for errors

### Issue: Values differ between charts and Excel

**Reason:** Might be looking at different DataFrames or columns.

**Solution:**
- Check the "Export Mean" metric shown above download buttons
- Verify column names in the "Donation Columns in Export" expander
- Ensure you're comparing the same configuration

---

## 9. Summary

**Excel files come from:**
1. ✅ **User runs simulation** → Orchestrators generate DataFrames
2. ✅ **DataFrames stored** in `st.session_state.simulation_results`
3. ✅ **Results page displays** results and provides export UI
4. ✅ **User clicks "📊 Download Excel"** → Excel file generated on-the-fly
5. ✅ **File contains:** One sheet with all agent data (traits + decisions)

**Key Points:**
- Excel files are **generated dynamically** when you click download
- Data comes **directly from simulation results** DataFrame
- `donation_default` column is the **final donation rate** [0-1]
- Only **one DataFrame** is exported in comparison modes
- Requires **openpyxl** package to be installed

---

*Document created: 2025-10-07*
*Last updated: 2025-10-07*

