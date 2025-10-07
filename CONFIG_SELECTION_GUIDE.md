# Configuration Selection Guide

## What is Configuration Selection?

Configuration selection is a feature that allows you to **"lock in" a specific donation decision configuration** to use across multiple simulations. It's like creating a template from one successful simulation that you can reuse.

---

## Why Use Configuration Selection?

### **The Problem It Solves**

Imagine you run the `donation_default` decision with "Compare all" mode, which generates **6 different configurations**:
- Copula + Categorical Income
- Copula + Continuous Income  
- Research Spec + Categorical Income
- Research Spec + Continuous Income ← **You like this one!**
- Research Baseline + Categorical Income
- Research Baseline + Continuous Income

You find that "Research Spec + Continuous Income" produces the most realistic results. Now you want to run a **complete simulation** with all 13 decisions using those exact same donation settings.

**Without Configuration Selection:** You'd have to:
1. Manually note all the settings (coefficients, sigma, anchor weights, etc.)
2. Manually reconfigure everything on Page 1
3. Hope you didn't miss anything

**With Configuration Selection:**
1. Click "🎯 Use This Config" button
2. Run your complete simulation
3. Done! All settings are automatically applied

---

## How Does It Work?

### **Step 1: Run Individual Donation Decision**

1. **Page 1:** Configure your simulation settings
   - Set Population Mode: e.g., "Compare all"
   - Set Income Spec Mode: e.g., "Compare both"
   - Configure other parameters (sigma, anchor weights, coefficients)

2. **Page 2:** Select **ONLY** the `donation_default` decision
   - This is important! Configuration selection only works for individual donation runs

3. **Click "🚀 Run Simulation"**
   - App generates multiple configurations (e.g., 6 results if Compare all + Compare both)
   - Results stored in `st.session_state.simulation_results`

### **Step 2: View Results & Select Configuration**

4. **Results Page** shows all generated configurations:

   ```
   📊 Copula + Categorical Income
   Mean: 42.3% | Std Dev: 12.1% | Median: 40.5% | Agents: 500
   Population: Copula    Income: Categorical
   [🎯 Use This Config]
   
   📊 Research Spec + Continuous Income  
   Mean: 38.7% | Std Dev: 10.5% | Median: 37.2% | Agents: 500
   Population: Research Spec    Income: Continuous
   [🎯 Use This Config]  ← YOU CLICK THIS
   
   ... (other configurations)
   ```

5. **Click "🎯 Use This Config"** on your preferred configuration
   - Configuration is saved to `st.session_state.selected_donation_config`
   - Button changes to "✅ Selected"

### **Step 3: What Gets Saved?**

When you select a configuration, the app saves:

```python
st.session_state.selected_donation_config = {
    'result_key': 'research_spec_continuous',
    'population_mode': 'Research Specification',
    'income_spec_mode': 'continuous only',
    
    # All regression coefficients
    'coefficients': {
        'intercept': 1.22985660120368,
        'beta_hh': 0.634001208840808,
        'beta_group': {'MidSub': ..., 'NoSub': ..., 'FullSub': ...},
        'beta_income_q': {'Q1': ..., 'Q2': ..., ...},
        'beta_income_linear': 0.0256,
        'beta_study': {'Incoming': ..., 'Law5yr': ..., ...}
    },
    
    # Stochastic parameters
    'stochastic_params': {
        'stochastic': {
            'sigma_value': 9.8995,
            'sigma_in_copula': False,
            'sigma_in_research': True,
            'raw_output': False
        },
        'anchor_weights': {
            'observed': 0.75,
            'predicted': 0.25
        }
    },
    
    # Result metrics (for reference)
    'metrics': {
        'mean_donation': 0.387,
        'std_donation': 0.105,
        'median_donation': 0.372,
        ...
    },
    
    'selected_timestamp': datetime.now(),
    'total_agents': 500,
    'source': 'individual_donation_run'
}
```

**Note:** The actual donation values are NOT saved (too memory intensive). Only the **parameters** that produced those values.

### **Step 4: Run Complete Simulation**

6. **Go back to Page 2** - You'll see:
   ```
   🎯 Selected Donation Configuration
   ✅ Selected Configuration: 📊 Research Spec + Continuous Income
   Selected at 14:23:15 - Avg Donation: 38.7%
   ```

7. **Select all decisions** you want to run (or multiple decisions including donation_default)

8. **Click "🚀 Run Simulation"** from sidebar

9. **Behind the scenes** (`app/simulation.py` lines 310-323):
   ```python
   if hasattr(st.session_state, 'selected_donation_config'):
       config = st.session_state.selected_donation_config
       
       # Override session state to match selected config
       st.session_state.population_mode = config['population_mode']  # "Research Specification"
       st.session_state.income_spec_mode = config['income_spec_mode']  # "continuous only"
       
       # Apply all coefficients and stochastic params
       apply_selected_donation_config(orchestrator, pop_mode, inc_mode)
   ```

10. **Simulation runs** with:
    - Population mode: Research Specification (not "Compare all")
    - Income mode: Continuous only (not "Compare both")
    - All coefficients from selected config
    - All stochastic parameters from selected config
    - **Plus** all other decisions you selected

---

## Does the Selected Config Show in the Excel File?

### **Short Answer: YES, Indirectly**

The Excel file **doesn't have a metadata field** saying "this used Research Spec + Continuous", but:

✅ **The DATA itself reflects the configuration:**
- All `donation_default` values were computed using the selected configuration's parameters
- The trait distributions match the selected population mode
- The regression coefficients used are from the selected config

✅ **The Results Page shows it explicitly:**
When you view results after using a selected config, you see:
```
🎯 Results using selected donation configuration: Research Specification + continuous only
```

### **What's in the Excel File?**

The Excel file contains:

```
| Honesty_Humility | Allowance | Study_Program | ... | donation_default | other_decisions |
|------------------|-----------|---------------|-----|------------------|-----------------|
|      3.45        |     3     |    CLEAM      | ... |      0.387       |      ...        |
|      2.87        |     5     |    BESS       | ... |      0.651       |      ...        |
|      ...         |    ...    |      ...      | ... |       ...        |      ...        |
```

**The `donation_default` values** were computed using:
- ✅ Regression coefficients from selected config
- ✅ Sigma/anchor weights from selected config
- ✅ Population mode from selected config (affects trait distributions)
- ✅ Income mode from selected config (categorical vs continuous regression)

So while the Excel doesn't have a "config name" column, **the data IS from that configuration**.

### **How to Verify Which Config Was Used**

**Option 1: Check Simulation Parameters Summary** (on Results page)
```
📊 Simulation Parameters Summary
Time & Market: ...
Product & Pricing: ...
Income & Agents: 
  - Distribution: normal
  - Range: $5000 - $100000
  - Mode: continuous only  ← HERE
  - Agents: 500
```

**Option 2: Check Results Page Header**
```
🎯 Results using selected donation configuration: Research Specification + continuous only
```

**Option 3: Export Parameters to JSON** (feature to add)
Currently not implemented, but you could add a feature to export:
```json
{
  "simulation_metadata": {
    "selected_config": "Research Spec + Continuous",
    "population_mode": "Research Specification",
    "income_spec_mode": "continuous only",
    "coefficients": {...},
    "stochastic_params": {...}
  },
  "data_file": "enhanced_simulation_results_20251007_142315.xlsx"
}
```

---

## Configuration Selection Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: RUN INDIVIDUAL DONATION DECISION                         │
│                                                                   │
│ Page 1: Set "Compare all" + "Compare both"                       │
│ Page 2: Select ONLY "donation_default"                           │
│ Click: "🚀 Run Simulation"                                       │
│                                                                   │
│ Results: 6 configurations generated                               │
│   - copula_categorical                                            │
│   - copula_continuous                                             │
│   - research_spec_categorical                                     │
│   - research_spec_continuous  ← You like this!                   │
│   - research_baseline_categorical                                 │
│   - research_baseline_continuous                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: SELECT YOUR PREFERRED CONFIGURATION                      │
│                                                                   │
│ Results Page shows cards for each configuration:                 │
│                                                                   │
│ ┌───────────────────────────────────────────────────────────┐   │
│ │ 📊 Research Spec + Continuous Income                       │   │
│ │ Mean: 38.7% | Std Dev: 10.5% | Median: 37.2%             │   │
│ │ Population: Research Spec    Income: Continuous            │   │
│ │ [🎯 Use This Config]  ← CLICK HERE                        │   │
│ └───────────────────────────────────────────────────────────┘   │
│                                                                   │
│ Action: Saves to st.session_state.selected_donation_config       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: RUN COMPLETE SIMULATION                                  │
│                                                                   │
│ Page 2: Select ALL decisions (or multiple including donation)    │
│         You'll see:                                               │
│         ┌─────────────────────────────────────────────────────┐ │
│         │ 🎯 Selected Donation Configuration                   │ │
│         │ ✅ Research Spec + Continuous Income                 │ │
│         │ Selected at 14:23:15 - Avg Donation: 38.7%          │ │
│         └─────────────────────────────────────────────────────┘ │
│                                                                   │
│ Click: "🚀 Run Simulation"                                       │
│                                                                   │
│ Behind the scenes:                                                │
│   - Detects selected_donation_config exists                       │
│   - Overrides population_mode → "Research Specification"         │
│   - Overrides income_spec_mode → "continuous only"               │
│   - Applies all coefficients from selected config                │
│   - Applies all stochastic params from selected config           │
│   - Runs simulation with ALL selected decisions                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: VIEW RESULTS & EXPORT                                    │
│                                                                   │
│ Results Page shows:                                               │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 🎯 Results using selected donation configuration:           │ │
│ │    Research Specification + continuous only                  │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│ Results contain:                                                  │
│   - All trait columns                                             │
│   - donation_default (computed with selected config)             │
│   - All other selected decisions                                 │
│                                                                   │
│ Excel Export:                                                     │
│   📊 Download Excel → enhanced_simulation_results_[timestamp].xlsx│
│   Contains: Full DataFrame with all decisions                    │
│   donation_default values reflect the selected configuration     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Implementation Details

### **Where Configuration Selection Happens**

| Step | File | Function | Lines |
|------|------|----------|-------|
| Display selection UI | `app/components.py` | `render_inline_selection_button()` | 110-144 |
| Save configuration | `app/pages/decision_execution.py` | `save_selected_configuration()` | 439-470 |
| Check if selected | `app/pages/decision_execution.py` | `is_configuration_selected()` | 596-602 |
| Apply to simulation | `app/simulation.py` | `apply_selected_donation_config()` | 595-671 |
| Override modes | `app/simulation.py` | `run_simulation_from_sidebar()` | 310-323 |

### **What Gets Stored**

```python
# Stored in session state
st.session_state.selected_donation_config = {
    'result_key': str,              # 'research_spec_continuous'
    'population_mode': str,         # 'Research Specification'
    'income_spec_mode': str,        # 'continuous only'
    'coefficients': dict,           # All regression coefficients
    'stochastic_params': dict,      # Sigma, anchor weights, etc.
    'metrics': dict,                # Mean, std, median, etc.
    'selected_timestamp': datetime, # When selected
    'total_agents': int,            # Number of agents in original run
    'source': str                   # 'individual_donation_run'
}
```

### **How It Affects Subsequent Simulations**

When `selected_donation_config` exists in session state:

1. **Population mode is overridden** (line 322 in `simulation.py`):
   ```python
   st.session_state.population_mode = config['population_mode']
   ```

2. **Income mode is overridden** (line 323 in `simulation.py`):
   ```python
   st.session_state.income_spec_mode = config['income_spec_mode']
   ```

3. **Coefficients are applied** (lines 600-605 in `simulation.py`):
   ```python
   orchestrator.config['donation_default']['regression_coefficients'].update(
       config['coefficients']
   )
   ```

4. **Stochastic parameters are applied** (lines 615-621 in `simulation.py`):
   ```python
   orchestrator.config['donation_default']['stochastic'].update(
       stoch_params['stochastic']
   )
   orchestrator.config['donation_default']['anchor_weights'].update(
       stoch_params['anchor_weights']
   )
   ```

5. **Simulation runs** with single mode (not comparison mode):
   - Generates **1 result** instead of 2, 3, or 6
   - Uses the selected configuration's parameters

---

## Common Questions

### **Q1: Can I select multiple configurations?**

**A:** No, only one configuration can be selected at a time. Selecting a new one replaces the previous selection.

### **Q2: Do I have to use the selected configuration?**

**A:** No! To stop using it:
- Click "🗑️ Clear Selection" on the Results page, OR
- Run an individual donation decision again (automatically clears selection)

### **Q3: Can I modify the selected configuration?**

**A:** Not directly. But you can:
1. Clear the selection
2. Change parameters on Page 1
3. Run individual donation decision again
4. Select the new configuration

### **Q4: What if I select a config then change Page 1 settings?**

**A:** The selected configuration **overrides** Page 1 settings for donation_default. Your Page 1 changes will be ignored for the donation decision (but apply to other decisions).

### **Q5: Does configuration selection work for other decisions?**

**A:** No, currently it only works for `donation_default`. Other decisions don't have the same complexity of parameters.

### **Q6: Can I export the selected configuration?**

**A:** Not currently, but you can see it displayed on Page 2 after selection. Consider adding a "Export Config as JSON" feature.

### **Q7: Is the configuration persistent across sessions?**

**A:** No, it's stored in `st.session_state` which clears when you close the browser tab. To persist, you'd need to:
- Save to a file (JSON/YAML)
- Add to database
- Use Streamlit's session state caching

### **Q8: How do I know which config my Excel file came from?**

**A:** Check the Results page header before downloading. It shows:
```
🎯 Results using selected donation configuration: [Mode] + [Income]
```

Add a timestamp to your Excel filename to track when it was generated.

### **Q9: Can I use a selected config for Monte Carlo runs?**

**A:** This depends on your implementation. Check if `run_monte_carlo_study()` respects the selected configuration. If not, it might need to be added.

### **Q10: What happens if I delete the selected config from session state?**

**A:** The system falls back to using current Page 1 settings. No errors - it gracefully handles the missing config.

---

## Benefits of Configuration Selection

✅ **Consistency:** Use exact same parameters across multiple simulations
✅ **Reproducibility:** Can recreate results by selecting the same config
✅ **Efficiency:** No manual parameter copying
✅ **Clarity:** Clear indication of which config is being used
✅ **Flexibility:** Easy to switch between configurations
✅ **Safety:** Original parameters are preserved, not modified

---

## Limitations

⚠️ **Only for donation_default:** Other decisions don't support config selection
⚠️ **Not persistent:** Clears when browser tab closes
⚠️ **No multi-select:** Can only use one config at a time
⚠️ **No config library:** Can't save multiple configs for later
⚠️ **No export:** Can't export config as JSON/YAML (yet)
⚠️ **Overrides Page 1:** Can't mix selected config with custom Page 1 changes

---

## Summary

**Configuration selection** is a **workflow optimization tool** that lets you:
1. Run donation decision once with comparison modes
2. Pick your favorite configuration
3. Use it automatically in complete simulations

**Excel files** reflect the selected configuration through:
- The actual donation_default values (computed with selected params)
- The trait distributions (from selected population mode)
- The regression approach (from selected income mode)

**Usage tip:** Use this feature when you've found a good donation configuration and want to run multiple complete simulations without manually reconfiguring every time.

---

*Document created: 2025-10-07*
*Last updated: 2025-10-07*
