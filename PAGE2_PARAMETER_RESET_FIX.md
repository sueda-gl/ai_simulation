# Page 2 Parameter Reset Fix - Complete Solution

## Problem Summary

When navigating from **Results → Decision Parameters** page, all configured parameter values were being reset to defaults:
- **Donation decision parameters**: sigma_coefficient, anchor_weight, sigma_in_copula, etc.
- **Default decision configurations**: probability sliders (disclose_income, purchase_vs_bid), checkbox selections (vendor_choice_weights), etc.

This happened even though Page 1 parameters persisted correctly.

## Root Cause

**Page 2 had NO page-level initialization**, unlike Page 1:

```python
# Page 1 (Working)
def render_page1():
    initialize_widget_keys()  # ← Called every time page loads
    # ... render widgets ...

# Page 2 (Broken)
def render_page2():
    # NO initialization here!
    # ... render widgets ...
```

### Why This Caused the Problem

1. **Widget keys** (`tab_sigma_coefficient`, etc.) were only initialized when the specific **tab** rendered
2. **Non-prefixed variables** (`sigma_coefficient`, etc.) used by simulation were not synced
3. **Default decision keys** were not initialized at all
4. When navigating from Results → Page 2, you land on the **Overview tab**
5. Individual decision tabs (like donation_default) don't render until clicked
6. Widget keys never get initialized → values are lost

## Solution Implemented

### Created `initialize_page2_widget_keys()` Function

**Location**: `app/pages/page2_decisions.py` (lines 14-155)

This function now runs **every time Page 2 loads** and performs 3 critical tasks:

#### 1. Initialize Donation Widget Keys

Initializes all donation-specific widget keys that preserve UI state:
- `tab_sigma_in_copula`, `tab_sigma_in_research`
- `tab_sigma_coefficient`, `tab_sigma_coefficient_research`, `tab_sigma_coefficient_compare`
- `tab_anchor_weight`
- `page2_tab_income_spec_mode`

#### 2. Sync Non-Prefixed Variables from Widget Keys

**CRITICAL**: Syncs the actual variables that simulation reads:
```python
# Based on population_mode, sync appropriate widget key
if population_mode == "Copula (synthetic)":
    st.session_state.sigma_coefficient = st.session_state.tab_sigma_coefficient
    st.session_state.sigma_value_ui = 9.8995 * st.session_state.tab_sigma_coefficient
    st.session_state.sigma_in_copula = st.session_state.tab_sigma_in_copula
```

This ensures the simulation uses the user's configured values, not defaults.

#### 3. Initialize ALL Default Decision Parameter Keys

Initializes keys for all 13 decisions' default configurations:
- **Probability decisions** (disclose_income, purchase_vs_bid, disclose_documents):
  - `{decision}_default_probability_y`
- **Checkbox decisions** (vendor_choice_weights):
  - `{decision}_default_params`
  - `{decision}_default_param_{param_key}`
- **Radio decisions** (rejected_transaction_option):
  - `{decision}_default_selection`
- **Prioritized decisions** (rejected_transaction_defaults):
  - `{decision}_priority_template`
- **Numeric decisions** (donation_default, final_donation_rate):
  - `{decision}_default_value`

### Called at Page Load

Added function call at the start of `render_page2()` (line 231):
```python
def render_page2():
    """Render Page 2: Decision-Specific Parameters"""
    st.markdown('<h2 class="page-header">Page 2: Decision-Specific Parameters</h2>', unsafe_allow_html=True)
    
    # Initialize widget keys to preserve values across navigation
    # CRITICAL: This ensures donation parameters persist when navigating from Results back to Page 2
    initialize_page2_widget_keys()  # ← NEW!
    
    # ... rest of page rendering ...
```

## Files Modified

1. ✅ **app/pages/page2_decisions.py**
   - Added `initialize_page2_widget_keys()` function (lines 14-155)
   - Called function in `render_page2()` (line 231)

2. ✅ **app/pages/page1_common_params.py**
   - Fixed attribute error for distribution parameters
   - Changed direct attribute access to `getattr()` with defaults
   - Fixed: `lognormal_mu`, `lognormal_sigma`, `lognormal_min`
   - Fixed: `gg_k`, `gg_c`, `gg_lambda`, `gg_min`
   - Fixed: `dagum_a`, `dagum_p`, `dagum_b`, `dagum_min`

## Testing the Fix

### Test Scenario 1: Donation Decision Parameters
1. Go to **Page 2 → Donation Default tab**
2. Set `sigma_coefficient = 1.5`, `anchor_weight = 0.6`
3. Run **"Donation Default Only"**
4. View results
5. Click **"← Back to Decision Parameters"** (goes to Overview tab)
6. Click **"🎯 Run Complete Simulation"**
7. ✅ **Expected**: Simulation uses `sigma_coefficient = 1.5` and `anchor_weight = 0.6`

### Test Scenario 2: Default Decision Configurations
1. Go to **Page 2 → Overview tab**
2. Configure default decisions:
   - Set `disclose_income` probability to 75%
   - Set `purchase_vs_bid` probability to 60%
   - Select only "Price" and "Quality" for vendor weights
3. Run **"🎯 Run Complete Simulation"**
4. View results
5. Click **"← Back to Decision Parameters"**
6. ✅ **Expected**: All default decision configurations preserved (75%, 60%, Price+Quality)

### Test Scenario 3: Page 1 Parameters (Regression Test)
1. Go to **Page 1**
2. Set `Number of Vendors = 5`, `Number of Periods = 10`
3. Navigate to **Page 2**
4. Navigate back to **Page 1**
5. ✅ **Expected**: Values still 5 and 10 (this already worked)

## Technical Details

### Why Both Widget Keys AND Non-Prefixed Variables?

1. **Widget keys** (`tab_*`): Store UI state, persist across reruns
2. **Non-prefixed variables** (`sigma_coefficient`): What simulation actually reads

The sync step is critical because:
- Widgets read/write widget keys
- Simulation reads non-prefixed variables
- Without sync, UI and simulation get out of sync

### Why Initialize Default Decision Keys?

Default decision widgets are **conditionally rendered** (only when decisions are unselected). Without page-level initialization:
- Keys only exist if the Overview tab with unselected decisions has rendered
- Navigating directly from Results → Page 2 might not render these widgets
- Keys get lost, values reset

### Pattern Matching Page 1

This fix brings Page 2 in line with Page 1's already-working pattern:
```python
# Page 1 Pattern
def render_page1():
    initialize_widget_keys()  # ← Page-level init
    # ... widgets use initialized keys ...

# Page 2 Pattern (NOW)
def render_page2():
    initialize_page2_widget_keys()  # ← Page-level init (NEW!)
    # ... widgets use initialized keys ...
```

## Status

✅ **COMPLETE** - All parameter persistence issues on Page 2 resolved.

**Date**: {{ current_date }}
**Issue**: Parameter reset when navigating Results → Decision Parameters
**Resolution**: Added page-level initialization to Page 2, matching Page 1 pattern


