# Parameter Persistence Investigation Report

## Issue Summary
When users navigate between pages (Page 1 ↔ Page 2), parameter values are being reset to defaults instead of persisting their changes. This affects:
1. Common Parameters (Page 1): e.g., number of vendors, periods, disclose income probability
2. Decision Parameters (Page 2): e.g., donation default settings
3. After running individual decisions and navigating back

## Root Cause Analysis

### 1. Widget Initialization Pattern
The application uses Streamlit widgets with a specific pattern that causes state loss:

#### Current Pattern (Problematic):
```python
# Example from page1_common_params.py (line 23-35)
if "page1_simulation_execution_mode" not in st.session_state:
    st.session_state.page1_simulation_execution_mode = "Snapshot" if st.session_state.sim_params.simulation_execution_mode == "snapshot" else "Live Simulation"

simulation_execution_mode = st.radio(
    "Execution Mode",
    ["Snapshot", "Live Simulation"],
    horizontal=True,
    key="page1_simulation_execution_mode"
)

# Sync the widget's state to sim_params
st.session_state.sim_params.simulation_execution_mode = "snapshot" if st.session_state.page1_simulation_execution_mode == "Snapshot" else "live"
```

**Problem**: The initialization block `if "page1_simulation_execution_mode" not in st.session_state` only runs ONCE per session. After that, the widget key exists in session state. However, the SYNC line (last line) runs on every page render, which means:
- When you change a value on Page 1, it gets stored in both the widget key AND synced to `sim_params`
- When you navigate to Page 2 and back, the widget reads from its key (which still has your value)
- BUT the initialization check doesn't run again (key already exists)
- **The problem is NOT with this pattern itself** - this pattern actually works

### 2. The Real Problem: Form Submit Behavior

Looking at the navigation code, I found the issue:

```python
# app/pages/navigation.py (lines 17-24)
def go_to_page1():
    restore_original_session_state()
    st.session_state.page = 'page1'

def go_to_page2():
    restore_original_session_state()
    st.session_state.page = 'page2'
```

These functions call `restore_original_session_state()` which may be overwriting values.

### 3. Specific Issue: Widget Keys Not Being Used Consistently

**Page 1 Example (number_input widgets)**:
```python
# Line 132-139 in page1_common_params.py
periods = st.number_input(
    "Number of Periods",
    min_value=1,
    max_value=100,
    value=st.session_state.sim_params.periods,  # ← PROBLEM: reads from object
    help="Number of periods for simulation run",
    key="periods_input",  # ← Widget key
    on_change=lambda: setattr(st.session_state.sim_params, 'periods', st.session_state.periods_input)  # ← Sync callback
)
```

**The Issue**:
1. The widget always initializes with `value=st.session_state.sim_params.periods` (the current value in the params object)
2. User changes it to a different value
3. The `on_change` callback updates `st.session_state.sim_params.periods`
4. User navigates to Page 2 (script reruns)
5. **When coming back to Page 1**, the widget reinitializes with `value=st.session_state.sim_params.periods`
6. BUT if something reset `st.session_state.sim_params.periods` to default (like object re-initialization or restore function), the widget shows the default

### 4. Investigation Findings

#### Files with Potential Issues:

1. **app/pages/page1_common_params.py**: 
   - Lines 67-76: `n_agents` input
   - Lines 132-139: `periods` input  
   - Lines 162-169: `num_vendors` input
   - Many other number_input and slider widgets follow same pattern

2. **app/pages/navigation.py**:
   - `restore_original_session_state()` function (lines 8-14) may be resetting values

3. **app/models.py**:
   - `SimulationParameters` class has default values (lines 16-236)
   - If this object gets re-instantiated, all values reset

4. **app/pages/decision_tabs/default_config.py**:
   - Line 156: Uses `.get()` with fallback to default
   - This is actually CORRECT, as it falls back only if key doesn't exist

## Verification Tests

To verify the issue, I need to check:

1. ✅ Are widget keys properly stored in session_state?
2. ✅ Are on_change callbacks working correctly?
3. ❓ Is `st.session_state.sim_params` being reset somewhere?
4. ❓ Does `restore_original_session_state()` affect these values?

## Recommended Solutions

### Solution 1: Fix Widget Value Initialization (RECOMMENDED)
Change widgets to read from their own key if it exists, falling back to the param object:

```python
# BEFORE (current)
periods = st.number_input(
    "Number of Periods",
    value=st.session_state.sim_params.periods,  # Always reads from object
    key="periods_input",
    on_change=lambda: setattr(st.session_state.sim_params, 'periods', st.session_state.periods_input)
)

# AFTER (fixed)
# Initialize widget key from params if not exists
if "periods_input" not in st.session_state:
    st.session_state.periods_input = st.session_state.sim_params.periods

periods = st.number_input(
    "Number of Periods",
    value=st.session_state.periods_input,  # Read from widget key (persists across reruns)
    key="periods_input",
    on_change=lambda: setattr(st.session_state.sim_params, 'periods', st.session_state.periods_input)
)
```

This ensures:
- Widget value persists in its own session_state key
- On first render, it initializes from the params object
- On subsequent renders, it keeps the user's value
- Changes sync back to params object via on_change

### Solution 2: Protect Session State During Navigation
Ensure `restore_original_session_state()` doesn't affect parameter values:

```python
def restore_original_session_state():
    """Restore original population and income mode values if they were overridden"""
    # Only restore specific keys, don't touch other parameters
    if hasattr(st.session_state, '_original_population_mode'):
        st.session_state.population_mode = st.session_state._original_population_mode
        st.session_state.income_spec_mode = st.session_state._original_income_spec_mode
        delattr(st.session_state, '_original_population_mode')
        delattr(st.session_state, '_original_income_spec_mode')
    # DO NOT reset sim_params or other parameter objects here
```

### Solution 3: Prevent sim_params Re-initialization
In `app/models.py`, ensure `initialize_session_state()` doesn't recreate the params object:

```python
def initialize_session_state():
    """Initialize all session state variables."""
    if 'page' not in st.session_state:
        st.session_state.page = 'page1'
    if 'sim_params' not in st.session_state:
        st.session_state.sim_params = SimulationParameters()
    # ^^^ This is correct - only creates if not exists
```

## Current Status

The application has a mix of patterns:
- ✅ Radio buttons: Use intermediate keys correctly (e.g., `page1_simulation_execution_mode`)
- ❌ Number inputs: Directly read from params object in `value=` parameter
- ❌ Sliders: Some directly read from params object
- ✅ Decision defaults: Use `.get()` with fallback correctly

## Next Steps

1. Apply Solution 1 to all widgets in Page 1 (common parameters)
2. Verify navigation.py doesn't reset parameters  
3. Test parameter persistence after changes
4. Apply same fix to Page 2 widgets if needed

