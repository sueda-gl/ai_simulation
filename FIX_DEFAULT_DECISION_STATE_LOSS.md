# Fix: Default Decision Parameter State Loss on Page Rerun

## Problem
When users change default decision parameters (e.g., probabilities, selections) and then rerun the simulation, the manually configured values get lost and reset to system defaults.

## Root Cause
The issue is caused by conditional rendering of Streamlit widgets combined with session state management:

1. **Conditional Widget Rendering**: Default decision parameter widgets are only rendered for unselected decisions in `app/pages/decision_tabs/default_config.py` (lines 19-33)

2. **Widget State Lifecycle**: When a Streamlit widget with a `key` parameter is not rendered during a script rerun, its value in session state can become stale or be overwritten by widget initialization

3. **State Reset Pattern**: The widgets initialize with `value=st.session_state[key]` AND use `key=key`, which can cause conflicts when:
   - The widget is not rendered (conditional rendering)
   - Session state is accessed before the widget is created
   - The page structure changes between reruns

## Affected Files
1. `/Users/suedagul/<sdg/app/pages/decision_tabs/default_config.py` - Widget definitions
2. `/Users/suedagul/<sdg/app/simulation.py` - `collect_decision_settings()` function (line 493)
3. `/Users/suedagul/<sdg/app/pages/page2_decisions.py` - Decision tab rendering

## Solution Options

### Option 1: Persistent State Initialization (Recommended)
Ensure all default decision parameters are initialized in session state at app startup and NEVER deleted during normal operation.

**Implementation:**
```python
# In app startup or initialization
def initialize_all_default_parameters():
    """Initialize all default decision parameters in session state"""
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES
    
    for decision_name, default_value in DEFAULT_DECISION_VALUES.items():
        if isinstance(default_value, dict):
            decision_type = default_value.get("type")
            
            if decision_type == "random_probability":
                key = f"{decision_name}_default_probability_y"
                if key not in st.session_state:
                    st.session_state[key] = default_value.get("probability_y", 0.5)
            
            elif decision_type == "checkbox_selection":
                key = f"{decision_name}_default_params"
                if key not in st.session_state:
                    st.session_state[key] = default_value.get("default_selection", [])
                
                # Initialize individual checkbox keys
                parameters = default_value.get("parameters", {})
                for param_key in parameters.keys():
                    checkbox_key = f"{decision_name}_default_param_{param_key}"
                    if checkbox_key not in st.session_state:
                        st.session_state[checkbox_key] = param_key in st.session_state[key]
            
            elif decision_type == "radio_selection":
                key = f"{decision_name}_default_selection"
                if key not in st.session_state:
                    st.session_state[key] = default_value.get("default_option", "")
        
        else:
            # Numeric or string value
            key = f"{decision_name}_default_value"
            if key not in st.session_state:
                st.session_state[key] = default_value
```

### Option 2: Use Callbacks Instead of Direct State Access
Instead of reading `value=st.session_state[key]`, let Streamlit manage the widget state entirely through the key parameter.

**Implementation:**
```python
# In default_config.py, remove the value= parameter when the key already exists
if prob_key not in st.session_state:
    st.session_state[prob_key] = default_probability

probability = st.slider(
    slider_label,
    min_value=0.0,
    max_value=1.0,
    # Remove: value=st.session_state[prob_key],  # Let Streamlit manage this
    step=0.01,
    help=slider_help,
    key=prob_key  # Streamlit automatically syncs to st.session_state[prob_key]
)
```

### Option 3: Store Parameters in Permanent Session State Key
Create a separate persistent storage for default parameters that's never affected by widget rendering.

**Implementation:**
```python
# Create a persistent storage key
if 'persistent_default_params' not in st.session_state:
    st.session_state.persistent_default_params = {}

# When widget changes, update both widget state AND persistent storage
def on_param_change(decision_name, param_key):
    widget_key = f"{decision_name}_default_{param_key}"
    if widget_key in st.session_state:
        if 'persistent_default_params' not in st.session_state:
            st.session_state.persistent_default_params = {}
        if decision_name not in st.session_state.persistent_default_params:
            st.session_state.persistent_default_params[decision_name] = {}
        st.session_state.persistent_default_params[decision_name][param_key] = st.session_state[widget_key]

# In collect_decision_settings(), read from persistent storage first
persistent_value = st.session_state.get('persistent_default_params', {}).get(decision_name, {}).get(param_key)
if persistent_value is not None:
    use persistent_value
else:
    fall back to widget keys
```

## Recommended Fix

**Implement Option 1** by adding initialization at app startup in the main app file.

### Step-by-Step Implementation:

1. **Add initialization function to `app/models.py`:**

```python
def initialize_default_decision_parameters():
    """Initialize all default decision parameters in session state on app startup"""
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES
    
    # Only initialize once per session
    if '_default_params_initialized' in st.session_state:
        return
    
    for decision_name, default_value in DEFAULT_DECISION_VALUES.items():
        if isinstance(default_value, dict):
            decision_type = default_value.get("type")
            
            if decision_type == "random_probability":
                key = f"{decision_name}_default_probability_y"
                if key not in st.session_state:
                    st.session_state[key] = default_value.get("probability_y", 0.5)
            
            elif decision_type == "checkbox_selection":
                key = f"{decision_name}_default_params"
                if key not in st.session_state:
                    st.session_state[key] = default_value.get("default_selection", []).copy()
                
                # Initialize individual checkbox keys
                parameters = default_value.get("parameters", {})
                default_selection = default_value.get("default_selection", [])
                for param_key in parameters.keys():
                    checkbox_key = f"{decision_name}_default_param_{param_key}"
                    if checkbox_key not in st.session_state:
                        st.session_state[checkbox_key] = param_key in default_selection
            
            elif decision_type == "radio_selection":
                key = f"{decision_name}_default_selection"
                if key not in st.session_state:
                    st.session_state[key] = default_value.get("default_option", "")
        
        else:
            # Numeric or string value
            key = f"{decision_name}_default_value"
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    # Mark as initialized
    st.session_state._default_params_initialized = True
```

2. **Call this function in the main app file near the top** (after imports, before page rendering):

```python
# In app_enhanced.py or main app file
from app.models import initialize_default_decision_parameters

# After session state initialization, before page routing
initialize_default_decision_parameters()
```

3. **CRITICAL: Remove the value= parameter from widgets in default_config.py** when they use a key:

```python
# BEFORE (problematic):
probability = st.slider(
    slider_label,
    min_value=0.0,
    max_value=1.0,
    value=st.session_state[prob_key],  # <-- REMOVE THIS
    step=0.01,
    key=prob_key
)

# AFTER (fixed):
# Initialize if needed (but don't read in value=)
if prob_key not in st.session_state:
    st.session_state[prob_key] = default_probability

probability = st.slider(
    slider_label,
    min_value=0.0,
    max_value=1.0,
    # value parameter removed - let Streamlit manage it via key
    step=0.01,
    key=prob_key
)
# Access the value via st.session_state[prob_key] after the widget
```

## Why This Works

1. **Session state keys are initialized once at app startup** - values persist across all reruns
2. **Widgets use ONLY the `key` parameter** - Streamlit automatically syncs between widget and session state
3. **No conditional dependency** - even if widgets aren't rendered, their session state values remain intact
4. **Clear single source of truth** - session state holds the values, widgets just provide UI

## Testing
After implementing the fix:

1. Navigate to Page 2 → Overview tab
2. Modify several default decision parameters (e.g., set disclose_income to 0.7, disclose_documents to 0.3)
3. Run simulation
4. Verify the configured values are used (check console output or debug expander)
5. Change another parameter
6. Run simulation again
7. Verify ALL previously configured values are still preserved

## Additional Notes

- The `reset_all_default_parameters()` function should PRESERVE this pattern by resetting to DEFAULT_DECISION_VALUES, not by deleting keys
- Consider adding a "Reset to Defaults" button per decision for granular control
- Monitor session state size if many parameters are added in the future

