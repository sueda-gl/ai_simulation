# Rejected Transaction Defaults - Prioritized Options Implementation

## Summary

Updated the **rejected_transaction_defaults** decision to support **prioritized lists of options per agent** instead of a single option for all agents. Each agent can now have 1-5 options in priority order, with the rule that **Option 5 (forgo_transaction) must be last if included**.

---

## Key Changes

### 1. **Decision Module** (`src/decisions/rejected_transaction_defaults.py`)

**Before:**
- Returned a single option string (e.g., `"forgo_transaction"`)
- All agents had the same default behavior

**After:**
- Returns a **list of option strings** in priority order
- Each agent can have a different prioritized list (1-5 options)
- Example: `["current_vendor_pn", "higher_price_category", "forgo_transaction"]`
- **Rule enforced**: If `"forgo_transaction"` is in the list, it's automatically moved to the end

**Implementation:**
```python
def rejected_transaction_defaults(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """
    Decision 4: Select prioritized defaults for handling rejected transactions
    
    Returns:
        dict: {"rejected_transaction_defaults": list of option strings in priority order}
    """
    # Uses priority_template from config or generates random prioritized list per agent
    # Ensures "forgo_transaction" is last if included
```

---

### 2. **Default Values Structure** (`app/pages/decision_execution.py`)

**Changes:**
- Type changed from `"radio_selection"` to `"prioritized_selection"`
- Added `priority_template` field (replaces `default_option`)
- Added `description` field explaining the prioritization rule

**New Structure:**
```python
"rejected_transaction_defaults": {
    "type": "prioritized_selection",
    "priority_template": ["forgo_transaction"],  # Default: all agents use Option 5 only
    "options": [
        ("higher_price_category", "Option 1: ..."),
        ("lower_pn_vendor", "Option 2: ..."),
        ("current_vendor_pn", "Option 3: ..."),
        ("place_bid", "Option 4: ..."),
        ("forgo_transaction", "Option 5: Forgo the purchase request")
    ],
    "description": "Each agent gets a prioritized list. If Option 5 is included, it must be last."
}
```

**Handler Function:**
```python
# Added new handler for prioritized_selection type
elif isinstance(base_value, dict) and base_value.get("type") == "prioritized_selection":
    priority_key = f"{decision_name}_priority_template"
    priority_template = st.session_state.get(
        priority_key,
        base_value.get("priority_template", ["forgo_transaction"])
    )
    return priority_template
```

---

### 3. **Configuration UI** (`app/pages/decision_tabs/default_config.py`)

**New Function:** `render_prioritized_default_config()`

**Features:**
- ✅ Display current priority list with order numbers
- ✅ Remove options from the list (via ❌ buttons)
- ✅ Add new options from dropdown selector
- ✅ Automatic enforcement: Option 5 always goes to the end
- ✅ Validation indicator: Shows if Option 5 is correctly positioned
- ✅ Reset to default button
- ✅ Metrics showing list size and modifications

**UI Layout:**
```
┌─────────────────────────────────────┬──────────────────────┐
│ Current Priority List:              │ Metrics              │
│ 1. Option 3: Purchase from current  │ Options in List: 3   │
│    vendor at PN price         [❌]  │ Available to Add: 2  │
│ 2. Option 1: Purchase from another  │ ⚙️ Modified          │
│    (higher) price category    [❌]  │                      │
│ 3. Option 5: Forgo the purchase     │ Validation:          │
│    request                    [❌]  │ ✅ Option 5 is last  │
│                                     │                      │
│ Add Option to List:                 │                      │
│ [Dropdown: Choose an option]        │                      │
│ [➕ Add to List]                    │                      │
│                                     │                      │
│ [🔄 Reset to Default]               │                      │
└─────────────────────────────────────┴──────────────────────┘
```

**Session State Key:** `rejected_transaction_defaults_priority_template`

---

### 4. **Results Visualization** (`app/pages/results/visualizations/transaction_viz.py`)

**Major Changes:**

#### a) Information Banner
```python
st.info("ℹ️ **Note**: Each agent has a prioritized list of default options (1-5 options). 
        The list shows their order of preference when transactions are rejected.")
```

#### b) Metrics Row
```
┌──────────────┬───────────────┬────────────────────┬────────────────────────────┐
│ Total Agents │ Simulation    │ Configured Default │ Agents with Priority Lists │
│    500       │   Snapshot    │   Option 5         │         500                │
│              │               │ (hover for desc)   │                            │
└──────────────┴───────────────┴────────────────────┴────────────────────────────┘
```

**Key Change:** The "Configured Default" metric now shows **"Option 5"** with a tooltip showing what it means:
- Metric label: `"Option 5"`
- Tooltip (help text): `"Forgo the purchase request"` (or the actual last option's description)

#### c) Configuration Display
Shows the configured priority template as:
```
✅ Configured Priority Template:
   Option 3 → Option 1 → Option 5

Priority Order:
1. Option 3: Purchase from the current vendor at PN price
2. Option 1: Purchase from another (higher) price category of the same vendor
3. Option 5: Forgo the purchase request

⚠️ Rule: If Option 5 (forgo_transaction) is included, it must be last in the priority list
```

#### d) Statistics Panel
Shows:
- Distribution of priority list lengths (e.g., "3 option(s): 250 agents (50%)")
- Most common 1st choice among all agents

#### e) Download Instructions
```
📋 Download Agent Priority Lists
Each agent's prioritized default options are included in the Excel/CSV export. 
Look for the 'rejected_transaction_defaults' column to see each agent's priority list.
```

---

## How It Works

### Configuration Flow

1. **Page 2 → Overview Tab** (if decision is NOT selected for customization):
   - User sees prioritized selection UI
   - Can add/remove options and set priority order
   - System validates Option 5 position
   - Configuration stored in `st.session_state.rejected_transaction_defaults_priority_template`

2. **Simulation Execution**:
   - `get_actual_default_value()` retrieves the priority template
   - Template is passed to decision module via `simulation_config`
   - Each agent receives the priority list (currently all agents use same template)
   - Future enhancement: Could vary by agent characteristics

3. **Results Page**:
   - Displays configured priority template (read-only)
   - Shows "Option 5" in Configured Default metric with tooltip
   - Shows statistics about priority list distribution
   - Explains that priority lists are downloadable in exports

---

## Data Structure Examples

### In DataFrame (Results)

Each agent has a list in the `rejected_transaction_defaults` column:

```python
# Agent 1
["current_vendor_pn", "higher_price_category", "forgo_transaction"]

# Agent 2
["place_bid", "forgo_transaction"]

# Agent 3
["forgo_transaction"]  # Only Option 5
```

### In Excel/CSV Export

The `rejected_transaction_defaults` column will show the list as a string:

```
Agent_Number | rejected_transaction_defaults
------------ | --------------------------------------------------------------
1            | ['current_vendor_pn', 'higher_price_category', 'forgo_transaction']
2            | ['place_bid', 'forgo_transaction']
3            | ['forgo_transaction']
```

Users can parse these lists in Excel/Python to analyze priority patterns.

---

## Validation Rules

1. **Option 5 Position Rule**:
   - If `"forgo_transaction"` is in the list, it MUST be last
   - Configuration UI enforces this automatically
   - Decision module enforces this when generating random lists
   - Validation indicator in UI shows ✅ or ⚠️

2. **List Length**:
   - Minimum: 1 option
   - Maximum: 5 options (all available options)

3. **No Duplicates**:
   - Each option can only appear once in the priority list

---

## Files Modified

1. ✅ `src/decisions/rejected_transaction_defaults.py` - Decision module
2. ✅ `app/pages/decision_execution.py` - Default values and handler
3. ✅ `app/pages/decision_tabs/default_config.py` - Configuration UI
4. ✅ `app/pages/results/visualizations/transaction_viz.py` - Results visualization

---

## Future Enhancements

### Potential Improvements:

1. **Agent-Specific Variation**:
   - Allow priority lists to vary by agent characteristics
   - Example: High-income agents prioritize different options than low-income
   
2. **Priority Weights**:
   - Instead of strict ordering, add weights to each option
   - Example: `{"Option 3": 0.5, "Option 1": 0.3, "Option 5": 0.2}`

3. **Custom Tab Configuration**:
   - Allow custom decision tabs to set agent-specific priority rules
   - Could use trait-based logic to determine priorities

4. **Analytics Dashboard**:
   - Show heat maps of which agents use which priorities
   - Correlation analysis between traits and priority patterns

---

## Testing Checklist

- [x] Decision module generates valid priority lists
- [x] Configuration UI enforces Option 5 positioning
- [x] Results page displays priority lists correctly
- [x] Configured Default metric shows "Option 5" with tooltip
- [x] Excel export includes priority lists per agent (verified with test script)
- [x] Priority lists export as string representations in CSV/Excel
- [x] No linter errors in modified files

**Export Test Results:**
```
CSV format: "['current_vendor_pn', 'higher_price_category', 'forgo_transaction']"
Excel format: Same as CSV (string representation)
Successfully tested with pandas 2.x and openpyxl
```

---

## Session State Keys

- `rejected_transaction_defaults_priority_template` - Configured priority template (list)
- Legacy keys (deprecated):
  - `rejected_transaction_defaults_default_selection` - Old single-option key
  - `rejected_transaction_defaults_option` - Old post-simulation key

---

## Notes

- The implementation currently uses a **single template for all agents** but the infrastructure supports **per-agent variation**
- Priority lists are stored as **Python lists** in the DataFrame
- Excel/CSV exports show lists as **string representations**
- The "Configured Default" metric specifically shows **only Option 5's meaning** as requested
- All agents' priority lists are downloadable in the standard export files

