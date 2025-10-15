# Default Decisions - Quick Reference Guide

## 🎯 At a Glance

Your codebase has **5 different patterns** for implementing default decisions, used across **13 decisions** in a **5-layer architecture**.

---

## 🏗️ The 5 Layers (Top to Bottom)

| Layer | Location | Purpose |
|-------|----------|---------|
| 1️⃣ **UI Config** | `app/pages/decision_tabs/default_config.py` | User configures defaults before simulation |
| 2️⃣ **Registry** | `app/pages/decision_execution.py` | Hard-coded defaults + priority resolution |
| 3️⃣ **Sim Config** | `app/simulation.py` | Collect settings & build config dict |
| 4️⃣ **Orchestrator** | `src/orchestrator.py` | Execute decisions, pass config |
| 5️⃣ **Decision Modules** | `src/decisions/*.py` | Check config & implement logic |

---

## 📋 The 5 Default Patterns

### Pattern 1: Complex Model with Simple Default

**When to use**: Decision has expensive computation that should only run when selected

**Example**: `donation_default`, `donation_default_stochastic`

```python
def donation_default(agent_state, params, rng, simulation_config=None, **kwargs):
    # CHECK: Is this decision unselected?
    if simulation_config and 'default_decisions_list' in simulation_config:
        if 'donation_default' in simulation_config.get('default_decisions_list', []):
            default_config = simulation_config.get('default_decisions', {}).get('donation_default')
            if default_config:
                return {"donation_default": float(default_config.get('value', 0.1))}
    
    # EXPENSIVE: Run full complex model (regression, scaling, stochastic, etc.)
    # ... 180 lines of logic ...
    return {"donation_default": computed_value}
```

**Key Points**:
- ✅ Early exit for unselected decisions
- ✅ Checks `default_decisions_list` (list of unselected decisions)
- ✅ Returns simple numeric value (e.g., 0.10)
- ✅ Otherwise runs full model

---

### Pattern 2: Random Probability (Y/N, Purchase/Bid)

**When to use**: Decision is binary choice with configurable probability

**Example**: `disclose_income`, `disclose_documents`, `purchase_vs_bid`

```python
def disclose_income(agent_state, params, rng, simulation_config=None):
    # CHECK: Is probability configured?
    if simulation_config and 'random_decisions' in simulation_config:
        prob_config = simulation_config['random_decisions'].get('disclose_income')
        if prob_config and prob_config.get("type") == "random_probability":
            probability_y = prob_config.get("probability_y", 0.5)
            options = prob_config.get("options", ["Y", "N"])
            
            # Weighted random choice
            if rng.random() < probability_y:
                return {"disclose_income": options[0]}  # Y (70% if configured to 0.7)
            else:
                return {"disclose_income": options[1]}  # N (30%)
    
    # FALLBACK: 50/50 random
    return {"disclose_income": rng.choice(["Y", "N"])}
```

**Key Points**:
- ✅ Checks `simulation_config['random_decisions']`
- ✅ Uses configured probability (e.g., 70% Y, 30% N)
- ✅ Uses RNG for reproducibility
- ✅ Falls back to 50/50

**Registry Entry**:
```python
"disclose_income": {
    "type": "random_probability",
    "probability_y": 0.5,      # Default 50%
    "options": ["Y", "N"]
}
```

---

### Pattern 3: Radio Selection (Multiple Options)

**When to use**: Decision has multiple mutually exclusive options

**Example**: `rejected_transaction_defaults`, `rejected_transaction_option`

```python
def rejected_transaction_defaults(agent_state, params, rng, simulation_config=None):
    # CHECK: Is option configured?
    if simulation_config and 'default_decisions' in simulation_config:
        config = simulation_config['default_decisions'].get('rejected_transaction_defaults')
        if config and config.get("type") == "radio_selection":
            selected_option = config.get("selected_option", "forgo_transaction")
            return {"rejected_transaction_defaults": selected_option}
    
    # FALLBACK: Default option
    return {"rejected_transaction_defaults": "forgo_transaction"}
```

**Key Points**:
- ✅ Checks `simulation_config['default_decisions']`
- ✅ Returns string identifier (e.g., "higher_price_category")
- ✅ All agents get same option (deterministic)

**Registry Entry**:
```python
"rejected_transaction_defaults": {
    "type": "radio_selection",
    "default_option": "forgo_transaction",
    "options": [
        ("higher_price_category", "Option 1: Purchase from another (higher) price category"),
        ("lower_pn_vendor", "Option 2: Purchase from another vendor at lower PN price"),
        ("current_vendor_pn", "Option 3: Purchase from current vendor at PN price"),
        ("place_bid", "Option 4: Place a bid"),
        ("forgo_transaction", "Option 5: Forgo the purchase")
    ]
}
```

---

### Pattern 4: Checkbox Selection (Parameter Weights)

**When to use**: Decision involves selecting subset of parameters with weights

**Example**: `vendor_choice_weights`

```python
def vendor_choice_weights(agent_state, params, rng, simulation_config=None):
    # CHECK: Is this decision unselected with configured weights?
    if simulation_config and 'default_decisions_list' in simulation_config:
        if 'vendor_choice_weights' in simulation_config.get('default_decisions_list', []):
            vendor_config = simulation_config['default_decisions'].get('vendor_choice_weights')
            if vendor_config and vendor_config.get("type") == "checkbox_selection":
                weights = vendor_config.get("weights", {})
                return {"vendor_choice_weights": weights}
    
    # FALLBACK: Equal weights
    return {"vendor_choice_weights": {
        "price": 0.25,
        "quality": 0.25,
        "proximity": 0.25,
        "sustainability": 0.25
    }}
```

**Key Points**:
- ✅ UI calculates weights based on selected checkboxes
- ✅ Decision module retrieves pre-calculated weights
- ✅ Returns dict of weights
- ✅ Falls back to equal weights

**Registry Entry**:
```python
"vendor_choice_weights": {
    "type": "checkbox_selection",
    "default_selection": ["price", "quality", "proximity", "sustainability"],
    "parameters": {
        "price": {"name": "Price", "description": "Cost of the product/service"},
        "quality": {"name": "Quality", "description": "Quality rating and reviews"},
        "proximity": {"name": "Proximity", "description": "Distance and convenience"},
        "sustainability": {"name": "Sustainability", "description": "Environmental impact"}
    }
}
```

---

### Pattern 5: Computed/Placeholder (No Simple Default)

**When to use**: Decision must be computed per agent, no meaningful simple default

**Examples**: 
- `consumption_quantity` - Random within agent's consumption limit
- `consumption_frequency` - Calculated from quantity / duration
- `bid_value` - Random within bidding range

**Example 1: Calculated from Previous Decision** (`consumption_frequency`)

```python
def consumption_frequency(agent_state, params, rng, simulation_config=None, **kwargs):
    # Get value from previous decision
    consumption_quantity = agent_state.get('consumption_quantity', 0)
    
    # Get simulation parameters
    periods = _get_simulation_param(simulation_config, 'periods', 1)
    duration_hours = _get_simulation_param(simulation_config, 'duration_hours', 1.0)
    term_duration = float(periods * duration_hours)
    
    # Calculate
    if term_duration > 0:
        frequency = consumption_quantity / term_duration
    else:
        frequency = 0.0
    
    return {"consumption_frequency": float(frequency)}
```

**Example 2: Random within Agent-Specific Bounds** (`consumption_quantity`)

```python
def consumption_quantity(agent_state, params, rng, simulation_config=None, **kwargs):
    # 1. Get/generate income
    income = agent_state.get('income') or _sample_income(simulation_config, rng)
    
    # 2. Assign to income category (1-10)
    income_category = _assign_income_category(income, simulation_config)
    
    # 3. Get consumption limit for that category
    consumption_limits = simulation_config.get('consumption_limits', {})
    limit = consumption_limits.get(f"cat_{income_category}", fallback)
    
    # 4. Random within limit
    quantity = int(rng.integers(0, limit + 1)) if limit > 0 else 0
    
    # 5. Generate purchase requests
    purchase_requests = [...] if quantity > 0 else []
    
    return {
        "consumption_quantity": quantity,
        "purchase_requests": purchase_requests,
        "income_category": income_category,
        "income": income
    }
```

**Key Points**:
- ✅ No default checking needed
- ✅ Always computed per agent
- ✅ May depend on previous decisions
- ✅ May generate additional state

**Registry Entry**:
```python
"consumption_quantity": "RANDOM_WITHIN_LIMIT",
"consumption_frequency": "CALCULATED",
"bid_value": "RANDOM_WITHIN_RANGE"
```

---

## 🔑 Key Session State Keys

### Pre-Configured Defaults (Overview Tab)
Set BEFORE simulation runs:

| Pattern | Key Format | Example |
|---------|-----------|---------|
| Probability | `{decision}_default_probability_y` | `disclose_income_default_probability_y = 0.7` |
| Radio | `{decision}_default_selection` | `rejected_transaction_defaults_default_selection = "place_bid"` |
| Checkbox | `{decision}_default_params` | `vendor_choice_weights_default_params = ["price", "quality"]` |
| Numeric | `{decision}_default_value` | `donation_default_default_value = 0.15` |

### Post-Simulation Adjustments (Results Page)
Set AFTER simulation runs:

| Pattern | Key Format | Example |
|---------|-----------|---------|
| Probability | `{decision}_probability_y` | `disclose_income_probability_y = 0.8` |
| Radio | `{decision}_selection` or `{decision}_option` | `rejected_transaction_defaults_option = "place_bid"` |
| Checkbox | `{decision}_selection` | `vendor_choice_weights_selection = ["price"]` |
| Numeric | `{decision}_config` | `donation_default_config = 0.20` |

### Priority Order
```
1. Pre-configured (decision_name_default_*)  ← HIGHEST
2. Post-simulation (decision_name_*)         ← MEDIUM
3. Hard-coded (DEFAULT_DECISION_VALUES)      ← FALLBACK
```

---

## 📊 Decision Summary Table

| # | Decision | Pattern | Default Value | Config Key |
|---|----------|---------|---------------|------------|
| 1 | disclose_income | Probability | 50% Y / 50% N | `random_decisions` |
| 2 | disclose_documents | Probability | 50% Y / 50% N (if income < threshold) | `random_decisions` |
| 3 | donation_default | Complex Model | 0.10 (10%) | `default_decisions_list` |
| 4 | rejected_transaction_defaults | Radio | "forgo_transaction" | `default_decisions` |
| 5 | vendor_choice_weights | Checkbox | Equal weights (0.25 each) | `default_decisions` |
| 6 | consumption_quantity | Computed | Random within limit | N/A |
| 7 | consumption_frequency | Computed | quantity / duration | N/A |
| 8 | vendor_selection | Computed | Deterministic (highest score) | N/A |
| 9 | purchase_vs_bid | Probability | 50% Purchase / 50% Bid | `random_decisions` |
| 10 | bid_value | Computed | Random within range | N/A |
| 11 | rejected_transaction_option | Radio | "forgo_transaction" | `default_decisions` |
| 12 | rejected_bid_value | Placeholder | "NA" | N/A |
| 13 | final_donation_rate | Numeric | 0.10 (10%) | `default_decisions` |

---

## 🚀 Quick Start: Adding a New Decision with Defaults

### Step 1: Choose Your Pattern

Ask yourself:
- **Simple value when unselected, complex when selected?** → Pattern 1 (Complex Model)
- **Binary choice with probability?** → Pattern 2 (Random Probability)
- **Multiple mutually exclusive options?** → Pattern 3 (Radio Selection)
- **Parameter weights?** → Pattern 4 (Checkbox Selection)
- **Must compute per agent?** → Pattern 5 (Computed)

### Step 2: Add to Registry

Edit `app/pages/decision_execution.py`:

```python
DEFAULT_DECISION_VALUES = {
    # ... existing decisions ...
    
    "my_new_decision": {
        "type": "random_probability",  # or radio_selection, checkbox_selection, numeric
        "probability_y": 0.5,
        "options": ["Option1", "Option2"]
    }
}
```

### Step 3: Implement Decision Module

Create `src/decisions/my_new_decision.py`:

```python
def my_new_decision(agent_state, params, rng, simulation_config=None):
    """My new decision with default behavior"""
    
    # CHECK: Is default configured?
    if simulation_config and 'random_decisions' in simulation_config:
        prob_config = simulation_config['random_decisions'].get('my_new_decision')
        if prob_config and prob_config.get("type") == "random_probability":
            probability = prob_config.get("probability_y", 0.5)
            options = prob_config.get("options", ["Option1", "Option2"])
            
            if rng.random() < probability:
                return {"my_new_decision": options[0]}
            else:
                return {"my_new_decision": options[1]}
    
    # FALLBACK
    return {"my_new_decision": rng.choice(["Option1", "Option2"])}
```

### Step 4: Add to Orchestrator

Edit `src/orchestrator.py`:

```python
self.decision_order = [
    'disclose_income',
    'disclose_documents',
    'donation_default',
    # ... other decisions ...
    'my_new_decision',  # Add in appropriate position
    # ... more decisions ...
]
```

### Step 5: Test

1. Run simulation with decision **unselected** → Should use default
2. Run simulation with decision **selected** → Should use custom config
3. Configure default in Overview tab → Should use configured value

---

## 💡 Common Gotchas

### Gotcha 1: Config Location Confusion
```python
# ❌ WRONG: Checking wrong config location
if 'default_decisions_list' in simulation_config:  # This is for Pattern 1
    if prob_config.get("type") == "random_probability":  # This is Pattern 2
        # Mixing patterns!

# ✅ CORRECT: Use right config for your pattern
# Pattern 1 (Complex Model): checks 'default_decisions_list'
# Pattern 2-4: checks 'random_decisions' or 'default_decisions'
```

### Gotcha 2: RNG Consistency
```python
# ❌ WRONG: Using Python's random (not reproducible)
import random
choice = random.choice(["Y", "N"])

# ✅ CORRECT: Using provided RNG (reproducible)
choice = rng.choice(["Y", "N"])
```

### Gotcha 3: State Dependencies
```python
# ❌ WRONG: Assuming previous decision ran
consumption_quantity = agent_state['consumption_quantity']  # May not exist!

# ✅ CORRECT: Use get() with fallback
consumption_quantity = agent_state.get('consumption_quantity', 0)
```

### Gotcha 4: Early Exit Not Early Enough
```python
# ❌ WRONG: Expensive computation before check
expensive_result = complex_model(agent_state)  # Run before checking!
if decision_is_unselected:
    return default_value

# ✅ CORRECT: Check first, compute only if needed
if decision_is_unselected:
    return default_value
expensive_result = complex_model(agent_state)  # Only run if selected
```

---

## 🔍 Debugging Checklist

Problem: **Default not being used**

1. ✅ Check `DEFAULT_DECISION_VALUES` has entry
2. ✅ Check session state key exists (`{decision}_default_*`)
3. ✅ Check `collect_decision_settings()` includes decision
4. ✅ Check decision in `default_decisions_list` (Pattern 1) or config passed (Pattern 2-4)
5. ✅ Check decision module checks correct config location
6. ✅ Add debug prints in decision module

Problem: **Wrong default being used**

1. ✅ Check priority: Pre-config → Post-sim → Hard-coded
2. ✅ Check session state for competing keys
3. ✅ Check `get_actual_default_value()` logic
4. ✅ Clear session state and retry

Problem: **Default works in UI but not in simulation**

1. ✅ Check `collect_decision_settings()` is called
2. ✅ Check `simulation_config` passed to orchestrator
3. ✅ Check orchestrator passes config to decision module
4. ✅ Add debug prints to trace config flow

---

## 📚 Key Files Reference

| File | Lines to Read | What to Look For |
|------|--------------|------------------|
| `app/pages/decision_execution.py` | 184-243 | `DEFAULT_DECISION_VALUES` registry |
| `app/pages/decision_execution.py` | 263-399 | `get_actual_default_value()` priority logic |
| `app/simulation.py` | 483-600 | `collect_decision_settings()` config collection |
| `app/pages/decision_tabs/default_config.py` | 47-273 | UI renderers for each pattern |
| `src/orchestrator.py` | 63-131 | Decision execution loop |
| `src/decisions/donation_default.py` | 25-36 | Pattern 1 example (Complex Model) |
| `src/decisions/disclose_income.py` | 6-23 | Pattern 2 example (Random Probability) |
| `src/decisions/rejected_transaction_defaults.py` | 3-15 | Pattern 3 example (Radio Selection) |
| `src/decisions/vendor_choice_weights.py` | 6-26 | Pattern 4 example (Checkbox Selection) |
| `src/decisions/consumption_frequency.py` | 40-101 | Pattern 5 example (Computed) |

---

## 🎯 TL;DR

1. **5 Patterns**: Complex Model, Random Probability, Radio Selection, Checkbox Selection, Computed
2. **5 Layers**: UI Config → Registry → Sim Config → Orchestrator → Decision Modules
3. **3 Priorities**: Pre-config (highest) → Post-sim → Hard-coded (fallback)
4. **2 Config Locations**: `random_decisions` (Pattern 2), `default_decisions` (Pattern 3-4)
5. **1 Rule**: Always check for defaults at START of decision function (early exit!)

---

**End of Quick Reference**

