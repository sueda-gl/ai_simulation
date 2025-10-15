# Default Decisions Architecture - Complete Analysis

## 📋 Table of Contents
1. [Overview](#overview)
2. [Architecture Layers](#architecture-layers)
3. [Data Flow](#data-flow)
4. [Implementation by Decision Type](#implementation-by-decision-type)
5. [Code Locations](#code-locations)
6. [Examples & Patterns](#examples--patterns)

---

## Overview

Your codebase implements a **sophisticated multi-layer architecture** for handling default decision values. When a decision is **NOT selected for custom configuration**, it uses default values that can be:
- Pre-configured in the UI (before simulation)
- Adjusted after simulation (in Results)
- Computed dynamically per agent
- Hard-coded as system fallbacks

The architecture supports **13 different decisions** with various types of default behaviors.

---

## Architecture Layers

### Layer 1: UI Configuration (Streamlit Frontend)
**Location**: `app/pages/decision_tabs/default_config.py`

**Purpose**: Allow users to configure defaults BEFORE running simulations

**Components**:
```
render_default_decisions_config()
  ├── render_decision_default_config()
  │   ├── render_probability_default_config()      [Y/N, Purchase/Bid]
  │   ├── render_radio_default_config()            [Multiple options]
  │   ├── render_checkbox_default_config()         [Parameter weights]
  │   ├── render_numeric_default_config()          [Percentages, rates]
  │   └── render_placeholder_default_config()      [Computed values]
  └── reset_all_default_parameters()
```

**Session State Keys** (Pre-configured):
- `{decision_name}_default_probability_y` - For probability decisions
- `{decision_name}_default_selection` - For radio selections
- `{decision_name}_default_params` - For checkbox selections
- `{decision_name}_default_value` - For numeric values

---

### Layer 2: Default Value Registry
**Location**: `app/pages/decision_execution.py`

**Hard-coded Default Values** (`DEFAULT_DECISION_VALUES`):
```python
DEFAULT_DECISION_VALUES = {
    # Numeric defaults
    "donation_default": 0.10,              # 10%
    "final_donation_rate": 0.10,           # 10%
    
    # Probability defaults (Y/N, Purchase/Bid)
    "disclose_income": {
        "type": "random_probability",
        "probability_y": 0.5,               # 50% Y, 50% N
        "options": ["Y", "N"]
    },
    "disclose_documents": {
        "type": "random_probability",
        "probability_y": 0.5,
        "options": ["Y", "N"]
    },
    "purchase_vs_bid": {
        "type": "random_probability",
        "probability_y": 0.5,               # 50% Purchase Now, 50% bid
        "options": ["Purchase Now", "bid"]
    },
    
    # Radio selection defaults (multiple options)
    "rejected_transaction_defaults": {
        "type": "radio_selection",
        "default_option": "forgo_transaction",
        "options": [...]                    # 5 options
    },
    "rejected_transaction_option": {
        "type": "radio_selection",
        "default_option": "forgo_transaction",
        "options": [...]
    },
    
    # Checkbox selection defaults (parameter weights)
    "vendor_choice_weights": {
        "type": "checkbox_selection",
        "default_selection": ["price", "quality", "proximity", "sustainability"],
        "parameters": {...}
    },
    
    # Computed/placeholder defaults
    "consumption_quantity": "RANDOM_WITHIN_LIMIT",
    "consumption_frequency": "CALCULATED",
    "bid_value": "RANDOM_WITHIN_RANGE",
    "vendor_selection": "deterministic",
    "rejected_bid_value": "NA"
}
```

**Priority Resolution** (`get_actual_default_value()`):
```python
def get_actual_default_value(decision_name):
    """
    Priority order:
    1. Pre-configured default from Overview tab (decision_name_default_*)
    2. Post-simulation adjustment from Results page (decision_name_*)
    3. Hard-coded default from DEFAULT_DECISION_VALUES
    """
    # Check pre-config first
    pre_config_key = f"{decision_name}_default_probability_y"
    if pre_config_key in st.session_state:
        return st.session_state[pre_config_key]
    
    # Then post-simulation
    post_sim_key = f"{decision_name}_probability_y"
    if post_sim_key in st.session_state:
        return st.session_state[post_sim_key]
    
    # Finally hard-coded
    return DEFAULT_DECISION_VALUES.get(decision_name)
```

---

### Layer 3: Simulation Configuration
**Location**: `app/simulation.py`

**Configuration Collection** (`collect_decision_settings()`):
```python
def collect_decision_settings():
    """
    Collect current default decision settings from session state.
    Returns a dict that gets passed to orchestrator as simulation_config.
    """
    decision_settings = {}
    
    for decision_name, default_value in DEFAULT_DECISION_VALUES.items():
        if isinstance(default_value, dict):
            # Handle different types
            if default_value.get("type") == "random_probability":
                current_prob = st.session_state.get(
                    f"{decision_name}_probability_y",
                    st.session_state.get(
                        f"{decision_name}_default_probability_y",
                        default_value.get("probability_y", 0.5)
                    )
                )
                decision_settings[decision_name] = {
                    "probability_y": current_prob,
                    "options": default_value.get("options"),
                    "type": "random_probability"
                }
            # ... similar for other types
    
    return decision_settings
```

**Configuration Passing** (`run_simulation_from_sidebar()`):
```python
def run_simulation_from_sidebar():
    # Collect decision settings
    prob_settings = collect_decision_settings()
    
    # Pass to orchestrator via simulation_config
    orchestrator.simulation_config['random_decisions'] = prob_settings
    orchestrator.simulation_config['default_decisions'] = prob_settings
    
    # Also pass which decisions are custom vs default
    orchestrator.simulation_config['custom_decisions'] = st.session_state.custom_decisions
    orchestrator.simulation_config['default_decisions_list'] = st.session_state.default_decisions
```

---

### Layer 4: Orchestrator Execution
**Location**: `src/orchestrator.py`

**Decision Execution**:
```python
def run_simulation(self, n_agents, seed, single_decision=None):
    """Run simulation, passing simulation_config to all decision modules"""
    
    for decision_name in decisions_to_run:
        if decision_name in self.decision_modules:
            params = self.config.get(decision_name, {})
            
            # Execute decision module
            # Pass simulation_config containing default decision settings
            decision_output = self.decision_modules[decision_name](
                agent_state, 
                params, 
                agent_rng, 
                simulation_config=self.simulation_config  # <-- Defaults passed here
            )
            
            # Update agent state
            agent_state.update(decision_output)
```

---

### Layer 5: Decision Module Implementation
**Location**: `src/decisions/*.py`

Each decision module receives `simulation_config` and checks if it should use defaults.

---

## Implementation by Decision Type

### Type 1: Numeric Defaults with Complex Model
**Example**: `donation_default`

**Implementation**:
```python
def donation_default(agent_state, params, rng, simulation_config=None, **kwargs):
    """Complex model with default fallback"""
    
    # CHECK 1: Is this decision using simple default? (unselected)
    if simulation_config and 'default_decisions_list' in simulation_config:
        if 'donation_default' in simulation_config.get('default_decisions_list', []):
            # This decision is unselected - use configured default value
            default_config = simulation_config.get('default_decisions', {}).get('donation_default')
            if default_config:
                if isinstance(default_config, dict) and default_config.get('type') == 'numeric':
                    return {"donation_default": float(default_config.get('value', 0.1))}
                elif isinstance(default_config, (int, float)):
                    return {"donation_default": float(default_config)}
    
    # Decision is SELECTED - use full complex model
    # ... 184 lines of regression, scaling, stochastic components ...
    
    return {"donation_default": donation_rate}
```

**Key Features**:
- ✅ Checks if decision is in `default_decisions_list`
- ✅ If YES: Returns simple configured default value (0.10)
- ✅ If NO: Runs full 6-step regression model with copula/research spec
- ✅ Supports both dict and direct numeric formats

**Similar Decisions**:
- `donation_default_stochastic.py` (same pattern)
- Both have identical default checking logic

---

### Type 2: Random Probability Decisions
**Examples**: `disclose_income`, `disclose_documents`, `purchase_vs_bid`

**Implementation** (`disclose_income.py`):
```python
def disclose_income(agent_state, params, rng, simulation_config=None):
    """Random Y/N decision with configurable probability"""
    
    # CHECK: Is probability configured in simulation config?
    if simulation_config and 'random_decisions' in simulation_config:
        prob_config = simulation_config['random_decisions'].get('disclose_income')
        if prob_config and prob_config.get("type") == "random_probability":
            # Use configured probability
            probability_y = prob_config.get("probability_y", 0.5)
            options = prob_config.get("options", ["Y", "N"])
            
            # Weighted random choice with proper RNG
            if rng.random() < probability_y:
                choice = options[0]  # Y
            else:
                choice = options[1]  # N
            return {"disclose_income": choice}
    
    # Fallback: 50/50 random choice
    choice = rng.choice(["Y", "N"])
    return {"disclose_income": choice}
```

**Key Features**:
- ✅ Checks `simulation_config['random_decisions']`
- ✅ Uses configured probability from UI (e.g., 70% Y, 30% N)
- ✅ Falls back to 50/50 if not configured
- ✅ Uses proper RNG for reproducibility

**Variation** (`disclose_documents.py`):
```python
def disclose_documents(agent_state, params, rng, simulation_config=None):
    """Only applies to agents with income < threshold"""
    
    # STEP 1: Generate or retrieve agent's income
    if 'income' not in agent_state:
        # Generate income using distribution from simulation_config
        income = _sample_from_distribution(simulation_config, rng)
        agent_state['income'] = income
    
    # STEP 2: Check eligibility
    agent_income = agent_state.get('income')
    threshold = simulation_config['simulation'].get('discount_income_threshold', 12500.0)
    
    if agent_income >= threshold:
        return {"disclose_documents": "NA"}  # Not applicable
    
    # STEP 3: Apply probability-based decision (same as disclose_income)
    if simulation_config and 'random_decisions' in simulation_config:
        prob_config = simulation_config['random_decisions'].get('disclose_documents')
        if prob_config:
            probability_y = prob_config.get("probability_y", 0.5)
            # ... weighted random choice ...
    
    return {"disclose_documents": choice}
```

**Key Features**:
- ✅ Generates income first (used by other decisions)
- ✅ Conditional applicability (only for low-income agents)
- ✅ Returns "NA" for ineligible agents
- ✅ Same probability pattern as other random decisions

---

### Type 3: Radio Selection Decisions
**Examples**: `rejected_transaction_defaults`, `rejected_transaction_option`

**Implementation** (`rejected_transaction_defaults.py`):
```python
def rejected_transaction_defaults(agent_state, params, rng, simulation_config=None):
    """Select from multiple rejection handling options"""
    
    # CHECK: Is configuration available?
    if simulation_config and 'default_decisions' in simulation_config:
        config = simulation_config['default_decisions'].get('rejected_transaction_defaults')
        if config and config.get("type") == "radio_selection":
            # Use the selected option from configuration
            selected_option = config.get("selected_option", "forgo_transaction")
            return {"rejected_transaction_defaults": selected_option}
    
    # Fallback to default option
    return {"rejected_transaction_defaults": "forgo_transaction"}
```

**Key Features**:
- ✅ Checks `simulation_config['default_decisions']`
- ✅ Returns selected option string (e.g., "higher_price_category")
- ✅ Falls back to "forgo_transaction"
- ✅ All agents get same option (not random)

**Variation** (`rejected_transaction_option.py`):
```python
def rejected_transaction_option(agent_state, params, rng, simulation_config=None):
    """Option selection with fallback import"""
    
    # Try to import dynamic default value function
    try:
        from app.pages.decision_execution import get_actual_default_value
        default_value = get_actual_default_value("rejected_transaction_option")
    except ImportError:
        # Fallback if import fails
        default_value = params.get("default_value", "NA")
    
    return {"rejected_transaction_option": default_value}
```

**Key Features**:
- ✅ Imports UI layer function directly
- ✅ Uses priority resolution (pre-config → post-sim → hard-coded)
- ✅ Graceful fallback if import fails

---

### Type 4: Checkbox Selection Decisions
**Example**: `vendor_choice_weights`

**Implementation** (`vendor_choice_weights.py`):
```python
def vendor_choice_weights(agent_state, params, rng, simulation_config=None):
    """Select weights for vendor choice parameters"""
    
    # CHECK: Is this decision using configured defaults? (unselected)
    if simulation_config and 'default_decisions_list' in simulation_config:
        if 'vendor_choice_weights' in simulation_config.get('default_decisions_list', []):
            # Unselected - use configured default values from Overview tab
            if 'default_decisions' in simulation_config:
                vendor_config = simulation_config['default_decisions'].get('vendor_choice_weights')
                if vendor_config and vendor_config.get("type") == "checkbox_selection":
                    # Use pre-calculated weights from configuration
                    weights = vendor_config.get("weights", {})
                    return {"vendor_choice_weights": weights}
    
    # Selected OR no configuration - use equal weights
    default_weights = {
        "price": 0.25,
        "quality": 0.25,
        "proximity": 0.25,
        "sustainability": 0.25
    }
    
    return {"vendor_choice_weights": default_weights}
```

**Key Features**:
- ✅ Checks `default_decisions_list` to see if unselected
- ✅ Uses pre-calculated weights from UI (e.g., {price: 0.5, quality: 0.5, others: 0})
- ✅ Falls back to equal weights (0.25 each)
- ✅ Returns dict of weights

---

### Type 5: Computed/Placeholder Decisions
**Examples**: `consumption_quantity`, `consumption_frequency`, `bid_value`

**Implementation** (`consumption_frequency.py`):
```python
def consumption_frequency(agent_state, params, rng, simulation_config=None, **kwargs):
    """Computed from consumption_quantity / term_duration"""
    
    # STEP 1: Get consumption quantity from previous decision
    consumption_quantity = agent_state.get('consumption_quantity', 0)
    
    # STEP 2: Get term duration from Page 1 parameters
    periods = _get_simulation_param(simulation_config, 'periods', 1)
    duration_hours = _get_simulation_param(simulation_config, 'duration_hours', 1.0)
    term_duration = float(periods * duration_hours)
    
    # STEP 3: Calculate frequency
    if term_duration > 0:
        frequency = consumption_quantity / term_duration
    else:
        frequency = 0.0
    
    return {"consumption_frequency": float(frequency)}
```

**Key Features**:
- ✅ No default checking needed - always computed
- ✅ Depends on previous decision output (`consumption_quantity`)
- ✅ Uses Page 1 simulation parameters
- ✅ Deterministic calculation

**Implementation** (`consumption_quantity.py`):
```python
def consumption_quantity(agent_state, params, rng, simulation_config=None, **kwargs):
    """Random quantity within consumption limit"""
    
    # STEP 1: Get or generate agent income (from disclose_documents)
    income = agent_state.get('income')
    if income is None:
        income = _sample_income_from_distribution(simulation_config, rng)
        agent_state['income'] = income
    
    # STEP 2: Assign to income category (1 to NFIC)
    income_category = _assign_income_category(income, simulation_config)
    
    # STEP 3: Get consumption limit for this category
    consumption_limits = simulation_config.get('consumption_limits', {})
    limit_key = f"cat_{income_category}"
    consumption_limit = consumption_limits.get(limit_key, fallback_max)
    
    # STEP 4: Generate random quantity in [0, limit]
    if consumption_limit > 0:
        total_quantity = int(rng.integers(0, consumption_limit + 1))
    else:
        total_quantity = 0
    
    # STEP 5: Generate purchase requests with timestamps
    if total_quantity > 0:
        term_duration = periods * duration_hours
        timestamps = sorted(rng.uniform(0, term_duration, size=total_quantity))
        
        purchase_requests = [
            {"request_id": i+1, "quantity": 1, "timestamp_hours": float(ts)}
            for i, ts in enumerate(timestamps)
        ]
    else:
        purchase_requests = []
    
    return {
        "consumption_quantity": int(total_quantity),
        "purchase_requests": purchase_requests,
        "income_category": int(income_category),
        "income": float(income)
    }
```

**Key Features**:
- ✅ No simple default - always computes per agent
- ✅ Uses income from previous decision
- ✅ Assigns income category based on threshold
- ✅ Random within consumption limit (from Page 1 config)
- ✅ Generates detailed purchase request objects

---

## Data Flow

### Complete Flow for Unselected Decision

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER: Configures defaults in Overview tab                   │
│    - Sets disclose_income probability to 70%                   │
│    - Stored in: st.session_state.disclose_income_default_      │
│                 probability_y = 0.7                            │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. SIMULATION START: collect_decision_settings()               │
│    - Reads: st.session_state.disclose_income_default_          │
│             probability_y = 0.7                                │
│    - Creates: decision_settings = {                            │
│         "disclose_income": {                                   │
│             "type": "random_probability",                      │
│             "probability_y": 0.7,                              │
│             "options": ["Y", "N"]                              │
│         }                                                       │
│     }                                                           │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. ORCHESTRATOR SETUP: run_simulation_from_sidebar()           │
│    - orchestrator.simulation_config['random_decisions'] =      │
│      decision_settings                                         │
│    - orchestrator.simulation_config['default_decisions'] =     │
│      decision_settings                                         │
│    - orchestrator.simulation_config['default_decisions_list']  │
│      = ['disclose_income', ...]  # Unselected decisions        │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. ORCHESTRATOR EXECUTION: orchestrator.run_simulation()       │
│    for each agent:                                             │
│      for each decision:                                        │
│        decision_output = decision_module(                      │
│            agent_state,                                        │
│            params,                                             │
│            rng,                                                │
│            simulation_config=self.simulation_config  # <--     │
│        )                                                        │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. DECISION MODULE: disclose_income()                          │
│    - Receives: simulation_config containing random_decisions   │
│    - Checks: simulation_config['random_decisions']             │
│              ['disclose_income']['probability_y'] = 0.7        │
│    - Executes: if rng.random() < 0.7:                          │
│                  return {"disclose_income": "Y"}  # 70% chance │
│                else:                                            │
│                  return {"disclose_income": "N"}  # 30% chance │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. RESULT: Agent state updated                                 │
│    agent_state["disclose_income"] = "Y"  (or "N")              │
└─────────────────────────────────────────────────────────────────┘
```

### Flow for Selected Decision (Custom Configuration)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER: Configures custom parameters in donation_default tab  │
│    - Sets complex regression coefficients                      │
│    - Sets anchor weights, stochastic params, etc.              │
│    - Stored in: st.session_state.donation_coeff_*              │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. SIMULATION START: Decision in selected_decisions list       │
│    - st.session_state.custom_decisions = ['donation_default']  │
│    - st.session_state.default_decisions = [other 12 decisions] │
│    - donation_default NOT in default_decisions_list            │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. ORCHESTRATOR: Passes simulation_config                      │
│    - simulation_config['default_decisions_list'] doesn't       │
│      include 'donation_default'                                │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. DECISION MODULE: donation_default()                         │
│    # First check: Is this in default_decisions_list?           │
│    if 'donation_default' in simulation_config.get(             │
│        'default_decisions_list', []):                          │
│        return {"donation_default": 0.10}  # Simple default     │
│    # NOT IN LIST - run full custom model                       │
│    else:                                                        │
│        # 184 lines of complex logic:                           │
│        # - Regression prediction                               │
│        # - Scaling to 0-100                                    │
│        # - Anchor calculation                                  │
│        # - Stochastic component                                │
│        # - Rescaling to [0,1]                                  │
│        return {"donation_default": complex_calculated_value}   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Code Locations

### Summary Table

| Layer | File | Key Functions/Classes |
|-------|------|----------------------|
| **UI Configuration** | `app/pages/decision_tabs/default_config.py` | `render_default_decisions_config()`<br>`render_probability_default_config()`<br>`render_radio_default_config()`<br>`render_checkbox_default_config()`<br>`render_numeric_default_config()` |
| **Default Registry** | `app/pages/decision_execution.py` | `DEFAULT_DECISION_VALUES` dict<br>`get_actual_default_value()`<br>`run_individual_decision()`<br>`run_combined_simulation()` |
| **Simulation Config** | `app/simulation.py` | `collect_decision_settings()`<br>`run_simulation_from_sidebar()` |
| **Orchestrator** | `src/orchestrator.py` | `Orchestrator.run_simulation()` |
| **Decision Modules** | `src/decisions/*.py` | Individual decision functions (13 files) |

### Decision Module Files

| Decision | File | Default Type |
|----------|------|--------------|
| 1. disclose_income | `src/decisions/disclose_income.py` | Random probability (Y/N) |
| 2. disclose_documents | `src/decisions/disclose_documents.py` | Random probability (Y/N, conditional) |
| 3. donation_default | `src/decisions/donation_default.py` | Numeric with complex model |
| 3. (alt) donation_default | `src/decisions/donation_default_stochastic.py` | Numeric with stochastic model |
| 4. rejected_transaction_defaults | `src/decisions/rejected_transaction_defaults.py` | Radio selection (5 options) |
| 5. vendor_choice_weights | `src/decisions/vendor_choice_weights.py` | Checkbox selection (weights) |
| 6. consumption_quantity | `src/decisions/consumption_quantity.py` | Computed per agent |
| 7. consumption_frequency | `src/decisions/consumption_frequency.py` | Calculated from #6 |
| 8. vendor_selection | `src/decisions/vendor_selection.py` | Deterministic |
| 9. purchase_vs_bid | `src/decisions/purchase_vs_bid.py` | Random probability (Purchase/Bid) |
| 10. bid_value | `src/decisions/bid_value.py` | Random within range |
| 11. rejected_transaction_option | `src/decisions/rejected_transaction_option.py` | Radio selection (5 options) |
| 12. rejected_bid_value | `src/decisions/rejected_bid_value.py` | NA placeholder |
| 13. final_donation_rate | `src/decisions/final_donation_rate.py` | Numeric (simple) |

---

## Examples & Patterns

### Pattern 1: Simple Numeric Default with Complex Model Fallback

**Used by**: `donation_default`, `donation_default_stochastic`

```python
def donation_default(agent_state, params, rng, simulation_config=None, **kwargs):
    """
    Pattern: Check if unselected → return simple default
             Otherwise → run complex model
    """
    
    # ========== DEFAULT CHECK BLOCK ==========
    if simulation_config and 'default_decisions_list' in simulation_config:
        if 'donation_default' in simulation_config.get('default_decisions_list', []):
            # Unselected - use simple default
            default_config = simulation_config.get('default_decisions', {}).get('donation_default')
            if default_config:
                if isinstance(default_config, dict) and default_config.get('type') == 'numeric':
                    return {"donation_default": float(default_config.get('value', 0.1))}
                elif isinstance(default_config, (int, float)):
                    return {"donation_default": float(default_config)}
    
    # ========== COMPLEX MODEL BLOCK ==========
    # Selected - run full model
    hh_score = agent_state['Honesty_Humility']
    income_level = agent_state['Assigned Allowance Level']
    # ... 180 lines of complex logic ...
    
    return {"donation_default": computed_rate}
```

**Key Insight**: Early exit strategy - check for simple default first, only run expensive computation if needed.

---

### Pattern 2: Probability-Based Random Choice

**Used by**: `disclose_income`, `disclose_documents`, `purchase_vs_bid`

```python
def disclose_income(agent_state, params, rng, simulation_config=None):
    """
    Pattern: Check for configured probability → weighted random choice
             Otherwise → 50/50 random choice
    """
    
    # ========== CONFIGURED PROBABILITY BLOCK ==========
    if simulation_config and 'random_decisions' in simulation_config:
        prob_config = simulation_config['random_decisions'].get('disclose_income')
        if prob_config and prob_config.get("type") == "random_probability":
            probability_y = prob_config.get("probability_y", 0.5)
            options = prob_config.get("options", ["Y", "N"])
            
            # Weighted random choice
            if rng.random() < probability_y:
                return {"disclose_income": options[0]}
            else:
                return {"disclose_income": options[1]}
    
    # ========== FALLBACK: 50/50 RANDOM ==========
    choice = rng.choice(["Y", "N"])
    return {"disclose_income": choice}
```

**Key Insight**: Uses RNG for reproducibility, supports custom probability from UI.

---

### Pattern 3: Conditional Applicability with Income Generation

**Used by**: `disclose_documents`

```python
def disclose_documents(agent_state, params, rng, simulation_config=None):
    """
    Pattern: Generate income → check eligibility → apply decision
    """
    
    # ========== INCOME GENERATION BLOCK ==========
    if 'income' not in agent_state:
        # Generate income using distribution from simulation_config
        income_dist = sim_params.get('income_distribution', 'lognormal')
        if income_dist == 'lognormal':
            mu = sim_params.get('lognormal_mu', 10.0)
            sigma = sim_params.get('lognormal_sigma', 0.5)
            min_val = sim_params.get('lognormal_min', 0.0)
            # ... sample from distribution ...
            income = min_val + Y
        agent_state['income'] = income
    
    # ========== ELIGIBILITY CHECK BLOCK ==========
    agent_income = agent_state.get('income', 50000.0)
    threshold = simulation_config['simulation'].get('discount_income_threshold', 12500.0)
    
    if agent_income >= threshold:
        return {"disclose_documents": "NA"}  # Not applicable
    
    # ========== DECISION BLOCK (same as disclose_income) ==========
    # Only applies to agents with income < threshold
    if simulation_config and 'random_decisions' in simulation_config:
        prob_config = simulation_config['random_decisions'].get('disclose_documents')
        # ... probability-based choice ...
    
    return {"disclose_documents": choice}
```

**Key Insight**: Generates state (income) used by later decisions, conditional logic for applicability.

---

### Pattern 4: Radio Selection from Config

**Used by**: `rejected_transaction_defaults`, `rejected_transaction_option`

```python
def rejected_transaction_defaults(agent_state, params, rng, simulation_config=None):
    """
    Pattern: Check config for selected option → return string
    """
    
    # ========== CONFIGURED SELECTION BLOCK ==========
    if simulation_config and 'default_decisions' in simulation_config:
        config = simulation_config['default_decisions'].get('rejected_transaction_defaults')
        if config and config.get("type") == "radio_selection":
            selected_option = config.get("selected_option", "forgo_transaction")
            return {"rejected_transaction_defaults": selected_option}
    
    # ========== FALLBACK: DEFAULT OPTION ==========
    return {"rejected_transaction_defaults": "forgo_transaction"}
```

**Key Insight**: Returns string identifier for option, all agents get same option (deterministic).

---

### Pattern 5: Checkbox Selection with Weight Calculation

**Used by**: `vendor_choice_weights`

```python
def vendor_choice_weights(agent_state, params, rng, simulation_config=None):
    """
    Pattern: Check if unselected → use pre-calculated weights
             Otherwise → use equal weights
    """
    
    # ========== CONFIGURED WEIGHTS BLOCK ==========
    if simulation_config and 'default_decisions_list' in simulation_config:
        if 'vendor_choice_weights' in simulation_config.get('default_decisions_list', []):
            vendor_config = simulation_config['default_decisions'].get('vendor_choice_weights')
            if vendor_config and vendor_config.get("type") == "checkbox_selection":
                # Use pre-calculated weights from UI
                weights = vendor_config.get("weights", {})
                return {"vendor_choice_weights": weights}
    
    # ========== FALLBACK: EQUAL WEIGHTS ==========
    default_weights = {
        "price": 0.25,
        "quality": 0.25,
        "proximity": 0.25,
        "sustainability": 0.25
    }
    return {"vendor_choice_weights": default_weights}
```

**Key Insight**: UI calculates weights in `render_checkbox_default_config()`, decision module just retrieves them.

---

### Pattern 6: Computed from Previous Decision

**Used by**: `consumption_frequency`

```python
def consumption_frequency(agent_state, params, rng, simulation_config=None, **kwargs):
    """
    Pattern: Get value from previous decision → compute derived value
    """
    
    # ========== GET PREVIOUS DECISION OUTPUT ==========
    consumption_quantity = agent_state.get('consumption_quantity', 0)
    
    # ========== GET SIMULATION PARAMETERS ==========
    periods = _get_simulation_param(simulation_config, 'periods', 1)
    duration_hours = _get_simulation_param(simulation_config, 'duration_hours', 1.0)
    term_duration = float(periods * duration_hours)
    
    # ========== COMPUTE DERIVED VALUE ==========
    if term_duration > 0:
        frequency = consumption_quantity / term_duration
    else:
        frequency = 0.0
    
    return {"consumption_frequency": float(frequency)}
```

**Key Insight**: Deterministic calculation, depends on decision execution order.

---

### Pattern 7: Random within Agent-Specific Bounds

**Used by**: `consumption_quantity`

```python
def consumption_quantity(agent_state, params, rng, simulation_config=None, **kwargs):
    """
    Pattern: Get agent income → assign category → get limit → random within limit
    """
    
    # ========== GET/GENERATE INCOME ==========
    income = agent_state.get('income')
    if income is None:
        income = _sample_income_from_distribution(simulation_config, rng)
    
    # ========== ASSIGN CATEGORY ==========
    income_category = _assign_income_category(income, simulation_config)
    
    # ========== GET LIMIT FOR CATEGORY ==========
    consumption_limits = simulation_config.get('consumption_limits', {})
    limit_key = f"cat_{income_category}"
    consumption_limit = consumption_limits.get(limit_key, fallback_max)
    
    # ========== RANDOM WITHIN LIMIT ==========
    if consumption_limit > 0:
        total_quantity = int(rng.integers(0, consumption_limit + 1))
    else:
        total_quantity = 0
    
    # ========== GENERATE PURCHASE REQUESTS ==========
    purchase_requests = []
    if total_quantity > 0:
        term_duration = periods * duration_hours
        timestamps = sorted(rng.uniform(0, term_duration, size=total_quantity))
        purchase_requests = [
            {"request_id": i+1, "quantity": 1, "timestamp_hours": float(ts)}
            for i, ts in enumerate(timestamps)
        ]
    
    return {
        "consumption_quantity": int(total_quantity),
        "purchase_requests": purchase_requests,
        "income_category": int(income_category),
        "income": float(income)
    }
```

**Key Insight**: Agent-specific bounds (based on income category), generates complex output (list of purchase requests).

---

## Summary of Default Mechanisms

### 5 Different Default Mechanisms:

1. **Simple Numeric Default** (donation_default)
   - Check if unselected → return configured value (0.10)
   - Otherwise → run complex model

2. **Random Probability** (disclose_income, disclose_documents, purchase_vs_bid)
   - Check for configured probability → weighted random choice
   - Otherwise → 50/50 random

3. **Radio Selection** (rejected_transaction_defaults, rejected_transaction_option)
   - Check for selected option → return string identifier
   - Otherwise → return default option

4. **Checkbox Selection** (vendor_choice_weights)
   - Check for pre-calculated weights → return weight dict
   - Otherwise → return equal weights

5. **Computed** (consumption_quantity, consumption_frequency, bid_value)
   - Always compute per agent
   - No simple default available
   - May depend on previous decisions or agent traits

### Key Design Principles:

✅ **Separation of Concerns**: UI layer handles configuration, decision modules handle logic
✅ **Priority System**: Pre-config → Post-sim → Hard-coded
✅ **Graceful Fallbacks**: Always have a default if configuration missing
✅ **RNG Consistency**: Use provided RNG for reproducibility
✅ **Early Exit**: Check for defaults first to avoid expensive computation
✅ **State Management**: Store generated values in agent_state for reuse

---

## Best Practices for Adding New Decisions

When adding a new decision with default behavior:

1. **Add to `DEFAULT_DECISION_VALUES`** in `app/pages/decision_execution.py`
   - Choose appropriate type: `random_probability`, `radio_selection`, `checkbox_selection`, numeric, or placeholder
   - Provide sensible default values

2. **Implement decision module** in `src/decisions/{decision_name}.py`
   - Accept `simulation_config` parameter
   - Check for default configuration at start
   - Implement fallback behavior
   - Use provided `rng` for randomness

3. **Add UI configuration** (if needed)
   - Update `render_decision_default_config()` routing
   - Create specialized renderer if new type
   - Add session state initialization

4. **Update orchestrator** (if needed)
   - Add to `decision_order` list
   - Ensure proper parameter passing

5. **Test priority resolution**
   - Pre-configured default works
   - Post-simulation adjustment works
   - Hard-coded fallback works
   - Selected vs. unselected behavior correct

---

**End of Architecture Analysis**

