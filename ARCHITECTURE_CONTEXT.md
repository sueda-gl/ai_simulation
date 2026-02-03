# AI Agent Simulation Framework - Architecture Context

**Purpose:** This document provides context for AI agents assisting with the codebase refactoring.

---

## 1. What We Are Building

An **AI Agent Behavioral Simulation Framework** that models how synthetic agents make decisions in a marketplace environment. The simulation uses statistical models (regression coefficients from research papers) to predict agent behaviors like income disclosure, donation rates, vendor selection, and purchasing patterns.

**Key Concept:** We simulate N agents (typically 100-1000) making 13 sequential decisions. Each decision uses either:
- **Statistical model** (research-based regression with stochastic elements)
- **Default implementation** (simple deterministic logic)

The user configures which decisions use the statistical model vs. defaults.

---

## 2. Technology Stack

- **Frontend:** Streamlit (Python web framework)
- **Backend:** Python with NumPy, Pandas, SciPy
- **Population Generation:** Gaussian Copula for correlated trait sampling
- **Configuration:** YAML files (`config/simulation.yaml`, `config/decisions.yaml`)
- **State Management:** Streamlit `st.session_state`

---

## 3. UI Structure

### Page 1: Global Simulation Parameters (`app/pages/page1_common_params.py`)

Configures simulation-wide settings:

| Section | Parameters |
|---------|-----------|
| **Time** | Periods, Duration (hours) |
| **Agents** | Number of agents, Random seed |
| **Vendors** | Number of vendors, Price range, Products per vendor |
| **Income Distribution** | Distribution type (lognormal/gamma/dagum), Parameters |
| **Purchasing Limits** | Per-category limits (optional) |
| **Population Mode** | Copula / Research Specification / Research Baseline |

### Page 2: Decision Configuration (`app/pages/page2_decisions.py`)

Tab-based interface for 13 decisions:

```
[Overview] [Disclose Income] [Donation Default] [Vendor Selection] ...
```

**Overview Tab:** Shows which decisions are selected for statistical modeling vs. defaults.

**Decision Tabs:** Each decision has its own tab with:
- Toggle: Use statistical model vs. default
- Model-specific parameters (coefficients, weights, probabilities)
- "Run This Decision Only" button
- "Run Complete Simulation" button (when config is saved)

### Results Page (`app/pages/results/`)

Displays simulation outputs:
- Summary statistics
- Visualizations (charts, distributions)
- Per-decision breakdowns
- Excel export functionality

---

## 4. The 13 Decisions (Execution Order)

Decisions execute sequentially. Later decisions can depend on earlier decisions' outputs.

| # | Decision | Description | Dependencies |
|---|----------|-------------|--------------|
| 1 | `disclose_income` | Agent decides whether to disclose income (Y/N) | None |
| 2 | `disclose_documents` | Agent decides whether to disclose documents (Y/N/NA) | Requires `disclose_income=Y` |
| 3 | `donation_default` | Agent's default donation rate (0-1) | Uses income from agent state |
| 4 | `rejected_transaction_defaults` | Default behavior for rejected transactions | None |
| 5 | `vendor_choice_weights` | Weights for vendor selection criteria | None |
| 6 | `purchasing_quantity` | How much the agent wants to purchase | None |
| 7 | `purchasing_frequency` | How often the agent purchases | None |
| 8 | `vendor_selection` | Which vendor the agent selects | Requires vendor weights |
| 9 | `purchase_vs_bid` | Purchase now vs. bid (deprecated) | - |
| 10 | `bid_value` | Bid amount (deprecated) | - |
| 11 | `rejected_transaction_option` | What to do when transaction rejected | - |
| 12 | `rejected_bid_value` | Bid value for rejected transactions | - |
| 13 | `final_donation_rate` | Final donation after all adjustments | Uses `donation_default` |

---

## 5. Population Modes

Three modes determine how agents are generated:

### Copula (Synthetic)
- Uses Gaussian Copula to generate correlated agent traits
- Traits: allowance group, household size, study programme, income quintile
- Most flexible - generates any number of agents with realistic correlations

### Research Specification
- Uses exact trait distributions from the research paper
- Applies stochastic model with specified variance (sigma)
- More constrained than Copula

### Research Baseline
- Uses exact point estimates from research paper
- No stochastic component (sigma = 0)
- Deterministic outputs for given inputs

---

## 6. Income Modes

How income affects decision calculations:

### Categorical
- Income divided into quintiles (Q1-Q5)
- Each quintile has a coefficient (beta_Q1, beta_Q2, etc.)
- Calculation: `effect = beta_Q[agent's quintile]`

### Continuous
- Uses raw income value with linear coefficient
- Calculation: `effect = beta_income_linear * income`

### Compare Both
- **NOT a calculation mode** - it's an orchestration instruction
- Runs simulation TWICE (once categorical, once continuous)
- Shows results side-by-side for comparison
- The actual calculation always uses either categorical OR continuous, never both

---

## 7. Key Architectural Components

### Orchestrators (`src/orchestrator*.py`)

Different orchestrators for different population modes:

| Orchestrator | Mode | Description |
|-------------|------|-------------|
| `orchestrator.py` | Copula | Main orchestrator for synthetic agents |
| `orchestrator_doc_mode.py` | Research Specification | Research mode with sigma |
| `orchestrator_baseline.py` | Research Baseline | Deterministic research mode |
| `orchestrator_depvar.py` | - | Dependent variable handling |

**Orchestrator Responsibilities:**
1. Generate agent population (traits)
2. Create RNG for each decision
3. Execute decisions in sequence
4. Collect and aggregate results

### Decision Modules (`src/decisions/*.py`)

Each decision is a Python function with signature:
```python
def decision_name(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    # Calculate decision outcome
    return {"decision_field": value}
```

**Key Decisions:**
- `disclose_income_stochastic.py` - Two-stage mediation model
- `donation_default_stochastic.py` - Regression-based donation calculation
- `income_utils.py` - Shared income processing utilities

### Session State (`st.session_state`)

Streamlit's mechanism for persisting state across reruns. Currently used for:
- Global parameters (`sim_params`)
- Decision parameters (`decision_params`)
- Widget values (with various key prefixes)
- Simulation results
- Selected configurations

---

## 8. Known Issues (Why We're Refactoring)

### Issue 1: Multi-Source Configuration Chaos
Same configuration stored in 6+ locations:
```
st.session_state.income_spec_mode
st.session_state.di_income_mode
st.session_state.di_tab_income_mode
st.session_state.page2_tab_income_spec_mode
st.session_state.disclose_income_tab_persistence['income_mode']
Widget keys with various prefixes
```

### Issue 2: Silent Mode Normalization
Code silently converts invalid modes to defaults:
```python
if 'continuous' in str(income_mode).lower():
    normalized_mode = 'continuous'
else:
    normalized_mode = 'categorical'  # Silent default!
```

### Issue 3: Magic Strings Everywhere
No enums or constants:
```python
if income_mode == "categorical only":  # Typo-prone
if config.get('source') != 'auto_implied_single_config':  # Magic string
```

### Issue 4: Implicit State Dependencies
Decisions implicitly depend on state created by earlier decisions:
- `donation_default` expects income from `disclose_documents`
- When running alone, it generates its own income (different RNG position)
- Causes discrepancies between individual runs and complete simulation

### Issue 5: RNG Sequence Contamination (FIXED)
Earlier decisions consumed RNG values, affecting later decisions.
**Fixed by:** Decision-specific RNG instances.

### Issue 6: Fragile Config Detection
Complex nested conditions to determine what mode to use:
```python
can_run, reason, config_count, block_type = result[:4]
blocking_issues = result[4] if len(result) > 4 else []
```

---

## 9. Proposed Architecture (Refactoring Goals)

### Goal 1: Single Source of Truth
Create `ConfigManager` class that:
- Stores all configuration in one place
- Provides typed access to settings
- Validates on write, not read

### Goal 2: Type Safety with Enums
```python
class IncomeMode(Enum):
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"

class PopulationMode(Enum):
    COPULA = "copula"
    RESEARCH_SPEC = "research_specification"
    RESEARCH_BASELINE = "research_baseline"
```

### Goal 3: Explicit Decision Interface
Each decision declares:
- **requires:** What state it needs (must exist)
- **provides:** What state it creates
- **rng:** Gets isolated RNG instance

### Goal 4: Separation of Concerns
- `IncomeMode` = actual calculation mode (CATEGORICAL or CONTINUOUS)
- `IncomeSelection` = UI selection (includes COMPARE_BOTH)
- Orchestrator handles "Compare Both" by running twice

---

## 10. File Structure

```
<sdg/
├── app/
│   ├── models.py                    # Data models, session state init
│   ├── simulation.py                # Main simulation orchestration (1500+ lines)
│   └── pages/
│       ├── page1_common_params.py   # Page 1 UI (1600 lines)
│       ├── page2_decisions.py       # Page 2 UI
│       ├── navigation.py            # Page navigation
│       ├── decision_execution.py    # Decision execution logic (1800+ lines)
│       ├── decision_tabs/           # Individual decision tab UIs
│       │   ├── disclose_income.py
│       │   ├── donation_default.py
│       │   └── ...
│       └── results/                 # Results display
│           ├── main_results.py
│           ├── config_selection.py
│           └── visualizations/
│
├── src/
│   ├── orchestrator.py              # Copula mode orchestrator
│   ├── orchestrator_baseline.py     # Research Baseline orchestrator
│   ├── orchestrator_doc_mode.py     # Research Specification orchestrator
│   ├── trait_engine.py              # Agent trait generation
│   └── decisions/                   # Decision modules
│       ├── disclose_income.py
│       ├── disclose_income_stochastic.py
│       ├── donation_default.py
│       ├── donation_default_stochastic.py
│       ├── income_utils.py          # Shared utilities
│       └── ...
│
├── config/
│   ├── simulation.yaml              # Global simulation defaults
│   └── decisions.yaml               # Decision-specific coefficients
│
└── outputs/                         # Simulation output files
```

---

## 11. Refactoring Priority Order

1. **Create `types.py`** - Enums for IncomeMode, PopulationMode, ConfigSource
2. **Create `ConfigManager`** - Single source of truth for all configuration
3. **Refactor Decision Interface** - Explicit requires/provides declarations
4. **Update Orchestrators** - Use ConfigManager, isolated RNG per decision
5. **Update Decision Modules** - Use typed params, clean interface
6. **Update Page 2 UI** - Use ConfigManager instead of session_state chaos
7. **Update Page 1 UI** - Lower priority (mostly working)
8. **Update Results/Export** - Consume clean data from ConfigManager

---

## 12. Critical Files to Read First

When starting refactoring, read these files first:

1. `app/models.py` - Current data models, ALL_DECISIONS list
2. `app/simulation.py` - Main simulation flow (see the mode selection spaghetti)
3. `app/pages/decision_execution.py` - How decisions are executed
4. `src/decisions/disclose_income_stochastic.py` - Example of statistical model
5. `config/decisions.yaml` - Coefficient definitions

---

## 13. Key Concepts to Understand

### Anchoring
Combines observed data with predicted values:
```
final_value = (anchor_weight * observed) + ((1 - anchor_weight) * predicted)
```

### Stochastic Component (Sigma)
Adds variance to predictions:
```
predicted = deterministic_prediction + (sigma * random_noise)
```

### Truncation
Clamps values to valid ranges (e.g., donation rate 0-1)

### Two-Stage Mediation Model (Disclose Income)
1. **Stage 1:** Calculate Prosocial Behavior score
2. **Stage 2:** Use Prosocial + other factors to predict disclosure

---

## 14. Session State Keys Reference

Key prefixes and their meanings:

| Prefix | Meaning |
|--------|---------|
| `sim_params.*` | Global simulation parameters object |
| `decision_params.*` | Decision configuration object |
| `tab_*` | Widget key for a tab-specific control |
| `page2_*` | Page 2 specific widget keys |
| `di_*` | Disclose income specific keys |
| `*_default_*` | Default decision values |
| `selected_*_config` | Saved configuration for a decision |

---

## 15. Testing Approach

When testing refactored code:

1. **Seed Reproducibility:** Same seed should produce same results
2. **Mode Equivalence:** Individual decision run should match complete simulation
3. **Compare Both:** Should produce same results as running categorical and continuous separately
4. **No Silent Defaults:** Invalid inputs should raise errors, not silently default

---

**Document Created:** February 2026
**For:** AI Agent Refactoring Assistance
**Codebase:** AI Agent Simulation Framework
