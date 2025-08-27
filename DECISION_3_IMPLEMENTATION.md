# Decision 3: Default Donation Suggestion - Implementation Details

## Overview

Decision 3 generates a default donation rate for each synthetic agent using a hybrid anchor approach that combines observed prosocial behavior with predicted behavior from a regression model, then adds stochastic variation and applies truncation rules.

## 5. Decision 3 Implementation

### 5.1 Deterministic Anchor Computation

For every agent *i*, we compute:

```
anchor_i = 0.75 × observed_i + 0.25 × ŷ_i
```

Where:
- `observed_i` is the agent's experimental prosocial score: `TWT+Sospeso [=AW2+AX2]{Periods 1+2}`
- `ŷ_i` is the predicted prosocial score from a fixed linear regression model

#### Regression Model for ŷ_i

The predicted prosocial score is computed using:

```
ŷ_i = β₀ + β_group[group_i] + β_income_q[income_q_i] + β_study[study_i] + β_hh × HH_i
```

**Coefficients (stored in `config/decisions.yaml`):**
- **Intercept (β₀):** 1.22985660120368
- **Group effects (reference: FullSub = 0):**
  - MidSub: 0.856140306694656
  - NoSub: -0.926633374153906
  - FullSub: 0.0 (reference)
- **Income quintile effects (reference: Q4_Q5 = 0):**
  - Q1: -0.520290427509808
  - Q2: 3.754612744416796
  - Q3: 4.001714810873598
  - Q4_Q5: 0.0 (reference)
- **Study programme effects (reference: Grad2yr = 0):**
  - Incoming: -6.920193024391676
  - Law5yr: -2.081331674770856
  - UG3yr: -2.139093511519692
  - Grad2yr: 0.0 (reference, includes CLEF/CLEAM/BIEF)
- **Honesty-Humility coefficient:** 0.634001208840808

### 5.2 Stochastic Variation

To reflect behavioral heterogeneity, we treat the anchor as the mean of a normal distribution:

```
x_i ~ N(μ = anchor_i, σ)
```

Where σ is computed using the `overall_sd_twt_sospeso` strategy (global standard deviation).

**Truncation and Rescaling Process:**
1. **Floor at zero:** `x_i = max(0, x_i)` (no negative donations)
2. **Personal maximum:** `x_max = μ + σ × Φ⁻¹(0.99)` where Φ⁻¹(0.99) ≈ 2.326
3. **Final rescaling:** `d_i = min(x_i, x_max) / x_max`, yielding a value in [0,1]

### 5.3 Implementation Architecture

#### File Structure and Responsibilities

**`config/trait_requirements.yaml`**
- Specifies required traits for Decision 3:
  - `Honesty_Humility` (continuous HH score)
  - `Assigned Allowance Level` (experimental income levels 1-5)
  - `Study Program` (academic programme indicator)
  - `Group_experiment` (experimental group: FullSub/MidSub/NoSub)
  - `TWT+Sospeso [=AW2+AX2]{Periods 1+2}` (observed prosocial behavior)

**`config/decisions.yaml`**
- Contains all regression coefficients and parameters
- Anchor weights: observed=0.75, predicted=0.25
- Stochastic parameters: sigma strategy and truncation rules
- Percentile maximum: 0.99 (99th percentile)

**`src/trait_engine.py`**
- Generates synthetic agents using Gaussian copula
- Preserves correlation structure from original 280 participants
- Maps income levels 1-5 to quintiles for regression model

**`src/decisions/donation_default.py`**
- **Interface:** `donation_default(agent_state: dict, params: dict, rng: np.random.Generator) -> dict`
- **Input:** agent_state contains agent's traits from TraitEngine
- **Input:** params auto-loaded from config/decisions.yaml
- **Input:** rng supplied by orchestrator for reproducibility
- **Output:** `{"donation_default": d_i}` where d_i ∈ [0,1]

**Implementation Steps in `donation_default.py`:**

```python
# Step 1: Extract required traits
hh_score = agent_state['Honesty_Humility']
income_level = agent_state['Assigned Allowance Level']  # 1-5
study_program = agent_state['Study Program']
group = agent_state['Group_experiment']
observed_prosocial = agent_state['TWT+Sospeso [=AW2+AX2]{Periods 1+2}']

# Step 2: Compute predicted prosocial using regression
predicted = regression['intercept']

# Add group effect (reference: FullSub)
if group in regression['beta_group']:
    predicted += regression['beta_group'][group]

# Map income level to quintile (1→Q1, 2→Q2, 3→Q3, 4-5→Q4_Q5)
income_quintiles = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4_Q5', 5: 'Q4_Q5'}
income_q = income_quintiles.get(int(income_level), 'Q4_Q5')
if income_q in regression['beta_income_q']:
    predicted += regression['beta_income_q'][income_q]

# Add study program effect with mapping:
# 'Incoming' → 'Incoming', 'Law' → 'Law5yr', 'UG' → 'UG3yr', else → 'Grad2yr'
if study_category in regression['beta_study']:
    predicted += regression['beta_study'][study_category]

# Add honesty-humility effect
predicted += regression['beta_hh'] * hh_score

# Step 3: Compute anchor (75% observed + 25% predicted)
anchor = 0.75 * observed_prosocial + 0.25 * predicted

# Step 4: Add stochastic variation
sigma = 5.0  # TODO: Compute from actual data
draw = rng.normal(anchor, sigma)

# Step 5: Apply truncation rules
draw = max(0.0, draw)  # Floor at zero
personal_max = anchor + norm.ppf(0.99) * sigma  # 99th percentile
if personal_max > 0:
    donation_rate = min(draw, personal_max) / personal_max
else:
    donation_rate = 0.0

# Step 6: Final clipping to [0,1]
donation_rate = np.clip(donation_rate, 0.0, 1.0)
```

**`src/orchestrator.py`**
- Coordinates trait sampling and decision execution
- Loads parameters from config/decisions.yaml
- Executes decisions sequentially with mutable agent state
- Manages random number generation with child RNGs per agent
- Supports both single-decision and full-simulation modes

### 5.4 Execution Modes

#### Single Decision Mode
Execute only Decision 3 for testing and validation:

```bash
python scripts/run_simulation.py --agents 1000 --seed 42 --decision donation_default
```

**Command Flow:**
1. `scripts/run_simulation.py` parses arguments
2. Creates `Orchestrator()` instance
3. Calls `orchestrator.run_simulation(n_agents=1000, seed=42, single_decision='donation_default')`
4. Orchestrator samples 1000 agents via `TraitEngine.sample()`
5. For each agent, executes only `donation_default()` function
6. Saves results to `outputs/simulation_seed42_agents1000_donation_default_TIMESTAMP.parquet`

#### Full Simulation Mode
Execute all 13 decisions in sequence:

```bash
python scripts/run_simulation.py --agents 10000 --seed 42 --decision all
```

**Decision Execution Order (defined in `src/orchestrator.py`):**
1. disclose_income
2. disclose_documents  
3. **donation_default** ← Decision 3
4. rejected_transaction_defaults
5. vendor_choice_weights
6. consumption_quantity
7. consumption_frequency
8. vendor_selection
9. purchase_vs_bid
10. bid_value
11. rejected_transaction_option
12. rejected_bid_value
13. final_donation_rate

#### Monte Carlo Mode
Execute Decision 3 multiple times with different seeds to quantify uncertainty:

```bash
python scripts/run_mc_study.py --agents 1000 --runs 100 --decision donation_default
```

**Monte Carlo Process:**
- **Inner loop:** For fixed parameters, draw n agents and generate stochastic donations; randomness in x_i captures person-to-person heterogeneity
- **Outer loop:** Repeat with different random seeds to capture sampling uncertainty
- **Output:** Distributions of aggregate statistics (mean donation rate, percentiles, etc.)

### 5.5 Data Flow and File Locations

```
Raw Data (data/):
├── Student Survey Results - Period 1.xlsx      → Honesty_Humility, Study Program
└── Student Experiment Results - Period 1-2.xlsx → Assigned Allowance Level, Group_experiment, TWT+Sospeso

↓ [Preprocessing via scripts/train_copula.py]

Fitted Model (config/):
├── trait_model.pkl                 → Gaussian copula for trait generation
├── trait_requirements.yaml         → Required traits per decision  
└── decisions.yaml                   → Regression coefficients and parameters

↓ [Simulation via scripts/run_simulation.py]

Results (outputs/):
├── simulation_*.parquet             → Individual simulation results
├── mc_summary_*.csv                 → Monte Carlo aggregate statistics
├── mc_detailed_*.csv                → Monte Carlo individual run results
└── mc_config_*.json                 → Monte Carlo configuration and metadata
```

## 6. Runtime Framework

### 6.1 Core Components

**TraitEngine (`src/trait_engine.py`)**
- Samples traits via fitted Gaussian copula
- Generates unlimited synthetic populations preserving correlations
- Only includes traits specified in `trait_requirements.yaml`

**Orchestrator (`src/orchestrator.py`)**
- Loads parameters from `config/decisions.yaml`
- Executes decisions sequentially with mutable agent state
- Manages reproducible random number generation
- Supports both single-decision and full-simulation modes

**Decision Modules (`src/decisions/*.py`)**
- All expose identical interface: `function_name(agent_state, params, rng) -> dict`
- Stateless functions that update agent state
- Parameters externalized to YAML for easy modification

### 6.2 Output Format

Each simulation produces a DataFrame where each row represents one agent:

```
Columns:
- All traits from TraitEngine (Honesty_Humility, Assigned Allowance Level, etc.)
- Decision outputs (donation_default, disclose_income, etc.)
- Agent ID and metadata
```

**Example Output Structure:**
```
agent_id | Honesty_Humility | Assigned Allowance Level | Study Program | Group_experiment | TWT+Sospeso [...] | donation_default | disclose_income | ...
---------|------------------|-------------------------|---------------|------------------|-------------------|------------------|-----------------|----
0        | 3.2              | 2.0                     | UG3yr         | MidSub           | 12.5              | 0.34             | NA              | ...
1        | 2.8              | 4.0                     | Law5yr        | FullSub          | 18.2              | 0.67             | NA              | ...
...
```

## 7. Extensibility Framework

### 7.1 Adding New Decisions

To implement a new decision (e.g., Decision 4):

1. **Update trait requirements (`config/trait_requirements.yaml`):**
```yaml
rejected_transaction_defaults:
  - required_trait_1
  - required_trait_2
```

2. **Regenerate copula model:**
```bash
python scripts/train_copula.py
```

3. **Add parameters (`config/decisions.yaml`):**
```yaml
rejected_transaction_defaults:
  coefficient_a: 1.23
  coefficient_b: -0.45
  stochastic_sd: 2.1
```

4. **Create decision module (`src/decisions/rejected_transaction_defaults.py`):**
```python
def rejected_transaction_defaults(agent_state: dict, params: dict, rng: np.random.Generator) -> dict:
    # Implementation logic here
    result = compute_decision(agent_state, params, rng)
    return {"rejected_transaction_defaults": result}
```

5. **Next simulation automatically includes the new decision:**
```bash
python scripts/run_simulation.py --agents 1000 --seed 42 --decision all
```

### 7.2 Parameter Uncertainty

For uncertainty-aware simulations (future enhancement):

```bash
python scripts/run_simulation.py --agents 10000 --seed 42 --posterior_draw
```

This would:
1. Draw new coefficient vector β (and σ) from stored posterior/bootstrap samples
2. Update parameters in memory (not YAML files)
3. Run simulation with new parameters
4. Repeat K times to get distributions combining estimation + behavioral uncertainty

The current implementation provides the foundation for this extension by externalizing all parameters to YAML configuration files.

## 8. Monte Carlo Implementation

The Monte Carlo framework (`scripts/run_mc_study.py`) implements uncertainty quantification by running multiple independent simulations with different random seeds and aggregating the results to compute confidence intervals and sampling distributions. The implementation follows a nested loop structure: the **inner loop** generates behavioral heterogeneity by sampling synthetic agents and running stochastic decision models for a fixed parameter set, while the **outer loop** quantifies sampling uncertainty by repeating the simulation with different random seeds. For each run *i* (where *i* = 1 to *K*), the system calls `run_simulation.py` with seed = `base_seed + i`, collects key statistics (mean, std, min, max) for each decision output, and stores results in temporary files. After all runs complete, the system aggregates results across runs to compute meta-statistics including the mean of means, standard deviation across runs, and 95% confidence intervals using the 2.5th and 97.5th percentiles. This approach separates **person-to-person heterogeneity** (captured within each run by the stochastic decision models) from **sampling uncertainty** (captured across runs by different agent populations), providing robust uncertainty quantification for academic reporting. The framework supports parallel execution, automatic cleanup of temporary files, and produces three output types: detailed run-by-run results, summary statistics with confidence intervals, and configuration files for exact reproducibility.

## 9. Complete Application Structure

```
AI Agent Simulation Framework/
├── README.md                           # Project documentation and quick start guide
├── DECISION_3_IMPLEMENTATION.md        # Detailed implementation documentation
├── deployment_guide.md                 # Deployment instructions for web interface
├── app.py                             # Streamlit web interface for interactive simulations
├── requirements.txt                    # Python dependencies for deployment
├── deploy.sh                          # Automated deployment setup script
│
├── config/                            # Configuration and trained models
│   ├── trait_requirements.yaml        # Required traits per decision module
│   ├── decisions.yaml                 # Decision parameters and regression coefficients
│   ├── trait_model.pkl                # Fitted Gaussian copula model
│   └── mc_seeds.txt                   # (Optional) Pre-generated seeds for MC studies
│
├── data/                              # Raw experimental and survey data
│   ├── Student Survey Results - Period 1.xlsx           # Survey responses (280 participants)
│   └── Student Experiment Results - Period 1-2.xlsx    # Experimental outcomes
│
├── src/                               # Core simulation engine
│   ├── __init__.py                    # Python package initialization
│   ├── trait_engine.py               # Gaussian copula for synthetic agent generation
│   ├── orchestrator.py               # Decision execution coordinator
│   ├── build_master_traits.py        # Extract required traits from YAML config
│   ├── validate_traits.py            # Validate trait availability in data
│   │
│   └── decisions/                     # Individual decision modules (13 total)
│       ├── __init__.py               # Package initialization
│       ├── donation_default.py       # Decision 3: Default donation rate (IMPLEMENTED)
│       ├── disclose_income.py        # Decision 1: Income disclosure (placeholder)
│       ├── disclose_documents.py     # Decision 2: Document disclosure (placeholder)
│       ├── rejected_transaction_defaults.py  # Decision 4: Transaction defaults (placeholder)
│       ├── vendor_choice_weights.py  # Decision 5: Vendor weights (placeholder)
│       ├── consumption_quantity.py   # Decision 6: Consumption quantity (placeholder)
│       ├── consumption_frequency.py  # Decision 7: Consumption frequency (placeholder)
│       ├── vendor_selection.py       # Decision 8: Vendor selection (placeholder)
│       ├── purchase_vs_bid.py        # Decision 9: Purchase vs bid (placeholder)
│       ├── bid_value.py              # Decision 10: Bid value (placeholder)
│       ├── rejected_transaction_option.py  # Decision 11: Rejected transaction option (placeholder)
│       ├── rejected_bid_value.py     # Decision 12: Rejected bid value (placeholder)
│       └── final_donation_rate.py    # Decision 13: Final donation rate (placeholder)
│
├── scripts/                           # Command-line simulation tools
│   ├── train_copula.py               # Fit Gaussian copula model on trait data
│   ├── run_simulation.py             # Single simulation execution (CLI)
│   └── run_mc_study.py               # Monte Carlo uncertainty quantification
│
└── outputs/                           # Generated simulation results
    ├── simulation_*.parquet           # Individual simulation results (agents × decisions)
    ├── mc_detailed_*.csv             # Monte Carlo run-by-run statistics
    ├── mc_summary_*.csv              # Monte Carlo aggregate statistics and CIs
    └── mc_config_*.json              # Monte Carlo configuration for reproducibility
```

### Key Architecture Principles

**Modular Design**: Each decision is an independent module with standardized interface, allowing easy addition of new decisions without modifying core framework.

**Configuration-Driven**: All parameters externalized to YAML files, enabling quick parameter updates and supporting future posterior sampling for uncertainty quantification.

**Reproducible**: Hierarchical random number generation ensures exact reproducibility across different execution modes (single run, Monte Carlo, web interface).

**Scalable**: Gaussian copula enables generation of unlimited synthetic populations while preserving statistical relationships from original 280-participant dataset.

**Multi-Interface**: Supports command-line execution for batch processing and web interface for interactive exploration and visualization.

**Academic-Ready**: Outputs formatted for academic reporting with proper uncertainty quantification, confidence intervals, and metadata for reproducibility.