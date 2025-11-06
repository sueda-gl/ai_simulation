# 🚨 CRITICAL BUG: Agent Mismatch in Multi-Configuration Export

## Problem Statement

When exporting multiple simulation configurations (e.g., copula_categorical + copula_continuous), the Excel file incorrectly aligns agents **by row position** rather than **by agent identity**. This causes:

- **Agent ID 22** in copula_categorical to be treated as the same agent as **Agent ID 22** in copula_continuous
- But they are actually **DIFFERENT AGENTS** with different traits!

## Evidence

From user's Excel file (Row 23, Agent ID 22):

| Config | Honesty_Humility | Allowance | donation_default |
|--------|------------------|-----------|------------------|
| Config 1 | 3.25 | 2 | 0.077388 |
| Config 2 | ? | ? | 0.066639 |
| Config 3 | ? | ? | **0.935537** |

The massive difference in donation_default (0.077 → 0.935) proves these are different agents with different traits being incorrectly aligned.

## Root Cause

### 1. **Each Configuration Creates NEW Agents**

In `app/simulation.py` lines 559-564:

```python
# For "Compare all" + "Compare both"
results[f"{pop_name}_categorical"] = _run(pop_type, "categorical", random_decision_probabilities)
results[f"{pop_name}_continuous"] = _run(pop_type, "continuous", random_decision_probabilities)
```

Each `_run()` call:
1. Creates a **new orchestrator instance**
2. Samples **new synthetic agents** via `trait_engine.sample(n_agents, seed)`
3. These are **INDEPENDENT random samples**

### 2. **Agent ID is Just Row Index**

In `src/orchestrator.py` line 117:

```python
agent_state['agent_id'] = idx + 1  # Agent IDs start at 1
```

- `agent_id` = DataFrame row index + 1
- **NOT a unique identifier** across different populations
- Agent ID 22 in one config ≠ Agent ID 22 in another config

### 3. **Export Assumes Row Alignment**

In `app/pages/results/components/export_section.py` line 186:

```python
combined_df[new_col_name] = config_df[col].values  # ❌ WRONG!
```

This copies values by row position, assuming:
- Row 0 in all configs = same agent
- Row 1 in all configs = same agent
- etc.

**BUT THIS IS FALSE!**

## Impact

### Affected Scenarios

1. ✅ **Single configuration**: NO BUG (only one DataFrame)
2. ❌ **Compare both income modes** (categorical + continuous): BUGGED
3. ❌ **Compare all population modes** (copula + research_spec + baseline): BUGGED
4. ❌ **Compare all × Compare both** (6 configurations): SEVERELY BUGGED

### Data Integrity

- **Traits don't match**: Agent 22's Honesty_Humility in config 1 ≠ Agent 22's HH in config 2
- **Decisions don't match**: donation_default values are from completely different people
- **Analysis invalid**: Any cross-config comparisons are meaningless

## Solution Options

### **Option 1: Sample Agents Once, Reuse Across Configs** (RECOMMENDED)

Modify `app/simulation.py` to:
1. Sample/load agents ONCE before running configurations
2. Pass the SAME agents_df to all orchestrators
3. Each orchestrator runs decisions on the SAME population

```python
def run_simulation_from_sidebar():
    # Sample agents ONCE
    seed = st.session_state.seed
    n_agents = st.session_state.n_agents
    
    # Determine which population to use
    if "Research" in st.session_state.population_mode:
        # Load original 280 participants
        agents_df = load_original_participants()
    else:
        # Sample from copula
        trait_engine = TraitEngine()
        agents_df = trait_engine.sample(n_agents, seed)
    
    # Run all configs with SAME agents
    results = {}
    for config_name, (pop_type, inc_mode) in configs:
        results[config_name] = _run_with_agents(agents_df, pop_type, inc_mode)
```

**Pros:**
- ✅ Ensures agent alignment
- ✅ Meaningful cross-config comparisons
- ✅ Preserves agent_id meaning

**Cons:**
- ⚠️ Requires refactoring orchestrator initialization
- ⚠️ Need to handle research mode vs copula mode agent selection

### **Option 2: Align by Traits in Export** (PARTIAL FIX)

Modify `export_section.py` to align DataFrames by trait signatures:

```python
# Instead of: combined_df[new_col_name] = config_df[col].values
# Use:
merged = combined_df.merge(
    config_df[['Honesty_Humility', 'Assigned Allowance Level', col]],
    on=['Honesty_Humility', 'Assigned Allowance Level'],
    how='left',
    suffixes=('', '_' + config_suffix)
)
combined_df[new_col_name] = merged[col + '_' + config_suffix]
```

**Pros:**
- ✅ Minimal code changes
- ✅ Works even if agents are different

**Cons:**
- ❌ Doesn't fix the root cause (agents still different)
- ❌ Merge may fail if traits don't match exactly
- ❌ Relies on trait uniqueness (may have duplicates)
- ❌ Still can't compare across population modes (copula vs research)

### **Option 3: Only Export Single Config** (WORKAROUND)

Disable multi-config export entirely:

```python
if export_all_configs:
    st.warning("⚠️ Multi-config export disabled due to agent mismatch bug")
    st.info("Please select a single configuration to export")
    return
```

**Pros:**
- ✅ Quick fix
- ✅ Prevents misleading data

**Cons:**
- ❌ Loses multi-config functionality
- ❌ Doesn't fix the underlying issue

## Recommended Solution

**Implement Option 1** (Sample Agents Once):

1. Refactor `run_simulation_from_sidebar()` to sample agents once
2. Create `run_with_agents()` method in all orchestrators
3. Pass the same agents_df to all configurations
4. Ensure agent_id is preserved and consistent

This ensures:
- ✅ Same agent in all configurations
- ✅ Valid cross-config comparisons
- ✅ Data integrity

## Implementation Plan

### Phase 1: Add `run_with_agents()` to Orchestrators

```python
# src/orchestrator.py
class Orchestrator:
    def run_with_agents(self, agents_df: pd.DataFrame, single_decision=None):
        """Run simulation with pre-sampled agents"""
        # Skip trait sampling, use provided agents_df
        # Process decisions as normal
        ...
```

### Phase 2: Refactor simulation.py

```python
# app/simulation.py
def run_simulation_from_sidebar():
    # Sample agents once
    agents_df = sample_agents_once()
    
    # Run all configs with same agents
    results = {}
    for config_key, (pop_type, inc_mode) in get_configs():
        results[config_key] = run_with_same_agents(agents_df, pop_type, inc_mode)
```

### Phase 3: Validation

1. Run diagnostic script to verify alignment
2. Check that traits match across configs
3. Verify donation_default values are correlated (when using same parameters)

## Testing

```python
# Test case: Agent traits should match across configs
def test_agent_alignment():
    results = st.session_state.simulation_results
    
    config1_df = results['copula_categorical']
    config2_df = results['copula_continuous']
    
    # Check agent 0
    agent1_traits = config1_df.iloc[0][trait_columns]
    agent2_traits = config2_df.iloc[0][trait_columns]
    
    assert all(agent1_traits == agent2_traits), "Agents don't match!"
```

## Timeline

- **Phase 1**: 2-3 hours (add run_with_agents methods)
- **Phase 2**: 1-2 hours (refactor simulation.py)
- **Phase 3**: 1 hour (testing and validation)

**Total**: 4-6 hours

## Status

- ❌ Bug discovered: 2025-11-06
- ⏳ Fix in progress
- ⏳ Testing pending
- ⏳ Deployment pending

