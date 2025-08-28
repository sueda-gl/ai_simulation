# Population Modes in AI Agent Simulation

## Overview

The simulation framework now supports three distinct population generation modes:

### 1. **Copula Mode** (Synthetic Agents)
- **Source**: Gaussian copula fitted on 280 original participants
- **Randomness**: From synthetic trait sampling via copula
- **Stochastic Component**: None (removed as copula provides randomness)
- **Advantages**: 
  - Unlimited synthetic agents
  - Preserves correlation structure
  - Natural trait variation

### 2. **Documentation Mode** (Original + Stochastic)
- **Source**: Original 280 participants from merged dataset
- **Randomness**: From Normal(anchor, σ) draws as per documentation
- **Stochastic Component**: Full implementation with σ = 9.8995
- **Advantages**:
  - Follows exact documentation methodology
  - Shows effect of stochastic component
  - Uses real participant data

### 3. **Dependent Variable Resampling Mode** (Outcome Only)
- **Source**: Empirical distribution of 280 donation rates
- **Randomness**: Bootstrap resampling from computed outcomes
- **Stochastic Component**: Already included in pre-computed values
- **Advantages**:
  - No trait sampling at all (as per professor's request)
  - Preserves exact outcome distribution
  - Simple and interpretable
  - Fast computation

## Key Differences

| Aspect | Copula Mode | Documentation Mode | Dependent Variable Mode |
|--------|-------------|-------------------|------------------------|
| **Population** | Synthetic agents | Original participants | Donation rates only |
| **Trait Variation** | From copula sampling | Fixed (original data) | Not applicable |
| **Stochastic Draw** | No | Yes - Normal(anchor, σ) | Pre-computed |
| **Personal Max** | Not used | 99th percentile scaling | Pre-computed |
| **Mean Donation** | ~15% | ~40-45% | ~40-45% |
| **Distribution** | Smooth, centered | Normal-like with some zeros | Exact empirical |
| **Trait Information** | Full | Full | None |
| **Use Case** | Study trait relationships | Follow documentation | Focus on outcome only |

## Dashboard Options

The Streamlit app now supports:

1. **Population Mode Selection**:
   - Copula (synthetic)
   - Documentation (original + stochastic)
   - Dependent variable resampling
   - Compare both

2. **Income Specification**:
   - Categorical only
   - Continuous only
   - Compare side-by-side

3. **Combined Comparisons**:
   - Up to 4 simultaneous comparisons (2 population × 2 income modes)
   - Side-by-side visualizations
   - Comparative statistics

## Implementation Details

### Copula Mode
```python
# Uses TraitEngine to sample from fitted copula
# No additional stochastic component
donation_rate = s100_anchor / 100.0
```

### Documentation Mode
```python
# Uses original participants
# Adds stochastic component
draw ~ Normal(anchor, σ)
personal_max = anchor + z_0.99 * σ
donation_rate = min(draw, personal_max) / personal_max
```

### Dependent Variable Mode
```python
# Pre-compute donation rates for all 280 participants
empirical_donations = [compute_donation(p) for p in participants]

# Generate new population by bootstrap resampling
new_donations = np.random.choice(empirical_donations, size=n_agents, replace=True)
```

## Usage Examples

### Single Mode
- Select "Copula (synthetic)" + "categorical only"
- Generates 1000 synthetic agents with categorical income treatment

### Comparison Mode
- Select "Compare both" + "compare side-by-side"
- Shows 4 histograms: Copula×Cat, Copula×Cont, Doc×Cat, Doc×Cont

### Monte Carlo
- Both modes support Monte Carlo studies
- Documentation mode bootstraps participants if n_agents > 280

## Expected Results

- **Copula Mode**: Lower mean (~15%), tighter distribution
- **Documentation Mode**: Higher mean (~40-45%), more spread, some zeros
- **Categorical vs Continuous**: Small differences in both modes

This dual approach allows professors to compare:
1. Pure copula-based heterogeneity (modern approach)
2. Traditional stochastic simulation (documentation approach)
3. Impact of income specification (categorical vs continuous)
