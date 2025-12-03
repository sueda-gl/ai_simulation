# Donation Default Decision: Why Agents with TWT+Sospeso = 0 Have Different Donation Rates

## Professor's Concern
Some agents with `TWT+Sospeso = 0` have higher donation rates compared to other agents who also have `TWT+Sospeso = 0`.

## Short Answer
**This is expected behavior by design.** The donation rate formula gives 25% weight to a **predicted** prosocial score based on agent characteristics (group, income, study program, Honesty-Humility). Even when the observed prosocial behavior (`TWT+Sospeso`) is zero, different agents have different predicted values, leading to different donation rates.

---

## The 6 Modes Explained

The simulation has 6 modes combining:
- **3 Population Modes:**
  1. **Copula** - Synthetic agents from Gaussian copula
  2. **Documentation** - Original 280 participants + stochastic noise
  3. **Baseline** - Original 280 participants, no stochastic noise
  
- **2 Income Modes:**
  1. **Categorical** - Income quintiles (Q1-Q5)
  2. **Continuous** - Linear income effect

**All 6 modes use the same donation_default calculation**, so this variation occurs in all modes.

---

## The Formula

### Step 1: Compute Predicted Prosocial Score
```
predicted = intercept 
          + β_group[group]           # Group effect
          + β_income[income_level]   # Income effect
          + β_study[study_program]   # Study program effect
          + β_hh × z(HH_score)       # Honesty-Humility effect
```

### Step 2: Scale to 0-100
```
s100_observed  = 100 × (TWT+Sospeso - 0) / (112 - 0)
s100_predicted = 100 × (predicted - (-4.0778)) / (7.203 - (-4.0778))
```

### Step 3: Compute Anchor (Weighted Average)
```
anchor = 0.75 × s100_observed + 0.25 × s100_predicted
       = 0.75 × observed_component + 0.25 × predicted_component
```

### Step 4-6: Adjust, Floor, and Scale
```
adjusted_anchor = anchor + shift_value  # shift_value = -4
donation_rate = max(adjusted_anchor, 0) / 100
```

---

## What Happens When TWT+Sospeso = 0

When an agent has `TWT+Sospeso = 0`:
- `s100_observed = 0`
- `anchor = 0.75 × 0 + 0.25 × s100_predicted = 0.25 × s100_predicted`

**The donation rate depends ENTIRELY on the predicted prosocial score.**

---

## Coefficients Driving the Variation

### Categorical Income Mode

| Factor | Coefficient | Effect |
|--------|-------------|--------|
| **Intercept** | 1.52 | Base value |
| **Group: MidSub** | +0.88 | Increases prediction |
| **Group: NoSub** | -0.91 | Decreases prediction |
| **Group: FullSub** | 0.00 | Reference |
| **Income: Q1** | 0.00 | Reference |
| **Income: Q2** | -0.42 | Decreases prediction |
| **Income: Q3** | -0.74 | Decreases prediction |
| **Income: Q4** | +3.54 | Large increase |
| **Income: Q5** | +3.78 | Large increase |
| **Study: Incoming** | -6.88 | Large decrease |
| **Study: Law5yr** | -2.00 | Moderate decrease |
| **Study: UG3yr** | -2.12 | Moderate decrease |
| **Study: Grad2yr** | 0.00 | Reference (CLEF, CLEAM, etc.) |
| **Honesty-Humility** | +0.60 | Per z-score |

---

## Example Calculations (TWT+Sospeso = 0)

### Agent A: Low Predicted (NoSub, Q1, Incoming, Low HH)
```
predicted = 1.52 - 0.91 + 0.00 - 6.88 + (0.60 × -1.60) = -7.24
s100_predicted = clipped to 0
anchor = 0.25 × 0 = 0
donation_rate = 0.00%
```

### Agent B: Medium Predicted (MidSub, Q3, Graduate, Average HH)
```
predicted = 1.52 + 0.88 - 0.74 + 0.00 + (0.60 × 0.01) = 1.67
s100_predicted = 50.9
anchor = 0.25 × 50.9 = 12.7
donation_rate = (12.7 - 4) / 100 = 8.74%
```

### Agent C: High Predicted (FullSub, Q5, Graduate, High HH)
```
predicted = 1.52 + 0.00 + 3.78 + 0.00 + (0.60 × 1.45) = 6.18
s100_predicted = 90.9
anchor = 0.25 × 90.9 = 22.7
donation_rate = (22.7 - 4) / 100 = 18.73%
```

---

## Evidence from Actual Data

**176 agents in the dataset have TWT+Sospeso = 0**

| Statistic | Donation Rate |
|-----------|---------------|
| Min | 0.00% |
| Max | 20.79% |
| Mean | 8.92% |
| Std | 5.45% |

This variation reflects differences in their other characteristics.

---

## Why This Design?

The research design gives 25% weight to predicted prosocial behavior because:

1. **Predictive Value**: Demographics and personality traits (like Honesty-Humility) can predict donation tendencies even when observed behavior is zero.

2. **Regression-Based**: The predicted component comes from a regression model fitted on real participant data.

3. **Heterogeneity**: Without the predicted component, agents with zero observed prosocial behavior would all have identical donation rates (after shift adjustment), which may not reflect realistic behavioral heterogeneity.

---

## If You Want Zero TWT+Sospeso → Zero Donation Rate

To make agents with `TWT+Sospeso = 0` always have `donation_rate = 0`, you would need to:

**Option 1:** Change anchor weights to 100% observed
```yaml
# In config/decisions.yaml
anchor_weights:
  observed: 1.0   # Was 0.75
  predicted: 0.0  # Was 0.25
```

**Option 2:** Add a special case in the code
```python
if observed_prosocial == 0:
    return {"donation_default": 0.0}
```

---

## Running the Diagnostic Script

To reproduce this analysis:
```bash
cd /Users/suedagul/<sdg
.venv/bin/python diagnose_donation_default.py
```

---

## Summary

| Question | Answer |
|----------|--------|
| Is this a bug? | **No** - expected behavior by design |
| Why do zero-TWT agents differ? | Different predicted scores from regression |
| What drives the variation? | Group, Income, Study Program, HH |
| Does this happen in all 6 modes? | **Yes** - same formula in all modes |
| Can it be changed? | Yes - adjust anchor weights or add special case |

