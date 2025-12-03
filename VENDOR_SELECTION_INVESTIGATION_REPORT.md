# Vendor Selection Investigation Report

**Date:** November 28, 2025  
**Issue:** Only one vendor selected for all 1000 customers across different random seeds  
**Status:** ✅ ROOT CAUSE IDENTIFIED

---

## 🔍 Problem Summary

When running simulations with 6 vendors and 1000 agents:
- **With seed 42:** Vendor 4 is selected by 100% of agents (1000/1000)
- **With seed 999:** Vendor 5 is selected by 100% of agents (1000/1000)
- This pattern occurs despite randomly generated proximity scores for each agent

**User's concern:** "The proximity matrix is randomly decided, so it cannot be that vendor 4 is most attractive to all 1000 customers."

---

## ✅ What IS Working Correctly

### 1. Vendor Attribute Randomization ✓
Vendors are correctly randomized with different attributes each time:

**Seed 42:**
- Vendor 1: $127.40, Quality: 4/5, Sustainability: 3/5
- Vendor 2: $119.74, Quality: 5/5, Sustainability: 2/5
- Vendor 3: $147.56, Quality: 4/5, Sustainability: 4/5
- **Vendor 4: $62.81, Quality: 4/5, Sustainability: 5/5** ⭐ (Best)
- Vendor 5: $87.08, Quality: 1/5, Sustainability: 5/5
- Vendor 6: $132.28, Quality: 4/5, Sustainability: 3/5

**Seed 999:**
- Vendor 1: $127.88, Quality: 1/5, Sustainability: 1/5
- Vendor 2: $125.03, Quality: 4/5, Sustainability: 1/5
- Vendor 3: $140.19, Quality: 1/5, Sustainability: 1/5
- Vendor 4: $141.09, Quality: 4/5, Sustainability: 3/5
- **Vendor 5: $105.74, Quality: 5/5, Sustainability: 4/5** ⭐ (Best)
- Vendor 6: $113.83, Quality: 3/5, Sustainability: 1/5

Different vendors are "best" with different seeds.

### 2. Proximity Score Generation ✓
Each agent gets unique proximity scores to each vendor:

**Agent 1 (seed 42):**
- Vendor 1: 0.0, Vendor 2: 36.4, Vendor 3: 23.8, Vendor 4: 74.6, Vendor 5: 70.0, Vendor 6: 81.3

**Agent 2 (seed 42):**
- Vendor 1: 1.4, Vendor 2: 53.5, Vendor 3: 39.9, Vendor 4: 48.4, Vendor 5: 64.0, Vendor 6: 95.6

**Agent 3 (seed 42):**
- Vendor 1: 22.3, Vendor 2: 37.3, Vendor 3: 51.6, Vendor 4: 99.8, Vendor 5: 62.9, Vendor 6: 74.8

Proximity scores vary correctly by agent.

### 3. Composite Score Calculation ✓
The math for calculating vendor scores is correct:

```python
score = (weight_price × norm_price) + 
        (weight_quality × norm_quality) + 
        (weight_proximity × norm_proximity) + 
        (weight_sustainability × norm_sustainability)
```

Where:
- `norm_price = 1.0 - (price - min_price) / (max_price - min_price)` (inverted: lower price = higher score)
- `norm_quality = (quality - 1) / 4.0`
- `norm_sustainability = (sustainability - 1) / 4.0`
- `norm_proximity = proximity / 100.0`

---

## ❌ ROOT CAUSES IDENTIFIED

### ROOT CAUSE #1: No Agent-Level Weight Variation

**Location:** `src/decisions/vendor_choice_weights.py`, lines 20-27

**Current Code:**
```python
default_weights = {
    "price": 0.25,
    "quality": 0.25,
    "proximity": 0.25,
    "sustainability": 0.25
}
return {"vendor_choice_weights": default_weights}
```

**Problem:**
- Every agent receives **IDENTICAL weights** (0.25 for all attributes)
- No randomization or agent-specific variation
- The `rng` parameter is not used
- Comment in code says: "For now, since there's no custom tab for vendor_choice_weights, always use equal weights"

**Impact:**
- All agents evaluate vendors using the **exact same criteria**
- Vendor ranking is **deterministic** (same for all agents)
- Only variation comes from proximity scores, which is not enough to overcome attribute differences

**Evidence:**
```
Weight Statistics (1000 agents):
                   Mean    Std     Min     Max
Weight_Price       0.25   0.00    0.25    0.25
Weight_Quality     0.25   0.00    0.25    0.25
Weight_Proximity   0.25   0.00    0.25    0.25
Weight_Sustain.    0.25   0.00    0.25    0.25
```

### ROOT CAUSE #2: One Vendor is Objectively Superior

With equal weights (0.25 each), vendors are ranked purely on their objective attributes.

**Seed 42 - Why Vendor 4 wins for ALL agents:**

Vendor 4 has:
- **Lowest price:** $62.81 → normalized: 1.000 (best possible)
- **Good quality:** 4/5 → normalized: 0.750
- **Highest sustainability:** 5/5 → normalized: 1.000 (best possible)
- Average proximity: ~60 (varies by agent)

**Average composite score: 0.8296** (33% higher than next best vendor)

Even agents with low proximity to Vendor 4 (e.g., proximity = 43.7) still rank it highest because:
```
Score = 0.25×1.000 + 0.25×0.750 + 0.25×(43.7/100) + 0.25×1.000
      = 0.250 + 0.188 + 0.109 + 0.250
      = 0.797
```

This beats all other vendors even for agents with high proximity to them.

**Seed 999 - Why Vendor 5 wins for ALL agents:**

Vendor 5 has:
- **Lowest price:** $105.74 → normalized: 1.000 (best possible)
- **Highest quality:** 5/5 → normalized: 1.000 (best possible)
- **High sustainability:** 4/5 → normalized: 0.750
- Average proximity: ~65

**Average composite score: 0.8629** (65% higher than next best vendor)

---

## 🧪 Test Results

### Test 1: Seed 42, 6 vendors, 1000 agents
```
Vendor Selection Results:
  Vendor 1:    0/1000 agents (  0.0%)
  Vendor 2:    0/1000 agents (  0.0%)
  Vendor 3:    0/1000 agents (  0.0%)
  Vendor 4: 1000/1000 agents (100.0%) ⚠️
  Vendor 5:    0/1000 agents (  0.0%)
  Vendor 6:    0/1000 agents (  0.0%)
```

### Test 2: Seed 999, 6 vendors, 1000 agents
```
Vendor Selection Results:
  Vendor 1:    0/1000 agents (  0.0%)
  Vendor 2:    0/1000 agents (  0.0%)
  Vendor 3:    0/1000 agents (  0.0%)
  Vendor 4:    0/1000 agents (  0.0%)
  Vendor 5: 1000/1000 agents (100.0%) ⚠️
  Vendor 6:    0/1000 agents (  0.0%)
```

**Conclusion:** Different seeds change which vendor is "best", but 100% concentration remains.

---

## 🎯 Why This Happens

1. **All agents have identical preferences** (same weights)
2. **Vendors are evaluated objectively** (not subjectively)
3. **One vendor randomly gets superior attributes** (low price + high quality/sustainability)
4. **Proximity variation is insufficient** to overcome the objective advantage
5. **Result:** All agents converge on the same "best" vendor

### Mathematical Example

With weights all at 0.25, the composite score formula is:

```
Score = 0.25×norm_price + 0.25×norm_quality + 0.25×norm_proximity + 0.25×norm_sustainability
```

For Vendor 4 (seed 42):
- Best case (high proximity): `0.25×1.00 + 0.25×0.75 + 0.25×0.99 + 0.25×1.00 = 0.937`
- Worst case (low proximity): `0.25×1.00 + 0.25×0.75 + 0.25×0.04 + 0.25×1.00 = 0.697`

For Vendor 1 (seed 42):
- Best case (high proximity): `0.25×0.24 + 0.25×0.75 + 0.25×1.00 + 0.25×0.50 = 0.623`
- Worst case (low proximity): `0.25×0.24 + 0.25×0.75 + 0.25×0.00 + 0.25×0.50 = 0.373`

Even Vendor 4's **worst case** (0.697) beats Vendor 1's **best case** (0.623)!

---

## 📊 Summary

| Component | Status | Details |
|-----------|--------|---------|
| Vendor attribute generation | ✅ Working | Randomized correctly each seed |
| Proximity score generation | ✅ Working | Varies by agent-vendor dyad |
| Composite score calculation | ✅ Working | Math is correct |
| Agent weight variation | ❌ **PROBLEM** | All agents identical (0.25 each) |
| Vendor selection diversity | ❌ **RESULT** | One vendor selected by 100% of agents |

---

## 💡 Why This Matters

In a realistic market:
- **Some customers prioritize price** (budget-conscious)
- **Some customers prioritize quality** (quality-conscious)
- **Some customers prioritize proximity** (convenience-conscious)
- **Some customers prioritize sustainability** (environmentally-conscious)

Currently, **all customers have identical preferences**, which leads to:
- Unrealistic monopoly behavior
- No market diversity
- All agents converging on one vendor
- Proximity becoming irrelevant

---

## 🔧 Solution (To Be Implemented)

The fix is to add **agent-level weight variation** in `src/decisions/vendor_choice_weights.py`:

```python
def vendor_choice_weights(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    # ... (keep existing config check logic) ...
    
    # Generate agent-specific weights using Dirichlet distribution
    # This ensures weights sum to 1.0 and vary across agents
    alpha = [2.0, 2.0, 2.0, 2.0]  # Concentration parameters
    weights_array = rng.dirichlet(alpha)
    
    weights = {
        "price": float(weights_array[0]),
        "quality": float(weights_array[1]),
        "proximity": float(weights_array[2]),
        "sustainability": float(weights_array[3])
    }
    
    return {"vendor_choice_weights": weights}
```

**Expected Result:**
- Agent 1 might get: `{price: 0.45, quality: 0.15, proximity: 0.30, sustainability: 0.10}`
- Agent 2 might get: `{price: 0.10, quality: 0.20, proximity: 0.60, sustainability: 0.10}`
- Agent 3 might get: `{price: 0.20, quality: 0.40, proximity: 0.15, sustainability: 0.25}`

This would lead to:
- Agent 1 selects cheap vendors (prioritizes price)
- Agent 2 selects nearby vendors (prioritizes proximity)
- Agent 3 selects high-quality vendors (prioritizes quality)
- **Diverse vendor selection** across the population

---

## 📁 Investigation Files Generated

1. **`investigate_vendor_simple.py`** - Diagnostic script
2. **`VENDOR_SELECTION_INVESTIGATION_REPORT.md`** - This report
3. Previous report: **`VENDOR_SELECTION_BUG_REPORT.md`** (November 20, 2025)

---

## ✅ Conclusion

**The proximity matrix IS randomly decided** (as you suspected). The problem is **not** with randomization.

**The issue is:**
- All agents have identical preferences (weights)
- With identical preferences, they all rank vendors the same way
- The vendor with the best "objective" attributes wins for everyone
- Proximity variation exists but is insufficient to overcome attribute advantages

**Next step:** Await user decision on whether to implement the fix for agent-level weight variation.

---

**Investigation completed:** November 28, 2025  
**Analyzed seeds:** 42, 999  
**Analyzed agents:** 1000  
**Analyzed vendors:** 6



