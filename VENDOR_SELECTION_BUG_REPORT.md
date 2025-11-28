# Vendor Selection Bug Report

**Date:** November 20, 2025  
**Issue:** All agents selecting the same vendor (Vendor 4)  
**Status:** ✅ ROOT CAUSE IDENTIFIED

---

## 🔍 Problem Statement

When running simulations with more than 1 agent, **ALL agents select the same vendor** (Vendor 4), regardless of their individual characteristics or proximity scores.

**Example:** In a simulation with 10 agents and 5 vendors:
- Vendor 1: 0 agents (0%)
- Vendor 2: 0 agents (0%)
- Vendor 3: 0 agents (0%)
- **Vendor 4: 10 agents (100%)** ⚠️
- Vendor 5: 0 agents (0%)

---

## 📊 Diagnostic Results

### Generated Files
1. **`vendor_score_diagnostics.xlsx`** - Complete analysis with 3 sheets:
   - **Vendor Scores** - All 50 score calculations (10 agents × 5 vendors)
   - **Proximity Matrix** - Agent-vendor proximity scores
   - **Summary** - Vendor selection results

### Key Findings

#### 1. Vendor 4 Has Highest Average Score
```
AVERAGE COMPOSITE SCORES BY VENDOR:
Vendor 4: 0.770 (WINNER - 33% higher than next best)
Vendor 1: 0.582
Vendor 2: 0.543
Vendor 3: 0.490
Vendor 5: 0.471
```

#### 2. Vendor 4 Has Objectively Superior Attributes
```
VENDOR ATTRIBUTES:
                Raw_Price  Raw_Quality  Raw_Sustainability  Norm_Price  Norm_Sustainability
Vendor 1         $127.40           4                   3        0.238               0.50
Vendor 2         $119.74           5                   2        0.328               0.25
Vendor 3         $147.56           4                   4        0.000               0.75
Vendor 4          $62.81           4                   5        1.000               1.00 ⭐⭐
Vendor 5          $87.08           1                   5        0.714               1.00
```

**Vendor 4 has BOTH:**
- **Lowest price:** $62.81 (normalized = 1.0)
- **Highest sustainability:** 5/5 (normalized = 1.0)

#### 3. All Agents Have Identical Weights ⚠️
```
WEIGHT STATISTICS (all 10 agents):
                   Mean    Std     Min     Max
Weight_Price       0.25   0.00    0.25    0.25
Weight_Quality     0.25   0.00    0.25    0.25
Weight_Proximity   0.25   0.00    0.25    0.25
Weight_Sustain.    0.25   0.00    0.25    0.25
```

**ALL agents weight attributes equally!** No variation across agents.

#### 4. Proximity Scores DO Vary (but not enough)
```
PROXIMITY SCORE VARIATION:
                        Mean        Std        Min         Max
Vendor_1_Proximity     83.85      11.15      62.72      100.00
Vendor_2_Proximity     59.20      19.95      29.50       94.09
Vendor_3_Proximity     45.89      10.46      32.95       66.91
Vendor_4_Proximity     33.19      16.45       3.48       57.01 (LOWEST on average)
Vendor_5_Proximity     16.98      15.70       0.00       42.33
```

Proximity scores vary correctly by agent, but Vendor 4's superiority in price and sustainability overwhelms the proximity disadvantage.

---

## 🎯 ROOT CAUSES

### **ROOT CAUSE #1: No Agent-Level Weight Variation**

The `vendor_choice_weights` decision returns the **same weights for ALL agents**:

```python
# src/decisions/vendor_choice_weights.py (lines 18-27)
def vendor_choice_weights(agent_state, params, rng, simulation_config=None):
    # ...
    default_weights = {
        "price": 0.25,
        "quality": 0.25,
        "proximity": 0.25,
        "sustainability": 0.25
    }
    return {"vendor_choice_weights": default_weights}
```

**Why this happens:**
- This is a "default decision" designed for population-level preferences
- No randomization or agent-specific variation is implemented
- The RNG parameter is not used
- Comment says: "For now, since there's no custom tab for vendor_choice_weights, always use equal weights"

**Impact:**
All agents evaluate vendors using the exact same criteria, so they naturally converge on the same "best" vendor.

### **ROOT CAUSE #2: Vendor 4 is Objectively Superior**

With equal weights (0.25 each), the composite score formula is:

```
Score = 0.25 × norm_price + 0.25 × norm_quality + 
        0.25 × norm_proximity + 0.25 × norm_sustainability
```

**Vendor 4's score for a typical agent:**
- Price component: 0.25 × 1.00 = **0.250** (best)
- Quality component: 0.25 × 0.75 = **0.188**
- Sustainability component: 0.25 × 1.00 = **0.250** (best)
- Proximity component: 0.25 × 0.33 = **0.083** (worst)
- **Total: 0.771**

**Vendor 1's score for a typical agent:**
- Price component: 0.25 × 0.24 = **0.060**
- Quality component: 0.25 × 0.75 = **0.188**
- Sustainability component: 0.25 × 0.50 = **0.125**
- Proximity component: 0.25 × 0.84 = **0.210** (best)
- **Total: 0.583**

Even though Vendor 1 has much better proximity, Vendor 4's superiority in price and sustainability gives it a **32% higher score**.

---

## 🔧 Potential Solutions

### **Solution 1: Add Weight Variation Across Agents** (Recommended)

Modify `vendor_choice_weights` to generate different weight preferences for each agent:

```python
def vendor_choice_weights(agent_state, params, rng, simulation_config=None):
    # ... (keep existing config check) ...
    
    # Generate agent-specific weights using Dirichlet distribution
    # This ensures weights sum to 1.0 and vary across agents
    alpha = [2.0, 2.0, 2.0, 2.0]  # Concentration parameters (higher = less variation)
    weights_array = rng.dirichlet(alpha)
    
    weights = {
        "price": float(weights_array[0]),
        "quality": float(weights_array[1]),
        "proximity": float(weights_array[2]),
        "sustainability": float(weights_array[3])
    }
    
    return {"vendor_choice_weights": weights}
```

**Result:** Agents will have different preferences:
- Some agents prioritize price (select Vendor 4)
- Some prioritize proximity (select Vendor 1)
- Some prioritize quality (select Vendor 2)
- Natural diversity in vendor selection

### **Solution 2: Increase Weight of Proximity**

If you want proximity to matter more, adjust default weights:

```python
default_weights = {
    "price": 0.20,
    "quality": 0.20,
    "proximity": 0.40,  # Double weight for proximity
    "sustainability": 0.20
}
```

This makes nearby vendors more attractive, even if they're more expensive.

### **Solution 3: Reduce Vendor Attribute Variance**

Ensure vendors have more balanced attributes during generation so no single vendor dominates:

```python
# In vendor_attribute_generator.py
# Constrain price and sustainability ranges to prevent extreme combinations
```

---

## 📈 Expected Behavior After Fix

With agent-level weight variation, a typical simulation should show:

```
Vendor Selection Distribution:
Vendor 1: 18 agents (18%)
Vendor 2: 23 agents (23%)
Vendor 3: 12 agents (12%)
Vendor 4: 31 agents (31%)  # Still popular but not exclusive
Vendor 5: 16 agents (16%)
```

---

## 🧪 Test Cases

### Test Case 1: Verify Weight Variation
```python
# Run simulation with 100 agents
# Check that vendor_choice_weights vary across agents
# Expected: Standard deviation > 0 for all weight dimensions
```

### Test Case 2: Verify Vendor Distribution
```python
# Run simulation with 100 agents and 5 vendors
# Check vendor selection distribution
# Expected: Multiple vendors selected (not just one)
# Expected: At least 3 vendors with >10% market share
```

### Test Case 3: Verify Proximity Impact
```python
# Agent 1: proximity to Vendor 1 = 100, Vendor 4 = 0
# Agent 2: proximity to Vendor 1 = 0, Vendor 4 = 100
# Expected: Agents select different vendors based on proximity
```

---

## 📝 Configuration Note

**Current simulation config has `num_vendors: 1`**

In `/Users/suedagul/<sdg/config/simulation.yaml`:
```yaml
simulation:
  num_vendors: 1  # ⚠️ Only 1 vendor generated!
```

If you're seeing "all agents select vendor 4" in the app, the config might have been temporarily changed to 5 vendors during testing, then the randomization happened to make Vendor 4 the best.

**To test with multiple vendors in the app:**
1. Open Page 1 (Common Parameters)
2. Set "Number of Vendors" to 5
3. Run simulation
4. Check Results → Vendor Choices

---

## 📊 Files Generated

1. **`vendor_score_diagnostics.xlsx`** - Complete scoring analysis
2. **`diagnose_vendor_scores.py`** - Diagnostic script (reusable)
3. **`VENDOR_SELECTION_BUG_REPORT.md`** - This document

---

## ✅ Next Steps

1. **Decide on solution approach:**
   - Option A: Add agent-level weight variation (recommended)
   - Option B: Adjust default weight distribution
   - Option C: Both

2. **Implement the fix** in `src/decisions/vendor_choice_weights.py`

3. **Test with diagnostic script:**
   ```bash
   python diagnose_vendor_scores.py
   ```

4. **Verify in app:**
   - Set num_vendors = 5
   - Run simulation with 100+ agents
   - Check vendor distribution is diverse

5. **Update tests** to verify weight variation

---

**Analysis completed:** November 20, 2025  
**Diagnostic files:** Available in workspace root



