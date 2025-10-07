# Final Donation Rate Override Implementation

## Summary

The `final_donation_rate` decision now uses `donation_default` values when they're available, making the Excel export consistent with the UI display.

---

## What Changed

### **Modified File:** `src/decisions/final_donation_rate.py`

**Before:**
```python
def final_donation_rate(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 13: Select donation rate after transaction accepted"""
    # Always use configured default value
    default_value = get_actual_default_value("final_donation_rate")
    return {"final_donation_rate": default_value}
```

**After:**
```python
def final_donation_rate(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 13: Select donation rate after transaction accepted"""
    
    # Check if donation_default was computed for this agent
    if 'donation_default' in agent_state:
        # Use the computed donation_default value (matches UI override behavior)
        return {"final_donation_rate": agent_state['donation_default']}
    
    # No donation_default available - use configured default value
    default_value = get_actual_default_value("final_donation_rate")
    return {"final_donation_rate": default_value}
```

### **Updated UI Messages:** `app/pages/results/decision_visualizations.py`

Added clarification that Excel values now match the UI display:
```python
st.caption("✅ The final_donation_rate values in your export match the donation_default distribution shown below")
```

---

## How It Works Now

### **Scenario 1: Both `donation_default` AND `final_donation_rate` Selected**

#### **Execution:**
```python
# Step 1: donation_default runs (Decision #3)
agent_state['donation_default'] = 0.423  # Computed value

# Step 2: final_donation_rate runs (Decision #13)
# Checks: Is 'donation_default' in agent_state? YES
# Returns: {"final_donation_rate": 0.423}  # Same as donation_default!
```

#### **Results DataFrame:**
```
| agent_id | donation_default | final_donation_rate |
|----------|------------------|---------------------|
|    1     |      0.423       |      0.423          | ✅ SAME
|    2     |      0.651       |      0.651          | ✅ SAME
|    3     |      0.287       |      0.287          | ✅ SAME
```

#### **UI Display:**
```
📊 Final Donation Rate (Custom Parameters)

📊 Using Distribution from Selected Donation Configuration
✅ The final_donation_rate values in your export match the donation_default distribution shown below

Mean Rate: 42.3% | Median Rate: 40.5% | Std Dev: 12.1%
[Shows histogram of donation_default distribution]
```

#### **Excel Export:**
```excel
| donation_default | final_donation_rate |
|------------------|---------------------|
|      0.423       |      0.423          | ✅ Consistent with UI!
|      0.651       |      0.651          |
|      0.287       |      0.287          |
```

---

### **Scenario 2: Only `final_donation_rate` Selected (No `donation_default`)**

#### **Execution:**
```python
# donation_default NOT selected/run
# agent_state does NOT have 'donation_default'

# final_donation_rate runs (Decision #13)
# Checks: Is 'donation_default' in agent_state? NO
# Returns: {"final_donation_rate": 0.10}  # Uses configured default
```

#### **Results DataFrame:**
```
| agent_id | final_donation_rate |
|----------|---------------------|
|    1     |        0.10         |
|    2     |        0.10         |
|    3     |        0.10         |
```

#### **UI Display:**
```
📊 Final Donation Rate (Custom Parameters)

💡 No donation configuration selected - Using simple rate configuration

Current Rate: 10%
[Shows slider to configure rate]
```

#### **Excel Export:**
```excel
| final_donation_rate |
|---------------------|
|        0.10         | ✅ Uses configured default
|        0.10         |
|        0.10         |
```

---

### **Scenario 3: Only `donation_default` Selected (No `final_donation_rate`)**

#### **Execution:**
```python
# Only donation_default runs
agent_state['donation_default'] = 0.423

# final_donation_rate NOT selected - doesn't run
```

#### **Results DataFrame:**
```
| agent_id | donation_default |
|----------|------------------|
|    1     |      0.423       |
|    2     |      0.651       |
|    3     |      0.287       |
```

No `final_donation_rate` column exists.

---

## Key Benefits

### **1. Consistency Between UI and Excel ✅**

**Before:**
- UI showed `donation_default` distribution
- Excel had different values in `final_donation_rate` column
- Confusing!

**After:**
- UI shows `donation_default` distribution
- Excel has same values in both columns
- Consistent!

### **2. Intuitive Behavior ✅**

When both decisions are selected:
- `donation_default` computes complex, agent-specific values
- `final_donation_rate` uses those same values (not a simple default)
- Makes sense: the "default" becomes the "final" rate

### **3. Backward Compatible ✅**

If `donation_default` is NOT selected:
- `final_donation_rate` still works as before
- Uses configured default value (e.g., 10%)
- No breaking changes

---

## Decision Execution Order

Both decisions execute in this order:

```
1. disclose_income
2. disclose_documents
3. donation_default          ← Runs FIRST, creates agent_state['donation_default']
4. rejected_transaction_defaults
5. vendor_choice_weights
6. consumption_quantity
7. consumption_frequency
8. vendor_selection
9. purchase_vs_bid
10. bid_value
11. rejected_transaction_option
12. rejected_bid_value
13. final_donation_rate      ← Runs LAST, checks agent_state['donation_default']
```

**Why this order matters:**
- `final_donation_rate` can access `donation_default` because it runs later
- If order were reversed, it wouldn't work

---

## Use Cases

### **Use Case 1: Complex Donation Modeling**

**Goal:** Model donation behavior using traits + regression, then use those rates throughout simulation

**Workflow:**
1. Select `donation_default` decision (computes complex rates)
2. Select `final_donation_rate` decision
3. Run simulation
4. `final_donation_rate` automatically uses the computed `donation_default` values
5. Both UI and Excel show consistent data

### **Use Case 2: Simple Fixed Rate**

**Goal:** Use a simple fixed donation rate (10%)

**Workflow:**
1. Do NOT select `donation_default`
2. Select `final_donation_rate` decision
3. Configure default value on Page 2 Default Config
4. Run simulation
5. All agents get the same 10% rate

### **Use Case 3: Override After Selection**

**Goal:** Compute donation_default, review it, then manually set final rate

**Currently:** `final_donation_rate` always uses `donation_default` when available

**Workaround:** If you want to override:
1. Run simulation with only `donation_default`
2. Review results
3. Run new simulation with only `final_donation_rate` and your custom rate

---

## Excel Export Examples

### **Example 1: Both Decisions Selected**

```excel
Sheet: Results
| agent_id | Honesty_Humility | Allowance | donation_default | final_donation_rate |
|----------|------------------|-----------|------------------|---------------------|
|    1     |      3.45        |     3     |      0.423       |      0.423          |
|    2     |      2.87        |     5     |      0.651       |      0.651          |
|    3     |      4.12        |     2     |      0.287       |      0.287          |
|    4     |      3.21        |     4     |      0.512       |      0.512          |
|    5     |      3.89        |     3     |      0.398       |      0.398          |

Mean donation_default:       42.3%
Mean final_donation_rate:    42.3%  ✅ SAME
```

### **Example 2: Only final_donation_rate Selected**

```excel
Sheet: Results
| agent_id | Honesty_Humility | Allowance | final_donation_rate |
|----------|------------------|-----------|---------------------|
|    1     |      3.45        |     3     |        0.10         |
|    2     |      2.87        |     5     |        0.10         |
|    3     |      4.12        |     2     |        0.10         |
|    4     |      3.21        |     4     |        0.10         |
|    5     |      3.89        |     3     |        0.10         |

Mean final_donation_rate:    10.0%  ✅ Uses configured default
```

---

## Verification Steps

### **How to Verify the Change Works:**

1. **Run simulation with both decisions:**
   - Page 2: Select `donation_default` + `final_donation_rate`
   - Configure any donation settings
   - Run simulation

2. **Check UI:**
   - Go to Results page
   - Look at `final_donation_rate` section
   - Should show: "✅ The final_donation_rate values in your export match the donation_default distribution shown below"

3. **Download Excel:**
   - Click "📊 Download Excel"
   - Open file

4. **Verify in Excel:**
   ```python
   import pandas as pd
   df = pd.read_excel('enhanced_simulation_results_20251007_142315.xlsx')
   
   # Check if values match
   assert (df['donation_default'] == df['final_donation_rate']).all()
   print("✅ Values match!")
   
   # Check they're not all the same (not a simple default)
   assert df['donation_default'].nunique() > 1
   print("✅ Values are varied (not all 10%)")
   ```

---

## Technical Details

### **Decision Function Signature**

```python
def final_donation_rate(
    agent_state: dict,      # Contains results from previous decisions
    params: dict,           # Configuration from decisions.yaml
    rng,                    # Random number generator
    simulation_config: dict = None  # Global simulation config
) -> dict:
```

### **Agent State Contents When final_donation_rate Executes**

```python
agent_state = {
    # Traits (always present)
    'Honesty_Humility': 3.45,
    'Assigned Allowance Level': 3,
    'Study Program': 'CLEAM',
    'Group_experiment': 'NoSub',
    'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': 56.0,
    
    # Previous decision outputs (if those decisions ran)
    'disclose_income': True,
    'disclose_documents': False,
    'donation_default': 0.423,  # ← Check if this exists
    # ... other decisions ...
}
```

### **Return Value**

```python
# If donation_default exists in agent_state
return {"final_donation_rate": 0.423}  # Uses that value

# If donation_default NOT in agent_state
return {"final_donation_rate": 0.10}   # Uses configured default
```

---

## Edge Cases

### **Edge Case 1: donation_default is "NA" or placeholder**

```python
agent_state['donation_default'] = "NA"

# Still uses it (as configured)
return {"final_donation_rate": "NA"}
```

If you want to skip "NA" values:
```python
if 'donation_default' in agent_state and agent_state['donation_default'] != "NA":
    return {"final_donation_rate": agent_state['donation_default']}
```

### **Edge Case 2: donation_default is 0.0**

```python
agent_state['donation_default'] = 0.0

# Uses it (0% is a valid donation rate)
return {"final_donation_rate": 0.0}
```

### **Edge Case 3: Multiple decision runs**

If you run decisions individually:
1. Run `donation_default` only → creates that column
2. Later run `final_donation_rate` only → won't have access to previous run's data

**Solution:** Run both decisions in the same simulation run.

---

## Troubleshooting

### **Issue: final_donation_rate still shows 0.10 for all agents**

**Cause:** `donation_default` was not run in the same simulation

**Solution:** 
- Verify both decisions are selected on Page 2
- Run simulation with both decisions together

### **Issue: Values in Excel are different from UI charts**

**Cause:** Looking at wrong column

**Solution:**
- Use `donation_default` OR `final_donation_rate` column (they're the same now)
- Both should show the same values

### **Issue: Want to use custom final rate, not donation_default**

**Cause:** Current implementation always uses `donation_default` when available

**Solution:**
- Run only `final_donation_rate` decision (don't select `donation_default`)
- OR modify the decision to check a flag in `simulation_config`

---

## Future Enhancements

### **Option 1: Add Toggle to Disable Override**

Add a checkbox on Page 2:
```
☐ Use donation_default for final_donation_rate
```

If checked: Use `donation_default` (current behavior)
If unchecked: Use configured default even if `donation_default` exists

### **Option 2: Add Slider Override**

Allow user to scale the `donation_default` values:
```python
if 'donation_default' in agent_state:
    scale_factor = simulation_config.get('final_donation_rate_scale', 1.0)
    return {"final_donation_rate": agent_state['donation_default'] * scale_factor}
```

### **Option 3: Add Conditional Logic**

Use `donation_default` only if it's above/below a threshold:
```python
if 'donation_default' in agent_state:
    if agent_state['donation_default'] > 0.5:  # Only if above 50%
        return {"final_donation_rate": agent_state['donation_default']}
```

---

## Summary

✅ **Implemented:** `final_donation_rate` now uses `donation_default` values when available
✅ **Result:** Excel exports match UI display
✅ **Benefit:** Consistent, intuitive behavior
✅ **Backward compatible:** Still works without `donation_default`

**Key Takeaway:** When both `donation_default` and `final_donation_rate` decisions are selected, they will have identical values in the exported data, matching what the UI shows.

---

*Implementation Date: 2025-10-07*
*Modified Files:*
- `src/decisions/final_donation_rate.py`
- `app/pages/results/decision_visualizations.py`

