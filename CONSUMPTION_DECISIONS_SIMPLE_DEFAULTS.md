# Consumption Decisions - Simple Random Defaults

## ✅ Implementation Complete

Simple random default values for `consumption_quantity` and `consumption_frequency` decisions have been implemented.

**Implementation Date:** October 1, 2025  
**Status:** ✅ Ready to Use

---

## 🎯 What Was Implemented

### **Decision 6: `consumption_quantity`**

**Simple Random Logic:**
```python
consumption_quantity = random integer in [0, consumption_limit]
```

**How it works:**
1. Get consumption limit for agent's income category
2. Generate random integer from 0 to limit
3. Return the quantity

**Example:**
- Consumption limit for Category 1 = 10 products per term
- Agent gets random quantity: could be 0, 1, 2, ..., 9, or 10

---

### **Decision 7: `consumption_frequency`**

**Calculated Formula:**
```python
consumption_frequency = consumption_quantity / term_duration
```

Where: `term_duration = periods × duration_hours`

**How it works:**
1. Get consumption_quantity from Decision 6
2. Get term duration from simulation config
3. Calculate frequency (units per hour)
4. Return the frequency

**Example:**
- consumption_quantity = 10 products
- periods = 1
- duration_hours = 1
- term_duration = 1 × 1 = 1 hour
- **consumption_frequency = 10 / 1 = 10.0 units per hour**

---

## 📁 Files Modified

### 1. **`src/decisions/consumption_quantity.py`** ✅

**Implementation:**
- Reads consumption limit from `simulation_config['consumption_limits']`
- Uses agent's `income_category` (defaults to 1 if not set)
- Generates random integer with `rng.integers(0, limit + 1)`
- Returns integer value

**Key Features:**
- ✅ Uses agent-specific RNG for reproducibility
- ✅ Respects consumption limits
- ✅ Fallback to default limit (50) if not configured
- ✅ Returns integer (can't consume 3.5 products)

### 2. **`src/decisions/consumption_frequency.py`** ✅

**Implementation:**
- Gets `consumption_quantity` from agent_state
- Gets `periods` and `duration_hours` from simulation_config
- Calculates frequency = quantity / (periods × hours)
- Returns float value (units per hour)

**Key Features:**
- ✅ Simple calculation (no randomness)
- ✅ Depends on Decision 6 output
- ✅ Uses term duration (not period)
- ✅ Returns 0.0 if no consumption

### 3. **`app/simulation.py`** ✅

**Added code to pass consumption limits:**
```python
# Pass consumption limits to orchestrator if enabled
if st.session_state.sim_params.apply_consumption_limits:
    orchestrator.simulation_config['consumption_limits'] = st.session_state.sim_params.consumption_limits
```

**Lines:** 283-290

---

## 🔄 Execution Flow

```
Simulation starts
      ↓
User configures consumption limits on Page 1:
  - Cat 1 (Lowest/Discount): 10 products
  - Cat 2: 15 products
  - Cat 3: 20 products
      ↓
Limits passed to orchestrator via simulation_config
      ↓
Agent reaches Decision 6: consumption_quantity
      ↓
Function reads agent's income_category (defaults to 1)
  → Gets limit for that category
  → Example: Category 1 → limit = 10
      ↓
Generates random quantity: rng.integers(0, 11)
  → Example result: 7 products
      ↓
Returns {"consumption_quantity": 7}
      ↓
Agent state updated: agent_state['consumption_quantity'] = 7
      ↓
Agent reaches Decision 7: consumption_frequency
      ↓
Function reads consumption_quantity from agent_state
  → quantity = 7
Function reads term duration from simulation_config
  → periods = 1, duration_hours = 1
  → term_duration = 1 hour
      ↓
Calculates frequency: 7 / 1 = 7.0
      ↓
Returns {"consumption_frequency": 7.0}
      ↓
Agent state updated: agent_state['consumption_frequency'] = 7.0
      ↓
Simulation continues with other decisions...
```

---

## 📊 Example Results

### **Example 1: Basic Case**

**Configuration:**
- Consumption limit (Cat 1): 10 products
- Term: 1 period × 1 hour = 1 hour

**Agent Results:**
```
Agent 1: quantity=7,  frequency=7.0 units/hour
Agent 2: quantity=3,  frequency=3.0 units/hour
Agent 3: quantity=10, frequency=10.0 units/hour
Agent 4: quantity=0,  frequency=0.0 units/hour
Agent 5: quantity=5,  frequency=5.0 units/hour
```

### **Example 2: Multi-Period**

**Configuration:**
- Consumption limit (Cat 1): 20 products
- Term: 4 periods × 2 hours = 8 hours

**Agent Results:**
```
Agent 1: quantity=15, frequency=15/8=1.875 units/hour
Agent 2: quantity=8,  frequency=8/8=1.0 units/hour
Agent 3: quantity=20, frequency=20/8=2.5 units/hour
Agent 4: quantity=12, frequency=12/8=1.5 units/hour
```

**Interpretation:**
- Agent 1 consumes 1.875 products per hour on average
- Over 8 hours, they consume 15 products total
- Frequency tells you the rate, not the total

---

## 🎓 Understanding the Values

### **Consumption Quantity**
- **What it means:** Total products consumed over the entire term
- **Units:** Integer number of products
- **Range:** 0 to consumption_limit
- **Distribution:** Uniform random

### **Consumption Frequency**
- **What it means:** Average consumption rate per hour
- **Units:** Products per hour (float)
- **Calculation:** Total quantity / Total hours
- **Example:** 10 products / 2 hours = 5.0 products/hour

---

## ⚙️ How to Use

### **Step 1: Configure Consumption Limits (Page 1)**

```
🛒 Consumption Limits
◉ Apply Consumption Limits? ● Yes

Cat 1 (Lowest/Discount): [10]
Cat 2: [15]
Cat 3: [20]
```

### **Step 2: Run Simulation**

From anywhere:
- Sidebar → "🚀 Run Simulation"
- Overview tab → "🚀 Run Complete Simulation"  
- Any decision tab → "🎯 Run Complete Simulation"

### **Step 3: View Results**

Results will show:
```
Agent | income_category | consumption_quantity | consumption_frequency
------|-----------------|---------------------|----------------------
1     | 1               | 7                   | 7.0
2     | 1               | 3                   | 3.0
3     | 2               | 12                  | 12.0
4     | 2               | 8                   | 8.0
```

---

## 🚀 Future Enhancements

This is the **simple default** implementation. For custom/sophisticated modeling, you can add:

### **Multi-Stage Model (Per Dovev's Explanation)**

1. **Total consumption** - Random within limit
2. **Purchase frequency** - Random number of purchases
3. **Base quantity** - Total / Frequency
4. **Stochastic variation** - Each purchase varies around base

### **Factors to Consider**

- Income level influence on consumption
- Product type preferences
- Seasonal variations
- Agent personality traits (Honesty_Humility, etc.)
- Budget constraints

---

## ✅ Summary

**What's implemented:**

✅ **consumption_quantity**: Random integer in [0, limit]  
✅ **consumption_frequency**: Calculated as quantity / term_duration  
✅ **Consumption limits**: Passed from UI to orchestrator  
✅ **Income categories**: Proper fallback handling  
✅ **No linting errors**: Clean, production-ready code  

**How to use it:**

1. Enable consumption limits on Page 1
2. Set limits for each income category
3. Run simulation
4. View results showing random quantities and calculated frequencies

**Status:** ✅ **Ready to use with simple random defaults!** 🎉

