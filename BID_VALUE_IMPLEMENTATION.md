# Bid Value Implementation - Random Selection from Range

## ✅ Implementation Complete

The `bid_value` decision now generates **random bid amounts** uniformly selected from a calculated range based on pricing parameters.

**Implementation Date:** October 1, 2025  
**Status:** ✅ Fully Implemented and Tested

---

## 🎯 What Was Implemented

### Core Feature
When an agent chooses to bid (rather than purchase immediately), the system now generates a random bid amount using the formula:

```
Baseline Price (Pc) = (1 + platform_markup) × vendor_price
Min Bid (Pmb) = (1 - price_range) × Pc
Max Bid (Ppn) = (1 + price_range) × Pc
Bid Value = Uniform random in [Pmb, Ppn]
```

### Example Calculation

**With default parameters:**
- Vendor Price (Pv) = €100.00
- Platform Markup (m) = 10% (0.1)
- Price Range (r) = 25% (0.25)

**Calculation:**
1. Baseline Price: Pc = (1 + 0.1) × 100 = €110.00
2. Min Bid: Pmb = (1 - 0.25) × 110 = €82.50
3. Max Bid: Ppn = (1 + 0.25) × 110 = €137.50
4. **Random Bid**: Uniform random from [€82.50, €137.50]

**Possible bid values:** €82.50, €95.23, €110.00, €125.47, €137.49, etc.

---

## 📁 Files Modified

### 1. ✅ `src/decisions/bid_value.py` (Main Implementation)

**Changes:**
- Replaced placeholder logic with actual random bid generation
- Implemented the bidding range formula
- Uses `rng.uniform()` for random selection
- Checks if agent chose to bid (returns "NA" if purchased)
- Rounds to 2 decimal places for currency

**Key Features:**
- ✅ Reads pricing parameters from `simulation_config`
- ✅ Uses agent-specific RNG for reproducibility
- ✅ Handles fallback to default values
- ✅ Returns "NA" when agent chose purchase option
- ✅ Properly documented with docstrings

**Lines:** 53 lines (increased from 13)

### 2. ✅ `config/decisions.yaml` (Configuration)

**Changes:**
- Updated `bid_value` configuration
- Changed `placeholder: true` → `placeholder: false`
- Added `uses_global_parameters` list
- Documented formula components
- Added distribution type (uniform)

**Lines:** 116-124 (modified)

### 3. ✅ `app/pages/decision_tabs/bid_value_tab.py` (New File)

**Purpose:** Dedicated UI tab for bid_value configuration

**Features:**
- 📐 Shows mathematical formula with LaTeX
- 🧮 Displays current parameters and calculated range
- 📊 Step-by-step example calculation
- ⚙️ Explains behavior and dependencies
- 🌐 Lists global parameters used
- 🔧 Technical implementation details
- 🚀 Simulation buttons (individual + complete)

**Lines:** 223 lines (new file)

### 4. ✅ `app/pages/decision_tabs/__init__.py` (Integration)

**Changes:**
- Added bid_value case to `render_decision_tab()`
- Imports and uses `render_bid_value_tab()`

**Lines:** 20-23 (added)

---

## 🏗️ Architecture

### Data Flow

```
Simulation Start
       ↓
Agent reaches purchase_vs_bid decision
       ↓
Agent chooses "bid" (50% probability by default)
       ↓
Agent reaches bid_value decision
       ↓
bid_value() function called with:
  - agent_state (contains purchase_vs_bid = "bid")
  - params (from decisions.yaml)
  - rng (agent-specific random number generator)
  - simulation_config (contains pricing parameters)
       ↓
Extract pricing parameters:
  - market_price from simulation_config['simulation']['market_price']
  - platform_markup from simulation_config['simulation']['platform_markup']
  - price_range from simulation_config['simulation']['price_range']
       ↓
Calculate bidding range:
  baseline_price = (1 + platform_markup) × market_price
  min_bid_price = (1 - price_range) × baseline_price
  max_bid_price = (1 + price_range) × baseline_price
       ↓
Generate random bid:
  bid_amount = rng.uniform(min_bid_price, max_bid_price)
       ↓
Round to 2 decimals:
  bid_amount = round(bid_amount, 2)
       ↓
Return:
  {"bid_value": bid_amount}  # e.g., {"bid_value": 115.73}
```

---

## 🔍 Technical Details

### Random Number Generation

**Agent-Specific RNG:**
- Each agent has their own RNG seeded from the global RNG
- Ensures reproducibility with the same simulation seed
- Different agents get different random bids
- Same agent with same seed gets same bid

**Distribution:**
- Uses `numpy.random.Generator.uniform(low, high)`
- Generates from continuous uniform distribution over [low, high)
- Equal probability for any value in the range

**Example Code:**
```python
# From src/orchestrator.py
rng_global = np.random.default_rng(seed)  # Global RNG with simulation seed

for idx, row in agents_df.iterrows():
    # Each agent gets child RNG
    agent_rng = np.random.default_rng(rng_global.integers(1e9))
    
    # Agent's bid is generated using agent_rng
    bid_amount = agent_rng.uniform(min_bid_price, max_bid_price)
```

### Parameter Sources

**Pricing parameters come from `config/simulation.yaml`:**

```yaml
simulation:
  market_price: 100.0           # Vendor price (Pv)
  platform_markup: 0.1          # 10% markup (m)
  price_range: 0.25             # 25% range (r)
```

**These are configurable in Page 1 of the UI:**
- Market Parameters section
- Sliders for platform_markup and price_range
- Number input for market_price

---

## 🎨 User Interface

### Bid Value Tab Layout

When user selects "bid_value" on Page 2:

```
┌─────────────────────────────────────────────────────────┐
│  🎯 Bid Value Configuration                             │
├─────────────────────────────────────────────────────────┤
│  ℹ️ When an agent chooses to bid, this decision         │
│     determines the bid amount they submit               │
├─────────────────────────────────────────────────────────┤
│  📐 Bidding Range Formula                               │
├──────────────────────────┬──────────────────────────────┤
│  Mathematical Formula:   │  Current Parameters:         │
│  • Pc = (1+m) × Pv      │  • Vendor Price: €100.00     │
│  • Pmb = (1-r) × Pc     │  • Platform Markup: 10.0%    │
│  • Ppn = (1+r) × Pc     │  • Price Range: 25.0%        │
│                          │  ─────────────────────────   │
│  Where:                  │  Calculated Range:           │
│  • Pv = Vendor price    │  • Baseline: €110.00         │
│  • m = Platform markup   │  • Min Bid: €82.50           │
│  • r = Price range       │  • Max Bid: €137.50          │
│  • Pc = Baseline price   │  ✅ Range: [€82.50, €137.50] │
├──────────────────────────┴──────────────────────────────┤
│  🧮 Example Calculation [Expandable]                    │
│  ⚙️ Bid Selection Behavior                              │
│  🌐 Global Parameters Used                              │
│  🔧 Technical Implementation [Expandable]               │
├─────────────────────────────────────────────────────────┤
│  🚀 Simulation Options                                  │
│  🔬 Run Bid Value Only │ 🎯 Run Complete Simulation    │
└─────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing Scenarios

### Test 1: Agent Chose to Bid

**Setup:**
- Agent's `purchase_vs_bid` decision = "bid"
- market_price = €100.00
- platform_markup = 0.1
- price_range = 0.25

**Expected Result:**
```python
{
    "bid_value": 107.35  # Random value in [€82.50, €137.50]
}
```

**Verification:**
- ✅ bid_value is a float
- ✅ 82.50 ≤ bid_value ≤ 137.50
- ✅ Rounded to 2 decimal places

### Test 2: Agent Chose to Purchase

**Setup:**
- Agent's `purchase_vs_bid` decision = "purchase"
- (pricing parameters don't matter)

**Expected Result:**
```python
{
    "bid_value": "NA"
}
```

**Verification:**
- ✅ bid_value is string "NA"
- ✅ No calculation performed

### Test 3: Different Parameters

**Setup:**
- market_price = €50.00
- platform_markup = 0.2 (20%)
- price_range = 0.5 (50%)

**Expected Calculation:**
- Baseline: (1 + 0.2) × 50 = €60.00
- Min Bid: (1 - 0.5) × 60 = €30.00
- Max Bid: (1 + 0.5) × 60 = €90.00

**Expected Result:**
```python
{
    "bid_value": 67.42  # Random value in [€30.00, €90.00]
}
```

### Test 4: Reproducibility

**Setup:**
- Run simulation with seed = 42
- Same agent configuration
- Same parameters

**Expected Result:**
- ✅ Same agent gets same bid_value in both runs
- ✅ Different agents get different bid_values
- ✅ Changing seed changes bid_values

---

## 🔄 Integration with Other Decisions

### Dependencies

**Upstream Decisions** (must run first):
1. **purchase_vs_bid** (Decision 9)
   - Determines if agent bids or purchases
   - bid_value only runs if agent chose "bid"

**Downstream Decisions** (use bid_value output):
- **rejected_transaction_option** (Decision 11)
- **rejected_bid_value** (Decision 12)
- May use bid_value for transaction logic

### Decision Sequence

```
Decision 9: purchase_vs_bid → "bid" or "purchase"
                ↓
Decision 10: bid_value → [€82.50, €137.50] or "NA"
                ↓
Decision 11: rejected_transaction_option
                ↓
Decision 12: rejected_bid_value
```

---

## 📊 Results Visualization

### In Results Page

The `render_bid_value()` function in `app/pages/results/decision_visualizations.py` displays:

1. **Current Parameters:**
   - Total Agents
   - Vendor Price
   - Baseline Price (Pc)
   - Range Parameter (r)

2. **Bidding Range Display:**
   - Formula components
   - Step-by-step calculations
   - Visual range: [€82.50, €137.50]

3. **Configuration:**
   - Default behavior explanation
   - Example bids

**Note:** The visualization already exists and will now show actual bid values instead of "RANDOM_WITHIN_RANGE" placeholder.

---

## ⚙️ Configuration Options

### Page 1: Common Simulation Parameters

Users can adjust:

1. **Market Price** (Vendor Price, Pv)
   - Default: €100.00
   - Range: €50.00 - €150.00
   - Location: Market Configuration section

2. **Platform Markup** (m)
   - Default: 10% (0.1)
   - Range: 0% - 50%
   - Location: Market Parameters section
   - Slider control

3. **Price Range** (r)
   - Default: 25% (0.25)
   - Range: 0% - 100%
   - Location: Market Parameters section
   - Slider control

### Page 2: Bid Value Tab

Users can:
- ✅ View current formula and parameters
- ✅ See calculated bidding range
- ✅ Understand behavior and logic
- ✅ Run individual bid_value test
- ✅ Run complete simulation
- ⚪ (Future) Configure distribution type
- ⚪ (Future) Set bid strategy (aggressive/conservative)

---

## 🚀 Future Enhancements

Possible future improvements:

### 1. Distribution Types
Allow users to select different distributions:
- **Uniform** (current): Equal probability
- **Normal**: Clustered around midpoint
- **Beta**: Skewed toward min or max
- **Triangular**: Peak at a specific point

### 2. Bidding Strategies
Add agent personality-based strategies:
- **Conservative**: Bias toward lower bids
- **Aggressive**: Bias toward higher bids
- **Strategic**: Based on agent traits (Honesty_Humility, etc.)

### 3. Dynamic Range
Calculate range based on:
- Agent's income level
- Product category
- Market conditions
- Historical bid success

### 4. Bid Optimization
Machine learning-based bidding:
- Learn from successful bids
- Adapt to market competition
- Maximize win rate vs. price

---

## 📋 Validation Checklist

### Implementation Quality
- ✅ Code follows existing patterns
- ✅ Proper error handling with fallbacks
- ✅ Comprehensive docstrings
- ✅ Type hints in function signature
- ✅ No linting errors
- ✅ Passes all orchestrators

### Functionality
- ✅ Correct formula implementation
- ✅ Proper parameter extraction
- ✅ Random number generation works
- ✅ Handles "purchase" vs "bid" cases
- ✅ Returns correct data types
- ✅ Rounds to 2 decimal places

### Integration
- ✅ Works with all orchestrators
- ✅ Uses simulation_config correctly
- ✅ Respects agent RNG
- ✅ Maintains state properly
- ✅ Compatible with existing decisions

### User Experience
- ✅ Clear UI with formula display
- ✅ Shows current parameters
- ✅ Explains behavior
- ✅ Provides examples
- ✅ Simulation buttons available

### Documentation
- ✅ Implementation guide created
- ✅ Technical details documented
- ✅ Testing scenarios provided
- ✅ Integration explained
- ✅ Future enhancements listed

---

## 🎓 Usage Guide

### For End Users

**To generate random bid values:**

1. **Configure Parameters** (Page 1):
   - Go to "Common Simulation Parameters"
   - Adjust Market Parameters:
     - Market Price (vendor price)
     - Platform Markup
     - Price Range
   
2. **Select Decision** (Page 2):
   - Navigate to "Decision-Specific Parameters"
   - Select "bid_value" from the decision list
   - View the bid_value tab

3. **Review Configuration**:
   - Check the calculated bidding range
   - Verify parameters are correct
   - Read the formula explanation

4. **Run Simulation**:
   - Click "🔬 Run Bid Value Only" (test)
   - Or click "🎯 Run Complete Simulation" (full)

5. **View Results** (Results page):
   - See actual bid values generated
   - Analyze distribution
   - Compare with expected range

### For Developers

**To modify bid selection logic:**

1. Edit `src/decisions/bid_value.py`
2. Modify the random generation section (lines 46-51)
3. Add custom distribution or strategy
4. Test with different seeds
5. Update UI tab if needed

**To add new parameters:**

1. Add to `config/simulation.yaml`
2. Update extraction in `bid_value()`
3. Add to UI display in `bid_value_tab.py`
4. Update documentation

---

## 🐛 Troubleshooting

### Issue: All bid values are "NA"
**Cause:** All agents chose "purchase" instead of "bid"  
**Solution:** Check `purchase_vs_bid` decision configuration

### Issue: Bid values outside expected range
**Cause:** Parameters not loading correctly  
**Solution:** Verify `simulation_config` contains correct values

### Issue: Same bid for all agents
**Cause:** RNG not working properly  
**Solution:** Check agent_rng is unique per agent

### Issue: Bid values not showing in results
**Cause:** Decision not in selected_decisions list  
**Solution:** Ensure bid_value is selected or run complete simulation

---

## ✅ Summary

The bid_value decision is now **fully implemented** with:

- ✅ **Correct formula** using pricing parameters
- ✅ **Random selection** from calculated range
- ✅ **Proper integration** with all orchestrators
- ✅ **Beautiful UI** with formula visualization
- ✅ **Comprehensive documentation**
- ✅ **Zero linting errors**
- ✅ **Ready for production use**

**Before:** Returns placeholder string "RANDOM_WITHIN_RANGE"  
**After:** Returns actual random bid amount like €115.73

**The implementation is complete, tested, and ready to use!** 🎉

