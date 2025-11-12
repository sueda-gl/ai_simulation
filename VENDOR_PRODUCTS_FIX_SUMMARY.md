# Vendor Products Parameter Fix - Summary

**Date:** November 6, 2025  
**Issue:** Vendor selection decision showing very few customers making purchases  
**Status:** ✅ **FIXED**

---

## 🐛 Problem Description

When setting "Average Products per Vendor/Period" to high values (e.g., 1000) in the Page 1 UI, very few customers were able to make purchases. Even increasing the value to 1000 did not improve the purchase success rate.

### User Report
> "For vendor selection decision default I want you to investigate why only very little customers make purchases, by the values configured on the first page and the issue persists even when I increase the number of Average Products per Vendor/Period and make it 1000"

---

## 🔍 Root Cause Analysis

### The Chain of Failure

1. **UI Configuration** ✅
   - User sets `vendor_products_min = 1000`
   - User sets `vendor_products_max = 1000`
   - User sets `vendor_products_avg = 1000`
   - Values correctly stored in `st.session_state.sim_params`

2. **Parameter Passing** ❌ **BROKEN HERE**
   - In `app/simulation.py` lines 368-442, when copying UI parameters to orchestrator
   - **Vendor product parameters were completely missing!**
   - Only basic market parameters were copied (`num_vendors`, `market_price`, etc.)
   - Vendor-specific parameters (`vendor_products_min/max/avg`) were NOT copied

3. **Orchestrator Fallback** ❌
   - Orchestrator doesn't receive UI values
   - Falls back to `config/simulation.yaml` defaults:
     - `vendor_products_min: 50`
     - `vendor_products_max: 150`
   - Creates vendors with only 50-150 products instead of 1000!

4. **Vendor Selection Logic** ⚠️
   - Vendors created with 50-150 products each
   - 100 agents each wanting ~5-10 products
   - Total demand: 500-1000 products
   - **Total supply: Only 50-150 products!**
   - Result: First ~10-30 agents purchase, remaining ~70-90 fail

### Key Code Issue

In `src/decisions/vendor_selection.py` lines 142-149:

```python
for rank, (vendor_id, score) in enumerate(vendor_scores, 1):
    # Check if this vendor has enough capacity for this agent's demand
    if remaining_capacity.get(vendor_id, 0) >= agent_demand:
        selected_vendor_id = vendor_id
        vendor_rank = rank
        # Reserve capacity for this agent
        remaining_capacity[vendor_id] -= agent_demand
        break
```

**The vendor must have capacity for ALL of an agent's purchase requests.** If the vendor has insufficient capacity (50-150 instead of 1000), most agents return `allocation_failed: True`.

---

## ✅ The Fix

### File Modified: `app/simulation.py`

Added missing vendor configuration parameters to the parameter passing section (after line 406):

```python
# Vendor configuration parameters - CRITICAL for vendor generation
# Without these, orchestrator falls back to YAML defaults regardless of UI settings
orchestrator.simulation_config['simulation']['vendor_config_mode'] = sim_params.vendor_config_mode
orchestrator.simulation_config['simulation']['vendor_price_source'] = sim_params.vendor_price_source

# Vendor pricing parameters (for random generation)
orchestrator.simulation_config['simulation']['vendor_price_min'] = sim_params.vendor_price_min
orchestrator.simulation_config['simulation']['vendor_price_max'] = sim_params.vendor_price_max

# Vendor products parameters (for random generation)
# These control quantity_offered per vendor, which determines total market supply
orchestrator.simulation_config['simulation']['vendor_products_min'] = sim_params.vendor_products_min
orchestrator.simulation_config['simulation']['vendor_products_max'] = sim_params.vendor_products_max
orchestrator.simulation_config['simulation']['vendor_products_avg'] = sim_params.vendor_products_avg

# Vendor carryover parameters
orchestrator.simulation_config['simulation']['vendor_carryover_probability'] = sim_params.vendor_carryover_probability
orchestrator.simulation_config['simulation']['override_carryover'] = sim_params.override_carryover
orchestrator.simulation_config['simulation']['global_carryover'] = sim_params.global_carryover

# Vendor configuration data (if uploaded via CSV)
if hasattr(sim_params, 'vendor_config_data') and sim_params.vendor_config_data is not None:
    orchestrator.simulation_config['simulation']['vendor_config_data'] = sim_params.vendor_config_data

# Legacy vendor parameters (for backward compatibility)
orchestrator.simulation_config['simulation']['products_per_vendor'] = sim_params.products_per_vendor
orchestrator.simulation_config['simulation']['carryover'] = sim_params.carryover
if hasattr(sim_params, 'vendor_prices') and sim_params.vendor_prices is not None:
    orchestrator.simulation_config['simulation']['vendor_prices'] = sim_params.vendor_prices
```

### What This Fixes

Now when you set "Average Products per Vendor/Period" to 1000 in the UI:

1. ✅ Value is stored in session state
2. ✅ **Value is passed to orchestrator** (this was the missing link!)
3. ✅ Vendors are created with `quantity_offered` in range [vendor_products_min, vendor_products_max]
4. ✅ Market has sufficient supply for customer demand
5. ✅ Most/all customers successfully make purchases

---

## 🧪 Verification

### Test Results

Created and ran `test_vendor_products_fix.py` with:
- **Configuration**: 3 vendors with 800-1200 products each (avg 1000)
- **Simulation**: 100 agents
- **Expected Total Supply**: ~3000 products

### Results ✅

```
📦 Vendor Details:
   Vendor 1: 973 products
   Vendor 2: 837 products
   Vendor 3: 1087 products

📊 Market Supply:
   Total Supply: 2897 products
   Average per Vendor: 965.7 products

🛒 Purchase Results:
   Successful: 99/100 agents (99.0%)
   Failed (Sold Out): 1/100 agents

VALIDATION CHECKS:
✅ CHECK 1 PASSED: All vendors have quantity_offered in configured range
✅ CHECK 2 PASSED: Total supply (2897) is within expected range (2400-3600)
✅ CHECK 3 PASSED: 99.0% of agents made purchases (≥80% threshold)
```

**Before Fix**: Only ~10-30% of agents could purchase (using default 50-150 products)  
**After Fix**: 99% of agents successfully purchased (using configured 1000 products)

---

## 📝 Parameters Added

The fix ensures these UI parameters are now properly passed to the orchestrator:

### Vendor Configuration
- `vendor_config_mode` - "random" or "upload"
- `vendor_price_source` - "random" or file-based

### Vendor Pricing (Random Mode)
- `vendor_price_min` - Minimum price per vendor
- `vendor_price_max` - Maximum price per vendor

### Vendor Products (Random Mode) ⭐ **PRIMARY FIX**
- `vendor_products_min` - Minimum products per vendor
- `vendor_products_max` - Maximum products per vendor
- `vendor_products_avg` - Target average products per vendor

### Vendor Carryover
- `vendor_carryover_probability` - Probability of carryover in random mode
- `override_carryover` - Override with global setting
- `global_carryover` - Global carryover (all/none)

### Legacy Parameters (Backward Compatibility)
- `products_per_vendor` - Legacy single-vendor mode
- `carryover` - Legacy global carryover flag
- `vendor_prices` - Legacy explicit vendor prices
- `vendor_config_data` - Uploaded CSV configuration

---

## 🎯 Impact

### Before This Fix
- UI settings for vendor products were **ignored**
- Simulations always used YAML defaults (50-150 products)
- Most agents failed to purchase due to insufficient supply
- Misleading user experience (changing UI had no effect)

### After This Fix
- UI settings are **correctly applied**
- Vendors created with user-specified capacity
- Purchase success rates match expected supply/demand
- UI changes immediately affect simulation behavior

---

## 🔄 Monte Carlo Mode

**Note**: Monte Carlo simulations run via `scripts/run_mc_study.py` do not use this code path. They read directly from `config/simulation.yaml`. To configure vendor products for Monte Carlo studies, edit the YAML file directly:

```yaml
# config/simulation.yaml
simulation:
  vendor_products_min: 800
  vendor_products_max: 1200
  vendor_products_avg: 1000
```

This is by design - Monte Carlo mode is intended for command-line batch processing with YAML configuration.

---

## ✨ User Action Required

**None!** The fix is automatic. Simply:

1. Go to **Page 1: Common Simulation Parameters**
2. Set "Average Products per Vendor/Period" to your desired value (e.g., 1000)
3. Run simulation
4. Vendors will now be created with the correct capacity
5. More customers will successfully make purchases

The fix ensures your UI configuration is respected by the simulation engine.

---

## 📚 Related Files

### Modified
- `app/simulation.py` - Added vendor parameter passing (lines 408-436)

### Related (No Changes)
- `src/orchestrator.py` - Uses vendor parameters to generate vendors
- `src/vendor_attribute_generator.py` - Generates vendor attributes
- `src/decisions/vendor_selection.py` - Allocates vendors to customers
- `app/pages/page1_common_params.py` - UI for vendor configuration
- `config/simulation.yaml` - Default configuration (for Monte Carlo)

---

**Fix verified and tested successfully! 🎉**


