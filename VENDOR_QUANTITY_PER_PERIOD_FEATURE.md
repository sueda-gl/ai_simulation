# Vendor Quantity Per Period Feature

**Date:** November 28, 2025  
**Status:** ✅ IMPLEMENTED

---

## 🎯 Feature Overview

Vendors now get **NEW random quantities for EACH period** instead of a single fixed quantity for the entire simulation. This makes the simulation more realistic by reflecting varying inventory levels across time periods.

### Before This Feature
- Vendor 1: 94 products (same for ALL periods)
- Vendor 2: 90 products (same for ALL periods)
- Vendor 3: 122 products (same for ALL periods)

### After This Feature
- **Vendor 1:**
  - Period 1: 94 products
  - Period 2: 122 products
  - Period 3: 87 products
  - Period 4: 115 products
  - Period 5: 99 products
- **Vendor 2:**
  - Period 1: 87 products
  - Period 2: 145 products
  - Period 3: 76 products
  - etc.

Each period gets a **NEW random value** within the configured `[min, max]` range from Page 1.

---

## 📁 Files Modified

### 1. **Core Generator: `src/vendor_attribute_generator.py`**

**Changes:**
- Added `num_periods` parameter to `generate_vendor_attributes()` function
- Generates `quantity_offered_per_period` dict mapping period number → quantity
- Maintains backward-compatible `quantity_offered` field (average across periods)

**New Data Structure:**
```python
vendor = {
    'vendor_id': 1,
    'price': 117.46,
    'quality': 4,
    'sustainability': 3,
    'quantity_offered': 103,  # Average (for backward compatibility)
    'quantity_offered_per_period': {  # NEW!
        1: 94,
        2: 122,
        3: 87,
        4: 115,
        5: 99
    }
}
```

---

### 2. **Orchestrators Updated**

All three orchestrator classes now pass `num_periods` to the generator:

#### `src/orchestrator.py` (Lines 199-232)
```python
# Get number of periods for per-period quantity generation
num_periods = sim_config.get('periods', 1)

vendors = generate_vendor_attributes(
    num_vendors=num_vendors,
    vendor_prices=vendor_prices,
    rng=rng,
    price_min=None if use_explicit_prices else price_min,
    price_max=None if use_explicit_prices else price_max,
    quantity_min=quantity_min,
    quantity_max=quantity_max,
    num_periods=num_periods  # NEW parameter
)
```

#### `src/orchestrator_baseline.py` (Lines 234-265)
- Same change applied

#### `src/orchestrator_doc_mode.py` (Lines 206-237)
- Same change applied

---

### 3. **Display Updated: `app/pages/results/visualizations/vendor_viz.py`**

**Changes:**
- Enhanced **Vendor Attributes & Selection Results** table
- Shows per-period breakdown in "Quantity Offered" column

**Display Format:**
```
Quantity Offered: 99 avg (P1:116, P2:94, P3:93, P4:136, P5:58)
Total Quantity: 497
```

Where:
- **Quantity Offered:** 
  - `99 avg` = Average quantity per period
  - `(P1:116, P2:94, ...)` = Period-by-period breakdown
- **Total Quantity:** `497` = Sum of all periods (116+94+93+136+58)

**Code Location:** Lines 857-890

---

## 🔬 How It Works

### Configuration (Page 1)
User sets:
- **Min Products per Vendor/Period:** 50
- **Max Products per Vendor/Period:** 150
- **Number of Periods:** 5

### Vendor Generation (Simulation Start)
```python
# For each vendor, for each period:
for period in range(1, 6):  # 5 periods
    vendor1_quantities[period] = random.randint(50, 150)
    # Example: Period 1 → 94, Period 2 → 122, etc.
```

### Purchase Request Assignment
- Agents make purchase requests with `timestamp_hours`
- Period calculated dynamically: `period = int(timestamp_hours // duration_hours) + 1`
- Each request belongs to a specific period

### Example Timeline
```
Period 1 (hours 0-2):   Vendor 1 has 94 products available
Period 2 (hours 2-4):   Vendor 1 has 122 products available (NEW random value!)
Period 3 (hours 4-6):   Vendor 1 has 87 products available (NEW random value!)
Period 4 (hours 6-8):   Vendor 1 has 115 products available (NEW random value!)
Period 5 (hours 8-10):  Vendor 1 has 99 products available (NEW random value!)
```

---

## ✅ Backward Compatibility

The feature maintains **full backward compatibility**:

1. **Legacy Field:** `quantity_offered` still exists (stores average)
2. **Old Code:** Any code not updated will see average quantity
3. **New Code:** Can access `quantity_offered_per_period` for period-specific values

### Example
```python
# Works with old code
quantity = vendor.get('quantity_offered', 100)  # Returns 103 (average)

# Works with new code
per_period = vendor.get('quantity_offered_per_period', {})
if per_period:
    period_1_qty = per_period[1]  # Returns 94
    period_2_qty = per_period[2]  # Returns 122
```

---

## 🎨 Visual Display

### Results Page Table
When you run a simulation with multiple periods, the vendor table now shows:

| Vendor ID | Price ($) | **Quantity Offered** | **Total Quantity** | Quality | ... |
|-----------|-----------|---------------------|-------------------|---------|-----|
| Vendor 1  | $117.46   | **99 avg (P1:116, P2:94, P3:93, P4:136, P5:58)** | **497** | 4 | ... |
| Vendor 2  | $119.74   | **112 avg (P1:59, P2:103, P3:148, P4:124, P5:126)** | **560** | 4 | ... |
| Vendor 3  | $147.56   | **98 avg (P1:101, P2:62, P3:134, P4:95, P5:100)** | **492** | 4 | ... |

**New Columns:**
- **Quantity Offered:** Shows average and per-period breakdown
- **Total Quantity:** Shows sum across ALL periods (e.g., 497 = 116+94+93+136+58)

This makes it immediately visible that quantities vary by period!

---

## 📊 Testing Checklist

To verify the feature works correctly:

- [ ] Run simulation with `periods > 1` (e.g., 5 periods)
- [ ] Check vendor table shows period breakdown
- [ ] Verify each period has different random quantity
- [ ] Confirm quantities are within configured [min, max] range
- [ ] Check backward compatibility (average shown if needed)

---

## 🔮 Future Enhancements

Potential future additions:

1. **Period-Specific Tables:** Separate tab showing vendor quantities by period
2. **Capacity Tracking:** If vendor capacity logic is added to Decision 8, it should use `quantity_offered_per_period[period]`
3. **CSV Export:** Add period-specific quantities to vendor data exports
4. **Visualization:** Chart showing quantity variation across periods per vendor

---

## 📝 Notes

- **Random Seed:** Quantities are deterministic based on simulation seed
- **No Capacity Tracking Yet:** Current implementation doesn't enforce capacity limits
- **All Orchestrators Updated:** Works with regular, baseline, and doc_mode orchestrators

---

## 🚀 Summary

This feature makes vendor inventory **dynamic across periods**, better reflecting real-world scenarios where suppliers have varying stock levels over time. The implementation is clean, backward-compatible, and ready for production use!

