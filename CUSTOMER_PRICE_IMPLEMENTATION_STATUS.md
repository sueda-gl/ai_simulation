# Customer Price Implementation Status

**Date:** November 12, 2025  
**Status:** ⚠️ NOT IMPLEMENTED - Placeholder values shown with user notices

---

## 📍 **Where Customer Prices Appear:**

### **Location #1: Vendor Selection - Purchase Requests Excel Export**
**File:** `app/pages/results/visualizations/vendor_viz.py`  
**Section:** "Purchase Request Level Data Export" (around line 706-710)  
**Column:** "Customer Paid Price"

**Current Behavior:**
```python
# Line 134-136
customer_paid_price = request.get('pricePaid',        # Doesn't exist
                                 request.get('price_paid',  # Doesn't exist
                                 request.get('price', vendor_price)))  # Fallback

# Uses vendor's base price as placeholder
```

**User Notice Added:**
> ℹ️ **Note on Customer Paid Price**: The 'Customer Paid Price' column currently shows vendor base prices as placeholder values. Final customer prices will be calculated based on customer type (Discount/Fixed/Regular), platform price type (PN/BID), and pricing parameters once the pricing algorithm integration is completed.

---

### **Location #2: Donation - Transaction-Level Excel Export**
**File:** `app/pages/results/visualizations/donation_viz.py`  
**Section:** "Transaction-Level Excel" download (around line 424-434)  
**Column:** "Customer Price"

**Current Behavior:**
```python
# Lines 47-50
market_price = 100.0  # Hardcoded, ignores actual vendor price
platform_markup = 0.1
baseline_price = (1 + 0.1) * 100.0  # Always uses $100 base
discount_price = 100.0 * 0.7  # $70

# Lines 115-133
if customer_type == "discount":
    customer_price = discount_price  # $70 (regardless of vendor)
elif customer_type == "fixed":
    customer_price = fixed_price  # $100 (regardless of vendor)
elif platform_price == "PN":
    customer_price = baseline_price  # $110 (regardless of vendor)
```

**User Notice Added:**
> ℹ️ **Note on Customer Price**: The 'Customer Price' column currently uses simplified placeholder calculations. Final customer prices will be calculated from actual vendor prices and customer-specific parameters once the pricing algorithm integration is completed.

---

## 🚨 **What's Wrong with Current Implementation:**

### **Problem #1: Wrong Base Price**
- **Should use:** Actual vendor's price (e.g., Vendor 4 = $62.81, Vendor 7 = $72.72)
- **Currently uses:** Either vendor base price OR hardcoded $100
- **Result:** Prices don't reflect actual vendor selection

### **Problem #2: Simplified Calculations**
**Donation export** uses hardcoded formulas:
- Discount: $100 × 0.70 = $70 (should be: vendor_price × (1 - discount_rate))
- Fixed: $100 (should be: vendor_price × (1 + platform_markup))
- PN: $110 (should be: vendor_price × (1 + markup) × (1 + range))

**Vendor export** just copies vendor base price:
- All transactions from Vendor 4 show $62.81
- Ignores customer type completely

### **Problem #3: Missing discount_rate Parameter**
- Discount rate is hardcoded as 30% (0.7 multiplier)
- No Page 1 parameter to configure it
- Should be configurable like other pricing parameters

---

## ✅ **What's Been Done:**

1. ✅ **Added user notices** to both export locations
2. ✅ **Documented** what's missing and why
3. ✅ **Explained** placeholder behavior to users

Users will now see clear information that prices are placeholders when they download the Excel files.

---

## 🔧 **Future Implementation Needed:**

When pricing algorithm is ready for integration:

### **Step 1: Add discount_rate to Page 1**
```python
# In app/models.py SimulationParameters class
discount_rate: float = 0.20  # 20% discount

# In app/pages/page1_common_params.py
discount_rate = st.slider(
    "Discount Rate (%)",
    min_value=0.0,
    max_value=0.5,
    value=0.20,
    help="Percentage discount for discount customers"
)
```

### **Step 2: Create Price Calculation Function**
```python
# In src/decisions/pricing_calculator.py (new file)
def calculate_customer_paid_price(request, vendors_data, simulation_config):
    """Calculate actual price based on vendor, customer type, and parameters"""
    # Implementation as outlined earlier
    pass
```

### **Step 3: Call After Vendor Selection**
```python
# Option A: Add to enrich_purchase_requests.py after line 118
request['pricePaid'] = calculate_customer_paid_price(request, vendors, config)

# Option B: Create new decision module that runs after vendor_selection
# Add to orchestrator.py decision_order after 'vendor_selection'
```

### **Step 4: Update Export Functions**
```python
# Both vendor_viz.py and donation_viz.py can then use:
customer_paid_price = request.get('pricePaid', np.nan)
# Will get actual calculated value instead of fallback
```

### **Step 5: Remove User Notices**
Once implemented, remove the info messages added in this update.

---

## 📋 **Files Modified (User Notices Added):**

1. `app/pages/results/visualizations/vendor_viz.py` (line 709)
2. `app/pages/results/visualizations/donation_viz.py` (line 426)

**Status:** Users are now informed about placeholder prices ✅




