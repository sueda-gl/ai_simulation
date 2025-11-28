# Critical Bugs Found - November 12, 2025

## 🚨 **BUG #1: Customer Paid Price is NOT Being Calculated**

### **Severity:** CRITICAL
### **Impact:** Excel exports show constant price (62.81136) for ALL transactions

### **Root Cause:**

The simulation creates purchase requests but **NEVER calculates the actual dollar amount** customers pay.

**Evidence from Excel:**
- Column O "Customer Paid Price": **62.81136** for EVERY transaction
- This is Vendor 4's base price - being used as fallback

**Code Flow:**

1. **`consumption_quantity.py` (lines 284-297):**
   ```python
   purchase_requests.append({
       "platformPrice": None,  # ❌ Will be "DISCOUNT", "FIXED", "PN", or "BID" (string)
       "bid_value": None       # ❌ Will be bid amount or "N/A" (not the final price)
       # ⚠️ NO 'pricePaid' or 'customer_paid_price' field created!
   })
   ```

2. **`enrich_purchase_requests.py` (lines 115-118):**
   ```python
   request['platformPrice'] = "PN"  # ❌ Sets STRING, not dollar amount
   request['bid_value'] = 85.50     # ❌ Only for BID, not actual paid price
   # ⚠️ NEVER calculates final customer paid price!
   ```

3. **`vendor_viz.py` Excel export (lines 128-130):**
   ```python
   customer_paid_price = request.get('pricePaid',        # ❌ Doesn't exist
                                    request.get('price_paid',  # ❌ Doesn't exist
                                    request.get('price', vendor_price)))  # ✅ Fallback to vendor price
   ```

### **What SHOULD Happen:**

After vendor selection and enrichment, calculate actual price based on customer type:

```python
# Pseudo-code for what's missing:

if customer_type == "discount":
    customer_paid_price = vendor_price * (1 - discount_rate)  # e.g., 20% off
    
elif customer_type == "fixed":
    customer_paid_price = vendor_price * (1 + platform_markup)  # Fixed markup
    
elif customer_type == "regular":
    if platformPrice == "PN":
        customer_paid_price = vendor_price * (1 + platform_markup) * (1 + price_range)
    elif platformPrice == "BID":
        customer_paid_price = bid_value  # Use actual bid
```

---

## 🚨 **BUG #2: Configuration Settings May Not Be Applied**

### **Severity:** HIGH
### **Impact:** Consumption limits and vendor settings may not be respected

### **Configuration Flow Validation Needed:**

**Page 1 Settings → sim_params → simulation_config → Decision Modules**

From your Excel, I need to check:

#### **A. Consumption Limits**
**Default (from simulation.yaml line 94):** `apply_consumption_limits: false`

**If disabled:**
- Fallback: `max_purchases_per_term = 50` (line 97)
- BUT: Your agents have 24,519 purchases from 4 agents = 6,129/agent!
- This exceeds 50 by **122x**! 

**Possible issues:**
1. Limits are disabled → fallback should be 50
2. If limits ARE enabled → need to check configured values
3. OR there's a bug where `consumption_limit` is being ignored

#### **B. Vendor Configuration**
**Your Excel shows:**
- All transactions use **Vendor 4**
- Vendor 4 price: **$62.81136**

**From Page 1 defaults:**
- `num_vendors: 1` (line 27)
- `vendor_price_min: 50.0` (line 29)
- `vendor_price_max: 150.0` (line 30)
- `vendor_config_mode: "random"` (line 37)

**Questions:**
1. How many vendors did you configure?
2. What are the actual vendor prices?
3. Are there supposed to be 10 vendors (as shown in your UI screenshot)?

#### **C. Periods & Duration**
**From Excel:** All transactions show "Period 1, H..." (Period 1)

**From defaults:**
- `periods: 1` (line 23)
- `duration_hours: 1.0` (line 24)
- **Term = 1 period × 1 hour = 1 hour total**

**With 6,129 purchases in 1 hour = 102 purchases per minute!**

This suggests either:
- Multiple periods are configured (not showing in Excel)
- OR extreme consumption quantities being generated

---

## 🔍 **DIAGNOSTIC STEPS NEEDED**

### **Step 1: Check Your Current Page 1 Configuration**

Go to Page 1 and tell me:

1. **Time Configuration:**
   - Number of Periods: ___?
   - Length of Period (hours): ___?
   - **Term Duration = Periods × Hours = ___?**

2. **Consumption Limits:**
   - Are limits **enabled** or **disabled**?
   - If enabled: What are the limits for cat_1 through cat_10?
   - If disabled: What is the Artificial Limit value?

3. **Vendor Configuration:**
   - Number of Vendors (N): ___?
   - Vendor Config Mode: Random or Upload?
   - If Random:
     - Min Price: ___?
     - Max Price: ___?
     - Min Products: ___?
     - Max Products: ___?

4. **Number of Agents:**
   - Total agents in simulation: ___?

### **Step 2: Verify Data Consistency**

Run this query on your Excel:

```excel
# Count agents by vendor:
=COUNTIF(G:G, "Vendor 1")  # Should match "4 agents" from UI

# Count total purchases by vendor:
=COUNTIF(G:G, "Vendor 1")  # Should match "24,519 purchases" from UI

# Check if any agent has >100 purchases:
# Create pivot table: Agent ID (rows) → Count of Transaction ID (values)
```

---

## ✅ **FIXES REQUIRED**

### **Fix #1: Add Price Calculation to Simulation**

Need to create a new decision or enhance existing decision to calculate `customer_paid_price`:

**Location:** Create new function in `src/decisions/enrich_purchase_requests.py` or separate module

```python
def calculate_customer_paid_price(request, agent_state, vendors_data, simulation_config):
    """
    Calculate the actual dollar amount customer pays for this request.
    
    Args:
        request: Purchase request dict with vendorID, platformPrice, bid_value
        agent_state: Agent state with customer_type
        vendors_data: List of vendor dicts with prices
        simulation_config: Global config with markup/discount parameters
    
    Returns:
        float: Actual price customer pays (rounded to 2 decimals)
    """
    # Get vendor price
    vendor_id = request.get('vendorID')
    vendor = next((v for v in vendors_data if v['vendor_id'] == vendor_id), None)
    
    if not vendor:
        return np.nan
    
    vendor_price = vendor['price']
    customer_type = request.get('customer_type', agent_state.get('customer_type'))
    platform_price_type = request.get('platformPrice')
    
    # Calculate based on customer type
    if customer_type == "discount":
        # Discount customers get reduced price
        discount_rate = simulation_config.get('discount_rate', 0.20)  # 20% off
        return round(vendor_price * (1 - discount_rate), 2)
    
    elif customer_type == "fixed":
        # Fixed customers get fixed markup price
        platform_markup = simulation_config.get('platform_markup', 0.10)
        return round(vendor_price * (1 + platform_markup), 2)
    
    elif customer_type == "regular":
        if platform_price_type == "BID":
            # Use their bid value
            bid_value = request.get('bid_value')
            if isinstance(bid_value, (int, float)):
                return round(bid_value, 2)
        
        # Purchase Now price
        platform_markup = simulation_config.get('platform_markup', 0.10)
        price_range = simulation_config.get('price_range', 0.25)
        baseline = vendor_price * (1 + platform_markup)
        return round(baseline * (1 + price_range), 2)
    
    return vendor_price  # Fallback
```

### **Fix #2: Verify Consumption Limits Are Applied**

Check `consumption_quantity.py` line 217-250 - consumption limit logic looks correct, but need to verify:

1. Is `consumption_limits` actually populated in `simulation_config`?
2. Is `apply_consumption_limits` being checked correctly?
3. Is the fallback `max_purchases_per_term` being used when limits disabled?

---

## 📋 **ACTION ITEMS**

1. ✅ **Document created** - this file
2. ⏳ **Need from you:** Your actual Page 1 configuration values
3. ⏳ **Implement:** Price calculation fix
4. ⏳ **Verify:** Consumption limits are being applied correctly
5. ⏳ **Test:** Re-run simulation and verify Excel export shows variable prices

---

**Created:** November 12, 2025  
**Status:** Awaiting configuration information to proceed with fixes







