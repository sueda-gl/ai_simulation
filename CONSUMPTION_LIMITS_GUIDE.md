# Consumption Limits Configuration Guide

## 📋 Overview

This guide explains the **Consumption Limits** system in the AI Agent Simulation, including how limits relate to income categories, the per-term duration concept, and special handling for discount customers.

---

## 🎯 Key Concepts

### 1. **Fixed Income Categories (NFIC)**

**What are they?**
- Fixed Income Categories are groupings of agents based on their **annual income levels**
- These categories are **NOT** related to product prices or price categories
- Categories are ordered from **lowest income (Category 1) to highest income**

**Purpose:**
- Determine customer status (Discount vs. Fixed pricing)
- Define consumption limits per income level
- Enable income-based policy modeling

**Example:**
```
If NFIC = 10 (10 fixed income categories):
- Category 1:  Lowest income level  → Discount customers
- Category 2:  Low income
- Category 3:  Low-mid income
- ...
- Category 10: Highest income level
```

---

### 2. **Consumption Limits**

**What are they?**
- Maximum number of products an agent can consume
- Applied **per term** (not per period)
- Category-specific (different limits for different income levels)

**Important Clarifications:**

#### A. **Per Term, Not Per Period**

**Term Definition:**
```
Term = Number of Periods × Length of Period (hours)
```

**Example:**
- Number of Periods = 3
- Length of Period = 2 hours
- **Term = 3 × 2 = 6 hours total**

If consumption limit for Category 1 = 10:
- Agent can consume **maximum 10 products over the entire 6-hour simulation**
- NOT 10 products per period (which would be 30 total)

#### B. **Based on Fixed Income Categories**

Consumption limits are tied to **fixed income categories**, not:
- ❌ Product price categories
- ❌ Price grid categories
- ❌ Vendor categories
- ✅ Fixed income categories (NFIC)

#### C. **Category 1 = Lowest Income**

**Critical Understanding:**
- **Category 1** always represents the **lowest income level**
- Category 1 limits apply to **discount customers**
- Higher category numbers = higher income levels
- Categories are ordered: 1 (lowest) → 2 → 3 → ... → N (highest)

---

## 👥 Discount Customers & Category 1

### **Who are Discount Customers?**

Discount customers are agents with income levels **below the discount threshold**:
- Set in "Discount Threshold Configuration" section
- Must also disclose income and documents
- Qualify for discounted pricing

### **Consumption Limits for Discount Customers**

**Key Rule:**
> **Discount customers use consumption limits from Category 1 (lowest income)**

**Why?**
- Discount customers have the lowest incomes by definition
- Category 1 represents the lowest income bracket
- This ensures consumption limits match income levels appropriately

**Example:**
```
Discount Threshold = $15,000
Fixed Income Categories (NFIC) = 5

Category Limits:
- Category 1: 10 products (lowest income, discount customers)
- Category 2: 15 products
- Category 3: 20 products
- Category 4: 25 products
- Category 5: 30 products (highest income)

Agent Income = $12,000 (below threshold)
→ Qualifies as Discount customer
→ Uses Category 1 limit = 10 products per term
```

---

## ⚙️ Configuration Options

### **Manual Entry (Only Option)**

Configure limits directly in the UI:

**UI Display:**
```
Fixed Income Categories (ordered from lowest to highest income)

Cat 1 (Lowest/Discount)  Cat 2  Cat 3  Cat 4  Cat 5
      [10]                [15]   [20]   [25]   [30]
```

**Features:**
- ✅ Visual input fields for all categories
- ✅ Clear labeling of Category 1 as lowest/discount
- ✅ Tooltips explain each category
- ✅ Dynamic based on NFIC value
- ✅ Immediate validation
- ✅ Real-time updates

---

## 📊 Income Categories Architecture

### **Complete Income Structure**

```
Total Income Distribution
          ↓
┌─────────────────────────────────────────┐
│ All Agents (sorted by income)           │
└─────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────┐
│ Split at Discount Threshold             │
├─────────────────┬───────────────────────┤
│ Below Threshold │ Above Threshold       │
│ (Discount)      │ (Fixed Price)         │
│                 │                       │
│ NDIC Categories │ NFIC Categories       │
│ (e.g., 1-3)     │ (e.g., 1-10)          │
└─────────────────┴───────────────────────┘
          ↓                    ↓
  Discount Customers    Fixed-Price Customers
  Use Category 1        Use Categories 1-N
  consumption limit     based on income level
```

### **Category Assignment Logic**

**Step 1: Assign Income Category (ALL customers, regardless of type):**
```python
# Income range is split into N equal intervals
min_income = 0  # From distribution parameters
max_income = 100000  # From distribution parameters
interval_width = (max_income - min_income) / N

# All agents assigned based purely on income amount
if agent.income in [0, interval_width):
    income_category = 1
elif agent.income in [interval_width, 2*interval_width):
    income_category = 2
# ... and so on ...
elif agent.income in [(N-1)*interval_width, max_income]:
    income_category = N
```

**Step 2: Determine Customer Type:**
```python
# Based on disclosure decisions and threshold
if agent.income < discount_threshold and disclosed_documents:
    customer_type = "Discount"
elif disclosed_income:
    customer_type = "Fixed"
else:
    customer_type = "Regular"
```

**Step 3: Apply Consumption Limit (based on customer type):**
```python
# Customer type determines WHICH limit to use
if customer_type == "Discount":
    consumption_limit = limits[category_1]  # Lowest category
elif customer_type == "Regular":
    consumption_limit = limits[category_N]  # Highest category  
elif customer_type == "Fixed":
    consumption_limit = limits[income_category]  # Their actual category
```

---

## 💡 Examples

### **Example 1: Basic Setup**

**Parameters:**
- Number of Periods = 1
- Length of Period = 1 hour
- **Term = 1 × 1 = 1 hour**
- NFIC = 5 categories
- Discount Threshold = $20,000

**Consumption Limits:**
```
Category 1 (Lowest/Discount): 5 products per term (1h total)
Category 2: 8 products per term
Category 3: 10 products per term
Category 4: 12 products per term
Category 5: 15 products per term
```

**Agent Examples:**
```
Agent A: Income = $15,000 (below threshold)
→ Discount customer
→ Category 1 limit = 5 products max over 1 hour

Agent B: Income = $25,000 (above threshold, low within fixed range)
→ Fixed-price customer
→ Category 1 or 2 limit = 5-8 products max over 1 hour

Agent C: Income = $80,000 (above threshold, high within fixed range)
→ Fixed-price customer
→ Category 5 limit = 15 products max over 1 hour
```

### **Example 2: Multi-Period Simulation**

**Parameters:**
- Number of Periods = 4
- Length of Period = 2 hours
- **Term = 4 × 2 = 8 hours**
- NFIC = 3 categories
- Discount Threshold = $25,000

**Consumption Limits:**
```
Category 1 (Lowest/Discount): 20 products per term (8h total)
Category 2: 30 products per term
Category 3: 40 products per term
```

**Important:**
- Limits are for the **entire 8-hour simulation**
- NOT 20 products per 2-hour period (which would be 80 total)
- Agent in Category 1 can consume maximum 20 products across all 4 periods

**Agent Example:**
```
Agent D: Income = $18,000 (discount customer)
→ Category 1 limit = 20 products total

Period 1 (hours 0-2): Consumes 6 products  → Remaining: 14
Period 2 (hours 2-4): Consumes 5 products  → Remaining: 9
Period 3 (hours 4-6): Consumes 4 products  → Remaining: 5
Period 4 (hours 6-8): Consumes 5 products  → Remaining: 0
→ Total: 20 products (reached limit)
```

### **Example 3: Different Category Strategies**

**Scenario:** Policy testing with varying generosity

**Strategy A: Progressive (generous to low income)**
```
Category 1 (Discount): 25 products
Category 2: 20 products
Category 3: 15 products
Category 4: 10 products
Category 5: 5 products
```
→ Lower income = higher consumption allowance

**Strategy B: Uniform (equal for all)**
```
Category 1 (Discount): 15 products
Category 2: 15 products
Category 3: 15 products
Category 4: 15 products
Category 5: 15 products
```
→ Same limit regardless of income

**Strategy C: Proportional (higher income = higher limit)**
```
Category 1 (Discount): 5 products
Category 2: 10 products
Category 3: 15 products
Category 4: 20 products
Category 5: 25 products
```
→ Higher income = higher consumption allowance

---

## 🔍 Common Questions

### **Q: Why is it "per term" and not "per period"?**

**A:** This provides flexibility in policy design:
- Allows modeling of real-world consumption constraints
- More realistic for benefit programs (monthly, quarterly limits)
- Prevents artificial inflation of limits with multiple periods
- Simpler to understand: one limit for entire simulation

### **Q: Can discount customers have different limits than Category 1?**

**A:** No. By design:
- Discount customers ARE Category 1 (lowest income)
- This ensures consistency between income level and limits
- If you want different limits, adjust the Category 1 value

### **Q: What if I have 10 NFIC but only set limits for 5 categories?**

**A:** This will cause an error. You must provide limits for **all NFIC categories**.

**Solution:**
- Set limits for all categories (1 to NFIC)
- Or reduce NFIC to match your desired number of categories

### **Q: Can Category 1 be higher income than Category 2?**

**A:** No. Categories are **always ordered** from lowest (1) to highest (N):
- Category 1 = Lowest income level
- Category 2 = Next lowest
- ...
- Category N = Highest income level

This ordering is **built into the system** and cannot be changed.

### **Q: How do I disable limits for certain categories?**

**A:** Set the limit to a very high value (e.g., 100) or:
- Disable consumption limits entirely (radio button)
- System will ignore all category limits

---

## 🎓 Best Practices

### **1. Align Limits with Income Levels**

```
Good Practice:
- Category 1 (Lowest):  10 products ✅
- Category 2:           15 products
- Category 3 (Highest): 20 products

Poor Practice:
- Category 1 (Lowest):  50 products ❌
- Category 2:            5 products
- Category 3 (Highest):  2 products
```

### **2. Consider Term Duration**

```
Short Term (1 period × 1 hour = 1h):
→ Lower limits (5-15 products)

Long Term (10 periods × 5 hours = 50h):
→ Higher limits (50-200 products)
```

### **3. Balance Realism and Flexibility**

```
Too Restrictive:
- Category 1: 1 product (might be unrealistic)

Too Generous:
- Category 1: 1000 products (no effective constraint)

Balanced:
- Category 1: 10-20 products (depends on term duration)
```

### **4. Document Your Strategy**

When configuring limits, document:
- What policy are you testing?
- Why these specific values?
- How do they relate to the research question?

---

## 📝 Configuration Checklist

Before running simulation, verify:

- [ ] NFIC (Fixed Income Categories) set appropriately
- [ ] Discount threshold configured
- [ ] Term duration calculated (periods × hours)
- [ ] Consumption limits enabled/disabled as desired
- [ ] Limits provided for ALL categories (1 to NFIC)
- [ ] Category 1 represents lowest income (discount customers)
- [ ] Limits make sense for the term duration
- [ ] CSV format correct (if using upload)
- [ ] All required columns present in CSV
- [ ] No duplicate category IDs in CSV

---

## 🚀 Summary

**Key Takeaways:**

1. ✅ **Consumption limits are per TERM** (periods × hours), not per period
2. ✅ **Categories refer to FIXED INCOME levels**, not product prices
3. ✅ **Category 1 = Lowest Income** and applies to discount customers
4. ✅ **Higher category numbers = Higher income levels**
5. ✅ **Discount customers always use Category 1 limits**
6. ✅ **Must configure limits for ALL categories** (1 to NFIC)
7. ✅ **Categories are ordered** and cannot be reordered

**Quick Reference:**
```
Term = Number of Periods × Length of Period (hours)
Category 1 = Lowest Income (Discount Customers)
Category N = Highest Income
Limit applies to entire term, not per period
Categories based on income, not prices
```

---

## 📚 Related Documentation

- `DEFAULT_DECISIONS_FEATURE.md` - Default decision configuration
- `POPULATION_MODES.md` - Population generation modes
- `ENHANCED_APP_GUIDE.md` - Overall application guide

---

**Last Updated:** October 1, 2025  
**Status:** ✅ Current and Accurate

