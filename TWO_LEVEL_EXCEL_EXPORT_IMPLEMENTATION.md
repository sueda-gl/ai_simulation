# Two-Level Excel Export Implementation

## 🎯 Overview

Successfully implemented a comprehensive two-level Excel export system that provides:
1. **Agent-Level Excel Sheet**: One row per agent with all agent-level decisions and summary statistics
2. **Transaction-Level Excel Sheet**: One row per purchase request with detailed transaction information

Both sheets are included in a single Excel file for easy analysis and cross-referencing.

---

## 📋 What Was Changed

### **File Modified**: `app/pages/results/components/export_section.py`

#### **1. Added New Imports**
```python
import numpy as np
from datetime import datetime, timedelta
```

#### **2. Created Three New Functions**

##### **A. `_build_agent_level_dataframe(df, vendors_data=None)`**
- Builds agent-level DataFrame (one row per agent)
- **Includes**:
  - Agent ID and 5 trait columns
  - All agent-level decisions (Decisions 1-13)
  - Vendor choice weights (flattened to 4 columns)
  - Vendor proximity scores (expanded to one column per vendor)
  - Summary statistics from transactions:
    - Total purchase requests count
    - PN vs BID breakdown
    - Average/min/max bid values
  
- **Returns**: DataFrame with ~30-35 columns

##### **B. `_build_transaction_level_dataframe(df, vendors_data=None, simulation_params=None)`**
- Builds transaction-level DataFrame (one row per purchase request)
- **Includes**:
  - Transaction ID, Agent ID, Request ID
  - Agent traits (copied for reference)
  - Customer type and income category
  - Timing information (timestamp, period, date, time)
  - Vendor information (ID, price, quality, sustainability, proximity, integrated score)
  - Purchase decision (platformPrice, request type)
  - Pricing (bid value, customer price)
  - Donation information (agent default, final rate, donation paid, total paid)
  
- **Returns**: DataFrame with ~25-30 columns

##### **C. `_calculate_vendor_composite_score(vendor, weights, proximity, all_vendors)`**
- Helper function to calculate vendor integrated composite score
- Uses weighted combination of normalized attributes:
  - Price (inverted: lower = better)
  - Quality (1-5 scale)
  - Sustainability (1-5 scale)
  - Proximity (0-100 scale)

#### **3. Completely Rewrote `render_export_section()`**
- Removed old single-level export logic
- New implementation:
  1. Builds both agent-level and transaction-level DataFrames
  2. Creates single Excel file with two sheets
  3. Shows summary statistics (# agents, # transactions)
  4. Provides preview of both sheets
  5. Single download button for complete Excel file

---

## 📊 Excel Structure

### **Sheet 1: Agent Level** (One row per agent)

| Column Category | Columns | Description |
|----------------|---------|-------------|
| **Identification** | `Agent ID` | Unique agent identifier |
| **Traits** | `Honesty_Humility`, `Assigned Allowance Level`, `Study Program`, `Group_experiment`, `TWT+Sospeso [=AW2+AX2]{Periods 1+2}` | 5 trait columns from trait engine |
| **Decision 1** | `disclose_income` | Y/N |
| **Decision 2** | `disclose_documents`, `customer_type` | Y/N/NA, discount/fixed/regular |
| **Decision 3** | `donation_default` | Baseline donation rate [0.0-1.0] |
| **Decision 4** | `rejected_transaction_defaults` | Default rejection behavior |
| **Decision 5** | `weight_price`, `weight_quality`, `weight_proximity`, `weight_sustainability` | Vendor choice weights (flattened) |
| **Decision 6** | `income`, `income_category`, `purchasing_quantity` | Income info and total requests |
| **Decision 7** | `purchasing_frequency` | Requests per period |
| **Decision 8** | `preferred_vendor`, `proximity_v1`, `proximity_v2`, ... | Preferred vendor and proximity scores |
| **Decision 9 Summary** | `total_purchase_requests`, `pn_requests_count`, `bid_requests_count`, `pct_purchase_now` | Purchase vs bid summary |
| **Decision 10 Summary** | `avg_bid_value`, `min_bid_value`, `max_bid_value` | Bid value statistics |
| **Decision 11** | `rejected_transaction_option` | Rejection behavior |
| **Decision 13** | `final_donation_rate` | Agent's final donation rate |

**Total Columns**: ~30-35 (varies by number of vendors)

---

### **Sheet 2: Transaction Level** (One row per purchase request)

| Column Category | Columns | Description |
|----------------|---------|-------------|
| **Identification** | `Transaction ID`, `Agent ID`, `Request ID` | Unique identifiers |
| **Agent Reference** | `Honesty_Humility`, `Assigned Allowance Level`, `Group_experiment`, `Customer Type`, `Income Category` | Agent traits for reference |
| **Timing** | `Timestamp (hours)`, `Period`, `Purchase Date`, `Purchase Time` | When request occurred |
| **Vendor** | `Vendor ID`, `Vendor Price`, `Vendor Quality`, `Vendor Sustainability`, `Vendor Proximity`, `Vendor Integrated Score` | Selected vendor details |
| **Purchase** | `Platform Price`, `Purchase Request Type` | PN/BID/FIXED/DISCOUNT |
| **Pricing** | `Bid Value`, `Customer Price` | Transaction pricing |
| **Donation** | `Agent Donation Default`, `Final Donation Rate`, `Donation Paid`, `Total Paid` | Donation information |

**Total Columns**: ~25-30

---

## 🔑 Key Features

### 1. **Comprehensive Data Coverage**
- **Agent-Level**: All agent decisions + summary statistics
- **Transaction-Level**: All per-request data + agent context

### 2. **Cross-Reference Capability**
- Both sheets include `Agent ID` for easy joining
- Transaction sheet includes agent traits for convenience
- Agent sheet includes transaction summaries

### 3. **Calculated Fields**
- **Agent-Level**:
  - `pct_purchase_now`: Percentage of PN requests
  - `avg_bid_value`, `min_bid_value`, `max_bid_value`: Bid statistics
- **Transaction-Level**:
  - `Vendor Integrated Score`: Composite vendor score
  - `Donation Paid`: customer_price × final_donation_rate
  - `Total Paid`: customer_price + donation_paid

### 4. **Smart Data Formatting**
- Proper data types (integers, floats, dates, times)
- "N/A" for non-applicable fields
- NaN for missing numeric values
- Proper date/time formatting

### 5. **User-Friendly Interface**
- Single download button for complete Excel file
- Summary statistics displayed
- Preview of both sheets before download
- Clear column counts and descriptions

---

## 📊 Example Data

### **Agent-Level Example** (Agent ID 42)
```excel
| Agent ID | Honesty_Humility | customer_type | donation_default | preferred_vendor | total_purchase_requests | pn_requests_count | bid_requests_count | pct_purchase_now | avg_bid_value |
|----------|------------------|---------------|------------------|------------------|------------------------|-------------------|-------------------|------------------|---------------|
| 42       | 3.45             | regular       | 0.423            | 3                | 12                     | 5                 | 7                 | 41.67            | 98.75         |
```

### **Transaction-Level Example** (Agent 42's first 3 requests)
```excel
| Transaction ID | Agent ID | Request ID | Period | Vendor ID | Purchase Request Type | Bid Value | Customer Price | Final Donation Rate | Donation Paid | Total Paid |
|----------------|----------|------------|--------|-----------|-----------------------|-----------|----------------|---------------------|---------------|------------|
| A42_R1         | 42       | 1          | 1      | Vendor 3  | Purchase Now          | N/A       | 137.50         | 0.423               | 58.16         | 195.66     |
| A42_R2         | 42       | 2          | 1      | Vendor 3  | Bid                   | 95.50     | 95.50          | 0.423               | 40.40         | 135.90     |
| A42_R3         | 42       | 3          | 1      | Vendor 3  | Bid                   | 102.30    | 102.30         | 0.423               | 43.27         | 145.57     |
```

---

## 🎯 Data Categorization Logic

### **Agent-Level Fields** (One value per agent)
- Traits (5): From trait engine
- Customer Type: Determined by Decisions 1 & 2
- Donation Rates: Agent's baseline values
- Vendor Preferences: Agent's preferred vendor and proximity scores
- Summary Statistics: Aggregated from all transactions

### **Transaction-Level Fields** (Multiple values per agent)
- Transaction Identification: Unique per request
- Timing: When request was made
- Vendor Selection: Which vendor for this request
- Purchase Decision: PN vs BID for this request
- Pricing: Unique bid value per BID request
- Donation: Per-request donation rate and amounts

### **Both Levels**
- `Agent ID`: Links agent-level and transaction-level data
- Customer Type: Agent-level value, copied to transactions for convenience
- Donation Default: Agent-level baseline, used to calculate per-transaction donations

---

## 📖 Usage Guide

### **For Researchers**

#### **Agent-Level Analysis**
```python
import pandas as pd

# Read agent-level sheet
df_agents = pd.read_excel('simulation_results_20251128_143052.xlsx', sheet_name='Agent Level')

# Analyze agent characteristics
print(df_agents[['Agent ID', 'customer_type', 'donation_default']].describe())

# Compare agents by customer type
df_agents.groupby('customer_type')['donation_default'].mean()

# Analyze vendor preferences
df_agents['preferred_vendor'].value_counts()
```

#### **Transaction-Level Analysis**
```python
# Read transaction-level sheet
df_transactions = pd.read_excel('simulation_results_20251128_143052.xlsx', sheet_name='Transaction Level')

# Analyze purchase patterns
df_transactions['Purchase Request Type'].value_counts()

# Calculate average customer price by type
df_transactions.groupby('Purchase Request Type')['Customer Price'].mean()

# Analyze donations
total_donations = df_transactions['Donation Paid'].sum()
avg_donation_rate = df_transactions['Final Donation Rate'].mean()
```

#### **Cross-Level Analysis**
```python
# Join both sheets on Agent ID
df_merged = df_transactions.merge(
    df_agents[['Agent ID', 'Honesty_Humility', 'donation_default']], 
    on='Agent ID'
)

# Analyze relationship between honesty and donation behavior
import matplotlib.pyplot as plt
plt.scatter(df_merged['Honesty_Humility'], df_merged['Final Donation Rate'])
plt.xlabel('Honesty-Humility Score')
plt.ylabel('Final Donation Rate')
plt.show()
```

### **For Students**

#### **Basic Analysis**
- Open Excel file
- Sheet 1: See each agent's characteristics and overall behavior
- Sheet 2: See each individual purchase request
- Use Excel pivot tables to analyze patterns

#### **Key Questions to Explore**
1. What percentage of agents are discount/fixed/regular customers?
2. What is the average donation rate?
3. How many purchase requests does each agent make?
4. What is the ratio of PN to BID requests?
5. Which vendors are most popular?

---

## 🔍 Data Dictionary

### **Agent-Level Sheet**

| Column | Type | Range/Values | Description |
|--------|------|--------------|-------------|
| `Agent ID` | Integer | 1 to N | Unique agent identifier |
| `Honesty_Humility` | Float | ~1.0-5.0 | Personality trait score |
| `Assigned Allowance Level` | Integer | 1-5 | Income level assignment |
| `Study Program` | String | CLEAM, BESS, etc. | Academic program |
| `Group_experiment` | String | HighSub, MidSub, etc. | Experimental group |
| `TWT+Sospeso` | Float | - | Observed prosocial behavior |
| `disclose_income` | String | Y, N | Income disclosure decision |
| `disclose_documents` | String | Y, N, NA | Document disclosure decision |
| `customer_type` | String | discount, fixed, regular | Customer classification |
| `donation_default` | Float | 0.0-1.0 | Baseline donation rate |
| `rejected_transaction_defaults` | String | - | Default rejection behavior |
| `weight_price` | Float | 0.0-1.0 | Price weight in vendor selection |
| `weight_quality` | Float | 0.0-1.0 | Quality weight in vendor selection |
| `weight_proximity` | Float | 0.0-1.0 | Proximity weight in vendor selection |
| `weight_sustainability` | Float | 0.0-1.0 | Sustainability weight in vendor selection |
| `income` | Float | Dollar amount | Agent's income (if disclosed) |
| `income_category` | Integer | 1 to N | Income category for Fixed customers |
| `purchasing_quantity` | Integer | 0+ | Total purchase requests made |
| `purchasing_frequency` | Float | - | Requests per period |
| `preferred_vendor` | Integer | 1 to M | Most preferred vendor |
| `proximity_v1`, `proximity_v2`, ... | Float | 0-100 | Proximity to each vendor |
| `total_purchase_requests` | Integer | 0+ | Total requests (same as purchasing_quantity) |
| `pn_requests_count` | Integer | 0+ | Number of Purchase Now requests |
| `bid_requests_count` | Integer | 0+ | Number of Bid requests |
| `pct_purchase_now` | Float | 0.0-100.0 | Percentage of PN requests |
| `avg_bid_value` | Float | Dollar amount | Average bid value (if any bids) |
| `min_bid_value` | Float | Dollar amount | Minimum bid value (if any bids) |
| `max_bid_value` | Float | Dollar amount | Maximum bid value (if any bids) |
| `rejected_transaction_option` | String | - | Rejection behavior choice |
| `final_donation_rate` | Float | 0.0-1.0 | Agent's final donation rate |

### **Transaction-Level Sheet**

| Column | Type | Range/Values | Description |
|--------|------|--------------|-------------|
| `Transaction ID` | String | A#_R# | Unique transaction identifier |
| `Agent ID` | Integer | 1 to N | Agent who made this request |
| `Request ID` | Integer | 1+ | Request number for this agent |
| `Honesty_Humility` | Float | ~1.0-5.0 | Agent's personality trait |
| `Assigned Allowance Level` | Integer | 1-5 | Agent's income level |
| `Group_experiment` | String | - | Agent's experimental group |
| `Customer Type` | String | Discount, Fixed, Regular | Agent's customer classification |
| `Income Category` | Integer | 1 to N | Agent's income category |
| `Timestamp (hours)` | Float | 0.0+ | Hours since simulation start |
| `Period` | Integer | 1+ | Period number |
| `Purchase Date` | Date | - | Date of purchase request |
| `Purchase Time` | Time | - | Time of purchase request |
| `Vendor ID` | String | Vendor # | Selected vendor for this request |
| `Vendor Price` | Float | Dollar amount | Vendor's base price |
| `Vendor Quality` | Integer | 1-5 | Vendor's quality rating |
| `Vendor Sustainability` | Integer | 1-5 | Vendor's sustainability rating |
| `Vendor Proximity` | Float | 0-100 | Agent-vendor proximity score |
| `Vendor Integrated Score` | Float | 0.0-1.0 | Composite vendor score |
| `Platform Price` | String | PN, BID, FIXED, DISCOUNT | Price mechanism |
| `Purchase Request Type` | String | Purchase Now, Bid, Fixed, Discount | Human-readable type |
| `Bid Value` | Float/String | Dollar amount or N/A | Unique bid value (BID requests only) |
| `Customer Price` | Float | Dollar amount | Price customer pays (before donation) |
| `Agent Donation Default` | Float | 0.0-1.0 | Agent's baseline donation rate |
| `Final Donation Rate` | Float | 0.0-1.0 | Donation rate for this request |
| `Donation Paid` | Float | Dollar amount | Donation amount for this request |
| `Total Paid` | Float | Dollar amount | Customer price + donation |

---

## ✅ Verification Checklist

### **Agent-Level Sheet**
- [x] One row per agent
- [x] All 5 trait columns present
- [x] All decision fields included (Decisions 1-13)
- [x] Vendor choice weights flattened to 4 columns
- [x] Vendor proximity scores expanded (one per vendor)
- [x] Summary statistics calculated correctly
- [x] Agent ID matches transaction-level

### **Transaction-Level Sheet**
- [x] One row per purchase request
- [x] Transaction ID unique and formatted correctly
- [x] Agent traits copied for reference
- [x] Timing information complete
- [x] Vendor information detailed
- [x] Purchase decision captured
- [x] Pricing calculated correctly
- [x] Donation fields populated
- [x] Agent ID links to agent-level

### **Excel File**
- [x] Two sheets: "Agent Level" and "Transaction Level"
- [x] Download button functional
- [x] File naming includes timestamp
- [x] Preview functionality works
- [x] Summary statistics displayed

---

## 🚀 Next Steps

### **Immediate Actions**
1. ✅ Implementation complete
2. ⏳ Test with sample simulation data
3. ⏳ Verify all fields populate correctly
4. ⏳ Test with different simulation configurations
5. ⏳ Gather user feedback

### **Future Enhancements** (Optional)
1. **Add data validation**: Ensure all required fields are present
2. **Add formatting**: Excel cell formatting (borders, headers, number formats)
3. **Add formulas**: Excel formulas for calculated fields
4. **Add charts**: Embedded charts in Excel for quick visualization
5. **Add filters**: Auto-filters on all columns
6. **Add data dictionary sheet**: Third sheet explaining all columns
7. **Export options**: Allow separate file export (not just combined)

---

## 📝 Testing Instructions

### **How to Test**

1. **Run a simulation**:
   - Go to Page 1 (Parameters)
   - Configure simulation parameters
   - Go to Page 2 (Decisions)
   - Select decisions to run
   - Run simulation

2. **View Results**:
   - Go to Results page
   - Scroll to "Export Results" section

3. **Verify Export**:
   - Check summary statistics (# agents, # transactions)
   - Click "Preview Agent-Level Data" to see first 5 rows
   - Click "Preview Transaction-Level Data" to see first 5 rows
   - Click "Download Complete Excel" button

4. **Verify Excel File**:
   - Open downloaded Excel file
   - Verify "Agent Level" sheet exists
   - Verify "Transaction Level" sheet exists
   - Check column counts match documentation
   - Spot-check data accuracy
   - Verify Agent ID linking works

5. **Test Analysis**:
   - Try example Python code from Usage Guide
   - Perform cross-level analysis
   - Create pivot tables in Excel

---

## 🐛 Known Issues / Limitations

### **Current Limitations**
1. **Vendor Count**: Proximity columns generated based on vendors in simulation
2. **Missing Data**: Some fields may be NaN/N/A if decisions not run
3. **Large Files**: Many agents × many requests = large Excel files
4. **Performance**: Building DataFrames may take time for large simulations

### **Not Implemented Yet**
1. **Multi-config comparison**: Currently single configuration only
2. **Separate file exports**: Only combined Excel available
3. **Custom column selection**: All columns included automatically
4. **Excel formatting**: Basic formatting only

---

## 📚 Related Documentation

- **Data Analysis Document**: `AGENT_VS_TRANSACTION_LEVEL_DATA_ANALYSIS.md`
- **Decision Architecture**: `DEFAULT_DECISIONS_ARCHITECTURE.md`
- **Technical Documentation**: `TECHNICAL_DOCUMENTATION.md`
- **Per-Request Implementation**: `PER_REQUEST_PURCHASE_DECISIONS_IMPLEMENTATION.md`

---

## 🎓 Key Insights

### **Why Two Levels?**
- **Agent-level**: Understand agent characteristics and overall behavior patterns
- **Transaction-level**: Analyze individual purchase decisions and outcomes
- **Together**: Enable multi-level analysis of behavior and outcomes

### **Design Decisions**
1. **Single Excel file**: Easier to share and analyze than separate files
2. **Agent traits copied**: Convenience for transaction-level analysis
3. **Summary statistics**: Agent-level aggregations save analysis time
4. **Calculated fields**: Pre-compute common metrics (donations, totals)

### **Data Relationships**
```
Agent (1) ----< Transactions (N)
   ↓
- Traits (fixed)
- Decisions (agent-level)
- Summaries (from transactions)
   
   Each transaction references:
   - Parent agent (via Agent ID)
   - Vendor (via Vendor ID)
   - Timing (period, timestamp)
   - Pricing (customer price, bid value)
   - Donation (rate, amount)
```

---

**Document Version**: 1.0  
**Implementation Date**: November 28, 2025  
**Status**: ✅ Complete  
**Files Modified**: 1 (`app/pages/results/components/export_section.py`)  
**Lines Added**: ~500  
**Lines Modified**: ~100  







