# Testing Guide: Two-Level Excel Export

## 🎯 Purpose

This guide provides step-by-step instructions for testing the new two-level Excel export functionality.

---

## ⚙️ Prerequisites

Before testing, ensure:
1. ✅ Implementation is complete (`app/pages/results/components/export_section.py` modified)
2. ✅ No linter errors
3. ✅ Streamlit app can start

---

## 🧪 Test Plan

### **Test 1: Basic Functionality**

#### **Objective**: Verify both Excel sheets are created correctly

#### **Steps**:
1. Start the Streamlit app:
   ```bash
   streamlit run app_enhanced_new.py
   ```

2. Configure simulation on Page 1:
   - Set parameters (use defaults or customize)
   - Note: Use small numbers for quick testing (e.g., 10 agents, 1 period)

3. Select decisions on Page 2:
   - Select at least: 
     - Decision 1 (Disclose Income)
     - Decision 2 (Disclose Documents)
     - Decision 3 (Donation Default)
     - Decision 6 (Purchasing Quantity)
     - Decision 9 (Purchase vs Bid)
   - Run simulation

4. Navigate to Results page

5. Scroll to "Export Results" section

6. **Verify UI displays**:
   - ✅ "Two export options" explanation text
   - ✅ Summary metrics (Total Agents, Total Transactions)
   - ✅ "Download Complete Excel (Both Levels)" button
   - ✅ "Preview Agent-Level Data" expander
   - ✅ "Preview Transaction-Level Data" expander

7. Click preview expanders:
   - ✅ Agent-Level preview shows first 5 agents
   - ✅ Transaction-Level preview shows first 5 transactions
   - ✅ Column counts displayed correctly

8. Click Download button:
   - ✅ Excel file downloads
   - ✅ Filename format: `simulation_results_YYYYMMDD_HHMMSS.xlsx`

9. Open Excel file:
   - ✅ File opens without errors
   - ✅ Two sheets present: "Agent Level" and "Transaction Level"

#### **Expected Results**:
- ✅ All UI elements display correctly
- ✅ Preview data looks reasonable
- ✅ Excel file downloads successfully
- ✅ Both sheets exist in Excel file

---

### **Test 2: Agent-Level Sheet Verification**

#### **Objective**: Verify agent-level data is complete and accurate

#### **Steps**:
1. Open downloaded Excel file
2. Go to "Agent Level" sheet

3. **Verify structure**:
   - ✅ One row per agent (row count = number of agents in simulation)
   - ✅ Agent ID in first column (values 1, 2, 3, ...)

4. **Verify trait columns** (should have 5):
   - ✅ Honesty_Humility
   - ✅ Assigned Allowance Level
   - ✅ Study Program
   - ✅ Group_experiment
   - ✅ TWT+Sospeso [=AW2+AX2]{Periods 1+2}

5. **Verify decision columns**:
   - ✅ disclose_income (Y/N)
   - ✅ disclose_documents (Y/N/NA)
   - ✅ customer_type (discount/fixed/regular)
   - ✅ donation_default (0.0-1.0)
   - ✅ rejected_transaction_defaults
   - ✅ weight_price, weight_quality, weight_proximity, weight_sustainability
   - ✅ income, income_category, purchasing_quantity
   - ✅ purchasing_frequency
   - ✅ preferred_vendor
   - ✅ proximity_v1, proximity_v2, ... (one per vendor)
   - ✅ total_purchase_requests, pn_requests_count, bid_requests_count, pct_purchase_now
   - ✅ avg_bid_value, min_bid_value, max_bid_value
   - ✅ rejected_transaction_option
   - ✅ final_donation_rate

6. **Verify data quality**:
   - ✅ No unexpected NaN values in required fields
   - ✅ Agent IDs are sequential (1, 2, 3, ...)
   - ✅ Donation rates in range [0.0, 1.0]
   - ✅ Vendor weights sum to ~1.0 or are meaningful
   - ✅ Request counts > 0 for active agents

7. **Verify summary statistics**:
   - ✅ total_purchase_requests = pn_requests_count + bid_requests_count
   - ✅ pct_purchase_now calculated correctly
   - ✅ Bid statistics (avg, min, max) only populated for agents with bids

#### **Expected Results**:
- ✅ ~30-35 columns (varies by number of vendors)
- ✅ All required columns present
- ✅ Data types correct (integers, floats, strings)
- ✅ Values within expected ranges

---

### **Test 3: Transaction-Level Sheet Verification**

#### **Objective**: Verify transaction-level data is complete and accurate

#### **Steps**:
1. Open downloaded Excel file
2. Go to "Transaction Level" sheet

3. **Verify structure**:
   - ✅ One row per purchase request
   - ✅ Row count = sum of all agents' purchasing_quantity
   - ✅ Transaction ID in first column (format: A#_R#)

4. **Verify identification columns**:
   - ✅ Transaction ID (unique, format A1_R1, A1_R2, etc.)
   - ✅ Agent ID (links to Agent Level sheet)
   - ✅ Request ID (sequential within each agent)

5. **Verify agent reference columns**:
   - ✅ Honesty_Humility (copied from agent)
   - ✅ Assigned Allowance Level (copied from agent)
   - ✅ Group_experiment (copied from agent)
   - ✅ Customer Type (copied from agent)
   - ✅ Income Category (copied from agent)

6. **Verify timing columns**:
   - ✅ Timestamp (hours) - numeric values
   - ✅ Period - integer values
   - ✅ Purchase Date - date format
   - ✅ Purchase Time - time format

7. **Verify vendor columns**:
   - ✅ Vendor ID (format: "Vendor #")
   - ✅ Vendor Price (numeric)
   - ✅ Vendor Quality (1-5)
   - ✅ Vendor Sustainability (1-5)
   - ✅ Vendor Proximity (0-100)
   - ✅ Vendor Integrated Score (0-1)

8. **Verify purchase decision columns**:
   - ✅ Platform Price (PN/BID/FIXED/DISCOUNT)
   - ✅ Purchase Request Type (human-readable)

9. **Verify pricing columns**:
   - ✅ Bid Value (numeric for BID requests, "N/A" for others)
   - ✅ Customer Price (numeric, varies by type)

10. **Verify donation columns**:
    - ✅ Agent Donation Default (reference value)
    - ✅ Final Donation Rate (per-request rate)
    - ✅ Donation Paid (calculated correctly)
    - ✅ Total Paid (customer price + donation)

11. **Verify calculations**:
    - ✅ Donation Paid = Customer Price × Final Donation Rate
    - ✅ Total Paid = Customer Price + Donation Paid
    - ✅ All monetary values have 2 decimal places

#### **Expected Results**:
- ✅ ~25-30 columns
- ✅ All required columns present
- ✅ Data types correct
- ✅ Calculations accurate
- ✅ Transaction IDs unique

---

### **Test 4: Cross-Level Consistency**

#### **Objective**: Verify data consistency between sheets

#### **Steps**:
1. Open Excel file with both sheets visible

2. **Pick a random agent** (e.g., Agent ID = 5):
   
   **In Agent Level sheet**:
   - Note: `purchasing_quantity` value (e.g., 12)
   - Note: `pn_requests_count` value (e.g., 7)
   - Note: `bid_requests_count` value (e.g., 5)
   - Note: `donation_default` value (e.g., 0.423)
   
   **In Transaction Level sheet**:
   - Filter by `Agent ID = 5`
   - ✅ Count rows → should equal `purchasing_quantity`
   - ✅ Count rows with Purchase Request Type = "Purchase Now" → should equal `pn_requests_count`
   - ✅ Count rows with Purchase Request Type = "Bid" → should equal `bid_requests_count`
   - ✅ Check `Agent Donation Default` → should match `donation_default` from Agent Level

3. **Repeat for 2-3 more agents** to verify consistency

4. **Verify totals**:
   - ✅ Sum of all agents' `purchasing_quantity` = Total rows in Transaction Level sheet

#### **Expected Results**:
- ✅ All counts match between sheets
- ✅ Agent IDs link correctly
- ✅ No orphaned transactions (transactions without matching agent)
- ✅ No missing transactions (agent has count but no transactions)

---

### **Test 5: Data Analysis Capability**

#### **Objective**: Verify Excel can be used for analysis

#### **Steps**:
1. **Agent-Level Analysis**:
   - Create pivot table on Agent Level sheet
   - ✅ Group by `customer_type`, calculate average `donation_default`
   - ✅ Create chart showing distribution of `preferred_vendor`

2. **Transaction-Level Analysis**:
   - Create pivot table on Transaction Level sheet
   - ✅ Group by `Purchase Request Type`, calculate sum of `Customer Price`
   - ✅ Create chart showing purchases by Period

3. **Cross-Sheet Analysis**:
   - Use VLOOKUP to link data
   - ✅ From Transaction sheet, lookup agent's `Honesty_Humility` from Agent sheet
   - ✅ Calculate correlation between Honesty_Humility and Final Donation Rate

#### **Expected Results**:
- ✅ Pivot tables create successfully
- ✅ Charts display correctly
- ✅ VLOOKUP functions work
- ✅ Data is analysis-ready

---

### **Test 6: Edge Cases**

#### **Objective**: Test behavior with unusual data

#### **Test 6.1: Agent with No Purchases**
- **Setup**: (This shouldn't happen in normal operation, but test robustness)
- **Expected**: Agent appears in Agent Level with 0 requests, no transactions in Transaction Level

#### **Test 6.2: All Purchase Now**
- **Setup**: Configure Decision 9 with 100% Purchase Now probability
- **Expected**: 
  - Agent Level: `bid_requests_count = 0`, `avg_bid_value = NaN`
  - Transaction Level: All rows have Purchase Request Type = "Purchase Now", Bid Value = "N/A"

#### **Test 6.3: All Bids**
- **Setup**: Configure Decision 9 with 100% Bid probability
- **Expected**:
  - Agent Level: `pn_requests_count = 0`
  - Transaction Level: All rows have Purchase Request Type = "Bid", Bid Value populated

#### **Test 6.4: Large Simulation**
- **Setup**: Run with 500 agents, 3 periods
- **Expected**:
  - Excel file size: ~1-5 MB
  - Both sheets open without performance issues
  - All data present and correct

---

## ✅ Test Checklist

### **Core Functionality**
- [ ] UI displays correctly
- [ ] Preview expanders work
- [ ] Download button functions
- [ ] Excel file downloads
- [ ] Both sheets present in Excel

### **Agent-Level Sheet**
- [ ] Correct number of rows (= # agents)
- [ ] All required columns present
- [ ] Trait columns populated
- [ ] Decision columns populated
- [ ] Vendor weights flattened correctly
- [ ] Proximity scores expanded
- [ ] Summary statistics calculated
- [ ] Data types correct
- [ ] Values within expected ranges

### **Transaction-Level Sheet**
- [ ] Correct number of rows (= total purchase requests)
- [ ] All required columns present
- [ ] Transaction IDs unique
- [ ] Agent reference columns populated
- [ ] Timing columns populated
- [ ] Vendor columns populated
- [ ] Purchase decision columns populated
- [ ] Pricing columns populated
- [ ] Donation columns populated
- [ ] Calculations correct

### **Cross-Level Consistency**
- [ ] Agent IDs match between sheets
- [ ] Purchase request counts match
- [ ] PN/BID counts match
- [ ] Donation rates consistent
- [ ] No orphaned data

### **Analysis Capability**
- [ ] Pivot tables work
- [ ] Charts can be created
- [ ] VLOOKUP functions work
- [ ] Data suitable for analysis

---

## 🐛 Troubleshooting

### **Issue**: Excel file won't download
- **Cause**: Browser blocking download or openpyxl not installed
- **Solution**: Check browser settings, verify openpyxl installed

### **Issue**: Missing columns in sheet
- **Cause**: Some decisions not run
- **Solution**: Expected behavior - columns only populated for selected decisions

### **Issue**: NaN values in donation fields
- **Cause**: Decision 3 (Donation Default) not selected
- **Solution**: Expected behavior - or select Decision 3

### **Issue**: Transaction count mismatch
- **Cause**: Bug in aggregation logic
- **Solution**: Check `_build_agent_level_dataframe` function

### **Issue**: Excel file too large
- **Cause**: Many agents × many requests
- **Solution**: Expected for large simulations - consider filtering or sampling

---

## 📝 Test Report Template

```
TWO-LEVEL EXCEL EXPORT TEST REPORT
==================================

Date: [Date]
Tester: [Name]
Simulation Config: [Brief description]

Test Results:
-------------

✅/❌ Test 1: Basic Functionality
✅/❌ Test 2: Agent-Level Sheet Verification  
✅/❌ Test 3: Transaction-Level Sheet Verification
✅/❌ Test 4: Cross-Level Consistency
✅/❌ Test 5: Data Analysis Capability
✅/❌ Test 6: Edge Cases

Issues Found:
-------------
[List any issues discovered]

Notes:
------
[Additional observations]

Conclusion:
-----------
✅ PASS / ❌ FAIL
```

---

## 🚀 Automated Test Script

For automated testing, use:
```bash
python3 test_two_level_export.py
```

This script:
- Creates mock data (10 agents, 4 vendors)
- Builds both DataFrames
- Verifies structure and content
- Checks cross-level consistency
- Exports test Excel file
- Reports pass/fail

---

## 📚 Documentation References

- **Implementation**: `TWO_LEVEL_EXCEL_EXPORT_IMPLEMENTATION.md`
- **Data Analysis**: `AGENT_VS_TRANSACTION_LEVEL_DATA_ANALYSIS.md`
- **Usage Guide**: See "Usage Guide" section in Implementation doc

---

**Testing Guide Version**: 1.0  
**Created**: November 28, 2025  
**Last Updated**: November 28, 2025







