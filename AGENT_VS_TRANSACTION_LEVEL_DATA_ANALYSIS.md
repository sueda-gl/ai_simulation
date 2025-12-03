# Agent-Level vs Transaction-Level Data Analysis

## Executive Summary

This document analyzes all data generated throughout the simulation decisions and categorizes each field as either:
- **AGENT-LEVEL**: One value per agent (e.g., personality traits, agent-level decisions)
- **TRANSACTION-LEVEL**: Multiple values per agent (one per purchase request)
- **BOTH**: Data that exists at both levels (agent average + per-transaction values)

## 📊 Data Structure Overview

### Current Simulation Architecture

```python
{
    'agent_id': 1,                          # AGENT-LEVEL
    'Honesty_Humility': 3.45,              # AGENT-LEVEL (trait)
    'donation_default': 0.423,              # AGENT-LEVEL
    'purchase_requests': [                  # TRANSACTION-LEVEL (list)
        {
            'request_id': 1,
            'vendorID': 3,
            'platformPrice': 'PN',
            'final_donation_rate': 0.423,
            ...
        },
        {
            'request_id': 2,
            'vendorID': 2,
            'platformPrice': 'BID',
            'bid_value': 95.50,
            'final_donation_rate': 0.423,
            ...
        }
    ]
}
```

---

## 🎯 Decision-by-Decision Data Categorization

### **Decision 1: Disclose Income**

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `disclose_income` | **AGENT** | String | "Y", "N" | One decision per agent |

**Excel Placement**: Agent-level only

---

### **Decision 2: Disclose Documents**

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `disclose_documents` | **AGENT** | String | "Y", "N", "NA" | One decision per agent |
| `customer_type` | **AGENT** | String | "discount", "fixed", "regular" | Derived from decisions 1 & 2 |

**Excel Placement**: Agent-level only
**Note**: `customer_type` is also included in each transaction for reference

---

### **Decision 3: Donation Default**

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `donation_default` | **AGENT** | Float | [0.0, 1.0] | Agent's baseline donation rate |
| `donation_default_raw` | **AGENT** | Float | - | Intermediate computation (optional) |
| `donation_default_raw_pos` | **AGENT** | Float | - | Intermediate computation (optional) |

**Excel Placement**: Agent-level only
**Note**: This is the agent's baseline rate, not the per-transaction rate

---

### **Decision 4: Rejected Transaction Defaults**

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `rejected_transaction_defaults` | **AGENT** | String | "wait", "try_another", "cancel", etc. | Default behavior when transaction rejected |

**Excel Placement**: Agent-level only

---

### **Decision 5: Vendor Choice Weights**

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `vendor_choice_weights` | **AGENT** | Dict | `{price, quality, proximity, sustainability}` | Agent's preference weights |

**Excel Placement**: Agent-level (flatten to 4 columns: `weight_price`, `weight_quality`, `weight_proximity`, `weight_sustainability`)

---

### **Decision 6: Purchasing Quantity**

#### Agent-Level Fields

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `income` | **AGENT** | Float | Dollar amount | Agent's annual income (if disclosed) |
| `income_category` | **AGENT** | Integer | 1 to N | Agent's income category (Fixed customers) |
| `purchasing_quantity` | **AGENT** | Integer | 0 to limit | Total number of purchase requests |

#### Transaction-Level Fields (in `purchase_requests`)

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `request_id` | **TRANSACTION** | Integer | 1, 2, 3... | Request number within agent |
| `quantity` | **TRANSACTION** | Integer | 1 (default) | Items per request |
| `timestamp_hours` | **TRANSACTION** | Float | [0.0, term_duration] | Hours since simulation start |
| `customer_id` | **TRANSACTION** | Integer | = agent_id | For reference |
| `customer_type` | **TRANSACTION** | String | "discount", "fixed", "regular" | Copied from agent |
| `vendorID` | **TRANSACTION** | Integer | 1, 2, 3... | Preferred/selected vendor |

**Excel Placement**: 
- Agent-level: `income`, `income_category`, `purchasing_quantity` (total count)
- Transaction-level: All fields in `purchase_requests`

---

### **Decision 7: Purchasing Frequency**

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `purchasing_frequency` | **AGENT** | Float | Requests per period | Calculated from total quantity / periods |

**Excel Placement**: Agent-level only

---

### **Decision 8: Vendor Selection**

#### Agent-Level Fields

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `vendor_selection` | **AGENT** | Integer | Vendor ID | Agent's preferred vendor (most common) |
| `preferred_vendor` | **AGENT** | Integer | Vendor ID | Same as vendor_selection |
| `vendor_proximity_scores` | **AGENT** | Dict | `{vendor_id: proximity}` | Agent-vendor proximity scores |

#### Transaction-Level Fields

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `vendorID` | **TRANSACTION** | Integer | Vendor ID | Vendor for this specific request |

**Excel Placement**:
- Agent-level: `preferred_vendor`, plus expanded `vendor_proximity_scores` (one column per vendor: `proximity_v1`, `proximity_v2`, etc.)
- Transaction-level: `vendorID`, plus vendor attributes (price, quality, sustainability, proximity) for the selected vendor

**Note**: Same agent can select different vendors for different requests

---

### **Decision 9: Purchase vs Bid**

#### Agent-Level Fields (Legacy)

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `purchase_vs_bid` | **AGENT** | String | "Purchase Now", "bid", "NA_discount", "NA_fixed" | Only for backward compatibility |

#### Transaction-Level Fields (Current Implementation)

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `platformPrice` | **TRANSACTION** | String | "PN", "BID", "FIXED", "DISCOUNT" | Decision for each request |

**Excel Placement**:
- Agent-level: Summary statistics (e.g., `pct_purchase_now`, `pct_bid`, `total_pn_requests`, `total_bid_requests`)
- Transaction-level: `platformPrice` for each request

**Note**: Regular customers can choose differently for each purchase request

---

### **Decision 10: Bid Value**

#### Agent-Level Fields (Legacy)

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `bid_value` | **AGENT** | Float | Dollar amount or NaN | Only for backward compatibility |

#### Transaction-Level Fields (Current Implementation)

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `bid_value` | **TRANSACTION** | Float | Dollar amount or "N/A" | Unique bid for each BID request |

**Excel Placement**:
- Agent-level: Summary statistics (e.g., `avg_bid_value`, `min_bid_value`, `max_bid_value`) - only for bid requests
- Transaction-level: `bid_value` for each request (only populated for BID requests)

**Note**: Each BID request generates a NEW random bid value

---

### **Decision 11: Rejected Transaction Option**

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `rejected_transaction_option` | **AGENT** | String | "wait", "try_another", "cancel", etc. | Agent's behavior after rejection |

**Excel Placement**: Agent-level only

---

### **Decision 12: Rejected Bid Value**

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `rejected_bid_value` | **AGENT** | String/Float | "NA" or value | Currently placeholder |

**Excel Placement**: Agent-level only (if implemented)

---

### **Decision 13: Final Donation Rate**

#### Agent-Level Fields

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `final_donation_rate` | **AGENT** | Float | [0.0, 1.0] | Agent's donation rate (often = donation_default) |

#### Transaction-Level Fields (Current Implementation)

| Field | Level | Type | Values | Notes |
|-------|-------|------|--------|-------|
| `final_donation_rate` | **TRANSACTION** | Float | [0.0, 1.0] | Donation rate for this specific request |

**Excel Placement**:
- Agent-level: `final_donation_rate` (baseline value)
- Transaction-level: `final_donation_rate` for each request

**Note**: Currently same for all requests from same agent, but architecture supports per-request variation

---

## 📋 Complete Field Lists

### **AGENT-LEVEL EXCEL** (One row per agent)

#### Identification
- `agent_id` (Integer)

#### Traits (from trait engine)
- `Honesty_Humility` (Float)
- `Assigned Allowance Level` (Integer 1-5)
- `Study Program` (String: CLEAM, BESS, CLEF, etc.)
- `Group_experiment` (String: HighSub, MidSub, NoSub, FullSub)
- `TWT+Sospeso [=AW2+AX2]{Periods 1+2}` (Float: observed prosocial behavior)

#### Decision 1: Disclose Income
- `disclose_income` (String: Y/N)

#### Decision 2: Disclose Documents & Customer Type
- `disclose_documents` (String: Y/N/NA)
- `customer_type` (String: discount/fixed/regular)

#### Decision 3: Donation Default
- `donation_default` (Float: 0.0-1.0)

#### Decision 4: Rejected Transaction Defaults
- `rejected_transaction_defaults` (String)

#### Decision 5: Vendor Choice Weights
- `weight_price` (Float: 0.0-1.0)
- `weight_quality` (Float: 0.0-1.0)
- `weight_proximity` (Float: 0.0-1.0)
- `weight_sustainability` (Float: 0.0-1.0)

#### Decision 6: Purchasing Quantity (Agent-Level)
- `income` (Float: dollar amount, if disclosed)
- `income_category` (Integer: 1 to N)
- `purchasing_quantity` (Integer: total requests)

#### Decision 7: Purchasing Frequency
- `purchasing_frequency` (Float: requests per period)

#### Decision 8: Vendor Selection (Agent-Level)
- `preferred_vendor` (Integer: vendor ID)
- `proximity_v1` (Float: 0-100, proximity to vendor 1)
- `proximity_v2` (Float: 0-100, proximity to vendor 2)
- ... (one column per vendor)

#### Decision 9: Purchase vs Bid (Summary Statistics)
- `total_purchase_requests` (Integer: total PN + BID requests)
- `pn_requests_count` (Integer: number of PN requests)
- `bid_requests_count` (Integer: number of BID requests)
- `pct_purchase_now` (Float: % of requests that are PN)

#### Decision 10: Bid Value (Summary Statistics)
- `avg_bid_value` (Float: average of all bid values, or NaN if no bids)
- `min_bid_value` (Float: minimum bid value, or NaN if no bids)
- `max_bid_value` (Float: maximum bid value, or NaN if no bids)

#### Decision 11: Rejected Transaction Option
- `rejected_transaction_option` (String)

#### Decision 13: Final Donation Rate
- `final_donation_rate` (Float: 0.0-1.0, agent's baseline)

**Total Columns**: ~30-35 (depending on number of vendors)

---

### **TRANSACTION-LEVEL EXCEL** (One row per purchase request)

#### Identification
- `agent_id` (Integer - links to agent-level)
- `request_id` (Integer - request number within agent)
- `transaction_id` (String - unique identifier)

#### Agent Traits (copied for reference)
- `Honesty_Humility` (Float)
- `Assigned Allowance Level` (Integer)
- `Group_experiment` (String)
- `customer_type` (String: discount/fixed/regular)
- `income_category` (Integer)

#### Transaction Timing
- `timestamp_hours` (Float: hours since simulation start)
- `period` (Integer: which period this request belongs to)
- `purchase_date` (Date: calculated from timestamp)
- `purchase_time` (Time: calculated from timestamp)

#### Vendor Selection (Per Transaction)
- `vendorID` (Integer: selected vendor for this request)
- `vendor_price` (Float: vendor's base price)
- `vendor_quality` (Integer: 1-5)
- `vendor_sustainability` (Integer: 1-5)
- `vendor_proximity` (Float: 0-100, agent-vendor proximity)
- `vendor_integrated_score` (Float: composite score)

#### Purchase Decision (Per Transaction)
- `platformPrice` (String: PN/BID/FIXED/DISCOUNT)
- `purchase_request_type` (String: Purchase Now/Bid/Fixed/Discount)

#### Pricing (Per Transaction)
- `bid_value` (Float: unique bid value for BID requests, N/A otherwise)
- `customer_price` (Float: actual price customer pays, before donation)

#### Donation (Per Transaction)
- `agent_donation_default` (Float: agent's baseline rate, for reference)
- `final_donation_rate` (Float: donation rate applied to this transaction)
- `donation_paid` (Float: calculated as customer_price × final_donation_rate)
- `total_paid` (Float: customer_price + donation_paid)

**Total Columns**: ~25-30

---

## 🎯 Key Insights for Excel Design

### 1. **Preferred Vendor vs Selected Vendor**
- **Agent-level**: `preferred_vendor` = the vendor the agent prefers most (highest score)
- **Transaction-level**: `vendorID` = the vendor actually selected for each request
- **Note**: Currently same for all requests (default behavior), but architecture supports different vendors per request

### 2. **Purchase vs Bid Decision**
- **Agent-level**: Summary statistics showing overall behavior pattern
- **Transaction-level**: `platformPrice` for each individual request
- **Key point**: Same agent can make different choices for different requests!

### 3. **Bid Values**
- **Agent-level**: Summary statistics (avg, min, max) across all bids
- **Transaction-level**: Unique bid value for each BID request
- **Key point**: Each BID generates a NEW random value!

### 4. **Donation Rates**
- **Agent-level**: `donation_default` (baseline) and `final_donation_rate` (agent's rate)
- **Transaction-level**: `final_donation_rate` per request (currently same as agent, but can vary)
- **Key point**: Architecture supports per-request donation rate variation

### 5. **Customer Type**
- **Agent-level**: Single value determined by Decisions 1 & 2
- **Transaction-level**: Copied to each request for convenience
- **Key point**: Customer type is agent-level, but included in both files for analysis

---

## 🔄 Data Relationships

```
AGENT-LEVEL (1 row per agent)
    ├── Traits (5 columns)
    ├── Agent-level decisions (20-25 columns)
    └── Summary statistics from transactions (5-10 columns)

TRANSACTION-LEVEL (N rows per agent, where N = purchasing_quantity)
    ├── Agent reference (agent_id)
    ├── Agent traits (copied for convenience)
    ├── Transaction-specific data (15-20 columns)
    └── Calculated fields (pricing, donations)
```

---

## 📊 Example Data

### Agent-Level Excel (Agent ID 42)
```
| agent_id | Honesty_Humility | customer_type | donation_default | preferred_vendor | pn_requests | bid_requests | avg_bid_value |
|----------|------------------|---------------|------------------|------------------|-------------|--------------|---------------|
| 42       | 3.45             | regular       | 0.423            | 3                | 5           | 7            | 98.75         |
```

### Transaction-Level Excel (Agent ID 42's requests)
```
| agent_id | request_id | period | vendorID | platformPrice | bid_value | customer_price | final_donation_rate | donation_paid | total_paid |
|----------|------------|--------|----------|---------------|-----------|----------------|---------------------|---------------|------------|
| 42       | 1          | 1      | 3        | PN            | N/A       | 137.50         | 0.423               | 58.16         | 195.66     |
| 42       | 2          | 1      | 3        | BID           | 95.50     | 95.50          | 0.423               | 40.40         | 135.90     |
| 42       | 3          | 1      | 3        | BID           | 102.30    | 102.30         | 0.423               | 43.27         | 145.57     |
| ...      | ...        | ...    | ...      | ...           | ...       | ...            | ...                 | ...           | ...        |
```

---

## ✅ Implementation Recommendations

### 1. **Two Separate Excel Files**
- `simulation_results_agent_level_YYYYMMDD_HHMMSS.xlsx`
- `simulation_results_transaction_level_YYYYMMDD_HHMMSS.xlsx`

### 2. **Or Single Excel with Two Sheets**
- Sheet 1: "Agent Level"
- Sheet 2: "Transaction Level"

### 3. **Column Ordering**
- **Agent-level**: Identification → Traits → Decisions (in order) → Summary Stats
- **Transaction-level**: Identification → Timing → Agent Reference → Vendor → Purchase → Pricing → Donation

### 4. **Data Types**
- Use appropriate Excel formats (integers, floats, dates, times)
- Format percentages as percentages (not 0.423 but 42.3%)
- Format currency with $ symbol and 2 decimal places

### 5. **Missing Data**
- Use "N/A" for text fields
- Use blank/null for numeric fields where appropriate
- Document what each N/A means (e.g., "N/A - Fixed customer doesn't bid")

---

## 🚀 Next Steps

1. ✅ Complete data categorization (this document)
2. ⏳ Implement agent-level export function
3. ⏳ Implement transaction-level export function
4. ⏳ Update export_section.py with both exports
5. ⏳ Test with sample simulation data
6. ⏳ Verify all fields are correctly populated
7. ⏳ Add user documentation

---

## 📝 Notes

- This analysis is based on the current codebase structure as of the review date
- Some fields may be optional depending on which decisions are selected
- Transaction-level data is stored in `purchase_requests` list within each agent's state
- Both Excel files should maintain consistent agent_id values for cross-referencing
- Consider adding a "Data Dictionary" sheet explaining all columns

---

**Document Version**: 1.0  
**Created**: [Current Date]  
**Last Updated**: [Current Date]



