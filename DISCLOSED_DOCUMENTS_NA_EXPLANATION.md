# Disclosed Documents Column - NA Explanation

**Date:** November 12, 2025  
**Question:** Why are most cells in "Disclosed documents" column blank?

---

## Quick Answer

**Blank cells = NA = "Not Applicable" = "Agent was never asked this question"**

This is **intentional and correct** - not missing data!

---

## The Three Meanings

| Excel Value | Meaning | Count (out of 100) |
|-------------|---------|-------------------|
| **BLANK** | NA = Not asked (not applicable) | 89 agents |
| **0** | Asked and agent said NO | 6 agents |
| **1** | Asked and agent said YES | 5 agents |

---

## Why NA (Not Asked)?

The simulation follows a **conditional decision flow**. Not every agent is asked about disclosing documents.

### Decision Flow

```
STEP 1: Disclose Income?
├─ NO (41 agents)
│  └─► Documents = NA (blank) → STOP, never asked
│     Customer type: Regular
│
└─ YES (59 agents)
   └─► Continue to STEP 2
   
       STEP 2: Check Income Level
       ├─ Income ≥ $12,500 (48 agents)
       │  └─► Documents = NA (blank) → STOP, never asked
       │     (Don't qualify for discount anyway)
       │     Customer type: Fixed
       │
       └─ Income < $12,500 (11 agents)
          └─► Continue to STEP 3
          
              STEP 3: Disclose Documents?
              ├─ NO (6 agents)
              │  └─► Documents = N (shows as 0)
              │     Customer type: Fixed
              │
              └─ YES (5 agents)
                 └─► Documents = Y (shows as 1)
                    Customer type: Discount
```

---

## Real Examples from Simulation

### CASE 1: Blank (NA) - Did NOT Disclose Income

**Agent 1:**
- Income: $29,667.28
- Q1: "Disclose income?" → **NO**
- Q2: "Disclose documents?" → **NOT ASKED** (blank)
- **Why?** Agent said no to Q1, so Q2 is irrelevant
- Customer type: **Regular**

**Agent 2:**
- Income: $13,079.63
- Q1: "Disclose income?" → **NO**
- Q2: "Disclose documents?" → **NOT ASKED** (blank)
- **Why?** Agent said no to Q1, so Q2 is irrelevant
- Customer type: **Regular**

---

### CASE 2: Blank (NA) - Disclosed Income But Too High

**Agent 7:**
- Income: $42,091.07
- Q1: "Disclose income?" → **YES**
- Check: Income > $12,500? → **YES**, not eligible for discount
- Q2: "Disclose documents?" → **NOT ASKED** (blank)
- **Why?** Agent doesn't qualify for discount anyway
- Customer type: **Fixed**

**Agent 8:**
- Income: $44,156.68
- Q1: "Disclose income?" → **YES**
- Check: Income > $12,500? → **YES**, not eligible for discount
- Q2: "Disclose documents?" → **NOT ASKED** (blank)
- **Why?** Agent doesn't qualify for discount anyway
- Customer type: **Fixed**

---

### CASE 3: Value = 0 - Asked and Said NO

**Agent 13:**
- Income: $10,433.95
- Q1: "Disclose income?" → **YES**
- Check: Income > $12,500? → **NO**, eligible for discount
- Q2: "Disclose documents?" → **NO** (shows as **0** in Excel)
- **Why?** Agent was asked but declined
- Customer type: **Fixed** (no discount)

---

### CASE 4: Value = 1 - Asked and Said YES

**Agent 20:**
- Income: $11,343.55
- Q1: "Disclose income?" → **YES**
- Check: Income > $12,500? → **NO**, eligible for discount
- Q2: "Disclose documents?" → **YES** (shows as **1** in Excel)
- **Why?** Agent was asked and agreed
- Customer type: **Discount** (gets discount!)

**Agent 33:**
- Income: $9,246.26
- Q1: "Disclose income?" → **YES**
- Check: Income > $12,500? → **NO**, eligible for discount
- Q2: "Disclose documents?" → **YES** (shows as **1** in Excel)
- **Why?** Agent was asked and agreed
- Customer type: **Discount** (gets discount!)

---

## Summary Table

| Disclosed Income | Income vs Threshold | Disclosed Documents | Excel Shows | Customer Type | Question Asked? |
|-----------------|---------------------|---------------------|-------------|---------------|----------------|
| NO | (any) | NA | **blank** | Regular | ❌ No |
| YES | ≥ $12,500 | NA | **blank** | Fixed | ❌ No |
| YES | < $12,500 | NO | **0** | Fixed | ✅ Yes, said no |
| YES | < $12,500 | YES | **1** | Discount | ✅ Yes, said yes |

---

## Breakdown of 100 Agents

```
89 Blank (NA) cells:
   ├─ 41 agents: Did NOT disclose income
   │             → Never asked about documents
   │             → Customer type: Regular
   │
   └─ 48 agents: Disclosed income BUT income > $12,500
                 → Don't qualify for discount
                 → Never asked about documents
                 → Customer type: Fixed

11 Non-blank cells:
   ├─ 6 agents (value = 0): Income < $12,500, asked, said NO
   │                        → Customer type: Fixed
   │
   └─ 5 agents (value = 1): Income < $12,500, asked, said YES
                            → Customer type: Discount
```

---

## Analogy

It's like asking someone **"Do you want to upgrade to business class?"**

- **If they didn't buy a plane ticket** → NA (blank) - question doesn't apply
- **If they bought economy ticket** → Asked, they answer:
  - No → 0
  - Yes → 1

You wouldn't ask someone to upgrade if they're not flying!

---

## Key Takeaways

✅ **Blank cells are NOT missing data** - they are intentional NA (Not Applicable)

✅ **NA means the question was never asked** because:
   - Agent didn't disclose income, OR
   - Agent's income is too high to qualify for discount

✅ **0 and 1 are actual responses** from agents who:
   - Disclosed income, AND
   - Have income below $12,500 (eligible), AND
   - Were asked the question

✅ **This design prevents illogical scenarios** like:
   - Asking for documents when we don't know their income
   - Asking for documents when they don't qualify anyway

---

## Files Referenced

- **Simulation data:** `outputs/simulation_seed42_agents100_all_20251112_072600.csv`
- **Excel export (fixed):** `agent_disclosure_customer_types_CORRECTED.xlsx`
- **Code location:** `src/decisions/disclose_documents.py` (lines 22-33)
- **Classification logic:** `src/decisions/income_utils.py` (line 526)

---

**Report Generated:** November 12, 2025  
**Status:** Data is correct - blanks are intentional NA values




