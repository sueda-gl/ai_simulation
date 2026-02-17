## Excel Export Structure

The disclose income decision produces multiple Excel export files depending on the run mode. There are three categories of exports:

### Export 1: Disclose Income Only — Single Configuration

**When produced**: Running disclose_income only with "Categorical only" or "Continuous only" income mode.

**File name**: `agent_disclose_income_data.xlsx` (from visualization) or `disclose_income_results_YYYYMMDD_HHMMSS.xlsx` (from export section)

**Structure**: Single sheet — "Agent Disclose Income Data" or "Disclose Income Results"

**Columns (19 total):**

| # | Column Name | Description | Source |
|---|-------------|-------------|--------|
| 1 | **Agent ID** | Agent identifier (1-based) | agent_id or index+1 |
| 2 | **Agreeable** | Raw Agreeableness trait (non-standardized) | Agreeable |
| 3 | **Openness** | Raw Openness to Experience (non-standardized) | OpennessBig5 |
| 4 | **Honesty_Humility** | Raw Honesty-Humility trait (non-standardized) | Honesty_Humility |
| 5 | **Extraversion** | Raw Extraversion (non-standardized) | ExtraversionBig5 |
| 6 | **Neuroticism** | Raw Neuroticism (non-standardized) | NeuroticismBig5 |
| 7 | **ReligiousAffiliation** | Raw binary value (0 or 1) | ReligiousAffiliation |
| 8 | **ReligiousService** | Raw ordinal value (0–4) | ReligiousService |
| 9 | **Religious** | Computed religiosity composite = (ReligiousAffiliation + ReligiousService/4) / 2 (non-z-scored) | disclose_income_religious_composite |
| 10 | **Assigned Allowance Level** | Income level (1–5) | Assigned Allowance Level |
| 11 | **income** | Actual income value (€ amount or derived from distribution) | income |
| 12 | **I-High** | Income-high indicator: 1 if level > 3 (categorical) or income > median (continuous), else 0 | disclose_income_income_high |
| 13 | **TWT+Sospeso** | Observed prosocial behavior (raw, non-standardized) | TWT+Sospeso [=AW2+AX2]{Periods 1+2} |
| 14 | **calc_PB** | Calculated prosocial behavior from Equation 1 (weighted_prosocial, before anchoring, non-z-scored) | disclose_income_weighted_prosocial |
| 15 | **WOPB** | Weight for observed prosocial behavior used in this run | disclose_income_wopb |
| 16 | **WPB** | Weight for prosocial behavior effect used in this run | disclose_income_wpb |
| 17 | **Intercept** | β₀ intercept value used in this run | disclose_income_intercept |
| 18 | **PB_i** | Anchored prosocial behavior (with stochastic draw if enabled) | disclose_income_anchored_pb |
| 19 | **Disclosure Income** | Continuous DI_i value (before Y/N classification) | disclose_income_di or disclose_income_raw |
| 20 | **Disclose Income (Y=1)** | Final binary decision: 1 = "Y" (disclosed), 0 = "N" (not disclosed) | disclose_income |

**Notes**:
- Columns 2–9 are the **input traits** (raw, non-standardized values — NOT z-scored).
- Columns 10–12 are the **income information**.
- Column 13 is the **observed prosocial behavior** input.
- Column 14 is the **Equation 1 output** (before anchoring).
- Columns 15–17 are the **configuration values** used for this run.
- Columns 18–19 are the **calculated intermediate and final values**.
- Column 20 is the **final binary decision**.
- All numeric values use "General" Excel format to preserve original decimal precision.

---

### Export 2: Disclose Income Only — Compare Both Income Modes

**When produced**: Running disclose_income only with "Compare both" income mode.

**File name**: `disclose_income_compare_both_YYYYMMDD_HHMMSS.xlsx`

**Structure**: Multiple sheets — one per income mode. You will always get **2 sheets**:

| Sheet Name | Description |
|-----------|-------------|
| **Categorical** | Results using categorical income treatment (level-specific intercepts) |
| **Continuous** | Results using continuous income treatment (β₀ + income coefficient) |

Each sheet contains the **full 19 columns** (identical structure to Export 1) with all values — PB_i, DI_i, and Disclose Income — **specific to that income mode**. This means the intermediate calculated values (which differ between categorical and continuous) are accurate per sheet.

The reason for separate sheets (rather than a single sheet with split columns) is that the two income modes produce **different intermediate values**: the direct_effect, its z-score, the income_high indicator, and the final DI_i all differ between categorical and continuous. Putting them on separate sheets avoids confusion about which intermediate values correspond to which final decision.

---

### Export 3: Disclose Income Only — Compare All Population Modes

**When produced**: Running disclose_income only with population mode set to "Compare all" (which generates 3 population modes). Combined with "Compare both" income mode, this can produce up to 6 sheets.

**File name**: `disclose_income_compare_all_YYYYMMDD_HHMMSS.xlsx`

**Structure**: Multiple sheets — one sheet per population mode × income mode combination. Each combination gets its own dedicated sheet.

**Possible sheet names (up to 6)**:

| Sheet Name | Population Mode | Income Mode |
|-----------|----------------|-------------|
| Copula_Cat | Copula (synthetic) | Categorical |
| Copula_Cont | Copula (synthetic) | Continuous |
| ResSpec_Cat | Research Specification | Categorical |
| ResSpec_Cont | Research Specification | Continuous |
| ResBase_Cat | Research Baseline | Categorical |
| ResBase_Cont | Research Baseline | Continuous |

Each sheet contains the **same 19-column structure as Export 1** (ending with a single `Disclose Income (Y=1)` column), with all values — PB_i, DI_i, and Disclose Income — specific to that particular configuration. Categorical and continuous results are never merged onto the same sheet; they always get separate sheets.
