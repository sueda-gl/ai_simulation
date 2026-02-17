# Decision 1 - Disclose Income Configuration

When customized, this decision uses a two-stage mediation model that predicts whether an agent will disclose their income based on their core personality traits (e.g., Agreeableness, Honesty-Humility, Extraversion, Neuroticism, Openness), their religiosity composite, their observed prosocial behavior, and their income level. It then produces a continuous score (DI_i) that determines a binary outcome: "Y" (disclose) or "N" (do not disclose).

This tab allows the user to design the agent logic for Decision 1: disclose income.

The process can be conceptualized as a seven-step procedure:

1. **Prosocial Behavior Prediction (calc_PB_i)**: Predict a prosocial behavior score based on the agent's personality traits (Agreeableness, Openness, Honesty-Humility, and Religiosity).

2. **Z-Scoring**: Standardize all individual traits and composite variables using fixed statistics from the original 280 experiment participants.

3. **Anchor Calculation**: Compute a weighted average of observed prosocial behavior (TWT+Sospeso) and the calculated prosocial behavior to create a deterministic "anchor" value (PB_i).

4. **Stochastic Component (σ)**: Optionally, introduce randomness by making a random draw from a normal distribution centered on the anchor value.

5. **Direct Effect Calculation**: Compute the direct personality and income effects on disclosure intention (using either categorical level-specific intercepts or a continuous income coefficient, depending on the income specification mode).

6. **Final Combining Equation (DI_i)**: Combine the intercept (β₀), the z-scored direct effect, and the z-scored prosocial behavior effect (weighted by income-high indicator) to produce a continuous disclosure intention score.

7. **Classification**: Convert to a binary decision — "Y" if DI_i > 0, "N" otherwise.

The controls on this tab enable the user to modify and review the steps of this process, determining the configuration of the agent model corresponding to a particular experiment participant.


---

## 1. The Two-Stage Mediation Model

The foundation of the model is a two-stage mediation structure where personality traits first predict prosocial behavior (Equation 1), which then mediates the effect on income disclosure intention (Equation 2).


### Equation 1: Calculated Prosocial Behavior (calc_PB_i)

The first stage is a linear combination of z-scored personality traits that predicts prosocial behavior:

```
calc_PB_i = 0.023776 × z_Agreeable_i + 0.016537 × z_Openness_i + 0.0295482 × z_HH_i + 0.0677157 × z_Religious_i
```

Where:

- **z_Agreeable**: Z-scored Agreeableness. Standardized using the original 280 participants' statistics (mean = 3.546071, sd = 0.3732712).

- **z_Openness**: Z-scored Openness to Experience (Big5). Standardized using (mean = 4.060714, sd = 0.5068274).

- **z_HH**: Z-scored Honesty-Humility. Standardized using (mean = 3.392187, sd = 0.5586641).

- **z_Religious**: Z-scored religiosity composite. This is a derived variable computed as follows:
  1. Scale ReligiousService to the 0–1 range: `rs_01 = (ReligiousService − 0) / (4 − 0)`
  2. Equal-weight average with ReligiousAffiliation: `Religious_composite = (ReligiousAffiliation + rs_01) / 2`
  3. Z-score using fixed statistics (mean = 0.1892857, sd = 0.2618114).

The coefficients (0.023776, 0.016537, 0.0295482, 0.0677157) are fixed from the original regression analysis and are not user-adjustable. The formula is displayed in the "Mathematical Model Formula" section of the tab, as:

```
calc_PB_i = 0.023776 × z_Agreeable + 0.016537 × z_Openness + 0.0295482 × z_HH + 0.0677157 × z_Religious
```


---

## 2. Z-Scoring (Standardization)

All z-scoring uses **fixed statistics from the original 280 experiment participants**. Statistics are NOT recomputed for each bootstrap sample or simulation run. This approach follows Stata's `egen z_var = std(var)` which standardizes once on the original data.

### Individual Trait Z-Scoring

| Trait | Mean | SD |
|-------|------|----|
| Agreeable | 3.546071 | 0.3732712 |
| Honesty_Humility | 3.392187 | 0.5586641 |
| ExtraversionBig5 | 3.557857 | 0.6989565 |
| NeuroticismBig5 | 2.702143 | 0.6839657 |
| OpennessBig5 | 4.060714 | 0.5068274 |
| Religious Composite | 0.1892857 | 0.2618114 |
| TWT+Sospeso (Observed PB) | 3.357143 | 9.899547 |

Formula: `z_trait = (raw_value − mean) / sd`

### Composite Variable Z-Scoring

After computing composite variables (weighted_prosocial, direct_effect, anchored_pb), these are also z-scored using fixed statistics:

| Composite Variable | Mean | SD |
|-------------------|------|----|
| weighted_prosocial (calc_PB) | 0.0 | 0.08608372 |
| weighted_disclosure_categorical (direct_effect) | 0.0 | 0.025040462 |
| anchored_pb (PB_i) | 0.0 | 0.7984211971 |

**Exception**: In continuous income mode, the direct_effect mean and SD are computed at runtime across all agents in the current simulation (because income values are stochastically generated each run).


---

## 3. Anchor Calculation (Combining Observed and Calculated PB)

The anchor value combines the agent's observed prosocial behavior with the trait-based prediction:

```
PB_i = W_OPB × z_obs_PB_i + (1 − W_OPB) × z_calc_PB_i
```

Where:

- **W_OPB** (Weight for Observed Prosocial Behavior): Default = **0.25**. This parameter can be directly modified using the "W_OPB" slider in the "Anchor Mix" section of the tab.

- **z_obs_PB**: Z-scored observed prosocial behavior (TWT+Sospeso from the experiment). Standardized using (mean = 3.357143, sd = 9.899547).

- **z_calc_PB**: Z-scored calculated prosocial behavior from Equation 1. Standardized using (mean = 0.0, sd = 0.08608372).

With the default W_OPB = 0.25, the anchor places **25% weight on observed behavior** and **75% weight on the trait-based prediction**. The user can adjust this balance using the slider.


---

## 4. Stochastic Component (σ)

Optionally, the model introduces randomness by drawing from a normal distribution centered on the anchor value. This captures the natural variability observed in prosocial behavior.

### The Draw

For each agent:

```
draw_k ~ Normal(μ = PB_i, σ = σ_scaled)
```

Where:
- **μ** = the agent's deterministic anchor value (PB_i from Step 3)
- **σ_scaled** = σ_raw × coefficient (scale factor)

So σ is the scale of natural variability in the experiment. The stochastic draw is applied to the anchor value (PB_i), NOT to the final DI_i output.

### σ Modes

The user can choose between two σ modes using the radio button in the "Stochastic Component" section:

#### Uniformly (Overall σ)

One common σ for all agents:

```
σ_overall = sd(TWT+Sospeso) = 9.899547
σ_scaled = σ_overall × coefficient
```

The **σ Coefficient** slider (default = 1.0, range 0.0–2.0) acts as a multiplier. For example:
- Coefficient = 1.0 → σ_scaled = 9.899547
- Coefficient = 0.1 → σ_scaled = 0.98995

#### Quintiles (Per-Income-Level σ)

Different σ for different income groups, computed from the standard deviation of TWT+Sospeso within each income quintile:

| Income Level | Budget | Base σ |
|-------------|--------|--------|
| Level 1 | €12 | 5.705052 |
| Level 2 | €32 | 3.069326 |
| Level 3 | €72 | 3.532226 |
| Level 4 | €128 | 12.219622 |
| Level 5 | €200 | 16.854622 |

Each level has its own coefficient slider (default = 1.0):

```
σ_scaled(q) = σ_quintile(q) × coefficient(q)
```

**Important**: Per-quintile σ values only apply to the **categorical** income specification. In continuous mode (or the continuous run of "Compare both"), the simulation always uses the overall σ, because level-specific sigmas are not meaningful for continuous income. The note on the UI dynamically reflects the current overall coefficient value.

### When Stochastic Is Applied

The stochastic component is controlled differently depending on the population mode:

| Population Mode | UI Control | Stochastic Behavior |
|----------------|------------|-------------------|
| **Copula (synthetic)** | "Add Normal(anchor, σ) draw to Copula runs" checkbox | ON when checked |
| **Research Specification** | "Use Normal(anchor, σ) draw in Research mode" checkbox | ON when checked |
| **Research Baseline** | No control (always off) | Always OFF — deterministic anchor values only |
| **Compare all** | Separate checkboxes for Copula and Research Spec | Independent per mode; Baseline always OFF |


---

## 5. Direct Effect Calculation

The direct effect captures the direct influence of personality traits (and income) on disclosure intention, separate from the prosocial behavior mediation path. This differs by income specification mode.

### Categorical Mode

Uses level-specific intercepts that absorb the income effect, with NO separate income coefficient:

```
direct_effect = β_income_q[quintile_i] + 0.00680238 × z_E_i + 0.0173732 × z_N_i + 0.0163905 × z_HH_i
```

**Income Quintile Effects (β_income_q):**

| Quintile | Budget | Intercept |
|----------|--------|-----------|
| Q1 | €12 | 0.0089007 |
| Q2 | €32 | 0.0055352 |
| Q3 | €72 | 0.0023109 |
| Q4 | €128 | −0.0032216 |
| Q5 | €200 | −0.0145324 |

The intercepts decrease with income level, reflecting that higher-income agents are less likely to disclose.

**I_high indicator** (categorical): `I_high = 1` if Assigned Allowance Level > 3, else `0`.

### Continuous Mode

Uses a single β₀ intercept and an explicit income coefficient:

```
direct_effect = 0.00680238 × z_E_i + 0.0173732 × z_N_i + 0.0163905 × z_HH_i + (−0.008988) × z_I_i
```

Where:
- **z_I**: Z-scored actual income of the agent, computed at runtime against the population: `z_I = (income − population_mean) / population_sd`
- **I_high indicator** (continuous): `I_high = 1` if agent income > population median, else `0`

### Income Specification Selection

The user selects the income specification mode using the radio button in the "Income Specification" section:

- **Categorical only**: Uses the 5 level-specific intercepts (β_income_q)
- **Continuous only**: Uses the single β₀ intercept + income coefficient
- **Compare both**: Runs both specifications and displays comparison results

The coefficient values (0.00680238, 0.0173732, 0.0163905, −0.008988) are fixed from the original regression and are not user-adjustable.


---

## 6. Final Combining Equation (DI_i)

The final disclosure intention score combines all components:

### Categorical Mode

```
DI_i = β_income_q[quintile_i] + (1 − W_PB) × z_direct_effect + W_PB × (z_PB_i × I_high)
```

Note: In categorical mode, the level-specific intercepts are part of the direct_effect before z-scoring. There is no separate β₀ (it defaults to 0.0).

### Continuous Mode

```
DI_i = β₀ + (1 − W_PB) × z_direct_effect + W_PB × (z_PB_i × I_high)
```

Where:

- **β₀ (Intercept)**: Default = **0.75**. This parameter can be modified using the "Intercept Override" section of the tab. Higher values increase the baseline probability of disclosure.

- **W_PB** (Weight for Prosocial Behavior effect): Default = **0.50**. This parameter can be modified using the "W_PB" slider in the "Anchor Mix" section.

- **z_direct_effect**: The z-scored direct effect from Step 5. Standardized using:
  - Categorical: fixed stats (mean = 0.0, sd = 0.025040462)
  - Continuous: runtime-computed stats

- **z_PB_i**: The z-scored prosocial behavior anchor (with optional stochastic draw). Standardized using (mean = 0.0, sd = 0.7984211971).

- **I_high**: Binary income indicator (1 if high income, 0 otherwise).

The `(1 − W_PB)` term weights the direct personality/income effect, while the `W_PB` term weights the prosocial behavior mediation effect. With the default W_PB = 0.50, both paths contribute equally.

**Important**: The prosocial behavior effect is multiplied by I_high, meaning that prosocial behavior only influences disclosure for agents with above-median (or above-level-3) income. For low-income agents (I_high = 0), the prosocial term drops out entirely.


---

## 7. Classification (Final Decision)

The continuous DI_i score is converted to a binary decision:

```
disclose_income = "Y" if DI_i > 0
disclose_income = "N" if DI_i ≤ 0
```

No additional thresholding or probability conversion is applied — the zero-crossing of DI_i is the decision boundary.


---

## Tab Controls Reference

### Left Column

#### Income Specification
Radio button with three options:
- **Categorical only**: Uses 5 income levels with level-specific intercepts
- **Continuous only**: Uses actual income amounts with a single β₀ and income coefficient
- **Compare both**: Runs both specifications for side-by-side comparison

### Right Column

#### Stochastic Component
Controls vary by population mode (set on Page 1):
- **Copula**: Checkbox to add Normal(anchor, σ) draw to Copula runs
- **Research Specification**: Checkbox to enable Normal(anchor, σ) draws
- **Research Baseline**: No controls — always deterministic
- **Compare all**: Separate checkboxes for Copula and Research Spec

When stochastic is enabled, additional controls appear:
- **σ mode**: Radio button — "Uniformly" (single σ) or "Quintiles" (per income level)
- **σ Coefficient slider** (Uniformly mode): Multiplier for base σ (range 0.0–2.0, default 1.0)
- **Per-quintile coefficient sliders** (Quintiles mode): Individual multipliers per income level

#### Anchor Mix
- **W_OPB slider**: Observed vs Calculated prosocial behavior weight (range 0.0–1.0, default 0.25)
- **W_PB slider**: Prosocial behavior effect weight in final equation (range 0.0–1.0, default 0.50)

### Full Width

#### Mathematical Model Formula
Expandable section showing the current equations with substituted parameter values. Updates dynamically based on income mode selection and current slider values.

#### Intercept Override
Three-column display:
- **Research Default**: Fixed reference value (0.75)
- **Override Value**: Number input to modify β₀ (range 0.0–5.0)
- **Impact Preview**: Shows the difference from research default

#### Actions & Management
- **Reset All**: Resets all disclose income configuration to research defaults
- **Reset Intercept**: Resets only the intercept to 0.75


---

## Default Parameter Values

| Parameter | Default Value | User-Adjustable | Location |
|-----------|--------------|-----------------|----------|
| β₀ (Intercept) | 0.75 | Yes | Intercept Override |
| W_OPB | 0.25 | Yes | Anchor Mix slider |
| W_PB | 0.50 | Yes | Anchor Mix slider |
| σ_overall | 9.899547 | No (base value) | Fixed |
| σ Coefficient | 1.0 | Yes | Stochastic slider |
| Stochastic enabled | Off | Yes | Stochastic checkbox |
| σ mode | Overall | Yes | σ mode radio |
| Income mode | Categorical only | Yes | Income Specification radio |
| Eq1 coefficients | Fixed | No | — |
| Eq2 coefficients | Fixed | No | — |
| Categorical intercepts | Fixed | No | — |
| Z-scoring statistics | Fixed | No | — |


---

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

---

### Export 4: Disclosure & Customer Types (from Decision 2 visualization)

> **Note**: This export is NOT produced by the Disclose Income (Decision 1) visualization. It comes from the **Disclose Documents (Decision 2)** visualization section — specifically the "Customer Type Distribution" area that appears below the disclose_documents results. It is included here for reference because it contains the disclose_income column and shows how Decision 1 feeds into customer types.

**When produced**: Running a simulation that includes disclose_documents. The download button appears inside the Decision 2 results section, under the Customer Type Distribution heading.

**File name**: `agent_disclosure_customer_types.xlsx`

**Structure**: Single sheet — "Agent Disclosure Data"

**Columns:**

| # | Column Name | Description |
|---|-------------|-------------|
| 1 | **Agent ID** | Agent identifier |
| 2 | **Honesty_Humility** | Raw trait value (rounded to 2 decimals) |
| 3 | **Assigned Allowance Level** | Income level (1–5) |
| 4 | **Study Program** | Agent's study program |
| 5 | **Group_experiment** | Experimental group assignment |
| 6 | **TWT+Sospeso [=AW2+AX2]{Periods 1+2}** | Observed prosocial behavior (rounded to 2 decimals) |
| 7 | **income** | Actual income value (rounded to 2 decimals) |
| 8 | **disclose_income** | Decision 1 result: 1 = Y, 0 = N |
| 9 | **disclose_documents** | Decision 2 result: 1 = Y, 0 = N, "N/A" = not applicable |
| 10 | **Regular** | Customer type indicator: 1 if Regular, 0 otherwise |
| 11 | **Fixed** | Customer type indicator: 1 if Fixed, 0 otherwise |
| 12 | **Discount** | Customer type indicator: 1 if Discount, 0 otherwise |

**Customer type logic**:
- **Regular** (disclose_income = N): Pays PN prices or places BIDs
- **Fixed** (disclose_income = Y, income ≥ threshold or docs not disclosed): Uses FIXED pricing
- **Discount** (disclose_income = Y, income < threshold, disclose_documents = Y): Gets DISCOUNT pricing

---

### Export 5: Agent-Level Excel (Full Simulation — Two-Level Export)

**When produced**: Running a complete simulation with all decisions.

**File name**: Part of the two-level export system (Agent-Level sheet).

**Disclose Income columns within the Agent-Level export:**

| Column Name | Description |
|-------------|-------------|
| **Agreeable** | Raw trait value |
| **Openness** | Raw trait value |
| **Honesty_Humility** | Raw trait value |
| **Extraversion** | Raw trait value |
| **Neuroticism** | Raw trait value |
| **ReligiousAffiliation** | Raw binary value |
| **ReligiousService** | Raw ordinal value |
| **Religious** | Computed religiosity composite |
| **Assigned Allowance Level** | Income level (1–5) |
| **income** | Actual income value |
| **income_category** | Income category label (or "N/A" for Regular customers) |
| **I-High** | Income-high indicator (1/0), or "N/A" if disclose_income was defaulted |
| **WOPB** | W_OPB value used, or "N/A" if defaulted |
| **WPB** | W_PB value used, or "N/A" if defaulted |
| **Intercept** | β₀ value used, or "N/A" if defaulted |
| **PB_i** | Anchored prosocial behavior, or "N/A" if defaulted |
| **DI_i** | Continuous DI value, or "N/A" if defaulted |
| **disclose_income** | Final decision: 1 = Y, 0 = N |
| **disclose_documents** | Decision 2: 1 = Y, 0 = N |
| **customer_type** | "regular", "fixed", or "discount" |

**Note**: When disclose_income was not explicitly selected (using default coin-flip), the calculated columns (I-High, WOPB, WPB, Intercept, PB_i, DI_i) show "N/A" because the two-stage mediation model was not executed.

The Agent-Level export also includes columns for all other decisions (donation_default, rejected_transaction priorities, vendor_choice_weights, purchasing_quantity, purchasing_frequency, vendor_selection, and transaction-level averages).

---

### Export 6: Transaction-Level Excel (Full Simulation — Two-Level Export)

**When produced**: Running a complete simulation with all decisions.

**File name**: Part of the two-level export system (Transaction-Level sheet).

This is a request-level export with one row per purchase request. Disclose income columns included:

| Column Name | Description |
|-------------|-------------|
| **disclose_income** | Decision 1 result: 1 = Y, 0 = N |
| **disclose_documents** | Decision 2 result: 1 = Y, 0 = N |
| **customer_type** | "Regular", "Fixed", or "Discount" (capitalized) |

These appear alongside per-request transaction details (vendor, price, quantity, period, etc.).

---

### Summary: All Disclose Income Excel Exports

| Export | File Name | When | Sheets | Key Feature |
|--------|-----------|------|--------|-------------|
| 1. Single Config | `agent_disclose_income_data.xlsx` | DI only, single mode | 1 | Full 19 columns |
| 2. Compare Both | `disclose_income_compare_both_*.xlsx` | DI only, Compare both | 1–2 | Split Cat/Cont columns |
| 3. Compare All | `disclose_income_compare_all_*.xlsx` | DI only, Compare all pop modes | Up to 6 | Sheet per config |
| 4. Disclosure + Customer Types | `agent_disclosure_customer_types.xlsx` | Decision 2 visualization | 1 | Customer type indicators (from Decision 2, not Decision 1) |
| 5. Agent-Level (Two-Level) | Two-level export | Full simulation | 1 | All decisions, N/A if defaulted |
| 6. Transaction-Level (Two-Level) | Two-level export | Full simulation | 1 | Per-request with customer type |


---

## Downstream Impact

The disclose_income decision directly determines the agent's **customer type**, which affects pricing and purchasing behavior throughout the rest of the simulation:

- **disclose_income = "N"** → Agent becomes a **Regular Customer** (pays PN prices or places BIDs)
- **disclose_income = "Y"** AND income ≥ discount threshold → Agent becomes a **Fixed Customer** (uses FIXED pricing)
- **disclose_income = "Y"** AND income < threshold AND disclose_documents = "Y" → Agent becomes a **Discount Customer** (gets discounted prices)

Customer types propagate to Decision 9 (Purchase Now vs Bid) and all subsequent transaction and vendor selection decisions.
