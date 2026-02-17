## Excel Export Structure

When running only the donation_default decision, there is one Excel export produced in the "Export Results" section at the bottom of the results page. The file structure depends on the combination of income mode and population mode selected.

---

### Scenario 1: Single Configuration

**When**: Single population mode (e.g., "Copula") + single income mode (e.g., "Categorical only").

**File name**: `donation_default_results_YYYYMMDD_HHMMSS.xlsx`

**Structure**: Single sheet — "Donation Results"

**Columns (7):**

| # | Column Name | Description |
|---|-------------|-------------|
| 1 | **Agent ID** | Agent identifier (1-based) |
| 2 | **Honesty_Humility** | Raw Honesty-Humility trait value |
| 3 | **Assigned Allowance Level** | Income level (1–5) |
| 4 | **Study Program** | Agent's university study program |
| 5 | **Group_experiment** | Experimental group (HighSub/MidSub/NoSub) |
| 6 | **TWT+Sospeso [=AW2+AX2]{Periods 1+2}** | Observed prosocial behavior |
| 7 | **donation_default** | Final donation rate as a proportion (0.0–1.0) |

**Notes**:
- Only the 5 trait columns used in the donation regression model are included.
- The donation_default value is the final output after the full process (regression → scaling → anchoring → adjustment → stochastic draw → floor at 0 → rescale to 0–1).
- Numeric columns display with 2 decimal places in Excel, but the underlying values retain full precision.

---

### Scenario 2: Compare Both Income Modes (Same Population)

**When**: Single population mode + "Compare both" income mode. This produces 2 configurations (categorical + continuous) with the **same agents**.

**File name**: `donation_all_configs_YYYYMMDD_HHMMSS.xlsx`

**Structure**: Single sheet — "All Configurations"

Because both income modes use the same agents (same population), they are combined into one sheet with separate donation columns:

**Columns (8):**

| # | Column Name | Description |
|---|-------------|-------------|
| 1 | **Agent ID** | Agent identifier |
| 2 | **Honesty_Humility** | Raw trait value |
| 3 | **Assigned Allowance Level** | Income level (1–5) |
| 4 | **Study Program** | Study program |
| 5 | **Group_experiment** | Experimental group |
| 6 | **TWT+Sospeso [=AW2+AX2]{Periods 1+2}** | Observed prosocial behavior |
| 7 | **donation_default_Categorical** | Donation rate from categorical income run (0.0–1.0) |
| 8 | **donation_default_Continuous** | Donation rate from continuous income run (0.0–1.0) |

**Why one sheet**: The agents are identical across both runs — only the donation rate differs. So the trait columns don't need to be duplicated.

---

### Scenario 3: Compare All Population Modes

**When**: "Compare all" population mode (which generates Copula + Research Spec + Research Baseline). May be combined with "Compare both" income mode.

**File name**: `donation_compare_all_YYYYMMDD_HHMMSS.xlsx`

**Structure**: Multiple sheets — one per population mode. Up to **3 sheets**:

| Sheet Name | Population Mode | Description |
|-----------|----------------|-------------|
| **Copula** | Copula (synthetic) | Synthetic agents generated from copula |
| **ResSpec** | Research Specification | Original 280 participants (bootstrap sampled) |
| **ResBase** | Research Baseline | Original 280 participants (sequential, deterministic) |

Each sheet uses **shortened column names**:

**Columns per sheet (up to 8):**

| # | Column Name | Original Name |
|---|-------------|---------------|
| 1 | **Agent_ID** | agent_id |
| 2 | **Honesty_Humility** | Honesty_Humility |
| 3 | **Income_Level** | Assigned Allowance Level |
| 4 | **Study_Program** | Study Program |
| 5 | **Group** | Group_experiment |
| 6 | **TWT_Sospeso** | TWT+Sospeso [=AW2+AX2]{Periods 1+2} |
| 7 | **donation_Categorical** | donation_default (categorical run) |
| 8 | **donation_Continuous** | donation_default (continuous run) |

**Why separate sheets**: Different population modes have **different agents** with different trait values, so they cannot be combined by row. Within each sheet, both income modes (if "Compare both" was selected) share the same agents, so they sit side by side as columns 7 and 8. If only one income mode was run, only one donation column appears.

---

### Summary

| Scenario | File Name | Sheets | Columns | Key Structure |
|----------|-----------|--------|---------|---------------|
| Single config | `donation_default_results_*.xlsx` | 1 ("Donation Results") | 7 | Traits + one donation rate |
| Compare both | `donation_all_configs_*.xlsx` | 1 ("All Configurations") | 8 | Traits + Cat/Cont donation columns |
| Compare all | `donation_compare_all_*.xlsx` | Up to 3 (Copula/ResSpec/ResBase) | Up to 8 per sheet | Shortened names, Cat/Cont per sheet |
