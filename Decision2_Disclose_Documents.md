Decision 2 – Disclose Documents Configuration

When customized, this decision uses a meta-analytic privacy-calculus model that predicts whether an agent will disclose supporting documents (in order to qualify for Discount Customer status) based on a subset of core personality traits (Extraversion, Neuroticism, Agreeableness) and the agent's income. The algorithm then produces a continuous score (DD_i) that determines a binary outcome: "Y" (disclose) or "N" (do not disclose). Because this decision is only offered to agents who have already disclosed their income and are eligible for the discount, agents who do not qualify are assigned "NA".

This tab allows the user to design the agent logic for Decision 2: disclose documents.

Note that Decision 2 is structurally different from Decision 1 (Disclose Income). It uses only three personality traits (Extraversion, Neuroticism, Agreeableness) — there is no Honesty-Humility, Openness, Religiosity, or observed-prosocial-behaviour anchoring. Income enters as an "inverse income" (Personal Incentive) term rather than through a prosocial mediation path, and the model uses a single baseline intercept (β₀).

The process can be conceptualized as a seven-step procedure:

Eligibility Gate: Determine whether the decision applies to the agent. Only agents who disclosed their income ("Y") and whose income is below the discount threshold are eligible; all other agents are assigned "NA".

Z-Scoring: Standardize the three personality traits (Extraversion, Neuroticism, Agreeableness) using fixed statistics based on the original 280 experiment participants.

Reduced-Form Trait Score: Combine the z-scored traits using the meta-analytically weighted coefficients that fold in the two Bansal et al. (2016) mediators — Privacy Concern and Trust — to produce the trait portion of the score.

Income Effect (Direct Effect): Add the income effect, using either a per-quintile income effect (β_PIcat_q, categorical income mode) or an inverse-income coefficient (continuous income mode).

Composite Standardization: Standardize the combined composite (weighted_dd) to a z-score using fixed statistics (categorical) or runtime statistics (continuous).

Final Combined Equation (DD_i): Add the baseline intercept (β₀) to the standardized composite to form the deterministic score, DD_i = β₀ + z_weighted_dd.

Stochastic Component (σ) and Classification: Optionally introduce randomness with a random draw from a normal distribution centered on DD_i (which already includes β₀); then convert to a binary decision — "Y" if the final DD_i > 0, "N" otherwise.

The controls on this tab enable the user to modify and review the steps of this process, determining the configuration of the agent model corresponding to a particular experiment participant.

Eligibility Gate (Who This Decision Applies To)

Decision 2 is a conditional decision — it only applies to agents who are candidates for the discount. Eligibility is determined by two conditions, both of which must hold:

The agent disclosed their income in Decision 1 (disclose_income = "Y"), and

The agent's income is below the discount income threshold (income < discount_income_threshold; default €12,500).

Agents who fail either condition are assigned disclose_documents = "NA" (not applicable) and do not receive a document-disclosure score for the decision itself. Because only the lowest income levels fall below the discount threshold, the qualified subgroup consists of the lowest-income agents. Note: the model still computes an (ungated) score for every agent for validation and inspection purposes — see disclose_documents_model_y in the Output section — but the platform decision itself respects the eligibility gate.

1. The Privacy-Calculus Model (Meta-Analytic Reduced Form)

The foundation of the model is a privacy-calculus specification in which document disclosure is driven by three constructs drawn from three separate studies: Privacy Concern, Trust, and Personal Incentive (the inverse of income). Because the three predictors originate from different studies, each is assigned a relative weight based on the precision of its reported coefficient, following a standard meta-analytic weighting procedure.

Meta-analytic weighting. For each predictor, the standard error is derived from its coefficient and the z-value implied by its significance level, and the weight is the inverse of the squared standard error:

SE(β) = |β| / z, where z = 1.96 (p < .05), 2.576 (p < .01), 3.291 (p < .001)

Weight (w) = 1 / SE(β)²

The three predictors of Disclose Documents and their weights are:

| Predictor | β | z (p<.01) | SE(β) | Weight | Relative weight |
| --- | --- | --- | --- | --- | --- |
| Privacy Concern → DD | −0.38 | 2.576 | 0.147516 | 45.954 | 48.98% |
| Trust → DD | 0.59 | 2.576 | 0.229037 | 19.063 | 20.32% |
| Personal Incentive → DD | 0.48 | 2.576 | 0.186335 | 28.801 | 30.70% |

The relative weights (sum of weights = 93.818) are: w_PC = 0.489822, w_T = 0.203189, w_PI = 0.306989.

Mediators. Privacy Concern and Trust are themselves predicted from the personality traits (Bansal et al. 2016). These are computed as intermediate constructs and emitted for inspection, but they do not separately alter the decision — the reduced form below already incorporates them:

Equation 1: Privacy Concern: PC_i = 0.12 × z_N_i + 0.14 × z_A_i

Equation 2: Trust: T_i = 0.13 × z_E_i + 0.0762 × z_A_i − 0.0204 × z_N_i

Reduced form. Substituting Equations 1 and 2 into the weighted Disclose Documents equation and collecting terms yields the reduced-form trait score used by the model. The formula is displayed in the "Mathematical Model Formula" section of the tab, as:

weighted_dd (trait part) = 0.015584630336545 × z_E_i − 0.024781455105683 × z_N_i − 0.016923520441338 × z_A_i

Where:

z_E: Z-scored Extraversion (Big5). Coefficient 0.015584630336545 = w_T × 0.59 × 0.13 (Extraversion enters only through Trust).

z_N: Z-scored Neuroticism (Big5). Coefficient −0.024781455105683 = w_PC × (−0.38) × 0.12 + w_T × 0.59 × (−0.0204) (Neuroticism enters through both Privacy Concern and Trust).

z_A: Z-scored Agreeableness. Coefficient −0.016923520441338 = w_PC × (−0.38) × 0.14 + w_T × 0.59 × 0.0762 (Agreeableness enters through both Privacy Concern and Trust).

The Personal Incentive term (income) is added separately in Section 3 (Income Effect). The reduced-form coefficients are fixed from the meta-analytic integration and are not user-adjustable. (These are the corrected reduced-form coefficients: they incorporate the corrected Privacy-Concern weight and the corrected Neuroticism-via-Trust path, and reproduce the professor's validated Stata results.)

2. Z-Scoring (Standardization)

All z-scoring uses fixed statistics from the original 280 experiment participants. Statistics are NOT recomputed for each bootstrap sample or simulation run. This approach follows Stata's egen z_var = std(var), which standardizes once on the original data.

Individual Trait Z-Scoring

| Trait | Mean | SD |
| --- | --- | --- |
| ExtraversionBig5 | 3.557857 | 0.6989565 |
| NeuroticismBig5 | 2.702143 | 0.6839657 |
| Agreeable | 3.546071 | 0.3732712 |

Formula: z_trait = (raw_value − mean) / sd

Composite Variable Z-Scoring

After the trait score and income effect are combined into the composite (weighted_dd), the composite is also z-scored using fixed statistics:

| Composite Variable | Mean | SD |
| --- | --- | --- |
| weighted_dd (categorical) | 0.0 | 0.13783441040509267 |

Exception: In continuous income mode, the composite mean and SD are computed at runtime across all agents in the current simulation (because income values are stochastically generated each run). The frozen fallback value is 0.15115619105427494, but it is overridden by the runtime statistic before any agent is scored.

3. Income Effect (Direct Effect)

The income effect captures the influence of the agent's income on document disclosure. It differs by income specification mode.

Categorical Mode

Uses per-quintile income effects (β_PIcat_q) that absorb the income effect, with NO separate income coefficient. The income effect is added to the reduced-form trait score:

weighted_dd = β_PIcat_q[quintile_i] + 0.015584630336545 × z_E_i − 0.024781455105683 × z_N_i − 0.016923520441338 × z_A_i

The income quintile is expressed as the inverse of income (Personal Incentive): PIcat = 200 if income-level = €12; 128 if = €32; 72 if = €72; 32 if = €128; 12 if = €200.

Income Quintile Effects (β_PIcat_q):

| Quintile | β_PIcat_q |
| --- | --- |
| Q1 (€200) | 0.1464773 |
| Q2 (€128) | 0.0902694 |
| Q3 (€72) | 0.0384204 |
| Q4 (€32) | −0.0522756 |
| Q5 (€12) | −0.2393718 |

Each value equals the regression base (−0.2393718) plus that quintile's differential income-quintile dummy. Because the quintiles are labeled by inverse income, Q1 (€200) corresponds to the lowest actual income (€12) and Q5 (€12) to the highest (€200); the effect therefore decreases as actual income rises, reflecting that higher-income agents are less likely to disclose documents.

Continuous Mode

Drops the per-quintile income effects and instead uses a single inverse-income (Personal Incentive) coefficient:

weighted_dd = 0.015584630336545 × z_E_i − 0.024781455105683 × z_N_i − 0.016923520441338 × z_A_i + 0.14735467793568 × z_PIcont_i

Where z_PIcont is the z-scored Personal Incentive (inverse income). Personal Incentive is defined as picont = maximum income − agent income, so that lower-income agents have a higher Personal Incentive. Because z(picont) = −z(income) exactly (the maximum-income term cancels in standardization), z_PIcont is computed at runtime as the negative of the agent's z-scored income against the current population, reusing the same income statistics computed for Decision 1.

Income Specification Selection

The user selects the income specification mode using the radio button in the "Income Specification" section:

Categorical only: Uses the 5 income quintile effects (β_PIcat_q).

Continuous only: Drops the income quintile effects and uses the single inverse-income coefficient.

Compare both: Runs both specifications and displays comparison results.

The reduced-form coefficients (0.015584630336545, −0.024781455105683, −0.016923520441338, 0.14735467793568) are fixed from the meta-analytic integration and are not user-adjustable.

4. Final Combining Equation (DD_i)

The final document-disclosure score adds the baseline intercept to the standardized composite:

DD_i = β₀ + z_weighted_dd_i

Where:

β₀ (Intercept): Default = −0.75. This parameter can be modified using the "Intercept Override" section of the tab. Higher (less negative) values increase the baseline probability of disclosure. β₀ is applied after the composite is standardized — it sets the baseline tendency to disclose on the standardized scale, which is why it does not appear inside the weighted_dd formula.

z_weighted_dd: The z-scored composite from Section 2 (Composite Standardization), i.e. (weighted_dd − mean) / sd, using the categorical fixed stats (mean = 0.0, sd = 0.13783441040509267) or the runtime-computed continuous stats.

If the stochastic component is enabled, a random draw is applied to DD_i before classification (see Section 5); otherwise the deterministic DD_i is used directly.

5. Stochastic Component (σ)

Optionally, the model introduces randomness by drawing from a normal distribution centered on the deterministic score DD_i (which already includes β₀). This captures the natural variability observed in the experiment.

The Draw

For each agent:

draw ~ Normal(μ = DD_i, σ = σ_scaled)

Where: μ = the agent's deterministic score DD_i (from Section 4); σ_scaled = σ_raw × coefficient (scale factor). Unlike Decision 1, no unit conversion is applied: the σ is applied directly to DD_i, which is already on the standardized score scale.

σ Provenance. The base σ is derived from the variability in the original research data corresponding to the observed measure of donation consumption (receiving) — specifically the two-period consumption of community donations (sospeso) plus received transfers, normalized to the [0, 1] range (consumedtransferssospeso2periods). This differs from Decision 1 and Decision 3, whose σ is sourced from prosocial giving (TWT + Sospeso).

σ Modes

The user can choose between two σ modes using the radio button in the "Stochastic Component" section:

Uniformly (Overall σ)

One common σ for all agents:

σ_overall = 0.1606568355; σ_scaled = σ_overall × coefficient

The σ Coefficient slider (default = 1.0, range 0.0–2.0) acts as a multiplier. For example: Coefficient = 1.0 → σ_scaled = 0.1606568; Coefficient = 0.1 → σ_scaled = 0.0160657; Coefficient = 0.0 → σ = 0 (no stochastic component).

Quintiles (Per-Income-Level σ)

Different σ for different income groups, computed from the standard deviation of the normalized two-period consumption within each income level:

| Income Level | Budget | Base σ |
| --- | --- | --- |
| Level 1 | €12 | 0.1690045319 |
| Level 2 | €32 | 0.1909536245 |
| Level 3 | €72 | 0.1077552751 |
| Level 4 | €128 | 0.1436159112 |
| Level 5 | €200 | 0.1562794728 |

Each level has its own coefficient slider (default = 1.0): σ_scaled(q) = Base σ_quintile(q) × Coefficient(q).

Important: Per-quintile σ values only apply to the categorical income specification. In continuous mode (or the continuous run of "Compare both"), the simulation always uses the overall σ, because level-specific σ are based on categorical budget levels and are not meaningful for continuous income.

When Stochastic Is Applied

The stochastic component is controlled differently depending on the population mode:

| Population Mode | UI Control | Stochastic Behavior |
| --- | --- | --- |
| Copula (synthetic) | "Add Normal(score, σ) draw to Copula runs" checkbox | ON when checked |
| Research Specification | "Use Normal(score, σ) draw in Research Specification mode" checkbox | ON when checked |
| Research Baseline | No control (always off) | Always OFF — deterministic score only |
| Compare all | Separate checkboxes for Copula and Research Spec | Independent per mode; Baseline always OFF |

Note that adding the stochastic element influences the disclosure rate depending on where agents sit relative to the decision threshold. Although the intercept is negative (β₀ = −0.75), the qualified (low-income) agents receive a positive income quintile effect (β_PIcat_q) and therefore sit well above the threshold, so the majority of the qualified subgroup would disclose deterministically. The stochastic draw adds symmetric variance, i.e., equal amounts of positive and negative deviations; because the majority of the qualified agents are already above the threshold, there are more agents at risk of being pushed below 0 (Y→N) than agents below 0 that could be pushed above (N→Y). Enabling the stochastic component therefore slightly lowers the disclosure rate for the qualified subgroup.

6. Classification (Final Decision)

The continuous DD_i score is converted to a binary decision:

disclose_documents = "Y" if DD_i > 0

disclose_documents = "N" if DD_i ≤ 0

(For agents who fail the eligibility gate, disclose_documents = "NA".)

No additional thresholding or probability conversion is applied — the zero-crossing of DD_i is the decision boundary.

Tab Controls Reference

Left Column

Income Specification

Radio button with three options:

Categorical only: Uses 5 income quintiles with per-quintile effects (β_PIcat_q).

Continuous only: Uses actual income amounts through a single inverse-income coefficient.

Compare both: Runs both specifications for side-by-side comparison.

Right Column

Stochastic Component

Controls vary by population mode (set on Page 1):

Copula: Checkbox to add Normal(score, σ) draw to Copula runs.

Research Specification: Checkbox to enable Normal(score, σ) draws.

Research Baseline: No controls — always deterministic.

Compare all: Separate checkboxes for Copula and Research Spec.

When stochastic is enabled, additional controls appear:

σ mode: Radio button — "Uniformly" (single σ) or "Quintiles" (per income level).

σ Coefficient slider (Uniformly mode): Multiplier for base σ (range 0.0–2.0, default 1.0).

Per-quintile coefficient sliders (Quintiles mode): Individual multipliers per income level.

Full Width

Mathematical Model Formula

Expandable section showing the current equations with substituted parameter values. Updates dynamically based on income mode selection and current slider values.

Intercept Override

Three-column display:

Research Default: Fixed reference value (−0.75).

Override Value: Number input to modify β₀ (negative values only; range −5.0 to 0.0).

Impact Preview: Shows the difference from research default.

Actions & Management

Reset All: Resets all disclose documents configuration to research defaults.

Reset Intercept: Resets only the intercept to −0.75.

Default Parameter Values

| Parameter | Default Value | User-Adjustable | Location |
| --- | --- | --- | --- |
| β₀ (Intercept) | −0.75 | Yes | Intercept Override |
| σ_overall | 0.1606568355 | No (base value) | Fixed |
| σ Coefficient | 1.0 | Yes | Stochastic slider |
| Stochastic enabled | Off | Yes | Stochastic checkbox |
| σ mode | Overall | Yes | σ mode radio |
| Income mode | Categorical only | Yes | Income Specification radio |
| Reduced-form coefficients | Fixed | No | — |
| Income quintile effects (β_PIcat_q) | Fixed | No | — |
| Z-scoring statistics | Fixed | No | — |

Output

The decision produces the following outputs per agent:

| Output Field | Description |
| --- | --- |
| disclose_documents | Binary decision: "Y", "N", or "NA" (not eligible) |
| disclose_documents_model_y | Ungated deterministic decision (1 if the model would disclose, 0 otherwise) — computed for every agent, ignoring the eligibility gate, for validation |
| disclose_documents_raw | Final DD_i value (with stochastic draw if enabled), before classification |
| disclose_documents_score | Deterministic DD_i value (without stochastic draw) |
| disclose_documents_intercept | β₀ value used |
| disclose_documents_trait_terms | Reduced-form trait score (before the income effect) |
| disclose_documents_z_extraversion | Z-scored Extraversion |
| disclose_documents_z_neuroticism | Z-scored Neuroticism |
| disclose_documents_z_agreeable | Z-scored Agreeableness |
| disclose_documents_z_picont | Z-scored Personal Incentive (continuous mode; 0 in categorical) |
| disclose_documents_privacy_concern | Privacy Concern mediator (trait part, Equation 1) |
| disclose_documents_trust | Trust mediator (trait part, Equation 2) |
| disclose_documents_agent_income | Agent's income (used to derive Personal Incentive) |
| disclose_documents_weighted_dd | Composite score (trait score + income effect), before standardization |
| disclose_documents_z_weighted_dd | Standardized composite |
| disclose_documents_sigma_used | σ actually applied (0 if stochastic disabled) |
| disclose_documents_income_mode | "categorical" or "continuous" |

These values are available in the Excel export from the results page.

Downstream Impact

The disclose_documents decision completes the customer-type assignment for agents who disclosed income and are below the discount threshold:

disclose_income = "Y" AND income < discount threshold AND disclose_documents = "Y" → Agent becomes a Discount Customer (receives discounted prices).

disclose_income = "Y" AND income < discount threshold AND disclose_documents = "N" → Agent remains a Fixed Customer (uses FIXED pricing).

(Agents who did not disclose income, or whose income is at or above the threshold, are unaffected by this decision — their customer type is already determined by Decision 1.)

Customer types propagate to Decision 9 (Purchase Now vs Bid) and all subsequent transaction and vendor-selection decisions.

Excel Export Structure

The disclose documents decision produces Excel export files whose structure depends on the combination of income mode and population mode selected. Every sheet — regardless of scenario — contains the same 15 columns, in the same order. The scenarios differ only in how many sheets the file has, not in the columns.

Columns present on every sheet

| No. | Column Name | Description |
| --- | --- | --- |
| 1 | Agent ID | Agent identifier (1-based) |
| 2 | Extraversion | Raw Extraversion trait (ExtraversionBig5), non-standardized |
| 3 | Neuroticism | Raw Neuroticism trait (NeuroticismBig5), non-standardized |
| 4 | Agreeable | Raw Agreeableness trait (Agreeable), non-standardized |
| 5 | Assigned Allowance Level | Budget/income level (1–5) |
| 6 | Income | Actual income value (€) |
| 7 | TWT+Sospeso | Observed prosocial behaviour (Periods 1+2), shown for reference; the disclose documents model does not use it |
| 8 | PersonalIncentive | Personal Incentive = max_income − income (max taken over the population) |
| 9 | Intercept | β₀ baseline intercept actually used in the run |
| 10 | PrivacyConcern | Privacy Concern mediator (Equation 1 trait part), standardized over the population (ddof=1). The Eq 1 β₀ is a per-agent constant and drops out of the standardized value, so it is not included here |
| 11 | Trust | Trust mediator (Equation 2 trait part), standardized over the population (ddof=1). The Eq 2 β₀ likewise drops out of the standardized value |
| 12 | Disclosure Document | Final DD_i value (after the optional stochastic draw), before Y/N classification. The deterministic pre-draw score is intentionally not exported (it equals this value when the stochastic component is off) |
| 13 | Disclose Income (Y=1) | Decision 1 outcome: 1 = Y, 0 = N, N/A if Disclose Income was not computed in the run |
| 14 | Disclose Documents (Y=1) | Final binary decision: 1 = Y, 0 = N; N/A unless the agent qualifies — i.e. Disclose Income = Y AND income ≤ the €12,500 discount threshold |
| 15 | customer_type | Discount / Fixed / Regular |

Note: the export deliberately carries only the final `Disclosure Document` value (column 12), not the internal `weighted_dd` / `z_weighted_dd` composites. The mediator columns (`PrivacyConcern`, `Trust`) are standardized over the exported population, so their values are relative to that sheet's agents.

Scenario 1: Single Configuration — single population mode + single income mode. File: `disclose_documents_results_YYYYMMDD_HHMMSS.xlsx`. One sheet named "Disclose Documents" with the 15 columns above.

Scenario 2: Compare Both Income Modes (Same Population) — single population mode + "Compare both". File: `disclose_documents_compare_YYYYMMDD_HHMMSS.xlsx`. Separate sheets for the two income runs, named "Categorical" and "Continuous" (the income effect, the composite, and the final DD_i differ between the two income modes, so they are not merged onto one sheet). Each sheet carries the same 15 columns.

Scenario 3: Compare All Population Modes — "Compare all" population mode (Copula + Research Specification + Research Baseline), each run for both income modes. File: `disclose_documents_compare_YYYYMMDD_HHMMSS.xlsx`. One sheet per population-mode × income-mode combination (up to six), named `Copula_Cat`, `Copula_Cont`, `ResSpec_Cat`, `ResSpec_Cont`, `ResBase_Cat`, `ResBase_Cont`. Different population modes have different agents, so they are never merged by row. Each sheet carries the same 15 columns.

The qualified subgroup (disclose_documents ≠ "NA") is reported separately from the ineligible ("NA") agents so that the disclosure rate is computed over eligible agents only.
