# Decision 4 – Rejected Transaction Defaults Configuration

When customized, this decision uses five separate sub-decision mechanisms that predict how an agent would like the platform to handle a purchase transaction that has been rejected. Four of the mechanisms are based on the agent's core personality traits (and, for two of them, on income): loyalty to the vendor, willingness to pay, risk-taking propensity, and flexibility. Each of these produces a priority ranking of the five rejected transaction options. The fifth mechanism, based on the agent's tendency to plan, produces the number of options the agent pre-selects (0–5). The four rankings are then reconciled into a single integrated default list per agent, which is truncated to the agent's options list length. Unlike Decisions 1 and 2, the outcome is not a binary classification but an ordered list of options.

The five rejected transaction options are:

- Option 1: Purchase from another (higher) price category of the same vendor
- Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor
- Option 3: Purchase from the current vendor at PN price
- Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)
- Option 5: Forgo the purchase request

The process can be summarised as a seven-step procedure:

1. Z-Scoring: Standardize the Big-5 personality traits, income, and the observed flexibility measure using fixed statistics based on the original 280 experiment participants.
2. Element Scores: Compute five weighted scores per agent — Tendency to Plan, Loyalty, Willingness to Pay, Risk-Taking, and Flexibility — each as a linear combination of the standardized traits, with its own intercept (β₀ to β₄).
3. Anchor Calculation (Flexibility only): Compute a weighted average of the agent's observed flexibility (the standard deviation in the agent's number of actions per cycle in the experiment) and the trait-based prediction of flexibility, to create the anchored flexibility score.
4. Stochastic Component (σ): Optionally, introduce randomness by making a random draw from a normal distribution centred on each element's score.
5. Segmentation: Rescale each score across the population and assign the agent to one of five score segments (1–5); for the Tendency to Plan element, the rescaling produces the options list length (0–5) directly.
6. Priority List per Element: Map each segment to a priority list of options, taken from the element's fixed priority sequence.
7. Integration: Reconcile the four priority lists into one consensus ranking using the Kemeny–Young method with a tie-breaking hierarchy, then cut the consensus to the agent's options list length and drop every option ranked after Option 5.

The controls on this tab enable the user to modify and review the steps of this process, determining the configuration of the agent model corresponding to a particular experiment participant.

The tab is organized into six sub-tabs — one for each of the five sub-decision mechanisms and one for the integration step. The income specification setting is placed in the left column and the stochastic component setting in the right column, above the sub-tabs, because both apply to the decision as a whole.

## 1. The Five Sub-Decision Mechanisms

| # | Sub-decision mechanism | Predictors | Output | Priority sequence |
|---|---|---|---|---|
| 1 | Options List Length (Tendency to Plan) | Extraversion, Agreeableness, Neuroticism, Conscientiousness, Education | Number of pre-selected options (0–5) | — |
| 2 | Loyalty Ranking | Extraversion, Openness, Agreeableness | Priority list of options | Option 3 → 1 → 4 → 5 → 2 |
| 3 | Willingness-to-Pay Ranking | Extraversion, Agreeableness, Income | Priority list of options | Option 3 → 2 → 1 → 4 → 5 |
| 4 | Risk-Taking Ranking | Extraversion, Openness, Agreeableness, Conscientiousness, Neuroticism, Income | Priority list of options | Option 4 → 2 → 1 → 3 → 5 |
| 5 | Flexibility Ranking | Extraversion, Openness, Neuroticism, Agreeableness, Conscientiousness, observed flexibility | Priority list of options | Option 2 → 4 → 3 → 1 → 5 |

The priority sequence lists the options from the one most consistent with a high score on that construct to the one least consistent with it. A highly loyal agent, for example, prefers Option 3 (stay with the current vendor and pay its Purchase Now price) and is least likely to choose Option 2 (switch to another vendor). Only the Willingness-to-Pay and Risk-Taking mechanisms use income; the Tendency to Plan, Loyalty, and Flexibility mechanisms are income-free and therefore produce identical results under both income specifications.

## 2. Z-Scoring (Standardization)

All z-scoring uses fixed statistics from the original 280 experiment participants. Statistics are not recomputed for each bootstrap sample or simulation run. This approach follows Stata's `egen z_var = std(var)`, which standardizes once on the original data.

### Individual Trait Z-Scoring

| Trait | Mean | SD |
|---|---|---|
| ExtraversionBig5 | 3.557857 | 0.6989565 |
| Agreeable | 3.546071 | 0.3732712 |
| NeuroticismBig5 | 2.702143 | 0.6839657 |
| ConscientiousnessBig5 | 3.657143 | 0.5596521 |
| OpennessBig5 | 4.060714 | 0.5068274 |
| stdactions (observed flexibility) | 1.1863376 | 0.6752843 |

Formula: z_trait = (raw_value − mean) / sd

Education enters the Tendency to Plan equation not as a z-score but as the binary indicator reducation = Education − 1, where 0 = Undergraduate and 1 = Graduate.

Income is standardized at runtime against the population of the current simulation run, using the sample standard deviation: z_Income = (income − population_mean) / population_sd. This follows Stata's `egen z_net_income = std(income)`. In the Research Baseline and Research Specification population modes each original participant carries their own income value from the research data; in the Copula population mode income is generated from the distribution configured on Page 1.

`stdactions` is the observed flexibility measure: the standard deviation in the number of actions the participant took per cycle over the eight experiment cycle-weeks. Because the experiment workbook records only per-participant totals, this variable is taken from the research data file and is carried as one of the traits of the copula model, so that synthetic agents reproduce its correlations with the Big-5 traits.

## 3. Mechanism 1: Options List Length (Tendency to Plan)

This mechanism determines how many default options the agent pre-selects. The score is a linear combination of four standardized traits plus education:

TTP_i = β₀ − 0.0152556564 × z_Extraversion_i + 0.0177638642 × z_Agreeableness_i + 0.01959 × z_Neuroticism_i + 0.00901465 × z_Conscientiousness_i + 0.0297 × Education_i

Where Education_i ∈ {0, 1}. The score is then rescaled across the population onto a 0–6 range and floored:

OptionsListLength05_i = ⌊(6 − 0.0001) × (TTP_i − min(TTP)) / (max(TTP) − min(TTP))⌋ ∈ {0, …, 5}

An agent whose options list length is 0 pre-selects no default at all; for such an agent, the handling of a rejected transaction is decided in real time under Decision 11. On the 280 original participants with β₀ = 0, the resulting allocation is 20 agents with length 0, 92 with length 1, 95 with length 2, 57 with length 3, 14 with length 4, and 2 with length 5.

The coefficients are fixed from the original meta-analysis and are not user-adjustable.

## 4. Mechanism 2: Loyalty Ranking

Loyalty to the vendor is predicted from three standardized traits:

Loyalty_i = β₁ − 0.009828468 × z_Extraversion_i + 0.01096706 × z_Openness_i + 0.0123046 × z_Agreeableness_i

The composite is then standardized across the population (`egen weighted_loyalty = std(bs_weighted_loyalty)`) and the intercept β₁ is added on that standardized scale. The standardized score is the one displayed in the results histogram and stored in the Excel export.

Priority sequence: Option 3 → Option 1 → Option 4 → Option 5 → Option 2. The most loyal agents prefer to stay with the current vendor and pay its Purchase Now price (Option 3); the least loyal are willing to switch vendors (Option 2).

On the 280 original participants the five loyalty segments hold 12, 87, 148, 31, and 2 agents respectively.

## 5. Mechanism 3: Willingness-to-Pay Ranking

Willingness to pay is predicted from two standardized traits and income:

WTP_i = β₂ + 0.078863062 × z_Extraversion_i − 0.012326128 × z_Agreeableness_i + 0.698 × z_Income_i

Priority sequence: Option 3 → Option 2 → Option 1 → Option 4 → Option 5. Agents with a high willingness to pay accept the current vendor's Purchase Now price (Option 3), while agents with a low willingness to pay forgo the transaction (Option 5).

On the 280 original participants the five willingness-to-pay segments hold 168, 82, 26, 3, and 1 agents respectively. This is the most skewed of the five distributions, because the income term dominates the equation and income is itself right-skewed.

## 6. Mechanism 4: Risk-Taking Ranking

Risk-taking propensity is predicted from five standardized traits and income:

RiskTaking_i = β₃ + 0.025942386297 × z_Extraversion_i + 0.023699214948 × z_Openness_i − 0.038734315188 × z_Agreeableness_i − 0.037739440732 × z_Conscientiousness_i − 0.025388697852 × z_Neuroticism_i + 0.006874197106 × z_Income_i

Priority sequence: Option 4 → Option 2 → Option 1 → Option 3 → Option 5. The most risk-taking agents place a new bid (Option 4); the most risk-averse forgo the transaction (Option 5).

On the 280 original participants the five risk-taking segments hold 20, 100, 110, 46, and 4 agents respectively.

## 7. Mechanism 5: Flexibility Ranking

Flexibility is the only mechanism that combines a trait-based prediction with an observed behavioural measure, in the same way as the anchor of Decision 3. The trait-based part uses all five Big-5 traits:

Flexibility_i = β₄ + 0.0206 × z_Extraversion_i + 0.0294118 × z_Openness_i − 0.04921357 × z_Neuroticism_i + 0.04339814 × z_Agreeableness_i + 0.04811179 × z_Conscientiousness_i

The trait-based score is then standardized across the population, the intercept β₄ is added on that standardized scale, and the result is blended with the agent's observed flexibility:

AnchoredFlexibility_i = W_OFlex × ObservedFlexibility_i + W_CFlex × Flexibility_i = 0.25 × ObservedFlexibility_i + 0.75 × Flexibility_i

Where ObservedFlexibility is the standardized `stdactions` value. Because the observed flexibility relates to the number of actions taken rather than to the choice of defaults for rejected transactions, the observed part receives a weight of only 25% by default. The user can change this balance with the W_OFlex slider in the "Anchor Mix" section of the Flexibility sub-tab; W_CFlex is always 1 − W_OFlex.

Priority sequence: Option 2 → Option 4 → Option 3 → Option 1 → Option 5. The most flexible agents readily switch to another vendor (Option 2); the least flexible forgo the transaction (Option 5).

On the 280 original participants the five flexibility segments hold 17, 85, 137, 37, and 4 agents respectively.

If an agent has no observed flexibility value — which can occur only for synthetic agents — the observed part of the anchor is treated as neutral (z = 0) and the case is flagged in the model output.

## 8. From Score to Segment

Each of the four ranking scores is converted into a 1–5 score segment by rescaling it across the population and flooring the result:

Score15_i = ⌊1 + (5 − 0.0001) × (Score_i − min(Score)) / (max(Score) − min(Score))⌋ ∈ {1, …, 5}

For the Flexibility mechanism the rescaling is applied to AnchoredFlexibility rather than to the trait-based score.

Important: the five segments are equal-width fifths of the observed score range, not quintiles containing 20% of the agents each. The segment boundaries are set by the two most extreme agents in the population, so the number of agents per segment can be very uneven — the willingness-to-pay distribution of the 280 participants, for instance, places 60% of the agents in segment 1. This follows the research specification, which defines the segments by a min-max rescaling of the score.

A consequence of this design is that the segment boundaries depend on the population being simulated. Because the Copula population mode generates synthetic agents whose extreme values differ from those of the original 280 participants, the same score can fall into a different segment under the Copula and Research population modes. The score distributions themselves are nearly identical across the two modes; the difference in the resulting segment allocations is driven almost entirely by the two agents that set the ends of the scale.

## 9. From Segment to Priority List

The agent's score segment selects a priority list from the element's priority sequence: segment 5 (the highest 20% of the score range) receives the full sequence starting with the element's top option, and each lower segment receives a shorter list, taken from the end of the sequence. Segment 1 (the lowest 20% of the score range) therefore receives a single option — the one least consistent with a high score on that construct.

### Loyalty (Option 3 → 1 → 4 → 5 → 2)

| Loyalty score segment | Priority list for rejected transaction options | Options list length |
|---|---|---|
| 5 (Highest 20% of Loyalty score segment) | 3 > 1 > 4 > 5 > 2 | 5 |
| 4 | 1 > 4 > 5 > 2 | 4 |
| 3 | 4 > 5 > 2 | 3 |
| 2 | 5 > 2 | 2 |
| 1 (Lowest 20% of Loyalty score segment) | 2 | 1 |

### Willingness-to-Pay (Option 3 → 2 → 1 → 4 → 5)

| Willingness-to-Pay score segment | Priority list for rejected transaction options | Options list length |
|---|---|---|
| 5 (Highest 20% of Willingness-to-Pay score segment) | 3 > 2 > 1 > 4 > 5 | 5 |
| 4 | 2 > 1 > 4 > 5 | 4 |
| 3 | 1 > 4 > 5 | 3 |
| 2 | 4 > 5 | 2 |
| 1 (Lowest 20% of Willingness-to-Pay score segment) | 5 | 1 |

### Risk-Taking (Option 4 → 2 → 1 → 3 → 5)

| Risk-Taking score segment | Priority list for rejected transaction options | Options list length |
|---|---|---|
| 5 (Highest 20% of Risk-Taking score segment) | 4 > 2 > 1 > 3 > 5 | 5 |
| 4 | 2 > 1 > 3 > 5 | 4 |
| 3 | 1 > 3 > 5 | 3 |
| 2 | 3 > 5 | 2 |
| 1 (Lowest 20% of Risk-Taking score segment) | 5 | 1 |

### Flexibility (Option 2 → 4 → 3 → 1 → 5)

| Flexibility score segment | Priority list for rejected transaction options | Options list length |
|---|---|---|
| 5 (Highest 20% of Flexibility score segment) | 2 > 4 > 3 > 1 > 5 | 5 |
| 4 | 4 > 3 > 1 > 5 | 4 |
| 3 | 3 > 1 > 5 | 3 |
| 2 | 1 > 5 | 2 |
| 1 (Lowest 20% of Flexibility score segment) | 5 | 1 |

Note that the list length implied by the score segment is a property of the individual mechanism. It is not the agent's options list length: that is set by the Tendency to Plan mechanism and is applied only to the integrated list, at the integration step described in section 13.

## 10. The Intercepts (β₀ – β₄)

Each mechanism carries its own intercept, which sets a baseline level for that element and can be modified in the "Intercept Override" section of the element's sub-tab:

| Symbol | Element | Research default | Scale on which it applies |
|---|---|---|---|
| β₀ | Options List Length (Tendency to Plan) | 0.05 | Standardized Tendency to Plan score |
| β₁ | Loyalty | 0.00 | Standardized loyalty score |
| β₂ | Willingness to Pay | 0.00 | Standardized willingness-to-pay score |
| β₃ | Risk-Taking | 0.00 | Standardized risk-taking score |
| β₄ | Flexibility | 0.00 | Standardized calculated flexibility score, before the anchoring with observed flexibility |

The intercepts follow fixed-cutoff semantics. The segment boundaries (and, when the stochastic component is enabled, the range of the drawn values) are computed from the intercept-free population scores and are then held fixed. The intercept is added to the individual agent's score afterwards, so that raising an intercept moves agents upward across the fixed boundaries and lowering it moves them downward, with agents accumulating in the end segments once they pass the extremes. Setting an intercept to 0 reproduces the intercept-free result exactly.

Because β₄ is added to the calculated flexibility score before the 25/75 anchoring, its effect on the anchored score is scaled by W_CFlex: a β₄ of 0.10 shifts the anchored flexibility score by 0.075 at the default weights.

Note on β₀: the research specification states a default of 0.05 for the Tendency to Plan intercept but, unlike β₁ to β₄, does not state the scale on which it applies. The simulation applies it on the standardized score, consistently with the other four elements. Applied instead to the unstandardized composite, whose range across the 280 participants is only 0.21, a value of 0.05 would shift every agent by approximately 1.4 list positions and would leave no agent with an options list length of 0.

## 11. The Stochastic Component (σ)

Every element is modelled as a deterministic component plus an optional stochastic element, configured once for the whole decision in the "Stochastic Component" section at the top of the tab. When enabled, the model makes one random draw per element for each agent:

draw_k ~ Normal(μ = anchor, σ = σ_base × coefficient)

The anchor is the agent's own continuous score for that element — the rescaled 0–6 Tendency to Plan value, the standardized loyalty score, the willingness-to-pay score, the standardized risk-taking score, and the anchored flexibility score respectively. The drawn values are then rescaled over the population and re-binned, exactly as the deterministic scores are, so the stochastic component can move an agent into a different segment and therefore change the agent's priority list.

The base σ values derive from the natural variability observed in the experiment: σ = (range of the element's score / 18) × mean(stdactions), where the division by 18 reflects that the choice set spans a 1–5 standardized range while the number of actions per cycle ranges from 0 to 17.

### σ Modes

The user can choose between two σ modes using the radio button in the "Stochastic Component" section.

Uniformly (Overall σ) applies one common σ per element to all agents:

| Element | Base σ |
|---|---|
| Options List Length | 0.395446 |
| Loyalty | 0.4665145336 |
| Willingness-to-Pay | 0.4526575455 |
| Risk-Taking | 0.39522204 |
| Flexibility | 0.4346228732 |

The σ Coefficient slider (default 1.0, range 0.0–2.0) multiplies each element's base σ. Setting the coefficient to 0 makes σ = 0, which effectively disables the stochastic component.

Quintiles (Per-Income-Level σ) applies a different σ per budget level, computed from the mean number of actions per cycle within each budget level:

| Budget level | Options List Length | Loyalty | Willingness-to-Pay | Risk-Taking | Flexibility |
|---|---|---|---|---|---|
| Level 1 (€12) | 0.3409412 | 0.3738315 | 0.3902672 | 0.3397903 | 0.3747184 |
| Level 2 (€32) | 0.3788994 | 0.4154515 | 0.4337170 | 0.3776204 | 0.4164371 |
| Level 3 (€72) | 0.4368824 | 0.4790282 | 0.5000888 | 0.4354077 | 0.4801646 |
| Level 4 (€128) | 0.4085648 | 0.4479787 | 0.4676743 | 0.4071857 | 0.4490415 |
| Level 5 (€200) | 0.4181634 | 0.4585033 | 0.4786616 | 0.4167519 | 0.4595910 |

Each budget level has its own coefficient slider (default 1.0, range 0.0–2.0). The σ settings are decision-wide: one mode and one set of coefficients apply to all five elements, while each element keeps its own base σ.

### When Stochastic Is Applied

The stochastic component is controlled differently depending on the population mode:

| Population Mode | UI Control | Stochastic Behavior |
|---|---|---|
| Copula (synthetic) | "Add Normal(anchor, σ) draw to Copula runs" checkbox | ON when checked |
| Research Specification | "Use Normal(anchor, σ) draw in Research Specification mode" checkbox | ON when checked |
| Research Baseline | No control (always off) | Always OFF — deterministic scores only |
| Compare all | Separate checkboxes for Copula and Research Specification | Independent per mode; Baseline always OFF |

All five draws are made for every agent whenever the stochastic component is enabled, even for an element whose σ coefficient is 0, so that each element's outcome is unaffected by the settings of the other elements.

## 12. Categorical vs Continuous Income Specification

The income specification applies to the Willingness-to-Pay and Risk-Taking mechanisms only; the Tendency to Plan, Loyalty, and Flexibility mechanisms use no income and are identical under both specifications.

Continuous only uses the agent's actual income, standardized at runtime across the population, with the income coefficients shown in sections 5 and 6.

Categorical only replaces the income term with a baseline intercept plus a budget-level effect, estimated by regressing the full continuous score on its income-free personality part and the budget-level indicators:

Score_categorical_i = β_income_q[quintile_i] + (the element's personality terms)

| Quintile | Budget | β_income_q (Willingness-to-Pay) | β_income_q (Risk-Taking) |
|---|---|---|---|
| Q1 | €12 | −0.6918430 | −0.0068307 |
| Q2 | €32 | −0.4246842 | −0.0042179 |
| Q3 | €72 | −0.1860714 | −0.0017752 |
| Q4 | €128 | 0.2493317 | 0.0024431 |
| Q5 | €200 | 1.1306280 | 0.0111505 |

Each value is the base intercept plus that budget level's differential effect, with Q1 as the base level. Everything downstream of the score — standardization, rescaling, segmentation, the mapping to a priority list, and the optional stochastic layer — is identical in both specifications.

The "Compare both" option runs the decision with both specifications, allowing for direct comparison. Because three of the five elements are income-free, the results page presents those three once for both specifications and repeats only the Willingness-to-Pay, Risk-Taking, and integrated results per specification.

## 13. Integrating the Five Mechanisms

The Loyalty, Willingness-to-Pay, Risk-Taking, and Flexibility mechanisms each produce a priority list of the five options per agent. These lists do not necessarily concur, so they are reconciled into one integrated ranking, in which all sub-decision mechanisms receive equal weight because there is no theoretical basis for assigning them different weights. Two rules are then applied:

1. The integrated list is truncated to the agent's Options List Length, from the Tendency to Plan mechanism.
2. Every option ranked after Option 5 in the integrated list is dropped.

Both rules apply to the integrated ranking only. An individual mechanism's own list is not truncated, so a Loyalty list may legitimately contain Option 2 after Option 5.

### The Aggregation Method: Kemeny–Young with a Tie-Breaking Hierarchy

The distance between two rankings is measured by the Kendall-tau distance: the number of option pairs that the two rankings order differently. With five options there are 10 such pairs; a pair (x, y) counts 1 when one ranking places x above y while the other places y above x, and 0 otherwise. The integrated ranking is the ordering of the five options whose total Kendall-tau distance to the four mechanism rankings is smallest (Kemeny, 1959; Kemeny & Snell, 1962). It is found by evaluating all 120 possible orderings.

Because a mechanism's list is usually shorter than five options, the options absent from a list are treated as tied at the bottom of that list: the mechanism prefers each listed option to each unlisted one and expresses no preference among the unlisted ones.

After applying the Kemeny–Young method, remaining ranking ties are resolved in two phases:

Phase 1 – If Kemeny returns several equally good orderings, the Schulze (2011) strongest-paths ordering is used to produce an initial ranking. Schulze is a Condorcet-consistent method that orders the options by their strongest paths of pairwise victories, and its ordering is in almost every case itself one of the Kemeny-optimal orderings, so this step selects among the optimal rankings rather than departing from them.

Phase 2 – leftover ties. After applying Kemeny and Schulze, remaining ties are resolved using Copeland (pairwise wins minus losses) and then by Spearman footrule (smallest total positional displacement). Any remaining ties are resolved by randomization which avoids any systematic bias. The random draw uses the agent's simulation seed, so runs remain reproducible.

### Diagnostics of the Integration

Because ties are frequent with only four input rankings, the integration records for each agent how its ranking was obtained. These diagnostics appear in the Excel export and, for the "Run Integrated Default List Only" run, on the results page:

| Field | Meaning |
|---|---|
| kemeny_status = unique | Exactly one of the 120 orderings minimizes the total Kendall-tau distance, and it is fully ordered. No tie-breaking is needed. |
| kemeny_status = unique_with_ties | Several orderings minimize the distance, but they are all the orderings of one ranking in which some options are tied; the ranking itself is unambiguous and only the tied positions have to be separated. |
| kemeny_status = multiple | Several minimizing orderings that do not reduce to one such ranking. The Schulze ordering is used as the initial ranking. |
| n_kemeny_optimal | The number of the 120 orderings that attain the minimum total Kendall-tau distance. It is 1 when the status is unique and larger otherwise. |
| is_kemeny_optimal | Whether the agent's final ranking, after all tie-breaking, is still one of those minimizing orderings. It can be false only when the random last resort was used. |
| settled_by | The stage at which the ranking became fully ordered: Kemeny, Schulze, Copeland, Footrule, or the random last resort. |
| truncated_by = none | The full five-option consensus was kept: the options list length was 5 and Option 5 was already last. |
| truncated_by = length | The list was cut only by the agent's Options List Length. |
| truncated_by = option5 | The list was cut only because Option 5 appeared before the length limit. Option 5 is kept as the final option of the list and everything after it is dropped. |
| truncated_by = both | The length limit and the position of Option 5 fall at the same place, so both rules bind. |

The results page also reports the percentage of agents with initial ties after Kemeny, which is the share of agents whose Kemeny status is not "unique" — that is, the share for whom the tie-breaking hierarchy had to be applied at all.

## 14. Tab Controls Reference

### Left Column

Income Specification. Radio button with three options:

- Categorical only: Willingness-to-Pay and Risk-Taking use the fitted budget-level effects (Quintiles 1–5)
- Continuous only: Willingness-to-Pay and Risk-Taking use the generated income, z-scored
- Compare both: Runs both specifications for side-by-side comparison

Options List Length, Loyalty, and Flexibility do not use income.

### Right Column

Stochastic Component. Controls vary by population mode (set on Page 1):

- Copula: Checkbox to add the Normal(anchor, σ) draw to Copula runs
- Research Specification: Checkbox to enable Normal(anchor, σ) draws
- Research Baseline: No controls — always deterministic
- Compare all: Separate checkboxes for Copula and Research Specification

When the stochastic component is enabled, additional controls appear:

- σ mode: Radio button — "Uniformly" (single σ per element) or "Quintiles" (σ per budget level)
- σ Coefficient slider (Uniformly mode): Multiplier for each element's base σ (range 0.0–2.0, default 1.0)
- Per-budget-level coefficient sliders (Quintiles mode): Individual multipliers per budget level

### Sub-Tabs

Each of the five element sub-tabs presents the element's equation with its coefficients substituted, the segmentation formula, the priority sequence, the segment-to-list mapping table, the stochastic component explanation, the Intercept Override section, a reset button that restores only that element's settings, and a Run button that runs the decision and presents only that element's results.

The Flexibility sub-tab additionally carries the Anchor Mix section with the W_OFlex slider (range 0.0–1.0, default 0.25), which sets the weight of observed flexibility against calculated flexibility.

The sixth sub-tab, "Integrated Default List (Rank Aggregation)", explains the aggregation method and the two output rules and carries its own Run button, which presents the integrated default list together with the tie-resolution statistics.

### Actions & Management

- Reset <Element> to Defaults: Resets only that element's intercept, and for Flexibility also the Anchor Mix weight. The decision-wide σ settings are not affected.
- Reset Decision 4 Settings to Defaults: Resets all Decision 4 configuration to the research defaults.

## 15. Default Parameter Values

| Parameter | Default Value | User-Adjustable | Location |
|---|---|---|---|
| β₀ (Options List Length) | 0.05 | Yes | Intercept Override, sub-tab 1 |
| β₁ (Loyalty) | 0.00 | Yes | Intercept Override, sub-tab 2 |
| β₂ (Willingness to Pay) | 0.00 | Yes | Intercept Override, sub-tab 3 |
| β₃ (Risk-Taking) | 0.00 | Yes | Intercept Override, sub-tab 4 |
| β₄ (Flexibility) | 0.00 | Yes | Intercept Override, sub-tab 5 |
| W_OFlex | 0.25 | Yes | Anchor Mix slider, sub-tab 5 |
| σ Coefficient | 1.0 | Yes | Stochastic slider |
| Stochastic enabled | Off for Copula, On for Research Specification | Yes | Stochastic checkboxes |
| σ mode | Uniformly | Yes | σ mode radio |
| Income mode | Continuous only | Yes | Income Specification radio |
| Base σ per element | Fixed | No (base value) | Fixed |
| Equation coefficients | Fixed | No | — |
| Priority sequences | Fixed | No | — |
| Categorical income effects | Fixed | No | — |
| Z-scoring statistics | Fixed | No | — |

## 16. Output

The decision produces the following outputs per agent:

| Output Field | Description |
|---|---|
| rejected_transaction_defaults | The integrated default list as option codes — the main output of the decision |
| rtd_weighted_ttp | Tendency to Plan score (with the intercept applied) |
| rtd_weighted_ttp06 | The Tendency to Plan score rescaled to the 0–6 range |
| rtd_choice_length_deterministic | Options list length before the stochastic draw |
| rtd_choice_length | Final options list length (0–5) |
| rtd_loyalty_score, rtd_wtp_score, rtd_rt_score, rtd_flex_score | The element's score; for Flexibility, the anchored score |
| rtd_loyalty_z, rtd_wtp_z, rtd_rt_z, rtd_flex_z | The element's standardized score |
| rtd_flex_ivw, rtd_flex_z_ivw | Calculated flexibility score, before and after standardization |
| rtd_z_stdactions | Standardized observed flexibility |
| rtd_*_segment_deterministic | Score segment before the stochastic draw |
| rtd_*_segment | Final score segment (1–5) |
| rtd_*_ranking | The element's priority list of option numbers |
| rtd_sigma_used_* | The σ actually applied to that element for this agent |
| rtd_default_list, rtd_default_list_length | The integrated default list and its length |
| rtd_consensus_ranking | The full consensus ranking of the five options, before truncation |
| rtd_consensus_kemeny_status, rtd_consensus_n_kemeny_optimal, rtd_consensus_is_kemeny_optimal, rtd_consensus_settled_by, rtd_consensus_truncated_by | The integration diagnostics described in section 13 |
| rtd_z_extraversion, rtd_z_agreeable, rtd_z_neuroticism, rtd_z_conscientiousness, rtd_z_openness, rtd_z_income, rtd_reducation | The standardized inputs used by the equations |

These values are available in the Excel export from the results page.

## 17. Downstream Impact

The integrated default list is the agent's standing instruction for how a rejected transaction should be handled. It is applied first, in the order of the list. Decision 11 (rejected_transaction_option) becomes relevant only when the list does not resolve the situation — that is, when the agent's options list length is 0, or when the pre-selected options have been exhausted — in which case the remaining choice is made in real time after the rejection. Where Option 4 is reached, Decision 12 (rejected_bid_value) determines the new bid value.

## 18. Simulation Options

Decision 4 can be run in several ways, each presenting a different level of detail:

- Run <Element> Only (five buttons, one per element sub-tab): runs the decision and presents only that element's results — the score distribution, the allocation of agents across options, and an Excel file containing only that element's variables.
- Run Integrated Default List Only (sixth sub-tab): presents the integrated default list — the distribution of default list lengths and of the first integrated default option — together with the tie-resolution statistics: the percentage of agents with initial ties after Kemeny, the breakdown of the Kemeny outcomes, the stage at which each ranking was settled, and the share of agents whose final ranking remains Kemeny-optimal.
- Run Rejected Transaction Defaults Only: presents every element's results as above, followed by the percentage of agents per first integrated default option. The tie statistics are not repeated here.
- Run Complete Simulation: presents only the distribution of integrated default list lengths and of the first integrated default option. The values of the individual elements are not displayed but are contained in the downloadable Excel file.

If the user has produced more than one Decision 4 configuration for comparison, the complete simulation cannot be run until one configuration has been selected with the "Use This Config" button beneath its results. Once a configuration has been selected, the results page presents only that configuration and its Excel export contains only that configuration.

### Presentation of the Results

Each element's score distribution is presented as a histogram using Stata's default binning rule — k = round(min(√N, 10 × log₁₀N)) equal-width bins spanning the observed range, which gives 17 bins for the 280 original participants — so that the histograms are directly comparable with the graphs in the research specification. The bar heights are proportions of the agent population and therefore sum to 1, and the agent population's mean is marked with a vertical line. The minimum and maximum of the score are reported beneath each histogram.

Each element's allocation chart presents the percentage of agents whose first-ranked option is each of the five options, ordered along the horizontal axis by the element's priority sequence in reverse, with the least likely option on the left and the most likely on the right. A companion table reports the same percentages for Options 1 to 5 in their natural order, and the five option descriptions are listed beneath the chart.

## 19. Excel Export Structure

### Export 1: A Single Element

When produced: running one element with its "Run <Element> Only" button.

File name: rejected_transaction_<element>_YYYYMMDD_HHMMSS.xlsx

Structure: a single sheet, named after the element ("Options List Length", "Loyalty", "Willingness-to-Pay", "Risk-Taking", "Flexibility").

Columns: Agent ID; each of the element's own independent variables, with each raw input followed immediately by the standardized value its equation uses (z_extraversionbig5, z_agreeable, z_neuroticismbig5, z_conscientiousnessbig5, z_opennessbig5, reducation, z_net_income, z_stdactions); the element's score and, where applicable, its standardized score; the deterministic and final score segment — for the Options List Length element, the rescaled 0–6 value and the deterministic and final list length instead; the σ applied; and choice1 to choice5, holding the agent's priority list, blank beyond the length of that list.

With the categorical income specification, the Willingness-to-Pay and Risk-Taking files carry "Assigned Allowance Level" in place of z_net_income, because those equations replace the income term with the budget-level effect.

### Export 2: The Whole Decision

When produced: running the whole decision with the "Run Rejected Transaction Defaults Only" button, or the "Run Integrated Default List Only" button.

File name: rejected_transaction_mechanisms_YYYYMMDD_HHMMSS.xlsx, or rejected_transaction_integrated_default_list_YYYYMMDD_HHMMSS.xlsx for the integration-only run.

Structure: the first sheet, "Integrated Default List", holds one row per agent with every variable the decision uses, grouped left to right:

| Column group | Contents |
|---|---|
| Identification | Agent ID |
| Raw inputs | ExtraversionBig5, Agreeable, NeuroticismBig5, ConscientiousnessBig5, OpennessBig5, Education, income (or Assigned Allowance Level in the categorical specification), stdactions |
| Standardized inputs | z_extraversionbig5, z_agreeable, z_neuroticismbig5, z_conscientiousnessbig5, z_opennessbig5, reducation, z_net_income |
| 1. Options List Length | weighted_ttp, weighted_ttp06, choice_length_deterministic, choice_length |
| 2. Loyalty | loyalty_score, z_loyalty, loyalty_segment_deterministic, loyalty_segment, loyalty_list |
| 3. Willingness-to-Pay | WTP_score, z_WTP, WTP_segment_deterministic, WTP_segment, WTP_list |
| 4. Risk-Taking | RT_score, z_RT, RT_segment_deterministic, RT_segment, RT_list |
| 5. Flexibility | Flexibility_calculated_ivw, z_Flexibility_calculated_ivw, z_stdactions, Flexibility_score, z_Flexibility, Flexibility_segment_deterministic, Flexibility_segment, Flexibility_list |
| 6. Integration | consensus_ranking, kemeny_status, n_kemeny_optimal, is_kemeny_optimal, settled_by, truncated_by, default_list_length, final_choice1 to final_choice5 |

For the whole-decision run this first sheet is followed by one sheet per element, each identical in structure to the corresponding single-element file described in Export 1.

### Export 3: Compare Both Income Modes or Compare All Population Modes

When produced: running Decision 4 with "Compare both" income mode or with the population mode set to "Compare all".

Structure: the same sheets as Export 2, with the sheet names prefixed by the configuration (Copula_Cat, Copula_Cont, ResSpec_Cat, ResSpec_Cont, ResBase_Cat, ResBase_Cont, or simply Cat and Cont when a single population mode is used). Once a configuration has been selected with "Use This Config", only that configuration is exported.

### Complete Simulation

In a complete simulation the Decision 4 variables are not exported separately. Every element's inputs, standardized inputs, scores, segments, and priority lists, together with the integration diagnostics and the final list, are carried in the agent-level Excel file of the complete simulation.

---

# Replacement text for the two Decision 4 passages elsewhere in the manual

The two passages below currently describe Decision 4 as it behaved before the model was implemented. They should be replaced with the text that follows each of them.

## A. In "1. Agent-Level Excel Export" (Complete Simulation section)

Replace the paragraph beginning "The agent's priority order for handling rejected transactions is split into five separate columns for easier analysis" and the five bullets that follow it with:

> Decision 4: Rejected Transaction Defaults
>
> The agent's integrated default list for handling rejected transactions is split into five separate columns for easier analysis:
>
> rejected_transaction_1_choice: First option of the integrated default list
> rejected_transaction_2_choice: Second option
> rejected_transaction_3_choice: Third option
> rejected_transaction_4_choice: Fourth option
> rejected_transaction_5_choice: Fifth option
>
> Each column shows the selected option — "Purchase from another (higher) price category of the same vendor", "Purchase from another vendor at PN price", "Purchase from the current vendor at PN price", "Place a bid", or "Forgo the purchase request" — or "N/A" beyond the length of the agent's list. The number of options that are filled is the agent's options list length, capped where Option 5 appears earlier in the list. An agent whose options list length is 0 has "N/A" in all five columns and resolves a rejected transaction under Decision 11 instead.
>
> When Decision 4 is run with its model, the agent-level file also carries the element variables behind the list: the standardized traits and income, each element's score, standardized score and score segment, the priority list of each element, and the integration diagnostics (consensus ranking, Kemeny status, number of Kemeny-optimal orderings, tie-break stage, and truncation rule).

## B. In "Transaction-Level Excel Export" (Complete Simulation section)

Replace the paragraph beginning "The agent's priority order for handling rejected transactions, split into five columns" and the five bullets that follow it with:

> Decision 4: Rejected Transaction Defaults
>
> The agent's integrated default list for handling rejected transactions, split into five columns:
>
> rejected_transaction_1_choice: First option of the integrated default list
> rejected_transaction_2_choice: Second option
> rejected_transaction_3_choice: Third option
> rejected_transaction_4_choice: Fourth option
> rejected_transaction_5_choice: Fifth option
>
> Each column shows the selected option or "N/A" beyond the length of the agent's list. The list is an agent-level decision made at registration, so it repeats on every transaction row of that agent.

## C. In "2. Configure Default Decision Parameters" (Page 2)

The existing Decision 4 entry describes the default behaviour that applies when the user does not select Decision 4 for modelling, and remains correct. One sentence should be added at its end so that the reader knows where the model is described:

> When the user selects Decision 4 for modelling, this single default is replaced by the five sub-decision mechanisms described in the "Decision 4 – Rejected Transaction Defaults Configuration" section below, which produce a different integrated default list for each agent.
