# Decision 4 — Rejected Transaction Defaults: model verification, 2026-09-16

Scope: the professor's five feedback items on Decision 4 (segment→option mapping direction;
TTP intercept β₀; Copula vs Research for Options List Length; formula/graph replication of
the Stata file; the categorical-income specification).

Sources used
* Design document, latest revision **040926** (converted text; Section 1 TTP, 2 Loyalty,
  3 WTP, 4 Risk-Taking, 5 Flexibility, 6 Integration).
* Professor's Stata file `Stata_File_Decision4_290826.dta` (280 participants, 233 variables),
  read directly with `pandas.read_stata`, plus the frozen extracts
  `data/stata_d4_verification.csv`, `data/stata_d4_flexibility_verification.csv`,
  `data/stata_stdactions.csv`.
* The app itself, run head-less through `src/orchestrator_baseline.py` (Research Baseline),
  `src/orchestrator_doc_mode.py` (Research Specification) and `src/orchestrator.py` (Copula).

Code touched: `src/decisions/rejected_transaction_defaults.py`,
`src/decisions/rtd_rank_aggregation.py`, `config/decisions.yaml` (Decision-4 section),
`tests/test_rejected_transaction_defaults.py`, `tests/test_rtd_flexibility.py`,
`tests/test_rtd_rank_aggregation.py`.

---

## 1. Segment → option-list mapping: direction flipped

### What changed

`_ranking_for_segment(mechanism, segment)` now returns **`seq[5 - segment:]`** — the *last*
`segment` options of the priority sequence — instead of `seq[segment - 1:]`.

| Segment (of the element's own score) | List the agent receives | Example: Flexibility (2 > 4 > 3 > 1 > 5) |
|---|---|---|
| 5 — highest 20 % | the **full** sequence, top option first | `2 > 4 > 3 > 1 > 5` |
| 4 | last 4 options | `4 > 3 > 1 > 5` |
| 3 | last 3 options | `3 > 1 > 5` |
| 2 | last 2 options | `1 > 5` |
| 1 — lowest 20 % | **one** option | `5` |

Applied to all four ranking elements: Loyalty `[3,1,4,5,2]`, WTP `[3,2,1,4,5]`,
Risk-Taking `[4,2,1,3,5]`, Flexibility `[2,4,3,1,5]`.

### Authority

The professor's written instruction of 2026-09-16 ("… 5 (Highest 20 % of Cognitive
Flexibility score segment) corresponds to 2>4>3>1>5, and (Lowest 20 % …) corresponds to 5
with option list length 1. Correct all other decision elements based on the updated Decision 4
document" / "The ranking graphs are flipped"). This agrees with the document's **prose**, e.g.
rev 040926 line 2819: *"the highest-Flexibility 20 % segment, `Flexibility_calculated15 == 5`,
takes Option 2 as its first choice"*, and the parallel sentences in Sections 2–4.

### ⚠️ Conflict with the .dta that the professor must be aware of

The document's **Stata `replace` statements** (rev 040926 lines 2831-2859 for Flexibility and
the equivalents in Sections 2-4) and therefore the **stored `choice1..5_*` columns of
`Stata_File_Decision4_290826.dta`** still implement the *opposite* direction —
`choice1_flex_deterministic = 2 if Flexibility_combined15 == 1`. The document's own code and
its own prose contradict each other; the professor's instruction settles it in favour of the
prose, so the app no longer matches the file's choice columns literally.

**The segments themselves are untouched** — still 280/280 against the file (see §4) — and the
verification tests still check the file: they assert that the model's list for segment *s* is
exactly the file's stored list for the **mirror segment 6 − s**, which holds 280/280 for all
four elements. The .dta's own per-segment lists (unchanged, for reference):

| Segment in the .dta | Loyalty | WTP | Risk-Taking | Flexibility |
|---|---|---|---|---|
| 1 | 3,1,4,5,2 | 3,2,1,4,5 | 4,2,1,3,5 | 2,4,3,1,5 |
| 2 | 1,4,5,2 | 2,1,4,5 | 2,1,3,5 | 4,3,1,5 |
| 3 | 4,5,2 | 1,4,5 | 1,3,5 | 3,1,5 |
| 4 | 5,2 | 4,5 | 3,5 | 1,5 |
| 5 | 2 | 5 | 5 | 5 |

### Substantive consequence — please read

Because the bins are **equal-width** (not quantiles), most participants sit in *low* segments.
Under the old direction a low segment received the *full* list headed by the element's top
option; under the new direction a low segment receives *one* option — the element's **last**
option, which for WTP, Risk-Taking and Flexibility is **Option 5 (forgo the transaction)**.
On the 280 participants, WTP puts 168/280 in segment 1, so 60 % of participants now have the
single-item WTP list `[5]`.

This propagates into the Section-6 integrated list (280 participants, β₀ = 0.05):

| | Old direction | New direction |
|---|---|---|
| Consensus first option (1/2/3/4/5) | 86 / 50 / 124 / 15 / 5 | 71 / 9 / 53 / 60 / **87** |
| Mean integrated default-list length | 1.879 | 1.421 |
| Integrated list length 0/1/2/3/4/5 | 17/91/97/61/12/2 | 17/138/115/10/0/0 |
| Truncated by length / Option-5 / both / none | 222 / 10 / 46 / 2 | 94 / 97 / 89 / 0 |
| Settled by kemeny / copeland / footrule / random | 30 / 143 / 19 / 88 | 17 / 40 / 62 / 161 |
| "% of initial ties" (Kemeny status ≠ `unique`) | 89.3 % | 93.9 % |

i.e. under the new mapping the Option-5 cut becomes a binding rule for a third of agents and
the integrated lists get materially shorter. That is the arithmetic consequence of the
instructed direction, not a bug — but it changes the headline outputs, so it should be
confirmed before publication.

### Not in this agent's scope

The UI mirrors of the mapping live in `app/pages/decision_tabs/rejected_transaction.py`
(line ~181) and `app/pages/results/visualizations/transaction_viz.py` (lines ~635-1021) and
are owned by another agent; they must be flipped to match, otherwise the tab tables and the
ranking graphs will disagree with the model.

---

## 2. TTP intercept β₀ — semantics changed to the standardized scale

### The professor's observation, reproduced exactly

> "When running the whole decision, I get no observations with 0 Option List Length. How come?"

Confirmed. With β₀ = 0.05 applied on the **raw `weighted_ttp` composite** (the app's behaviour
before today) the 280 research participants gave:

| β₀ = 0.05, OLD raw-composite scale | 0 | 1 | 2 | 3 | 4 | 5 | mean |
|---|---|---|---|---|---|---|---|
| Research baseline, 280 | **0** | 6 | 59 | 95 | 86 | 34 | 3.296 |
| Copula, 1000 agents, seed 42 | **0** | 9 | 57 | 246 | 371 | 317 | **3.930** |

The copula figure **3.93** is exactly the number in the professor's message, which confirms the
diagnosis.

**Why**: `weighted_ttp` on the 280 spans only `[-0.0718, 0.1380]`, a range of **0.2098**, with
SD 0.0353. A raw-scale β₀ = 0.05 is therefore ~24 % of the entire range, i.e.
`0.05 × (6 − 0.0001) / 0.2098 = ` **+1.43 list positions for every agent**. Zero-length lists
become arithmetically impossible.

### Decision taken (flagged for the professor's confirmation)

**β₀ now applies on the STANDARDIZED TTP scale**, exactly like β₁/β₂/β₃ — the raw-scale
equivalent is `β₀ × sd₀(weighted_ttp)`, with `sd₀` the intercept-free population SD. The
fixed-cutoff semantics are unchanged (all population statistics stay β-free, so a non-zero β
genuinely moves agents across frozen bin boundaries; β = 0 is bit-identical to no intercept).
The research default stays **0.05** in `config/decisions.yaml`.

Rationale:
* The document writes β₁–β₄ as "*XXX **after standardization*** (to apply to the final
  equation)" (lines 480, 1049, 1995, 2739) but gives β₀ no scale qualifier (line 156) — it does
  not settle the question.
* The `.dta` embeds **no** intercept anywhere, so it cannot arbitrate either.
* Empirically the raw reading is untenable (above); the standardized reading makes
  β₀ = 0.05 a 0.05-SD nudge = **+0.050 list positions**, which preserves the Stata allocation.
* One consistent semantics across all five elements ("β is measured in SDs of the element's own
  score") is far easier to document and to reason about on the UI sliders.

### Resulting TTP allocations (280 research participants, deterministic)

| β₀ (standardized) | 0 | 1 | 2 | 3 | 4 | 5 | mean |
|---|---|---|---|---|---|---|---|
| **0.00** (= the Stata file) | 20 | 92 | 95 | 57 | 14 | 2 | 1.854 |
| **0.05** (research default) | **17** | 88 | 97 | 59 | 17 | 2 | 1.918 |
| −0.05 | 23 | 93 | 96 | 53 | 13 | 2 | 1.807 |
| +0.50 | 5 | 54 | 96 | 88 | 32 | 5 | 2.368 |
| −0.50 | 60 | 96 | 87 | 32 | 4 | 1 | 1.382 |

`.dta` `choice_length_deterministic`: **{0:20, 1:92, 2:95, 3:57, 4:14, 5:2}** — reproduced
exactly at β₀ = 0, and at most 5 agents per bin away at β₀ = 0.05, with zero-length lists intact.

Scale reference on the 280: `sd₀ = 0.035285`, `range₀ = 0.209788`;
raw shift at β₀ = 0.05 is `0.001764` = **0.0505 list positions** (was 1.4300 on the old scale).

---

## 3. Copula vs Research for the Options List Length

### (a) How much of the gap was the β₀ raw-scale effect

The β₀ bug was responsible for the **absolute inflation** of both numbers (and for the missing
zero-length lists), not for the copula-vs-research *gap*:

| | Research Spec (1000) | Copula (1000, seed 42) | gap |
|---|---|---|---|
| β₀ = 0.05, OLD raw scale | 3.288 | **3.930** | 0.642 |
| β₀ = 0.05, NEW standardized scale | **1.890** | **2.606** | 0.716 |
| β₀ = 0 | 1.826 | 2.561 | 0.735 |

So after the fix the research-mode mean drops from 3.29 to 1.89 (and matches the Stata file),
the copula mean drops from 3.93 to 2.61, and a residual gap of ≈ 0.72 list positions remains.

### (b) `Education` in the copula — **no bug, no fix needed**

`config/trait_model.pkl` emits `Education` as a NumPy **int64 with exactly two values {1, 2}**
(1000 agents, seed 42: 613 / 387). `reducation = Education − 1` is therefore exactly
`{0.0, 1.0}` on every population tested (research 0.379 / 0.378 mean, copula 0.400 at n=280 and
0.387 at n=1000). No rounding or clipping was added, because coercion would silently mask a
future copula retrain; instead `compute_rtd_population_stats` now prints a **warning** if
`reducation` is ever outside {0, 1} (the education term 0.0297 is the largest single
coefficient in `weighted_ttp`, so a broken coding would distort the whole allocation).

### (c) The residual gap is the **min-max window**, not the scores — and it affects all five elements

The document's Stata code bins every element with
`floor(low + (span − 0.0001) · (x − min)/(max − min))`, where `min`/`max` are the **population
extremes**. Two agents therefore set the whole scale. The copula and the research population
produce almost identical *score distributions* but noticeably different *extremes*:

| Element | score mean (research / copula) | score SD (research / copula) | window research | window copula |
|---|---|---|---|---|
| TTP | 0.011244 / 0.011056 | 0.035285 / 0.035130 | [−0.071788, 0.137999] | [−0.095179, 0.112988] |
| Loyalty | 0.000000 / 0.000645 | 0.035864 / 0.035900 | [−0.108719, 0.121961] | [−0.101251, 0.096541] |
| WTP | 0.000000 / −0.001580 | 0.696472 / 0.698225 | [−1.225587, 3.127499] | [−1.131730, 5.473251] |
| Risk-Taking | −0.000000 / 0.000654 | 0.065258 / 0.065559 | [−0.172739, 0.218111] | [−0.222282, 0.221359] |
| Flexibility | 0.000000 / −0.004197 | 0.799798 / 0.802643 | [−2.376135, 2.898073] | [−3.124111, 2.505341] |

Swapping only the window proves the point (research = baseline 280, copula = 1000 agents,
seed 42, β = 0 throughout):

| Element | research on own window | copula on own window | **copula on research window** | **research on copula window** |
|---|---|---|---|---|
| TTP | 1.854 | 2.561 | **1.897** | **2.557** |
| Loyalty | 2.861 | 3.079 | **2.886** | **3.050** |
| WTP | 1.900 | 1.338 | **1.869** | **1.371** |
| Risk-Taking | 2.693 | 3.002 | **2.728** | **3.014** |
| Flexibility | 2.736 | 3.267 | **2.741** | **3.271** |

In every element the copula scored on the research window reproduces the research mean, and the
research scored on the copula window reproduces the copula mean. **Essentially 100 % of the
copula-vs-research difference is the min-max window, i.e. the two extreme agents — not a
different score distribution and not a model bug.**

For TTP the window displacement happens to be the largest relative to the range: the copula
window is shifted down by ≈ 0.023-0.025 on a 0.21 range (≈ 11-12 % ≈ 0.7 list positions). The
extreme research participant is a rare joint combination (z_E = −2.80, z_A = +2.47, z_N = +1.90,
graduate) that the copula does not reproduce, while the copula's low extreme (z_N = −2.49)
is more extreme than the research low.

### Full element-by-element comparison, β₀ = 0.05 (new semantics), seed 42

| Run | TTP mean | Loyalty | WTP | Risk-Taking | Flexibility |
|---|---|---|---|---|---|
| Research baseline, 280 | 1.918 | 2.861 | 1.900 | 2.693 | 2.736 |
| Research spec, 280 | 1.918 | 2.861 | 1.832 | 2.700 | 2.736 |
| Research spec, 1000 | 1.890 | 2.873 | 1.747 | 2.649 | 2.765 |
| Copula, 280 | 2.682 | 3.139 | 1.764 | 2.689 | 3.196 |
| Copula, 1000 | 2.606 | 3.079 | 1.338 | 3.002 | 3.267 |

Segment distributions (research spec 1000 vs copula 1000):

| Element | Research Spec 1000 | Copula 1000 |
|---|---|---|
| TTP length 0-5 | 51/335/347/210/54/3 | 24/112/313/363/161/27 |
| Loyalty 1-5 | 41/260/490/203/6 | 47/216/393/299/45 |
| WTP 1-5 | 448/401/117/24/10 | 704/260/31/4/1 |
| Risk-Taking 1-5 | 75/364/406/147/8 | 19/238/488/232/23 |
| Flexibility 1-5 | 52/302/493/135/18 | 8/135/474/348/35 |

### Recommendation (not implemented — needs the professor's decision)

If the copula population is meant to be comparable with the research population, the segment
cutoffs should be **frozen on the original-280 window** rather than recomputed per population
(or a robust range, e.g. 1st/99th percentile, should replace min/max). This is a specification
change, so no code change was made; the current behaviour follows the document's Stata code
literally.

---

## 4. Formula verification against the document + Stata replication

All numbers below come from running the app's own functions (`compute_rtd_scores`,
`compute_rtd_population_stats`, `rejected_transaction_defaults`) on the `.dta`'s own inputs,
with `intercepts = 0` and `income_mode = continuous` unless stated.

### 4.1 Coefficients, re-derived from the document's own weights × betas (rev 040926)

| Element | Coefficient | Document (product of its stated weight × β) | Code | Verdict |
|---|---|---|---|---|
| **TTP** | z_E | 0.1753523722 × −0.087 = −0.01525566 | −0.0152556564 | ✔ |
| | z_A | 0.328960448 × 0.054 = 0.01776386 | 0.0177638642 | ✔ |
| | z_N | 0.15424536 × 0.127 = 0.01958916 | 0.01959 | ✔ (doc rounds) |
| | z_C | 0.0288008 × 0.313 = 0.00901465 | 0.00901465 | ✔ |
| | reducation | 0.312641 × 0.095 = 0.02970090 | 0.0297 | ✔ (doc rounds) |
| **Loyalty** | z_E | 0.476815 × −0.02061275 = −0.00982847 | −0.009828468 | ✔ |
| | z_O | 0.256046 × 0.042832382 = 0.01096706 | 0.01096706 | ✔ |
| | z_A | doc equation: 0.**6**7139 × 0.04606063 = 0.03092465 | 0.03092465 | ⚠ see below |
| **WTP** | z_E | 0.21259182 × 0.37096 = 0.07886306 | 0.078863062 | ✔ |
| | z_A | 0.05267576 × −0.234 = −0.01232613 | −0.012326128 | ✔ |
| | z_income | 0.734732428 × 0.95 = 0.69799580 | 0.698 | ✔ (doc rounds) |
| **Risk-Taking** | z_E | 0.2078115876 × 0.124836091165 = 0.025942386 | 0.025942386297 | ✔ |
| | z_O | 0.1687536495 × 0.140436755094 = 0.023699215 | 0.023699214948 | ✔ |
| | z_A | 0.2078115876 × −0.186391507961 = −0.038734315 | −0.038734315188 | ✔ |
| | z_C | 0.2078115876 × −0.181604121136 = −0.037739441 | −0.037739440732 | ✔ |
| | z_N | 0.1687536495 × −0.150448289120 = −0.025388698 | −0.025388697852 | ✔ |
| | z_income | 0.0390579381 × 0.176 = 0.006874197 | 0.006874197106 | ✔ |
| **Flexibility** | z_E | 0.21082365 × 0.09771 = 0.0205996 | 0.0206 | ✔ |
| | z_O | 0.17227968 × 0.1707212 = **0.0294118** | 0.0294118 | ⚠ doc prints 0.0293241 |
| | z_N | 0.13677769 × −0.359807 = **−0.04921357** | −0.04921357 | ⚠ doc prints −0.053781925 |
| | z_A | 0.24859134 × 0.17457623 = **0.04339814** | 0.04339814 | ⚠ doc prints 0.04921357 |
| | z_C | 0.23152734 × 0.20780175 = 0.04811179 | 0.04811179 | ✔ |

#### ⚠ Loyalty: the Agreeableness weight is still mis-typed in rev 040926

Line 460 of the document establishes `W_(Agreeableness) = 0.267139`; with
`W_E = 0.476815` and `W_O = 0.256046` the three weights sum to **exactly 1.000000**. The final
equation on line 464, however, multiplies by **0.67139** (the leading "2" dropped), giving
`0.03092465` instead of the arithmetically correct
`0.267139 × 0.04606063 = 0.0123045906` — a factor of 2.51.

The `.dta` was built with the mis-typed 0.03092465, so the app keeps it (that is why the model
matches the file 280/280). **Impact if corrected: 59 of 280 participants change Loyalty
segment**, the distribution moving from `{1:14, 2:71, 3:137, 4:56, 5:2}` to
`{1:12, 2:87, 3:148, 4:31, 5:2}`. This needs the professor's decision; no change was made.

#### ⚠ Flexibility O / N / A: corrected values retained (by design)

The document's printed equation (line 2675 / the `gen` on line 2755) is arithmetically
inconsistent with its own weight table — note in particular that the printed Agreeableness
coefficient `0.04921357` is the *Neuroticism* magnitude. The professor adopted the corrected
values in the 2026-09 review; the model uses them (`FLEX_COEFFS`), keeping the document/​`.dta`
literals as `DTA_FLEX_COEFFS` for verification. Effect vs the file: **8 of 280 segments move**,
`{1:17, 2:83, 3:141, 4:35, 5:4}` → `{1:17, 2:85, 3:137, 4:37, 5:4}`.

### 4.2 Reproduction of every `.dta` column (280 participants, β = 0)

| Column | max |app − .dta| |
|---|---|
| `z_extraversionbig5` | 2.9e-07 |
| `z_agreeable` | 1.3e-06 |
| `z_neuroticismbig5` | 4.9e-07 |
| `z_conscientiousnessbig5` | 4.8e-07 |
| `z_opennessbig5` | 7.1e-07 |
| `z_net_income` | 1.2e-07 |
| `reducation` | 0.0 (exact) |
| `weighted_ttp` | 1.8e-08 |
| `weighted_ttp06` | 3.5e-07 |
| `bs_weighted_loyalty` | 4.6e-08 |
| `weighted_loyalty` (z) | 2.1e-07 |
| `WTP_calculated` | 5.6e-06 |
| `z_WTP_calculated` | 8.1e-06 |
| `RT_calculated_hs` | 2.4e-08 |
| `z_RT_calculated_hs` | 1.7e-07 |
| `Flexibility_calculated_ivw` (with `DTA_FLEX_COEFFS`) | 1.0e-07 |
| `z_Flexibility_calculated_ivw` | 1.8e-07 |
| `z_stdactions` | 1.6e-07 |
| `anchored_flexibility` | 1.5e-07 |
| `z_anchored_flexibility` | 2.1e-07 |

All within float32 storage tolerance (the `.dta` stores floats; the frozen YAML z-scoring
constants are 7-digit roundings of the exact 280-sample statistics).

### 4.3 Segment / bin parity

| Column | parity | app distribution | `.dta` distribution |
|---|---|---|---|
| `choice_length_deterministic` (TTP) | **280/280** | 20/92/95/57/14/2 | 20/92/95/57/14/2 |
| `weighted_loyalty15` | **280/280** | 14/71/137/56/2 | 14/71/137/56/2 |
| `WTP_calculated15` | **280/280** | 168/82/26/3/1 | 168/82/26/3/1 |
| `RT_calculated15` | **280/280** | 20/100/110/46/4 | 20/100/110/46/4 |
| `Flexibility_combined15` (`DTA_FLEX_COEFFS`) | **280/280** | 17/83/141/35/4 | 17/83/141/35/4 |
| `Flexibility_combined15` (model's corrected coefficients) | 272/280 *by design* | 17/85/137/37/4 | 17/83/141/35/4 |

Choice lists: **280/280 for all four elements** against the **mirror** of the stored
`choice1..5_*` columns (see §1).

### 4.4 Histogram replication (document's graphs)

Research baseline, 280 participants, all intercepts 0. For each variable, 16 equal-width bins
spanning `min..max` of the pooled app+`.dta` values; bin counts compared one by one.

| Variable | 16-bin counts identical? | max bin difference |
|---|---|---|
| `weighted_ttp` | **yes** | 0 |
| `weighted_loyalty` (z) | **yes** | 0 |
| `z_WTP_calculated` | **yes** | 0 |
| `z_RT_calculated_hs` | **yes** | 0 |
| `z_anchored_flexibility` | **yes** | 0 |

PNGs (side-by-side `.dta` vs app), written to
`/private/tmp/claude-501/-Users-suedagul--sdg/fc29ec89-bc10-44b9-a898-1bf25562a207/scratchpad/d4/`:

* `hist_weighted_ttp.png`
* `hist_weighted_loyalty_z.png`
* `hist_z_WTP_calculated.png`
* `hist_z_RT_calculated_hs.png`
* `hist_z_anchored_flexibility.png`
* `first_choice_new_mapping.png` — first-choice option counts per element under the **new**
  mapping direction (the "flipped ranking graphs" the professor asked about)

Scripts: `.../scratchpad/d4/verify.py` (parity + histograms),
`.../scratchpad/d4/copula_compare.py` (Task 3), `.../scratchpad/d4/harness.py` (head-less
orchestrator runner). Machine-readable results: `verify_report.json`, `copula_compare.json`.

### 4.5 Stochastic σ constants (rev 040926)

Every range the document quotes was re-checked directly against the `.dta`:

| Element | doc range | `.dta` range | σ_overall doc | σ_overall config | Verdict |
|---|---|---|---|---|---|
| TTP | (6/18) scaling | `weighted_ttp` 0.209788 | 0.395446 | 0.395446 | ✔ |
| Loyalty | 6.432015 | `weighted_loyalty` 6.432014 | 0.423918958 | 0.423918958 | ✔ |
| WTP | 6.868056 | `z_WTP_calculated` 6.868057 | 0.4526575455 | 0.45265807275 | ⚠ 1.2e-6 relative |
| Risk-Taking | 5.979747 | `z_RT_calculated_hs` 5.979747 | 0.332208167 | 0.332208167 | ✔ (see note) |
| Flexibility | 6.6140623 | `z_anchored_flexibility` 6.614062 | 0.4359172665 | 0.4359172665 | ✔ (see note) |

Notes and corrections to earlier internal documentation:
* **Flexibility — the previously flagged "typo" is narrower than thought.** The document writes
  `σ_overall = (6.494119/18)·mean(stdactions) = 0.367448·1.1863376`. The *numerator* 6.494119 is
  wrong, but the *multiplier* 0.367448 is exactly `6.6140623/18`, i.e. derived correctly from the
  document's own (and the `.dta`'s) range. The app's constant is therefore correct and internally
  consistent; only the printed fraction is a typo. Code comments updated accordingly.
* **Risk-Taking** — the document's own formula states `(range/18)·mean(stdactions)` but computes
  `5.979747/18 = 0.332208167` *without* the `mean(stdactions)` factor, and its quintile table uses
  the multiplier `0.28` (≈ 5/18) rather than 5.979747/18. Both inconsistencies are internal to the
  document; the app follows the printed tables, as before.
* **WTP** — rev 040926 updated the range from 6.868064 to 6.868056, so its σ becomes
  0.4526575455 (and the quintile multiplier 0.3815586667) against the config's 0.45265807275 /
  0.3815592397. The difference is 1.2e-6 relative, with no visible effect; the config value was
  **kept** so earlier stochastic runs stay reproducible. One-line change if preferred.
* Per-quintile σ tables for TTP, Loyalty, Risk-Taking and Flexibility match `config/decisions.yaml`
  entry-for-entry.

### 4.6 Section 6 — rank aggregation still matches the document

Verified against rev 040926 Section 6 text: Kemeny-Young over all 120 permutations with equal
weights → Schulze when Kemeny returns several equally good orderings → Copeland → Spearman
footrule → random last resort; then the two output rules (truncate to the TTP choice length,
drop everything after Option 5), **applied to the integrated consensus list only**. The
reference stage shares from the document's 100,000-case experiment (Kemeny alone 7.7 %,
Copeland 43.3 %, Footrule 22.7 %, last resort 26.3 %; Kemeny-optimal in 99.93 %) are reproduced
by `simulate_stage_shares` and asserted in `tests/test_rtd_rank_aggregation.py`.

**Wording fix applied**: `rtd_rank_aggregation.py` now states explicitly that neither output
rule applies to a *mechanism's own* priority list — those are tails of a fixed priority
sequence and may legitimately continue past Option 5 (e.g. Loyalty's full list
`3 > 1 > 4 > 5 > 2` ends with Option 2 after Option 5) and are never cut to the TTP length.

#### Output-field definitions (for the documentation agent)

| Field | Meaning |
|---|---|
| `rtd_consensus_kemeny_status` | What the Kemeny step (Phase 1) returned. `'unique'` — exactly **one** optimal permutation, a fully ordered ranking, the procedure ends there. `'unique_with_ties'` — several optimal permutations that are exactly the linear extensions of **one** weak order, i.e. a single optimal list that still contains tied options. `'multiple'` — several optimal orderings that are **not** the extensions of one weak order, so Schulze supplies the Phase-1 ordering instead. |
| `rtd_consensus_n_kemeny_optimal` | How many of the 120 permutations attain the minimum total Kendall-tau distance (1 iff status is `'unique'`). Mean 6.64 on the 280 participants under the new mapping. |
| `rtd_consensus_is_kemeny_optimal` | Whether the **final** consensus ranking is one of those optimal permutations. Only the random last resort can make this `False` (document: 99.93 % optimal; 280/280 on the research participants). |
| `rtd_consensus_settled_by` | The stage that produced the final fully ordered ranking: `'kemeny'`, `'schulze'`, `'copeland'`, `'footrule'` or `'random'` — the last stage that still had work to do. |
| `rtd_consensus_truncated_by` | Which output rule actually bound on the integrated list. `'none'` — neither cut anything (only possible when the TTP length is 5 and Option 5 is last). `'length'` — the TTP choice length cut it, at or before the Option-5 position. `'option5'` — the Option-5 cut bound (Option 5 appears before position 5, at or before the length cut). `'both'` — both rules cut at the **same** position. |
| **"% of initial ties"** | The share of agents whose **Kemeny step did not yield a unique fully ordered ranking**, i.e. `rtd_consensus_kemeny_status != 'unique'`. It is *not* the share settled by the random last resort. On the 280 research participants: **93.9 %** under the new mapping (89.3 % under the old). |
| `rtd_consensus_phase1` | `'kemeny'` if the Kemeny weak order was carried into Phase 2, `'schulze'` if the status was `'multiple'` and Schulze's ordering was carried in instead. |

---

## 5. Categorical-income specification

### 5.1 What the app implements

Only **WTP** and **Risk-Taking** use income. `TTP`, `Loyalty` and `Flexibility` are income-free
and are **bit-identical** in both modes (verified: identical scores and segments on all 280).

Setting `income_mode: categorical` replaces the continuous income term
`β_I · z_net_income` with fitted per-budget-level effects, exactly as in the document's `cond()`
code (Section 3 for WTP, Section 4 for RT):

```
score_categorical = intercept + level_dummy(assignedallowancelevel) + personality_part
```

where `personality_part` is the element's own equation **minus** the income term, entering at
**coefficient 1.0** (doc-literal: the document's operative `cond()` code applies it at slope 1;
the fitted slope on `WTP_noincome` / `RT_noincome_hs` — 0.5439353 and 0.9991365 in the
regression output the document prints — is **not** applied).

**WTP** (`assignedallowancelevel` 1..5 ↔ `totalallowance` 12/32/72/128/200 EUR; level 1 = base):

| Term | Value |
|---|---|
| intercept (`_cons`) | −0.691843 |
| level 2 (EUR 32) | +0.2671588 |
| level 3 (EUR 72) | +0.5057716 |
| level 4 (EUR 128) | +0.9411747 |
| level 5 (EUR 200) | +1.822471 |
| personality part | `0.078863062 · z_extraversionbig5 − 0.012326128 · z_agreeable` |

**Risk-Taking**:

| Term | Value |
|---|---|
| intercept (`_cons`) | −0.0068307 |
| level 2 | +0.0026128 |
| level 3 | +0.0050555 |
| level 4 | +0.0092738 |
| level 5 | +0.0179812 |
| personality part | `0.025942386297 · z_E + 0.023699214948 · z_O − 0.038734315188 · z_A − 0.037739440732 · z_C − 0.025388697852 · z_N` |

**Downstream is identical to continuous mode**: standardize (`egen z_… = std(…)`, + the
element's β), min-max rescale into five equal-width segments, then the segment → option-list
tail mapping of §1, then the optional stochastic layer (doc: *"replacing `z_WTP_calculated` with
`z_WTP_categorical`"* / *"replacing `z_RT_calculated_hs` with `z_RT_categorical`"*).
Min-max rescaling is invariant to standardization and to additive intercepts, so binning the raw
categorical score gives identical segments to binning its standardized version — verified as a
test.

The professor has not yet specified categorical-specific σ values, so the continuous σ
configuration is reused unchanged.

### 5.2 Propagation to the integrated decision

The categorical WTP and RT segment → option lists enter the Section-6 Kemeny aggregation on
exactly the same footing as their continuous counterparts; TTP, Loyalty and Flexibility
contribute identical inputs in both modes, and the TTP list length that truncates the integrated
list is also identical. So the income mode changes the integrated default list **only** through
the WTP and Risk-Taking rankings.

### 5.3 Parity against the `.dta`

| Check | Result |
|---|---|
| `weighted_RT_categorical`, max abs diff | 2.3e-08 |
| RT segment parity | **280/280** — app 21/103/109/43/4, `.dta` 21/103/109/43/4 |
| `z_RT_categorical`, max abs diff | 2.0e-07 |
| `weighted_WTP_categorical`, max abs diff | 0.0595 (one branch, see below) |
| WTP segment parity | **279/280** — app 67/102/57/—/54, `.dta` 67/101/58/—/54 |

The single WTP difference is the known `.dta` construction quirk, now **proved numerically**:
the file's **level-1 branch multiplies the RAW `agreeable`**, not `z_agreeable`. On the 57
level-1 participants, `weighted_WTP_categorical` matches
`−0.691843 + 0.078863062·z_E − 0.0123242128·agreeable` to **8.8e-08**, but differs from the
`z_agreeable` version by up to **0.0595**. The document's own `cond()` code in rev 040926 writes
`z_agreeable` in that branch (with the coefficient printed as −0.0123242128 rather than the
−0.012326128 used at every other level, a 1.6e-4 relative difference that is numerically
irrelevant), consistent with its own `WTP_noincome` construction — so the app uses `z_agreeable`
at **all** levels. Because the level-1 values feed the population min/max, the resulting shift
moves exactly **one** participant (a level-3 one) across a segment boundary: 279/280.

WTP segment 4 is **genuinely empty** in categorical mode in both the app and the `.dta`: the
level-5 dummy (+1.822471) opens a gap in the score range that the equal-width bins map to an
unpopulated bin.

---

## 6. Summary of code changes

| File | Change |
|---|---|
| `src/decisions/rejected_transaction_defaults.py` | `_ranking_for_segment` flipped to `seq[5-s:]`; `_intercept_raw_shift` applies TTP β₀ on the standardized scale (`β·sd₀`); `reducation` sanity warning added; module docstring rewritten for both changes plus the rev-040926 verification result and the corrected Flexibility-σ note |
| `src/decisions/rtd_rank_aggregation.py` | Docstring states the two output rules apply to the **integrated** list only; `apply_output_rules` and `aggregate_rankings` docstrings now define `truncated_by`, `kemeny_status`, `n_kemeny_optimal`, `is_kemeny_optimal`, `settled_by` and "% of initial ties" precisely |
| `config/decisions.yaml` (D4 section) | Mapping-direction comment rewritten; intercepts comment rewritten (all five on the standardized scale, with the β₀ rationale); σ comments corrected and the WTP σ drift documented |
| `tests/test_rejected_transaction_defaults.py` | Choice-column assertions now check the **mirror** of the stored `.dta` columns via a new `_stored_lists_by_segment` helper; new `test_segment_to_option_mapping_direction`, `test_ttp_intercept_is_on_the_standardized_scale`, `test_research_default_ttp_intercept_keeps_zero_length_lists`; intercept/saturation tests updated to the new β₀ scale |
| `tests/test_rtd_flexibility.py` | `test_flexibility_choice_lists_match_stata_columns` rewritten as a mirror check, asserting the professor's two anchor cases (segment 5 → `2>4>3>1>5`, segment 1 → `[5]`) |

Test status: **74 passed** for
`tests/test_rejected_transaction_defaults.py tests/test_rtd_flexibility.py tests/test_rtd_rank_aggregation.py`
(excluding `test_apptest_aggregation_subtab_and_results_section`, which exercises UI owned by
another agent).

## 7. Open items needing the professor's decision

1. **Mapping vs the `.dta`** — the app now contradicts the stored `choice1..5_*` columns by
   design (§1). The document's Stata code should be updated to match its prose, otherwise the
   next `.dta` regeneration will reintroduce the conflict.
2. **TTP β₀ scale** — confirm the standardized reading (§2), or state a raw-scale default that
   is small relative to the 0.21 composite range (e.g. 0.0018 ≈ 0.05 SD).
3. **Loyalty Agreeableness weight** — 0.67139 in the equation vs 0.267139 in the weight table;
   correcting it moves 59/280 segments (§4.1).
4. **Min-max window across populations** — copula and research results are not comparable while
   the cutoffs are recomputed from each population's two extreme agents (§3c).
5. **WTP σ_overall** — 0.45265807275 (config) vs 0.4526575455 (rev 040926); 1.2e-6 relative.
6. **Categorical personality slope** — the document's `cond()` code applies the personality part
   at slope 1, while its own regression fits 0.5439353 (WTP) / 0.9991365 (RT).
7. **Categorical σ** — no categorical-specific σ has been specified; continuous values are reused.
8. **UI mirrors** — the tab and results-visualisation copies of the mapping
   (`app/pages/decision_tabs/rejected_transaction.py`,
   `app/pages/results/visualizations/transaction_viz.py`) must be flipped by their owner.
