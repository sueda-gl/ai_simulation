# Decision 4 (Rejected Transaction Defaults) — Review Notes

> **RESOLUTION UPDATE (Issue 1, professor's ruling):** Prof. Lavie confirmed the
> segment→option mapping in his Stata code was a bug — the document's PROSE is the
> intended behavior ("the 20% highest scores opt for the top option first"). The
> implementation now uses the MIRRORED mapping: segment s receives the tail of the
> priority sequence from position 6−s (highest segment → full list starting with the
> top option; lowest segment → only the last option). Verified: the WTP first-choice
> distribution on the 280 participants is now {Option 5: 168, Option 4: 82,
> Option 1: 26, Option 2: 3, Option 3: 1} — exactly the professor's stated expectation.
> Consequence: the choice1–5_* columns in Stata_File_Decision4_050626.dta are
> KNOWN-FLIPPED; tests assert the file's structure as-is but the model's output
> against the mirror. A corrected .dta is expected for direct parity. Issue 1
> discussion below is retained for the historical record.

What verification found, how each conflict was resolved (Stata file arbitrated, per project rule),
and exactly which source was implemented. For manual checking, the easiest tool is
**`data/stata_d4_verification.csv`** — extracted from the professor's
`Stata_File_Decision4_050626.dta`, it has every input and every derived column side by side,
so you can open it in Excel and see everything described below.

---

## Issue 1 — The doc's prose and the doc's own Stata code point in opposite directions (the big one)

**What the prose says** (Loyalty section, search the docx for this sentence): *"the 20% highest
loyalty scores will opt for Option 3 as their first choice, the next 20% loyalty scores will opt
for Option 1..."* — i.e., **most loyal → Option 3 first** (pay the PN price to stay with your
vendor, the most loyal behavior). That direction also matches the theory section at the start of
the doc.

**What the Stata code right below it says**: `replace choice1_loyalty_deterministic = 3 if
weighted_loyalty15 == 1` and `... = 2 if weighted_loyalty15 == 5`. Since `weighted_loyalty15` is
built by min-max rescaling, **segment 1 = the lowest loyalty scores, segment 5 = the highest**.
So as coded: the **least** loyal customers get Option 3 first, and the **most** loyal customers
get Option 2 (switch to another vendor — the least loyal behavior) as their only choice. The
`.dta` contains exactly the coded version, all 280 participants.

You can see this in the CSV: filter `weighted_loyalty15 == 5` — those rows have the *highest*
`weighted_loyalty` values, and their `choice1_loyalty_deterministic` is `2` with choices 2–5
empty. Filter `weighted_loyalty15 == 1` — the *lowest* loyalty values, first choice `3`, full
list 3,1,4,5,2.

The same inversion exists in **WTP** (prose: highest WTP → Option 3 first; code/data: the
highest-WTP segment gets Option 5 = *forgo the transaction*, which is the lowest-WTP behavior)
and in **RT** — where the doc even contradicts *itself* in prose: one sentence says the
highest-RT segment takes Option 4, but the code two lines later gives Option 4 to segment 1.

There's a second, subtler consequence baked into the same code: because segment *s* receives
only the **tail** of the sequence from position *s*, the segment number also determines **list
length** (segment 1 → 5 options, segment 5 → 1 option). So the ranking mechanisms implicitly
encode a list-length decision that is independent of — and can contradict — the
Tendency-to-Plan length mechanism. The doc never reconciles the two.

**What was implemented:** exactly what the Stata code and the `.dta` do (the Stata file
arbitrates), reproducing it 280/280. Nothing about the prose interpretation exists in the code.
**What to ask the professor:** which direction is intended — and if the prose is right, the fix
is trivial (reverse the segment→sequence mapping) but it changes every choice column, so it
needs a corrected `.dta` to verify against.

## Issue 2 — "20% of observations" is not what the code produces

The doc says each of the five segments contains "observations in the corresponding 20% range."
But `floor(1 + (5−0.0001) × minmax(score))` creates five **equal-width slices of the score
range**, not equal-count groups. With skewed scores the counts are far from 20%: in the `.dta`,
the WTP segments hold **168 / 82 / 26 / 3 / 1** participants (60% in segment 1). Check: count
rows per `WTP_calculated15` in the CSV. If the professor genuinely wants 20% of *people* per
segment, the code should use percentile cuts (Stata `xtile`) instead — that's a substantive
spec decision, not a bug fix.

## Issue 3 — The WTP stochastic draw isn't anchored where the doc says

The doc says the random draw is centered on the **binned 1–5 score** (`Mean =
WTP_calculated15`). The data says otherwise: in the CSV, compute `sWTP_calculated −
WTP_calculated` (the raw continuous score) — its mean is ≈ 0 and its spread ≈ 0.468, consistent
with one common σ = 0.4527 for everyone. Compute `sWTP_calculated − WTP_calculated15` instead —
its mean is ≈ **−1.54**, which is impossible for a draw centered on that value. So the actual
Stata run anchored the draw on the **raw score**, with the **single overall σ** (the
per-budget-group σ option fits the data measurably worse). Continuous-score anchoring is the
default for all four mechanisms in the implementation (it's also the only reading under which
the σ formulas' scaling makes dimensional sense), and the doc-literal "binned" option is
selectable in each ranking sub-tab's Advanced expander.

## Issue 4 — Verified-harmless quirks (no action needed)

- The doc's loyalty code says to standardize the composite and add β₀ — the `.dta` shows
  neither was done. Provably irrelevant: min-max rescaling wipes out any standardize/shift, so
  every segment and choice is identical either way. Same for the never-materialized β₁/β₂
  intercepts — they mathematically cannot affect any output, which is why the tab exposes no
  intercept control.
- The doc's WTP weight sum says 4764 but 1013+251+3501 = **4765**. The `.dta` embeds the
  4764-based coefficients; the error scales all three coefficients by the same factor, so
  outputs are bit-identical. The `.dta` values were kept.
- The loyalty σ table's €32 row reads 0.04975281 — a dropped digit; the row's own formula
  (1.1366981 × 0.0268299279) gives **0.03**04975281. The corrected value is used; anyone
  copying the table instead of the formula would apply ~63% too much noise to that budget group.
- The RT σ is stated as `(5.979747/18) × mean(stdactions) = 0.332208167` — but 5.979747/18
  alone is 0.332208; the ×mean(stdactions) multiplication that TTP, Loyalty, and WTP all apply
  is missing from the arithmetic. The doc's printed number is followed (there's no `.dta` to
  arbitrate since stochastic RT was never generated), but pattern-consistency would give
  0.394111.

## Issue 5 — Three loyalty meta-analysis numbers can't be reproduced from the doc itself (doc-level, worth raising)

The final loyalty equation (0.09·z_E + 0.0273·z_O + 0.0045·z_A) is internally consistent
*given* the doc's tables, and it's what the `.dta` embeds — so that's what was implemented. But
three of the inputs behind those tables don't follow from the doc's own stated formulas:

1. **Leon & Dixon, Extraversion row**: the doc lists SE = 0.008748 with weight 114.4165 — but
   1/SE² = 13,067; the printed 114.4165 is 1/SE (not squared). Every neighboring row uses
   1/SE². If squared correctly, Leon & Dixon would carry ~86% of the extraversion weight and
   the pooled β would drop from 0.4128 to ≈0.037 — the 0.09 coefficient would change
   drastically.
2. **Ho & Wong SE(ESTLtotal) = 0.023723**: this drives Ho & Wong's dominant 82.5% weight, but
   no combination of the doc's own listed path SEs reproduces it via the Sobel/delta method
   (recomputation gives ≈0.140).
3. **Agreeableness combined SE = 0.031375866**: applying the doc's own formula √Σ(SEᵢ²·Wᵢ²) to
   the doc's own numbers gives 0.011546. This matters twice over — it sets the cross-trait
   weights (corrected, the equation would look like ≈0.055/0.017/0.020 instead of
   0.09/0.0273/0.0045), and it's the only reason Agreeableness survives the p < 0.15 inclusion
   rule (stated p = 0.142; corrected p ≈ 0.00007).

These likely trace to the professor's side worksheets, so only they can say which numbers are
the intended ones. TTP and Risk-Taking, by contrast, verified perfectly — every number in those
sections reproduces to 9+ significant digits.

---

# Which source was implemented, per issue

Wherever the doc and the .dta conflicted, the **.dta** was implemented (the Stata file
arbitrates). Wherever the .dta was silent (things never generated in it, like stochastic
loyalty/RT), the **doc as literally written** was implemented. Nothing was invented; where a
genuine judgment call existed, the .dta-consistent choice is the default and the doc-literal
reading is available as a toggle.

| Issue | What was implemented | Source used | Which side is "wrong" |
|---|---|---|---|
| 1. Mapping direction | Segment 1 (lowest score) → full list starting with Option 3; segment 5 → one option | **.dta** (= the doc's Stata code) | The doc's **prose** contradicts what was actually run. Substantively, the prose matches the theory (loyal people should get the loyal option first), so the *code/data* may be the real mistake — only the professor can say |
| 2. "20% of observations" | Equal-width fifths of the score range | **.dta** (= the doc's Stata code) | The doc's **prose** — the code never produces 20% groups |
| 3. WTP stochastic anchor | Draw centered on the raw continuous score, one common σ = 0.4527 | **.dta** | The doc's **text** (says binned anchor, offers per-quintile σ) — the data proves neither was used |
| 3b. Loyalty/RT stochastic anchor | Continuous score by default (consistent with the WTP ground truth); "binned" doc-literal option in the Advanced expander | .dta *pattern* (no direct arbiter exists) | Unresolvable — the .dta has no stochastic loyalty/RT columns at all |
| 4a. Loyalty std()+β₀ step | Raw composite, no standardization, no β₀ | **.dta** | The doc's **code listing** (step was never executed) — provably zero effect on outputs either way |
| 4b. WTP weight sum 4764 vs 4765 | The 4764-based coefficients | **.dta** | The doc's **arithmetic** (sum is 4765) — provably zero effect on outputs |
| 4c. Loyalty σ, €32 row | 0.0304975281 | Doc's own **formula** (no .dta arbiter) | The doc's **table cell** (0.04975281, a dropped digit) |
| 4d. RT σ | 0.332208167 as printed | Doc **as written** (no .dta arbiter) | The doc is internally inconsistent (its formula says ×mean(stdactions), its number omits it); the printed number is followed |
| 5. Loyalty coefficients 0.09 / 0.0273 / 0.0045 | Exactly those values | **Doc and .dta — they agree**, so no conflict | Possibly **both**: three upstream meta-analysis inputs don't follow from the doc's own formulas, so the final coefficients may be wrong at the source — but there was nothing to arbitrate against |

One clean pattern worth noticing: in every doc-vs-data conflict, the .dta matches the doc's
**Stata code listings**, and it's the doc's **narrative prose** that disagrees. So the
implementation equals "what the professor's code actually did," verified 280/280 — and the open
question for the professor is whether what the code did is what they *meant* (Issue 1
especially, since there the theory sides with the prose, not the code).

---
---

# ADDENDUM — Doc revision 130826, from-zero re-verification

New document: `Decision 4 - Rejected Transaction Defaults 130826.docx` (4,342 lines vs 3,328 in
the 100726 revision; extracted text used for line references below). Verified by three fresh,
independent agents given **none** of the findings above, arbitrating against the same
`Stata_File_Decision4_050626.dta` (note: the .dta predates this revision, so the newly added
variables have no data counterpart — inventoried below).

## A. Every previous finding was independently re-confirmed; none was fixed

The from-zero check re-derived the entire original list on its own: the segment-mapping
inversion (Issue 1 — now present in **all five** mechanisms including the new Flexibility one);
the "20% of observations" claims vs equal-width binning (Issue 2); the WTP stochastic draws
anchored on the raw continuous score with one common σ, per-quintile σ statistically rejected
(Issue 3); the loyalty std()+β step never executed (4a); 4764 vs 4765 (4b); the €32 σ-table
dropped digit (4c); the RT σ ×mean(stdactions) omission (4d — now repeated in the Flexibility
σ, stated "(6.361869/18)×mean(stdactions)" but computed as 6.361869/18 = 0.3534372); and all
three non-reproducible loyalty meta-analysis inputs (Issue 5). Independent recomputation again
shows the corrected loyalty equation would be ≈ 0.055·E + 0.017·O + 0.020·A (vs 0.09/0.0273/
0.0045) and that Agreeableness's combined SE should be ≈0.0115 (z≈3.99, p≈0.00007 — highly
significant, not the doc's marginal p=0.142). Nothing in the implementation is contradicted.

Also unchanged: intercepts remain undefined placeholders (now β0–β4 = "XXX"); the loyalty
stochastic block still conditions every `choice*_loyalty_stochastic` on the **deterministic**
`weighted_loyalty15` (a no-op as written; the WTP/RT/flex stochastic blocks correctly use
their s*15 variables); `min_2weighted_loyalty` typo persists.

## B. NEW: verified categorical-income specifications (WTP + RT) — the good news

The revision finally defines the categorical income mode, via regression rather than
substitution: build the score **without** the income term, regress the full continuous score
on it plus budget-level dummies (`i.totalallowance`, base €12), and replace the income term
with the fitted intercept + dummies. **Both embedded regression outputs reproduce bit-for-bit
from the .dta** (two agents independently) — the strongest-verified new material:

**WTP** (doc ln 1303–1320): `WTP_noincome = 0.0788796127824·z_E − 0.012328716·z_A`;
regress `WTP_calculated` on WTP_noincome + i.totalallowance:

| Term | Coef | SE | t |
|---|---|---|---|
| WTP_noincome | 0.5439389 | 0.2115558 | 2.57 |
| €32 | +0.2672136 | 0.0529004 | 5.05 |
| €72 | +0.5058749 | 0.0551847 | 9.17 |
| €128 | +0.9413666 | 0.0540135 | 17.43 |
| €200 | +1.822843 | 0.0545341 | 33.43 |
| intercept | −0.6919842 | 0.0380311 | −18.20 |

R² = 0.8331 (adj 0.8301), F = 273.61, RMSE = 0.28707. Sanity check verified: intercept+dummies
essentially recover the group means of the replaced 0.698·z_net_income term
(−0.692/−0.425/−0.186/+0.249/+1.131 vs actual −0.694/−0.429/−0.179/+0.248/+1.132).

**RT** (doc ln 2477–2492): `RT_noincome_hs` = H&S RT equation minus income term; same design:
slope 0.9991365 (t 379.15); dummies €32 +0.0026128, €72 +0.0050555, €128 +0.0092738,
€200 +0.0179812; intercept −0.0068307. R² = 0.9981, F = 29252.50, RMSE = 0.00285. Every
coefficient, SE, t, CI, and the full SS decomposition reproduce exactly.

**Two defects in the WTP categorical code block (fix before implementing):**
1. The `cond()` branch for allowance level 1 uses raw `agreeable` instead of `z_agreeable`
   (ln 1324) — as printed it shifts level-1 scores by ≈−0.06 to −0.03. Unambiguous bug;
   correct answer: `z_agreeable`.
2. The constructed `weighted_WTP_categorical` applies the personality part at coefficient
   **1.0**, silently discarding the fitted slope **0.5439389**. In RT the same substitution is
   harmless (slope 0.9991 ≈ 1) but for WTP it materially re-weights personality vs income —
   the doc never states this is intentional. Needs the professor's ruling (apply the slope, or
   confirm slope-1 substitution by design).
Also: the inline "Therefore:" pseudo-Stata lines (ln 1322, 2494) are garbled (unbalanced
parentheses); the operative `cond()` code below each is correct (modulo defect 1).

None of the categorical variables (`WTP_noincome`, `weighted_WTP_categorical`,
`z_WTP_categorical`, `RT_noincome_hs`, `weighted_RT_categorical`, `z_RT_categorical`) exist in
the 05.06.26 .dta — the doc's regressions were evidently run on a later file but are exactly
replicable from this one, so implementation parity targets can be self-generated.

## C. NEW: the Flexibility section (§5, ~900 new lines) — complete mechanism, broken arithmetic

For the first time Flexibility is a full computable fifth mechanism: 6-study pool (Odaci,
Daban, Latzman [DV-reversed], Smith & Konik; Hooi & Tan and Nair excluded), per-trait IVW and
H&S pooling, IVW retained, final equation on z-scored Big-5 (no income), blended with observed
behavior as `anchored = 0.25·z_stdactions + 0.75·z_Flexibility_calculated_ivw`, min-max floor
into 5 segments, priority sequence **2 ≻ 4 ≻ 3 ≻ 1 ≻ 5**, and a stochastic layer
(σ_overall = 6.361869/18 = 0.3534372; per-allowance rows at multiplier 0.297923 — all rows
internally consistent). **However, the printed final equations are arithmetically wrong and
must not be implemented as-is:**

| # | Error (doc ln) | As printed | Correct value |
|---|---|---|---|
| F1 | IVW summary "Rel. Weight" column (3273–3297) | 28.32/16.56/18.58/19.14/12.41% | Copy-pasted **from the RT section** (sums to 95% because RT's income weight is dropped); bears no relation to flexibility |
| F2 | Cross-trait w_N (3300–3304) | 0.1240639657 (RT's value) | 695.4518/4700.4381 = **0.147955** |
| F3 | Cross-trait w_A (3300–3304) | 0.1479546675 (N's correct value, shifted) | 1278.011/4700.438 = **0.271892** |
| F4 | Final IVW N coefficient (3308) | −0.0439714703 | **−0.0524390** (with doc's own table values) |
| F5 | Final IVW A coefficient (3308) | −0.0258293681 (also sign-typo: product is positive) | **+0.0474659** (doc's table values) |
| F6 | H&S Extraversion coefficient (3363, code 3391) | 0.30846193 = 0.15827863 **+** 0.1501833 (added, not multiplied) | 0.15827863 × 0.1501833 = **0.0237709** (13× smaller) |
| F7 | Latzman N weight 1/SE² (3031) | 26.09779355 (β multiplied in by mistake) | 1/0.1490775² = **44.9962** → weights 48.17/36.21/6.40/9.21% |
| F8 | Daban N β in combination (3072) | 0.378 | **0.376** → corrected β_N,IVW = **−0.3598072**, SE 0.0377111 |
| F9 | A→Flex H&S Var (3171–3173) | 0.00911805288 (digit '9' inserted) | 6.1304862/5193 = **0.0011805** → SE 0.034361, z 5.42, **p 6e-8 (highly significant)** — overturns the doc's "H&S makes A insignificant" rationale for preferring IVW |
| F10 | C→Flex SE(β_comb) (3245) | 0.0363351 (uses Daban's weight 0.297² twice) | **0.0289851** (z 7.17; restores the IVW identity 1/SE² = Σ1/SE² = 1190.28) |
| F11 | C→Flex H&S deviations (3254–3258) | −0.02938…/−0.10938… | −0.02968…/−0.10968… → SE 0.0496803 (conclusion unchanged) |
| F12 | H&S ΣN expression (3317) | "1302+1731+1731+1731+1731+1731 = 8226" | one 1731 too many listed; **8226 = 1302 + 4×1731** is what's used (correct) |
| F13 | Anchor weight (3399 vs 3408) | prose 25%, code "0.2z_stdactions" (also missing `*`) | **0.25** — the doc's own quoted min/max (−2.82353/3.538339) reproduce from the .dta only with 0.25/0.75 |
| F14 | O-section derivation lines (2986–2987) | show E's numbers (√0.000922639; 0.09771/0.030375) | O's own: sum 0.001129053, SE 0.03360152 (the tabled results are right; the shown work is E's) |

**Fully corrected final IVW equation** (holding the doc's own SE conventions fixed, correcting
F2/F3/F7/F8/F10; Σ1/SE² = 5141.00):

    Flexibility_i = β4 + 0.020600·E + 0.029412·O − 0.049213·N + 0.043390·A + 0.048109·C

(vs printed β4 + 0.0225304·E + 0.0321685·O − 0.0439715·N + 0.0258294·A + 0.0334856·C).
Note the corrected coefficients change the score range (≈6.594 vs the doc's 6.361869), so the
flexibility σ would change too. The corrected H&S Extraversion coefficient is 0.0237709 (F6).

**Specification questions with NO computable "correct answer" (professor must rule):**
- **F-Q1**: Daban's Extraversion effect (β=−0.044, p=0.308) is *included* with the largest
  weight (49.5%) although the stated rule and the section's own sentence say non-significant
  effects are excluded — the exact IVW pathology the WTP section cites to justify H&S.
  Excluding it per the stated rule gives β_E = 0.2368 instead of 0.09771.
- **F-Q2**: SE derivation uses unexplained z = 5 or 6 for several studies although exact
  t-statistics are reported (using the t's would ≈double Odaci's weights). No stated rule
  covers this.
- **F-Q3**: pooling-method rationale is now self-contradictory across sections: WTP prefers
  H&S because sample sizes differ widely (N 251–3501); Flexibility prefers IVW claiming that
  condition doesn't hold (N 258–620 — it held *more* strongly for WTP) and because "H&S
  reveal two insignificant coefficients" — a rationale partly based on the F9 typo.
- **F-Q4**: the segment→option inversion (Issue 1) is repeated verbatim in the flexibility
  block (ln 3425 prose: highest segment takes Option 2; code ln 3431: segment 1 → Option 2).
- No flexibility variable exists in the .dta, so there is no parity target; implementation
  should wait for corrected coefficients (or professor's sign-off on the corrections above).

## D. Other findings new to this revision

- **Section 6 "Integrating the model effects" is entirely empty** (ln 3679 ff. are blank
  bullets) — the rank-aggregation step the whole decision builds toward is still unwritten,
  though the intro promises "a unified predictive model".
- The RT section still contains three mutually contradictory mapping statements (prose
  "highest → Option 3" [WTP copy-paste], parenthetical "RT15==5 takes Option 4", code
  "4 if RT15==1"); the .dta matches the code.
- Duplicated study summaries persist (Buelow & Cayton, Joseph & Zhang; the WTP priorities
  paragraph appears twice); the Decision-2 disclosure annex and figure fragment are still in
  the file; "3.1/3.2/3.3" headings sit inside §4; ln 2607 ends mid-sentence; the summary
  figure mixes coefficient levels across mechanisms and contains "0.0.0237***" and an
  income→RT label (0.0077) matching neither the H&S (0.0069) nor IVW (0.0088) coefficient.
- Minor new transcription slips confined to dropped rows or non-operative text: Udo-Imeh
  t = 15.009 vs 5.009; misplaced "z = 4.2"; O→WTP inner sum 0.001186 vs 0.0010186 (SE
  0.034438 → 0.031916; trait dropped either way); C→WTP H&S squaring omission (SE 0.420 →
  0.271; still dropped); assorted mislabeled headings ("Agreeableness → Conscientiousness",
  H&S blocks labeled "βExtraversion" for other traits).

## E. Status of the .dta vs this revision

Present in the .dta (44 vars): all four original deterministic mechanisms + WTP stochastic —
everything the implementation is parity-tested against. Absent (56 vars): all categorical
branches, RT-IVW, the entire Flexibility mechanism, TTP/loyalty/RT stochastic outputs,
z_stdactions, and every intercept except the Decision-2 leftover `beta0 = 0.1`. A refreshed
.dta from the professor would allow parity tests for the new material; until then, the two
categorical regressions (§B) are the only new blocks that can be — and have been — verified.
