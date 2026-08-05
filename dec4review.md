# Decision 4 (Rejected Transaction Defaults) — Review Notes

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
