# Open questions for the professor

The migration split the logic from the screens. It did **not** re-derive the model.
Along the way seven things turned up that look like modelling decisions rather than
programming decisions, so they were left exactly as they were and written down here.
Three are still open (items 1–3). Items 4–7 have since been ruled on by the owner and
applied, and are kept below as records; the one thing left from item 7 is a data-file
note, at the end.

Each open item says three things: **what the code does today**, **why it looks
suspicious**, and **what would change if it were altered**. None of them is a bug the
migration introduced; every one of them predates it.

---

## 1. Three customer-price conventions in the exports

**What the code does today.** Three exports compute "Customer Price" for the same
purchase requests, in three different ways.

| export | source file | vendor price used | Fixed / Discount rows |
|---|---|---|---|
| Transaction-Level sheet | `app/reports/transaction_level.py` | **never** the vendor's own price — always `(1 + price_range) × (1 + platform_markup) × market_price` | included, but the price column shows `'N/A'` |
| Donation transaction export | `app/reports/donation.py` | the vendor's own price when the request names a vendor, market price otherwise; **also** defines Fixed = vendor price and Discount = vendor price × 0.7 | included, with a price |
| Purchase-vs-bid export | `app/reports/purchase.py` | the vendor's own price when available, market price otherwise | **skipped entirely** (`continue`) |

The `× 0.7` in the donation export is a hard-coded "assume 30 % discount" that
appears nowhere else in the model, and `app/reports/donation.py` still carries
`DEFAULT_MARKET_PRICE = 100.0`, `DEFAULT_PLATFORM_MARKUP = 0.1`,
`DEFAULT_PRICE_RANGE = 0.25` as module defaults that the page does not override.

**Why it looks suspicious.** The same transaction can appear in two workbooks with
two different prices. In a multi-vendor run the Transaction-Level sheet prices every
request at the market price while the other two price it at its vendor's price.

**What would change if it were altered.** Nothing in the model — the price columns
are report output only, never fed back into a decision. Making them consistent
changes the numbers in the Transaction-Level sheet and the Excel files students
download, and would need a decision on what the "actual price" of a Fixed or
Discount purchase is, which the code currently declines to state.

**Related, same family.** Two more inconsistencies were left untouched:

* **The vendor score is computed by two different formulas.** The Agent-Level sheet
  (`app/reports/agent_level.py`) contains an inline copy that sets
  `norm_price = 0.5` when the price bounds are equal, while the shared function
  `calculate_vendor_score_with_breakdown` in `src/vendor_attribute_generator.py`
  returns `1.0` in that case. With the default single vendor the Page-1 bounds
  **are** equal, so the Agent-Level `avg_vendor_score` and the Transaction-Level
  `vendor_integrated_score` disagree for exactly the same vendor.
* **Rounding is baked into the stored values,** not applied at display time
  (`float(f"{customer_price:.2f}")` in `app/reports/purchase.py`, and `round(…, 2)`
  in the disclosure export), so the rounded number is what leaves the program.

**The question.** Which convention is authoritative, and which function is the
authoritative vendor score?

---

## 2. Three frozen Decision 1 reference files that disagree

**What the code does today.** `data/` holds three files that all claim to be
Decision 1 (disclose income) results for the same 280 participants. They disagree,
and the repository takes no position on which is right:

| file | rows × cols | Decision 1 outcome | how it was produced |
|---|---|---|---|
| `data/stata_step5_results.csv` | 280 × 32 | `disclose_categorical` sums to **165** (58.9 %) | `stata/step5_full_pipeline.do` — the only one with a script in the repository |
| `data/stata_results.csv` | 280 × 32 | **252** `Y` (90.0 %) | **no script in this repository produces it**; its variable names differ from step 5 (`z_hh`, `rs_01`, `anchored_pb`, `di_i` vs step 5's `z_honesty_humility`, `religiousservice_01`, `anchored_prosocial_behavior`, `fs_deterministic_categorical`) |
| `data/python_verification.csv` | 280 × 12 | **232** `Y` (82.9 %) | written by `experiments/verify_against_stata.py` |

Two further mechanical problems:

* `data/python_verification.csv` has `participant_id` = 0, 1, 2 … 279 — row indices,
  not participant IDs — because the script reads a frame that has no
  `Participant ID` column. The comparison script then merges it with
  `data/stata_results.csv` (whose IDs are real: 831, 924, 900 …) **on that column**,
  so the two files are compared row-against-wrong-row. Any agreement rate computed
  that way is meaningless.
* `data/stata_incomes.csv` — the frozen income column the tests inject — cannot be
  reproduced by the engine. Its values form clean, disjoint bands per allowance
  level, but they are not the bands of the configured `lognormal(mu = 10,
  sigma = 0.5)` (the 40th percentile of that distribution is about 19 400, below the
  level-2 band's maximum of 20 263). The parameters that generated it are unknown.

**Why it looks suspicious.** 165, 232 and 252 out of 280 are not rounding
differences. At least two of the three files describe a model that is not the one
in the code.

**What would change if it were altered.** Nothing in the running model — none of
these files is read at run time. They matter because they are the evidence base:
`experiments/verify_against_stata.py` compares against `stata_results.csv`, and
`tests/test_disclose_documents.py` injects `stata_incomes.csv`. If the wrong file is
the reference, the verification is measuring the wrong thing.

**The question.** Which of the three is the reference for Decision 1, and what
produced `stata_results.csv` and the frozen income column?

---

## 3. `config/income_reference_breaks.json` was deleted as dead

**What the code does today.** Nothing — the file was removed as part of the dead-code
cleanup (ruling R37), because no `.py`, `.yaml` or `.md` file in the repository
referenced it. It is recoverable from git history (`git show 80ee344:config/income_reference_breaks.json`).

It contained the income quintile breakpoints derived from the 280 experiment
participants, together with the distribution they came from:

```json
"reference_params": { "min": 1000.0, "max": 10000.0, "avg": 5000.0,
                      "mu": 8.229370054791982, "sigma": 0.7587135646925732 },
"quintile_breakpoints": [1418.05, 2518.71, 3749.47, 5581.65],
"verification": { "q1_count": 57, "q2_count": 61, "q3_count": 52,
                  "q4_count": 56, "q5_count": 54 }
```

**Why it looks suspicious.** Three things.

* Its income range (1 000 – 10 000, average 5 000) is the range the app used *before*
  the current defaults (0 – 100 000, average 25 000), so the breakpoints describe a
  scale the app no longer uses.
* `mu` is exactly `ln(avg) - sigma²/2`, but `sigma = 0.7587135646925732` cannot be
  derived from min, max and average by any obvious formula (the closest relation
  found is `ln(max/min) / 3.035 = 0.75868`). Where it came from is unknown.
* It is the only place the 57 / 61 / 52 / 56 / 54 quintile counts of the original 280
  are written down.

**What would change if it were altered.** Nothing runs differently either way — no
code path reads it. The question is whether a piece of documented provenance was
thrown away.

**The question.** Are those reference quintiles still meaningful, and should they be
restored (as documentation, or as the reference the income mapping should be checked
against)? If yes, how was `sigma` obtained?

---

## 4. Vendor price bounds in the engine (ruling R8) — RESOLVED 2026-09-04

**Resolved by the owner; no longer open.** The owner confirmed that R8 covers the
engine's vendor choice, not only the reported scores: the engine must normalise
vendor price with the Page-1 min/max bounds exactly as the reports do, the Page-1
"Average Price per Vendor" (`sim_params.market_price`) is the reference price
wherever "the price" is needed, and the bounds only shape the random vendor prices.
Everything else stays identical.

**What was wrong.** `_calculate_preferred_vendor` in
`src/decisions/purchasing_quantity.py` read `vendor_price_min` / `vendor_price_max`
as **top-level** keys of `simulation_config`, which nothing sets, so it always
normalised on the hard-coded `[50, 150]`. The Page-1 values live one level down, in
`simulation_config['simulation']`, which every other decision reads through
`get_simulation_param()` (`src/decisions/income_utils.py`) and which
`src/engine/vendors.py` draws the vendor prices from.

**What was changed.** The two reads now go through
`get_simulation_param(simulation_config, 'vendor_price_min', 50.0)` and
`('vendor_price_max', 150.0)`; the 50 / 150 fallbacks remain for runs without Page-1
parameters. Nothing else in `src/` scored vendors or read a reference price from the
wrong place: `src/engine/vendors.py` already reads the bounds and `market_price`
from `['simulation']`, `src/decisions/bid_value.py` already takes `market_price`
through `get_simulation_param()` as the fallback reference price, and
`src/decisions/vendor_selection.py` only echoes the Decision-6 choice.

**Effect.** With the default single vendor (Page-1 bounds both equal to the Average
Price per Vendor, 100) the engine's normalised price moves from
`1 - (100-50)/100 = 0.5` to `1.0` — the value `calculate_vendor_score_with_breakdown`
returns whenever `max <= min` — matching the reports. With one vendor the choice is
unchanged, so no journey number moved. Runs with several vendors and non-default
bounds now rank vendors on the same scale the results page reports. Register row
Q-10 and `acceptance-report.md` §5 record the change.

---

## 5. Two different standard-deviation formulas — RESOLVED 2026-09-04 (R-SD)

**Resolved by the owner; no longer open.** The income standard deviation used for the
continuous-income z-scores of Decisions 1 and 2 must divide by N−1, like Stata's
`egen std()` — the formula every z-score in `stata/*.do` uses.

**What was wrong.** The shared population income statistics at the end of Pass 1
(`src/engine/core.py`) were computed with the *population* formula,
`np.std(all_incomes)` (`ddof=0`), while the composite statistics of Decisions 1 and 2
and Decision 4's own income statistics already used the *sample* formula
(`ddof=1`). Decision 4 computed its own income SD precisely because the shared one
was `ddof=0`.

**What was changed.** One line: the shared `income_stats['sd']` is now
`np.std(all_incomes, ddof=1)` (falling back to the population formula only when
fewer than two incomes exist, the same guard `compute_rtd_population_stats` uses).
Decision 4 (`src/decisions/rejected_transaction_defaults.py`) was left untouched — it
already used N−1 — and the Decision 1 / Decision 2 composite statistics were already
`ddof=1`.

**Proof on the professor's data.** His corrected Decision 2 `.dta` carries
`z_net_income` and `z_picont`; both equal `±(income − mean) / sd` with the
*sample* SD to 3 × 10⁻⁷, so his Stata standardised income with N−1. On the 280
participants the change moves every continuous score by at most 1.6 × 10⁻³ while the
closest score to the threshold is 0.0116 away, so the professor's continuous table
(63 / 280 = 22.50 %, every per-allowance-level cell) and his `disclosedoc_cont`
column (280 / 280) are reproduced under **both** formulas — the old value was not
wrong on the 280, only inconsistent.

**Effect on the app.** Only continuous-income runs in which the Decision 1 / Decision 2
*model* runs can change a decision, and only for an agent within about 0.2 % (n = 280)
or 1 % (n = 50) of its threshold. In the 17 reference journeys nothing flipped: the
three continuous complete runs (J2, J4, J6 — Decision 2 as a default) moved only the
five `disclose_documents_*` analytic columns in the third decimal, the "DD Raw Values"
histogram and the Agent Disclose Documents export; every categorical journey, every
Decision 4 column and every donation column is identical. A dedicated Research
Specification × continuous run with both decisions selected (n = 280, seed 42) flipped
0 of 280 `disclose_income` and 0 of 280 `disclose_documents` values. Register row
Q-31 and `acceptance-report.md` §10 record the change.

---

## 6. `disclose_income.stochastic.scale_factor: 0.1` — RESOLVED 2026-09-04 (R-DI01)

**Resolved by the owner; no longer open.** The value `0.1` in `config/decisions.yaml`
is **correct and intended**: it is the Disclose Income tab's "σ Coefficient
(multiplier)" default of 0.10, so the Research Specification draw is
`Normal(anchored_pb, sigma_overall × 0.10)` by design. Nothing was changed. Register
row Q-59 records the ruling; the `shift_value: -4.0` half of Q-14 is unchanged and
was never part of this question.

---

## 7. Period 2 is merged by row position, not by participant ID — RESOLVED 2026-09-04 (R-P2)

**Resolved by the owner; no longer open.** The row-position merge in
`src/build_dd_sigma.py` (`reconstruct_consumed_2periods`: Period 2's 269 rows laid over
the first 269 master rows, the 11 participants absent from Period 2 contributing 0) is
**exactly the professor's Stata procedure** and is correct. The Decision 2 sigma
constants derived from it (`sigma_overall = 0.1606568355` and the five per-level values
in `config/decisions.yaml`) stand. Nothing was changed. Register row Q-60 records the
ruling.

---

## Data-file note for the professor (not a question)

In `Student Experiment Results - Period 2.xlsx` the `Participant ID` column is offset
relative to its own data rows: joining the two periods **by ID** mismatches 89 of the
280 participants, while the row-position merge reproduces the validated `.dta`
280 / 280. Since the positional merge is the intended procedure this changes nothing
in the model — it is only worth knowing that the ID column of that export file is
shifted, in case the file is ever re-exported or joined by ID elsewhere.
