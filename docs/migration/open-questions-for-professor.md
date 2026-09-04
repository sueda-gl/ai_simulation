# Open questions for the professor

The migration split the logic from the screens. It did **not** re-derive the model.
Along the way seven things turned up that look like modelling decisions rather than
programming decisions, so they were left exactly as they were and written down here.

Each item says three things: **what the code does today**, **why it looks
suspicious**, and **what would change if it were altered**. None of them is a bug the
migration introduced; every one of them predates it.

---

## 1. Two different standard-deviation formulas

**What the code does today.** The population income statistics are computed with the
*population* formula:

```python
# src/engine/core.py, end of Pass 1
self.simulation_config['income_stats'] = {
    'mean': float(np.mean(all_incomes)),
    'sd':   float(np.std(all_incomes)),      # ddof = 0
}
```

Those statistics are what Decisions 1 and 2 use to standardise income in
*continuous* income mode (`src/decisions/disclose_income_stochastic.py`,
`src/decisions/disclose_documents_stochastic.py`).

Everywhere else the *sample* formula is used, with a comment saying it matches
Stata's `egen std()`:

* `src/decisions/disclose_income_stochastic.py` — the continuous DE composite
  statistics, `np.std(all_de, ddof=1)`;
* `src/decisions/disclose_documents_stochastic.py` — the continuous DD composite
  statistics, `np.std(vals, ddof=1)`;
* `src/decisions/rejected_transaction_defaults.py` — Decision 4's own income
  statistics and every mechanism-score statistic, `.std(ddof=1)`;
* `src/build_dd_sigma.py` — the Decision 2 sigma constants.

**Why it looks suspicious.** These are the same operation on the same 280
participants, and Stata's `egen std()` is the sample formula in every one of them.
Decision 4 goes as far as computing its **own** income statistics with `ddof=1`
rather than reusing the shared ones, which is only necessary because the shared ones
use `ddof=0`.

**What would change if it were altered.** Switching the shared income statistics to
`ddof=1` divides every continuous-mode income z-score by `sqrt(n/(n-1))`:
about 1.0018 at n = 280 and about 1.0102 at n = 50 — so z-scores shrink by roughly
0.2 % and 1 % respectively. That moves every Decision 1 and Decision 2 continuous
result slightly, and flips the agents that sit within that margin of their
threshold. Decision 4 would be unaffected (it already uses `ddof=1`), and
categorical mode would be unaffected (it uses fixed statistics from the file).

**The question.** Should the shared income standard deviation be the sample
standard deviation, like every other standard deviation in the model?

---

## 2. `disclose_income.stochastic.scale_factor: 0.1`

**What the code does today.** In `config/decisions.yaml`, Decision 1 carries

```yaml
disclose_income:
  stochastic:
    sigma_overall: 9.899547
    scale_factor: 0.1
```

and the code multiplies them:

```python
# src/decisions/disclose_income_stochastic.py
sigma_raw    = stochastic_params.get('sigma_overall', 9.899547)
scale_factor = stochastic_params.get('scale_factor', 0.1)
sigma_scaled = sigma_raw * float(scale_factor)
stochastic_anchored_pb = rng.normal(anchored_pb, sigma_scaled)
```

so the actual noise is `Normal(anchored_pb, 0.9899547)`.

**Why it looks suspicious.** `9.899547` is the standard deviation of **TWT+Sospeso**
in its own units — the same file records `TWT_Sospeso: {mean: 3.357143, sd: 9.899547}`.
But the variable it is added to, `anchored_pb`, is a weighted average of two
z-scores (`0.25 × z_obs_PB + 0.75 × z_weighted_prosocial`), whose population standard
deviation the same file records as `0.7984211971`. A raw-units standard deviation is
being used as the width of a draw on a z-scale variable, and `0.1` is the only thing
bridging the two. Nothing in the code or the configuration says where `0.1` comes
from, and no other decision has a scale factor other than `1.0`.

The comparison with Decision 2 makes the asymmetry visible. Decision 2 perturbs
`dd_deterministic = beta0 + z_weighted_dd` (standard deviation 1 by construction)
with `sigma_overall = 0.1606568355 × scale_factor 1.0`. So, relative to the variable
being perturbed:

| | width of the draw | standard deviation of the variable | ratio |
|---|---|---|---|
| Decision 1 | 0.98995 | 0.79842 | **1.24** |
| Decision 2 | 0.16066 | 1.0 | **0.16** |

Decision 1 injects roughly eight times as much noise, relative to its own variable,
as Decision 2 does.

**What would change if it were altered.** Setting `scale_factor: 1.0` multiplies
Decision 1's noise by ten — a draw of width 9.9 around a variable whose spread is
0.8, which would make the disclosure outcome almost pure noise. Setting it so the
two decisions match in relative terms (about `0.013`) would make Decision 1 nearly
deterministic. Either way, only Research Specification runs and Copula runs with the
Decision 1 σ tick box on are affected; Research Baseline draws no noise at all.

**The question.** Is `0.1` a deliberate modelling choice ("use one tenth of the
observed spread"), or a leftover from tuning? And is `sigma_overall` for Decision 1
meant to be the standard deviation of TWT+Sospeso in raw units, or of the z-scored
anchor?

---

## 3. Three customer-price conventions in the exports

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

## 4. Three frozen Decision 1 reference files that disagree

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

## 5. Period 2 is merged by row position, not by participant ID

**What the code does today.** The Decision 2 sigma constants come from
`consumedtransferssospeso2periods`, the two-period count of sospeso and transfers
each participant consumed. `src/build_dd_sigma.py` reconstructs it:

```python
def reconstruct_consumed_2periods(period1_df, period2_df):
    c1 = _consumed(period1_df)          # 280 rows, the master order
    c2_vals = _consumed(period2_df)     # 269 rows
    c2 = np.zeros(len(c1), dtype=int)
    c2[: len(c2_vals)] = c2_vals        # aligned by POSITION
    return c1 + c2
```

Period 2 has 269 rows and is laid over the first 269 master rows; the 11 participants
absent from Period 2 contribute 0.

**Why it looks suspicious.** This is not a join. The module docstring explains why:
the `Participant ID` column in the Period 2 export is offset relative to its own data
rows, so joining the two periods **by ID** produces 89 of 280 per-participant
mismatches. The grand total (351) is the same either way — the same 269 values are
simply attributed to adjacent participants. The positional merge was chosen because
it reproduces the professor's validated `.dta` exactly (280 of 280), which is strong
evidence that the original Stata merge was also positional.

**What would change if it were altered.** The per-participant values change for 89
participants, so the derived sigmas change: the overall σ (`0.1606568355`) and all
five per-level σ values in `config/decisions.yaml`. Every Decision 2 stochastic draw
in Research Specification mode moves. The deterministic Decision 2 score does not
change — it does not use these constants.

**The question.** Was the original merge positional on purpose, or is the offset in
the Period 2 export a data-export defect that should be repaired at the source? If it
is a defect, the sigma constants need re-deriving.

---

## 6. `config/income_reference_breaks.json` was deleted as dead

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

## 7. Vendor price bounds in the engine (ruling R8, only half applied)

**What the code does today.** Ruling **R8** says vendor scoring should normalise
vendor price with the Page-1 price bounds. The results page and the exports do that
— `app/pages/results/components/export_section.py` reads
`sim_params.vendor_price_min / vendor_price_max` and passes the values into
`app/reports/agent_level.py`, `transaction_level.py` and `vendor.py`.

The engine does not:

```python
# src/decisions/purchasing_quantity.py, _calculate_preferred_vendor
price_min_config = simulation_config.get('vendor_price_min', 50.0)
price_max_config = simulation_config.get('vendor_price_max', 150.0)
```

Those are **top-level** keys of `simulation_config`, and nothing sets them. The
Page-1 values live one level down, in `simulation_config['simulation']`, which every
other decision reads through `get_simulation_param()`
(`src/decisions/income_utils.py`). So the engine always normalises on the hard-coded
`[50, 150]`, whatever the user typed on Page 1.

**Why it looks suspicious.** The engine picks the preferred vendor with one price
scale and the results page reports that vendor's score with another. With the default
single vendor (Page-1 bounds both equal to the Average Price per Vendor, 100) the
engine computes a normalised price of `1 - (100-50)/100 = 0.5` while the reports
compute `1.0`.

**What would change if it were altered.** Changing the two lines to
`get_simulation_param(simulation_config, 'vendor_price_min', 50.0)` (and `_max`)
changes `preferred_vendor` — and therefore `vendorID` on every purchase request, and
therefore vendor-dependent prices and bids — in **every run whose Page-1 bounds are
not exactly 50 and 150**. That is every default single-vendor run. It is the largest
single numerical change of any item on this page, which is why it has been reported
rather than applied.

**The question.** Confirm that R8 was meant to cover the engine's vendor choice, and
not only the reported scores.
