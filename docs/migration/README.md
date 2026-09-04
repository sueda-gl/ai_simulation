# The logic / UI split — reader's guide

This folder explains one change to the COOPECON simulation: **the simulation logic
was separated from the Streamlit screens.** Nothing about the model itself changed.
The screens look the same, and for the same settings and the same seed the numbers
are the same as before.

If you have never opened this project before, read the top-level
[`README.md`](../../README.md) first (how to install and start the app), then come
back here.

## Why the split was done

Before, the code that decided *what to simulate* and the code that *drew the screen*
were the same code. Three consequences:

* the model could only be run from inside a running Streamlit app — you could not
  call it from a script, a notebook or a test without a browser session;
* a decision module could reach back into a page module for a default value, so the
  same run gave different numbers depending on which tab you had opened;
* the same quantity (a sigma, a price, an income mode) was written in several places
  and the copies had drifted apart.

After the split there is one engine, one place where a run is described, and one
place where the app hands that description to the engine.

## The five documents

| File | Read it when you want to know |
|---|---|
| [`architecture.md`](architecture.md) | how the code is laid out now, what happens when you press a Run button, how the random numbers are produced, and how to run the engine without the UI |
| [`rulings-and-quirks.md`](rulings-and-quirks.md) | every odd behaviour that was found in the old code, what the owner decided about it, and where it lives now |
| [`open-questions-for-professor.md`](open-questions-for-professor.md) | the handful of modelling questions that were deliberately **not** decided by the migration and still need the professor's judgement |
| [`how-to-add-a-decision.md`](how-to-add-a-decision.md) | the step-by-step recipe for adding decision number 14 |
| [`session-state.md`](session-state.md) | every session-state key the screens use, which of the four layers it belongs to, which keys reach the engine, and the registry test that keeps the list honest |

## The rules the migration worked under

1. **Screens identical.** Every page renders exactly the same elements as before.
2. **Numbers identical.** For the same settings and seed, every result column and
   every exported file is unchanged, except where an owner ruling deliberately
   changed it (those are the rows marked *fixed* in
   [`rulings-and-quirks.md`](rulings-and-quirks.md)).
3. **The engine never imports Streamlit.** Enforced by a test, not by convention —
   see "The import boundary" in [`architecture.md`](architecture.md).
4. **The model was not re-derived.** No coefficient, no sigma, no configuration file
   value was recomputed. `config/trait_model.pkl` was never retrained.

## Known gap

None open. When this document was first written one owner ruling — **R8**, register
row Q-10 — was only half in place: the results page and the exports normalised vendor
price with the Page-1 bounds, but `_calculate_preferred_vendor` in
`src/decisions/purchasing_quantity.py` read `vendor_price_min` / `vendor_price_max`
as top-level keys nothing sets and so always fell back to the hard-coded `50.0 /
150.0`. That was closed on 2026-09-04: the engine now reads both bounds from
`simulation_config['simulation']` through `get_simulation_param()` in
`src/decisions/income_utils.py`, exactly as `src/engine/vendors.py` and the reports
do, with the Page-1 "Average Price per Vendor" as the reference price wherever one
is needed. See Q-10 in the register, item 4 of
[`open-questions-for-professor.md`](open-questions-for-professor.md) (resolved) and
§5 of [`acceptance-report.md`](acceptance-report.md).

On the same day the owner ruled on three of the questions that had been referred to
the professor: the shared income standard deviation now divides by N−1 like Stata's
`egen std()` (R-SD, Q-31, one line in `src/engine/core.py`; no decision in the
reference journeys flipped), the Decision 1 `scale_factor: 0.1` is correct as it stands
(R-DI01, Q-59), and the Period-2 row-position merge behind the Decision 2 sigma
constants is the professor's own procedure (R-P2, Q-60). See items 5–7 of
[`open-questions-for-professor.md`](open-questions-for-professor.md) and §10 of
[`acceptance-report.md`](acceptance-report.md).
