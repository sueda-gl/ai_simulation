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

One owner ruling (**R8**, register row Q-10) is only half in place. The ruling says
vendor scoring should normalise vendor price with the Page-1 price bounds. The
results page and the exports do that. The engine does not:
`_calculate_preferred_vendor` in `src/decisions/purchasing_quantity.py` reads
`simulation_config['vendor_price_min' / 'vendor_price_max']` — top-level keys that
nothing sets — so it always falls back to the hard-coded `50.0 / 150.0`. The Page-1
bounds live one level down, in `simulation_config['simulation']`, and are read
everywhere else through `get_simulation_param()` in
`src/decisions/income_utils.py`. See Q-10 in the register and
[`open-questions-for-professor.md`](open-questions-for-professor.md) for what changes
if it is corrected.
