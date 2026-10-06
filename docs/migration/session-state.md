# Session state — the four layers, and the registry that pins them down

Streamlit keeps everything the user typed in one dictionary, `st.session_state`.
It is the app's only memory: close the browser tab and it is gone; press a button
and the whole script runs again from the top with that dictionary still in place.

Before the split, that dictionary was also the *interface to the model* — the
simulation reached into it while it ran. It no longer does (see
[`architecture.md`](architecture.md) §2, "the seam"). But the screens still use it
heavily, the same value still lives in up to four places at once, and nothing in
the code said which places those were. This page says it, and
[`app/state/registry.py`](../../app/state/registry.py) says it in a form a test can
check.

---

## 1. The registry

`app/state/registry.py` holds one `KeySpec` per session-state key, or per key
*family* when the name is built with an f-string:

```python
KeySpec(name="di_sigma_enabled",
        default=True,
        initialised_by="decision_tabs.disclose_income."
                       "initialize_disclose_income_session_state",
        category="page2-mirror",
        engine_input=True,
        notes="R18: the tab default is ON, so an absent key must read True too")
```

| Field | Meaning |
|---|---|
| `name` | the literal key, or a pattern such as `{decision_name}_default_value` |
| `default` | the value the initialiser stores. `WIDGET` means Streamlit owns it and nothing seeds it; `"<expr>"` in angle brackets means the value is copied from `expr` when the key is first created |
| `initialised_by` | which function creates the key |
| `category` | which of the layers below it belongs to |
| `engine_input` | the value reaches the model, through the seam snapshot |
| `is_widget` | some `st.<widget>(…, key=…)` binds it |
| `write_only` | nothing in `app/` reads it back (the seam may still read it — §5) |

It is plain data. It imports nothing from the app and never touches Streamlit, so
you can read it, or import it from a script, without starting a session.

**`tests/test_session_key_registry.py` keeps it honest.** The test parses every
module under `app/` with Python's own `ast` module and collects every way a key can
be named — `st.session_state.x`, `st.session_state["x"]`, `.get`, `.pop`,
`.setdefault`, `"x" in st.session_state`, `getattr`/`hasattr`/`setattr`/`delattr`,
and every `key=` handed to a Streamlit call. Then it checks both directions:

* every key the code touches has a registry entry — you cannot add a key quietly;
* every registry entry is touched by the code — a key that leaves the app has to
  leave the registry, so the list cannot rot into fiction.

It also checks that the `is_widget` and `write_only` flags still match reality, that
every `engine_input` key is one the seam actually names, and that no key is covered
by two equally specific family patterns. The test never imports the app, so it stays
green without Streamlit installed and cannot be fooled by an import-time side effect.

---

## 2. The four layers

Take one setting — the Disclose Income σ tick box — and follow it.

```
  ┌─ 1. widget key ──────────┐   di_tab_sigma_enabled      Streamlit owns it
  │        on_change ↓       │
  ├─ 3. tab persistence ─────┤   disclose_income_tab_persistence['sigma_enabled']
  │        restore ↑         │
  ├─ 2. canonical key ───────┤   di_sigma_enabled          the seam reads this
  └──────────────────────────┘
     4. _persistent_defaults      (default-decision parameters only)
```

### Layer 1 — the widget key

The string handed to a Streamlit widget as `key=`. Streamlit owns the value: it
writes the user's input into it before the script reruns, and it refuses to let you
change it after the widget has been created in the same run. **A widget key only
survives while its widget keeps rendering.** Switch to another tab and the widget
stops rendering; Streamlit is then free to drop the key.

**The browser has its own copy.** Streamlit sends a key's value to the browser only
in the run in which the key is *written* through `st.session_state`. A widget drawn
without `value=` / `index=` whose key was seeded in an *earlier* run reaches a
browser that has not drawn it before (fresh session, a decision tab just selected,
a page navigated back to, a changed label or range) with only its built-in default
— min / `False` / option 0 — and the browser sends that back on the next rerun. So
every keyed widget whose value lives in session state is drawn through
`stateful(st.<widget>, …, key=…)` from `app/state/widgets.py`: it re-writes the key
(unchanged) whenever the widget was not drawn in the previous run with the same
arguments, using a per-run render log (`_widget_render_log`, rotated by
`begin_script_run()` at the top of `app_enhanced_new.py`). `tests/browser_sim.py`
models the browser for AppTest; `tests/test_widget_browser_state.py` uses it, and
also checks statically that no value-less keyed widget bypasses `stateful()`.

Registry categories: `page1-widget`, `page2-widget`. Names follow a convention —
Page 1 uses `*_input` / `*_slider` / `page1_*`, the decision tabs use a `*_tab_*`
infix (`tab_sigma_in_copula`, `di_tab_income_mode`, `rtd_tab_intercept_{mech}`).

### Layer 2 — the canonical key

The value the rest of the app reads, and the only layer the seam looks at. It is
written unconditionally right after the widget call, on every render, so the two
cannot drift apart while the tab is open:

```python
sigma_enabled = st.checkbox("Use Normal(anchor, σ) draw in Research Specification mode",
                            value=res_val, key="di_tab_sigma_enabled", ...)
st.session_state.di_sigma_enabled = sigma_enabled
```

Registry categories: `page1-mirror`, `page2-mirror`. Page 1's canonical layer is one
object rather than one key per value — `sim_params`, a `SimulationParameters`
instance whose *attributes* the widgets mirror.

Because layer 2 is written only while the tab renders, an **unopened tab has no
layer-2 value at all**, and the seam falls back to the default recorded in the
registry. That is why several defaults carry a ruling number: `di_sigma_enabled`,
`dd_sigma_enabled` and `rtd_sigma_enabled` default to `True` (R18) precisely so that
an unopened tab runs the same σ the tab would have shown.

### Layer 3 — the tab persistence dict

One dict per decision tab:

| Dict | Tab |
|---|---|
| `donation_tab_persistence` | Donation Default (decision 3) |
| `disclose_income_tab_persistence` | Disclose Income (decision 1) |
| `disclose_documents_tab_persistence` | Disclose Documents (decision 2) |
| `rejected_transaction_tab_persistence` | Rejected Transaction Defaults (decision 4) |

It exists because of layer 1's weakness. It is written **only** from an `on_change`
callback (`save_to_donation_storage`, `save_to_disclose_income_storage`,
`save_to_dd_storage`, `save_to_rtd_storage`) — that is, only when the user actually
changes something — and it is read **before** the widget renders, by each tab's
`restore_widget_from_storage`, which restores in this order:

1. the entry in the persistence dict, if there is one;
2. otherwise the widget key, if it is still there;
3. otherwise the hard default.

So a value the user typed survives leaving the tab and coming back, even though the
widget key did not.

Registry category: `tab-persistence`.

### Layer 4 — `_persistent_defaults`

A shadow store for one specific family: the **default-decision parameters** the
Overview tab configures for decisions the user did *not* select
(`{decision_name}_default_probability_y`, `_default_params`, `_default_selection`,
`_priority_template`, `_default_value`).

Those widgets live on the Overview tab, which is often never opened. Their session
keys are also re-seeded from `DEFAULT_DECISION_VALUES` on every Page-2 render, which
would overwrite a user's choice. `_persistent_defaults` sidesteps both problems:
`default_config.save_to_persistent_storage(key)` copies the value in, and the seam
consults it **before** the session key. The full order for one default decision is

1. the post-simulation override (`{decision_name}_probability_y` and friends — §6);
2. `_persistent_defaults[{decision_name}_default_probability_y]`;
3. `st.session_state[{decision_name}_default_probability_y]`;
4. the hard-coded value in `DEFAULT_DECISION_VALUES`.

Registry category: `defaults`.

---

## 3. The other categories

The registry has ten categories; six are the layers above. The rest describe keys
that are not settings at all:

| Category | What it holds | Examples |
|---|---|---|
| `navigation` | which screen renders | `page` |
| `saved-config` | the "Use This Config" pins | `selected_decision_configs`, `custom_coefficients` |
| `run-flag` | what the click that started a run recorded | `custom_decisions`, `default_decisions`, `_pending_decisions_restore` |
| `results` | run output, and the chart / download widgets that show it | `simulation_results`, `mc_results`, `vendors`, `_run_metadata` |

`selected_decision_configs` deserves its own note: one entry per decision the user
pinned, each holding the seed, the agent count, the population mode, that decision's
parameters, and the sha256 of the columns the pinned run produced (R14). A later
complete simulation re-checks that hash, so "reproduce what I looked at" is verified
rather than assumed.

---

## 4. Which keys the seam reads

The seam takes **one** snapshot of the session state per click
(`take_snapshot(st.session_state)`, ruling R15) and hands that frozen copy to
`app/seam/build_plan.py`, which reads it and writes nothing. Nothing later in the
run can observe a value another part of the run wrote.

These are the entries marked `engine_input=True`. They are the complete list of
session-state values that can change a number.

| Group | Keys |
|---|---|
| Run size and seed | `sim_params`, `n_agents`, `seed` / `seed_input`, `base_seed` / `base_seed_input` |
| Population and income | `population_mode`, `income_spec_mode` |
| Which decisions run | `decision_params`, `custom_decisions`, `default_decisions` |
| Donation (3) | `sigma_in_copula`, `sigma_in_research`, `sigma_coefficient`, `anchor_observed_weight`, `donation_sigma_strategy`, `donation_quintile_scale_factors`, `donation_adjustment_shift`, `donation_coeff_{name}_{cat\|cont}` |
| Disclose Income (1) | `di_intercept`, `di_wopb`, `di_wpb`, `di_income_mode`, `di_sigma_enabled`, `di_sigma_in_copula`, `di_sigma_strategy`, `di_scale_factor`, `di_quintile_scale_factors` |
| Disclose Documents (2) | `dd_intercept`, `dd_income_mode`, `dd_sigma_enabled`, `dd_sigma_in_copula`, `dd_sigma_strategy`, `dd_scale_factor`, `dd_quintile_scale_factors` |
| Rejected Transaction Defaults (4) | `rtd_income_mode`, `rtd_intercept_{mech}` (five elements), `rtd_flex_observed_weight`, `rtd_aggregation_enabled`, `rtd_sigma_enabled`, `rtd_sigma_in_copula`, `rtd_sigma_strategy`, `rtd_scale_factor`, `rtd_quintile_scale_factors` |
| Default decisions | `_persistent_defaults`, `{decision_name}_default_probability_y`, `_default_params`, `_default_selection`, `_priority_template`, `_default_value`, `final_donation_rate_default_value` |
| Saved configs | `selected_decision_configs` |

Two things are easy to get wrong here.

* **The donation coefficients the engine runs are the suffixed ones.** The Donation
  tab shows and edits the flat `donation_coeff_*` keys, but `build_plan` reads
  `donation_coeff_{name}_cat` / `_cont` (ruling R10). The flat set is display state.
* **`sigma_value_ui` is not read.** It is the product of the σ constant and the
  coefficient slider, kept for the screen. R12 says there is one σ constant, and the
  seam recomputes the product itself from `sigma_coefficient`.

One more path exists outside the seam: `n_runs`, `n_agents`, `base_seed`,
`anchor_observed_weight`, `population_mode` and the selected decisions are *also*
passed to the Monte-Carlo subprocess as command-line arguments by
`app/simulation.py`. Of these only `n_runs` is not a seam input (the seam reads
`base_seed` for the Monte-Carlo seed and `anchor_observed_weight` for the donation
anchor weight). Since 2026-10-07 the Monte-Carlo runner also builds the ordinary run
plan from the same snapshot and hands its sub-run to the subprocess as a plan file
(`app/seam/mc.py`), so every tab setting reaches the Monte-Carlo runs exactly as it
reaches a single run (Q-29 in [`rulings-and-quirks.md`](rulings-and-quirks.md)).
A selected ("Use This Config") saved configuration pins the Monte-Carlo agent count
and population exactly as it pins a complete run's (the subprocess gets the plan's
`n_agents`, not Page 1's); only the seed still varies (`base_seed + i`), and the
screen says so ("🔑 Using saved config: agents N, population P; seeds vary per run").

---

## 5. Written but never read

The scan found four keys that something writes and nothing ever reads back. They are
**left in place** — none of them costs anything, and removing state is exactly the
kind of change that turns into a numbers difference. They are listed here so the
owner can decide.

| Key | Written by | Why it looks dead |
|---|---|---|
| `save_results` | `models.initialize_session_state` defaults dict | the "save results to disk" feature was disabled in the UI; the flag was kept for backward compatibility and nothing consults it |
| `simulation_running` | same | a spinner flag from an older version of the run button |
| `individual_results` | same | seeded to `{}`; the per-decision results now live in `simulation_results` |
| `_default_params_initialized` | `models.initialize_default_decision_parameters` | a debug marker; the comment above it already says the logic no longer branches on it |

One more key is written and never read **inside `app/`** — but it is not a
candidate, because the seam reads it from the snapshot: `donation_adjustment_shift`.
The registry marks it `write_only=True, engine_input=True`, which is exactly that
situation. (`rtd_sigma_enabled` / `rtd_sigma_in_copula` used to be in the same
position; since the September 2026 Decision 4 "Use This Config" port the saved-config
store reads them too.)

One family is stale in the other direction. `clear_input_field_cache()` in the
Donation tab deletes sixteen `donation_coeff_*_input` keys when the income mode
changes, but no widget binds those keys any more — the coefficient number inputs
were removed — so the list only ever finds them absent. Harmless, and also a removal
candidate.

## 6. Read but never written

The mirror image: the seam looks for a value the app never writes, so the lookup
always falls through to the next priority. Again, nothing was changed.

| Key | Looked up by | Falls through to |
|---|---|---|
| `{decision_name}_probability_y` | `build_plan._collect_decision_settings`, priority 1 | `_persistent_defaults`, then the session key, then `DEFAULT_DECISION_VALUES` |
| `{decision_name}_selection` (concretely `vendor_choice_weights_selection`) | same, priority 1 | same |
| `rejected_transaction_defaults_option`, `rejected_transaction_option_selection` | same, priority 1 for the two radio decisions, and `decision_execution.get_actual_default_value` | same |
| `{decision_name}_config` | `decision_execution.get_actual_default_value`, priority 2 | the hard-coded default |
| `donation_coeff_q45` | the legacy Q4+Q5 fallback, read when `donation_coeff_q5` is missing | `0.0` |

These are the remains of a "adjust the default on the results page and re-run"
feature that no screen offers any more. The priority chains still have a slot for it.

---

## 7. Bulk operations

Three places delete keys wholesale. They matter because they are the only way a key
disappears, and the registry's initialisers are what put it back.

| Operation | What it deletes | Then |
|---|---|---|
| A decision tab's **Reset to research defaults** | every key starting with `di_`, `dd_` or `rtd_` (whichever tab), plus that tab's persistence dict. Disclose Income / Documents only raise `_di_/_dd_reset_to_defaults_pending` on the click and do the deletion at the top of the next run, before any of the tab's widgets exists (September 2026) | calls the tab's `initialize_*_session_state()` (DI/DD: re-seeds the widget keys from the defaults) |
| Decision 4's per-element reset | only `rtd_intercept_{mech}` and its widget key (for Flexibility also `rtd_flex_observed_weight` and its slider key), plus those entries in the persistence dict | same; σ is decision-wide and deliberately untouched |
| **Clear Results** on the results page | *every* key in `st.session_state` | calls `models.initialize_session_state()` and stays on the results page |

The prefix list lives in the registry as `PREFIX_DELETE_PREFIXES` and the test
checks it still matches the code.

---

## 8. Keys named at run time

A handful of accesses name their key through a variable the scanner cannot resolve —
a helper's parameter, or a loop over `st.session_state.keys()`. They are reviewed by
hand and listed in `DYNAMIC_KEY_VARIABLES`, so that a *new* unresolvable access fails
the test rather than slipping past it:

* `key` — the parameter of `save_to_persistent_storage` / `restore_from_persistent_storage`,
  and the loop variable of the prefix resets and the Clear Results wipe;
* `k` — the per-category purchasing-limit key on Page 1, bound in a lambda default;
* `chart_key` — the plotly key the shared Decision-4 chart helpers receive.

---

## 9. Adding a key

1. Give it a name that says which layer it belongs to (`*_tab_*` for a widget key on
   a decision tab, no infix for the canonical value).
2. Write it next to the widget, every render, if it is a canonical key; add the
   `on_change` save into the tab's persistence dict if the tab can be left and
   re-entered.
3. Add a `KeySpec` to `app/state/registry.py` — category, default, who initialises
   it, and whether the seam reads it.
4. Run `pytest tests/test_session_key_registry.py`. If the key is a new engine input,
   the test will also insist the seam actually names it.
