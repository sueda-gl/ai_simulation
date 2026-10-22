# Session state — the three layers, and the registry that pins them down

Streamlit keeps everything the user typed in one dictionary, `st.session_state`.
It is the app's only memory: close the browser tab and it is gone; press a button
and the whole script runs again from the top with that dictionary still in place.

The simulation itself never reads that dictionary while it runs: the seam takes one
snapshot per click and works from the copy (see [`README.md`](README.md) §2). But the
screens use it heavily, the same value lives in up to three places at once, and
nothing in the code would otherwise say which places those are. This page says it,
and [`app/state/registry.py`](../app/state/registry.py) says it in a form a test can
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
        notes="the tab default is ON, so an absent key must read True too")
```

| Field | Meaning |
|---|---|
| `name` | the literal key, or a pattern such as `{decision_name}_default_value` |
| `default` | the value the initialiser stores. `WIDGET` means Streamlit owns it and nothing seeds it; `"<expr>"` in angle brackets means the value is copied from `expr` when the key is first created |
| `initialised_by` | which function creates the key |
| `category` | which of the layers below it belongs to |
| `engine_input` | the value reaches the model, through the seam snapshot |
| `is_widget` | some `st.<widget>(…, key=…)` binds it |
| `write_only` | nothing in `app/` reads it back (the seam may still read it, §5) |

It is plain data. It imports nothing from the app and never touches Streamlit, so
you can read it, or import it from a script, without starting a session.

**`tests/test_session_key_registry.py` keeps it honest.** The test parses every
module under `app/` with Python's own `ast` module and collects every way a key can
be named: `st.session_state.x`, `st.session_state["x"]`, `.get`, `.pop`,
`.setdefault`, `"x" in st.session_state`, `getattr`/`hasattr`/`setattr`/`delattr`,
and every `key=` handed to a Streamlit call or to `stateful()`. A key held in a
variable counts when the variable is bound to a literal or an f-string, or loops (a
`for` statement or a comprehension) over a literal list of them. In the two modules
that draw the same widgets for several tabs (§8) a key is an f-string over a
per-tab row of the registry's own table, and the test expands it to one concrete
key per tab. Then it checks both directions:

* every key the code touches has a registry entry, so you cannot add a key quietly;
* every registry entry is touched by the code, so a key that leaves the app has to
  leave the registry and the list cannot rot into fiction. The seam reads a
  snapshot rather than `st.session_state`, so an `engine_input` entry the seam
  names counts as touched (the read-only fallbacks of §6).

It also checks that the `is_widget` and `write_only` flags still match reality, that
every `engine_input` key is one the seam actually names, and that no key is covered
by two equally specific family patterns. The test imports only the registry (and
`conftest.py` the Streamlit-free seam), never a screen, so it stays green without
Streamlit installed and cannot be fooled by an import-time side effect.

---

## 2. The three layers

Take one setting, the Disclose Income σ tick box, and follow it.

```
  ┌─ 1. widget key ──────────┐   di_tab_sigma_enabled      Streamlit owns it
  │   written after the      │
  │   widget, every render ↓ │
  │   restore (when absent)↑ │
  ├─ 2. canonical key ───────┤   di_sigma_enabled          the seam reads this
  └──────────────────────────┘
     3. _persistent_defaults      (default-decision parameters only)
```

### Layer 1 — the widget key

The string handed to a Streamlit widget as `key=`. Streamlit owns the value: it
writes the user's input into it before the script reruns, and it refuses to let you
change it after the widget has been created in the same run. **A widget key only
survives while its widget keeps rendering.** Every selected decision's tab is drawn
in every Page-2 run (switching tabs happens in the browser), but leave the page or
unselect the decision and the widget stops rendering; Streamlit is then free to drop
the key.

**The browser has its own copy.** Streamlit sends a key's value to the browser only
in the run in which the key is *written* through `st.session_state`. A widget drawn
without `value=` / `index=` whose key was seeded in an *earlier* run reaches a
browser that has not drawn it before (fresh session, a decision tab just selected,
a page navigated back to, a changed label or range) with only its built-in default,
min / `False` / option 0, and the browser sends that back on the next rerun. So
every keyed widget whose value lives in session state is drawn through
`stateful(st.<widget>, …, key=…)` from `app/state/widgets.py`: it re-writes the key
(unchanged) whenever the widget was not drawn in the previous run with the same
arguments, using a per-run render log (`_widget_render_log`, rotated by
`begin_script_run()` at the top of `app_enhanced_new.py`). `tests/browser_sim.py`
models the browser for AppTest; the browser-driven journeys of
`tests/test_golden_screens.py` use it and record each widget the browser would
show with another value than the script holds (`browser_mismatches`).

Registry categories: `page1-widget`, `page2-widget`. Names mostly follow a
convention: Page 1 uses `*_input` / `*_slider` / `page1_*`, the decision tabs use a
`*_tab_*` infix (`tab_sigma_in_copula`, `di_tab_income_mode`,
`rtd_tab_intercept_{mech}`). The exceptions are the intercept and shift inputs
(`override_categorical_intercept`, `override_continuous_intercept`,
`override_adjustment_shift`, `di_override_intercept`, `dd_override_intercept`), the
Disclose Income anchor sliders (`di_wopb_widget`, `di_wpb_widget`), the buttons, the
Decision 4 `rtd_subtabs` container, and on Page 1 `single_vendor_carryover`,
`show_individual_agents_checkbox` and `purchasing_limit_{i}`. The Overview's
default-decision widgets bind the layer-3 keys themselves (category `defaults`).

### Layer 2 — the canonical key

The value the rest of the app reads, and the layer the seam looks at (besides layer
3 and the two Page-1 seed widget keys `seed_input` / `base_seed_input`, which it
reads before `seed` / `base_seed`). It is written unconditionally right after the
widget call, on every render, so the two cannot drift apart while the tab is open:

```python
# decision_tabs/_controls.py, drawn for the Disclose Income tab with
# keys = CONTROL_KEYS['disclose_income'] (wp='di_tab_', sp='di_', research='sigma_enabled')
research_widget_key = f'{keys.wp}{keys.research}'        # di_tab_sigma_enabled
research_canonical_key = f'{keys.sp}{keys.research}'     # di_sigma_enabled
restore(research_widget_key, True)
sigma_enabled = stateful(st.checkbox,
                         f"Use Normal({noun}, σ) draw in Research Specification mode",
                         key=research_widget_key, ...)
st.session_state[research_canonical_key] = sigma_enabled
```

(`stateful()` refuses `value=` / `index=` / `default=`: the widget's value lives in
the key, which `restore` seeds.) The four modelled tabs draw their
stochastic controls and their income-mode radio through this one component,
`app/pages/decision_tabs/_controls.py`, each with its own `TabControls` spec (its
keys, its σ constants, its wording); the Disclose Income and Disclose Documents
tabs share their intercept override, resets and action buttons the same way
(`_disclosure.py`, a `DisclosureTab` spec each).

**The canonical key is also where an absent widget key is seeded from.** Layer 1's
weakness - Streamlit drops the key of a widget that is not drawn - is covered by
layer 2: the canonical key is never dropped and holds the last rendered value, so
`_tab_state.restore` (`app/pages/decision_tabs/_tab_state.py`), called right before
every widget the table lists, seeds the key from it. `registry.TAB_WIDGET_CANONICAL`
is the table it reads: widget key → (canonical key, entry), where the entry is `None` for
a flat key and `'1'`..`'5'` for the per-quintile sliders, which map to entries of
`di_` / `dd_` / `rtd_quintile_scale_factors`; the Decision 4 intercept inputs map
to `rtd_intercept_{mech}`, the Flexibility anchor slider to
`rtd_flex_observed_weight`, the Disclose Income anchor sliders to `di_wopb` /
`di_wpb`, the Donation anchor slider to `anchor_observed_weight` and the Donation
σ coefficient slider to `donation_sigma_coefficient_setting` - a key of its own,
because `sigma_coefficient`, the engine input, is deliberately 1.0 while the σ mode
is `quintile` (the Donation tab writes it after the σ controls: the slider's value
in overall mode, 1.0 in quintile mode), so it cannot say what the hidden slider
showed.
Every tab restores in the same order:

1. the live widget key, if it is still there;
2. otherwise the canonical value, if the key (and, for a dict, the entry) exists;
3. otherwise the hard default.

The key is written only when it is absent. Until 2026-10-22 the Donation, Disclose
Income and Disclose Documents tabs did it the other way round and wrote the stored
value over the widget key on every rerun; Streamlit re-sends a key written through
the Session State API to the browser with that run's response, and that overwrote
the +/- clicks of a number input that arrived while a rerun was in flight (two to
four of six fast clicks landed on the Donation intercept inputs; Decision 4,
restored widget-first, kept all six).

So a value the user typed survives leaving the tab and coming back, even though the
widget key did not. Until 2026-10-22 each tab kept a *fourth* place for this, a
`*_tab_persistence` dict written only from the widgets' `on_change` callbacks and
read before the canonical key; it only ever held what the canonical key held (the
canonical key is rewritten by every render, the dict only by a change), so the four
dicts and the callbacks that filled them are gone. The callbacks that remain do
something else: the two radios mirror their canonical key before the Overview reads
it in the same run, Decision 4's intercept inputs and Flexibility slider write
theirs, and the intercept / adjustment override inputs save the typed value.
`tests/test_session_key_registry.py` holds `TAB_WIDGET_CANONICAL` to the
`CONTROL_KEYS` rows, to the registry rows and to every `restore()` call in `app/`.

**The Donation reset and the σ controls (2026-10-22).** The Donation tab's "Reset
Config to Defaults" returns the σ controls to the file's defaults, as its help text
says: it writes the canonical keys (`donation_sigma_strategy`,
`donation_sigma_coefficient_setting`, the quintile dict) from the file and deletes
their widget keys, which `restore()` re-seeds from the canonical values right before
each widget is drawn (a σ control hidden at the reset is re-seeded on its next
draw). Until that day the reset left the σ controls alone, and a coefficient edited
before the reset silently reverted on the next navigation; both are gone.

**One rule keeps the widget-first order safe: never write a widget key in a run that
does not draw its widget.** Streamlit keeps such a write in its old session state
under the user key, never refreshes that entry once the widget exists (later writes
go under the widget id), and lets it resurface, stale, as soon as the widget is
culled - in front of the canonical value. That is why Page 2 no longer pre-seeds the
tabs' keys (`initialize_page2_widget_keys` only mirrors the canonical keys, for the
Overview tab that is drawn before the decision tabs), why each tab seeds its own keys
right before its widgets, and why a reset writes only the keys of the widgets the
same run draws (the Disclose Income / Documents per-quintile sliders drop their keys
on a reset and are seeded again on their next draw). The one exception keeps its
guard elsewhere: the Donation intercept reset writes both intercept keys, the one the
income mode draws and the one it does not, so those two inputs are seeded from the
typed override not only while their key is absent but every time they come back
after not being drawn (`donation_default._seed_intercept_input`, which asks the
render log of `app/state/widgets.py`) - the run in which the value is pushed to the
browser anyway, so a steady rerun still writes nothing.

Registry categories: `page1-mirror`, `page2-mirror`.

### Layer 3 — `_persistent_defaults`

A shadow store for one specific family: the **default-decision parameters** the
Overview tab configures for decisions the user did *not* select
(`{decision_name}_default_probability_y`, `_default_params`, `_default_selection`,
`_priority_template`, `_default_value`).

Those widgets are drawn on the Overview tab only while their decision is not
selected, and only while Page 2 shows; otherwise Streamlit may drop their keys. At
the top of every script run `models.initialize_default_decision_parameters` (called
by `models.initialize_session_state`) seeds every **absent** key from
`DEFAULT_DECISION_VALUES` - the hard default, not the user's choice.
`_persistent_defaults` keeps the choice: `default_config.save_to_persistent_storage(key)`
copies the value in (the widgets' `on_change`), `restore_from_persistent_storage`
writes it back before the widget is drawn, and the seam consults it **before** the
session key. The full order for one default decision is

1. the post-simulation override (`{decision_name}_probability_y` and friends, §6);
2. `_persistent_defaults[{decision_name}_default_probability_y]`;
3. `st.session_state[{decision_name}_default_probability_y]`;
4. the hard-coded value in `DEFAULT_DECISION_VALUES`.

Registry category: `defaults`.

---

## 3. The other categories

The registry has nine categories; five are the layers above. The rest describe keys
that are not settings at all:

| Category | What it holds | Examples |
|---|---|---|
| `navigation` | which screen renders | `page` |
| `saved-config` | the "Use This Config" pins | `selected_decision_configs` |
| `run-flag` | what the click that started a run recorded (the run itself travels as a `RunRequest`, not through session state); also the Donation tab's one-shot reset flags | `custom_decisions`, `default_decisions`, `_reset_config_to_defaults_flag` |
| `results` | run output, and the chart / download widgets that show it | `simulation_results`, `mc_results`, `vendors`, `_run_metadata` |

`selected_decision_configs` deserves its own note: one entry per decision the user
pinned, each holding the seed, the agent count, the population mode, that decision's
parameters, and the sha256 of the columns the pinned run produced. A later complete
simulation re-checks that hash, so "reproduce what I looked at" is verified rather
than assumed.

---

## 4. Which keys the seam reads

The seam takes **one** snapshot of the session state per click
(`take_snapshot(st.session_state)`) and hands that frozen copy to
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
| Default decisions | `_persistent_defaults`, `{decision_name}_default_probability_y`, `_default_params`, `_default_selection`, `_priority_template`, `_default_value`, `final_donation_rate_default_value`, and the seam's read-only fallbacks (§6) |
| Saved configs | `selected_decision_configs` |

Two things are easy to get wrong here.

* **The donation coefficients the engine runs are the suffixed ones.** `build_plan`
  reads `donation_coeff_{name}_cat` / `_cont`, and the Donation tab's formula tables
  show that set too. The flat `donation_coeff_*` keys (the active income mode's set)
  are display state: a pinned donation configuration stores them
  (`state.saved_configs.get_current_coefficients`), and the intercept override
  writes the active intercept into them as well as into the suffixed key.
* **The seam does not read `sigma_value_ui`.** It is the product of the σ constant
  and the coefficient slider, stored with a pinned donation configuration
  (`state.saved_configs.get_current_stochastic_params`). There is one σ constant, in
  `config/decisions.yaml`, and the seam recomputes the product itself from
  `sigma_coefficient`.

One more path exists outside the seam: `n_runs`, `base_seed`, `population_mode` and
`income_spec_mode` are *also* read by the in-process Monte-Carlo loop in
`app/simulation.py` (the seeds, and which sub-run the study repeats). Of these only
`n_runs` is not a seam input. The Monte-Carlo runner builds the ordinary run plan
from the same snapshot and repeats one of its sub-runs in process
(`app/seam/mc.py`), so every tab setting reaches the Monte-Carlo runs exactly as it
reaches a single run. A selected ("Use
This Config") saved configuration pins the Monte-Carlo agent count and population
exactly as it pins a complete run's; only the seed varies (`base_seed + i`), and the
screen says so.

---

## 5. Written but never read

None at present: no registry entry is `write_only`. The test holds the registry to
the scan, so a key that becomes written-but-never-read in `app/` fails it until the
registry marks it `write_only=True` and this section lists it. Such a key is still
not a removal candidate if it is also `engine_input` (the seam reads the snapshot).

`donation_adjustment_shift` looks write-only but is not: besides the seam,
`donation_default.reload_coefficients_for_income_mode` reads it to carry it across a
coefficient reload.

## 6. Read but never written

The mirror image: the seam looks for a value the app never writes, so the lookup
always falls through to the next priority.

| Key | Looked up by | Falls through to |
|---|---|---|
| `{decision_name}_probability_y` | `build_plan.collect_decision_settings`, priority 1 for the random-probability decisions | `_persistent_defaults`, then the session key, then `DEFAULT_DECISION_VALUES` |
| `{decision_name}_selection` (concretely `vendor_choice_weights_selection`) | same, priority 1 for the checkbox decision | same |
| `rejected_transaction_option_selection` | same, priority 1 for the radio decision (11) | same |
| `rejected_transaction_defaults_option` | named in the same radio branch, which decision 4 (a priority list) never reaches | - |
| `donation_coeff_q45` | never written; `state.saved_configs` still names it as the fallback for `donation_coeff_q5` when saving a configuration | `0.0` |

These are the remains of an "adjust the default on the results page and re-run"
feature that no screen offers any more. The priority chains still have a slot for it.

---

## 7. Bulk operations

Five places delete keys wholesale, and the Donation tab's reset deletes its σ
widget keys. They
matter because they are the only way a key disappears (besides Streamlit dropping an
undrawn widget's key), and the registry's initialisers are what put it back.

| Operation | What it deletes | Then |
|---|---|---|
| A decision tab's **Reset Config to Defaults** (Disclose Income, Disclose Documents; on Decision 4 **Reset Decision 4 Settings to Defaults**) | every key starting with `di_`, `dd_` or `rtd_` (whichever tab): widget keys and canonical keys alike. Disclose Income / Documents only raise `_di_/_dd_reset_to_defaults_pending` on the click and do the deletion at the top of the next run, before any of the tab's widgets exists | Decision 4 calls `initialize_rtd_session_state()`; Disclose Income / Documents write the reset widget keys and canonical keys from the file (`_disclosure.apply_config_to_widget_keys`, plus the tab's own `_apply_extra_widget_keys`), and the tab's `initialize_*_session_state()` runs next in the same run. A widget key the reset does not write (the per-quintile sliders, and on Disclose Income also the σ coefficient and σ mode) is seeded from the reset canonical key on its next draw |
| Decision 4's per-element reset | only `rtd_intercept_{mech}` and its widget key (for Flexibility also `rtd_flex_observed_weight` and its slider key) | same; σ is decision-wide and deliberately untouched |
| The Donation tab's **Reset Config to Defaults** | only the σ controls' widget keys: `donation_tab_sigma_strategy_stochastic`, `tab_sigma_coefficient_stochastic` and `donation_tab_sigma_q{level}_stochastic`; otherwise it *writes*: the intercept, adjustment, income-mode, anchor-weight and σ tick-box widget keys (Copula off, Research Specification on), the canonical income mode, and the canonical σ mode, σ coefficient (`donation_sigma_coefficient_setting`) and quintile dict from the file's `donation_default.stochastic` (`overall`, `scale_factor`, `quintile_scale_factors`), and empties the two override-value dicts | `_reset_config_to_defaults_flag` is consumed at the top of the tab in the next run, before the tab's widgets; the σ mode radio and coefficient slider are seeded from the reset canonical keys in that run, the quintile sliders (hidden once the mode is `overall`) on their next draw |
| Page 1's **Reset to Default Values** | every `purchasing_limit_{i}`, `purchasing_limits_temp` and the five widget keys that pass `value=` (the three "Maximum Value" text inputs, `uniform_purchasing_limit_input`, `show_individual_agents_checkbox`); every other Page-1 widget key is *written* | `sim_params` fields go back to the `SimulationParameters` dataclass defaults and the Page-1 session values to `models.SESSION_DEFAULTS`; the widget keys are written from `page1_widget_values`, the same mapping `initialize_widget_keys` seeds a fresh session from |
| The Overview's **Reset All Defaults** | for every unselected decision, its `{decision_name}_default_*` and `_priority_template` keys (and the `_default_param_*` checkbox keys), plus their `_persistent_defaults` entries | `models.initialize_default_decision_parameters` seeds them again at the top of the next run |
| **Clear Results** on the results page | *every* key in `st.session_state` | calls `models.initialize_session_state()` and stays on the results page |

The prefix list lives in the registry as `PREFIX_DELETE_PREFIXES` and the test
checks it still matches the code.

---

## 8. Keys named at run time

A handful of accesses name their key through a variable the scanner cannot resolve:
a helper's parameter, or a loop over `st.session_state.keys()`. They are reviewed by
hand and listed in `DYNAMIC_KEY_VARIABLES`, so that a *new* unresolvable access fails
the test rather than slipping past it:

* `key` — the parameter of `state.widgets.sync_widget_key` (every `stateful()`
  widget's key), of `save_to_persistent_storage` / `restore_from_persistent_storage`
  and of the Decision-4 download helper; the loop variable of the Disclose Income /
  Documents initialisers, the prefix resets (`_disclosure.apply_pending_reset`,
  Decision 4) and the Clear Results wipe;
* `k` — the per-category purchasing-limit key on Page 1, bound in a lambda default;
* `chart_key` — the plotly key the shared Decision-4 chart helpers receive;
* `widget_key`, `canonical_key` — the tabs' seeding helpers
  (`decision_tabs/_tab_state.py`: `restore`, `seed_if_absent`,
  `seed_if_changed`; `widget_key` also in `donation_default._seed_intercept_input`):
  the widget key they are handed, and the canonical key `restore` looks it up in
  `TAB_WIDGET_CANONICAL`. Every caller still binds its widget key literally (`key=`),
  or through the registry's key-parameter table below, and every canonical key of the
  table is written literally next to its widget; the test holds the table to both.

**Keys built from the registry's per-tab tables.** Two modules draw the same
widgets for several tabs: `decision_tabs/_controls.py` (the four modelled tabs'
stochastic controls and income-mode radio) and `decision_tabs/_disclosure.py` (the
Disclose Income / Documents intercept override, resets and buttons). They name
their keys with f-strings over a row of `registry.CONTROL_KEYS` /
`registry.DISCLOSURE_KEYS` — `f"{keys.wp}sigma_coefficient{keys.suffix}"` is
`tab_sigma_coefficient_stochastic` on the Donation tab and
`rtd_tab_sigma_coefficient` on Decision 4 — and read those rows from the registry,
so the keys the app uses *are* the table's values. `registry.KEY_PARAMETER_TABLES`
maps each module to its table; the test expands every such f-string of the module
over every row (a placeholder `{<name>.<field>}` whose last segment is a field of
the row; `{level}` stays a family placeholder) and holds each expansion against the
registry, which keeps one row per concrete key as for every other key. An
expansion the registry does not hold fails the test like any other unregistered
key; a dotted placeholder that is not a field, or any such placeholder in a module
without a table, is a dynamic key and is refused unless it is in
`DYNAMIC_KEY_VARIABLES` (`tests/test_session_key_registry.py` has a negative test
for each).

---

## 9. Adding a key

1. Give it a name that says which layer it belongs to (`*_tab_*` for a widget key on
   a decision tab, no infix for the canonical value).
2. Write it next to the widget, every render, if it is a canonical key; seed the
   widget key from it with `_tab_state.restore` right before the widget, and add
   the pair to `registry.TAB_WIDGET_CANONICAL`.
3. Add a `KeySpec` to `app/state/registry.py`: category, default, who initialises
   it, and whether the seam reads it.
4. Run `pytest tests/test_session_key_registry.py`. If the key is a new engine input,
   the test will also insist the seam actually names it; a widget seeded through
   `restore` that is not in the table fails it too.
