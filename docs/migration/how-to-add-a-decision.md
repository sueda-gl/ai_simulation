# How to add a decision

This is the recipe for adding decision number 14. It assumes you have read
[`architecture.md`](architecture.md), in particular §3 (where the random numbers come
from) — because the single most important rule here follows from it.

> **Add the new decision at the END of `DECISION_ORDER`.**
> A decision's position in that list is what seeds its random numbers
> (`decision_rng = default_rng(agent_base_seed + position * 1000)`). Appending a
> fourteenth entry leaves all thirteen existing streams untouched. Inserting one in
> the middle shifts every later decision's offset and silently changes every later
> result in every mode. If the model genuinely requires the new decision to run
> before an existing one, that is a change to the model, not a refactor — get it
> ruled first, and expect every golden number to move.

Throughout, the example decision is called `loyalty_choice`.

---

## Step 1 — Write the decision module

Create `src/decisions/loyalty_choice.py`. The module must export a function with the
decision's own name and this exact signature:

```python
# src/decisions/loyalty_choice.py
def loyalty_choice(agent_state: dict, params: dict, rng,
                   simulation_config: dict = None) -> dict:
    """Decision 14: ..."""
    ...
    return {"loyalty_choice": value}
```

(A decision that receives `pop_context=` — see the noise gate below — takes
`**kwargs` as well, exactly as `donation_default` and the other three modelled
decisions do.)

What each argument is:

| argument | what it holds |
|---|---|
| `agent_state` | the agent's traits **plus everything the earlier decisions returned** — this is how a decision reads a previous one's output (e.g. `agent_state['disclose_income']`) |
| `params` | this decision's block of `config/decisions.yaml`, already patched with whatever the UI configured |
| `rng` | a `numpy.random.Generator` seeded for this agent × this decision. **Draw only from this.** Never `random.random()`, never `np.random.*` |
| `simulation_config` | the whole run configuration. Read Page-1 parameters with `get_simulation_param(simulation_config, 'market_price', 100.0)` from `src/decisions/income_utils.py` — **not** `simulation_config.get('market_price')`, which reads a top-level key nothing sets (this mistake is live in the code today; see Q-10 in [`rulings-and-quirks.md`](rulings-and-quirks.md)) |

Return a dictionary. Its keys become columns of the results frame, and they are
merged into `agent_state` so later decisions can read them.

Three rules the import-boundary test will enforce for you:

* **Never `import streamlit`** and **never `from app… import …`**, not even inside a
  function body. If you need a value that lives in the UI, it must be passed in
  through `simulation_config` — that is what the seam is for.
* Do nothing at import time. No file reads, no `sys.exit`.
* Raise on bad input. Do not substitute a placeholder value and continue (ruling R4).

If your decision needs *population-level* statistics (a median, a standard deviation
over all agents), add a hook in `Engine._compute_population_stats` in
`src/engine/core.py` and store the result on `simulation_config`; Pass 1 runs before
any decision does. Follow the pattern of `rtd_population_stats`.

If your decision should add noise, gate it with the shared rule rather than writing
your own:

```python
from src.utils.stochastic import should_use_stochastic
if should_use_stochastic(params.get('stochastic', {}), kwargs.get('pop_context', 'copula')):
    ...
```

and add `'loyalty_choice'` to `POP_CONTEXT_DECISIONS` in `src/engine/core.py` so the
engine passes `pop_context=` to it.

## Step 2 — Register it in the engine

`src/engine/core.py`:

```python
DECISION_ORDER = [
    ...,
    'final_donation_rate',    # 13
    'loyalty_choice',         # 14   <-- appended
]
```

That is all the engine needs. It imports `src/decisions/<name>.py` and takes the
function called `<name>` out of it automatically. (The two exceptions are Decisions 1
and 2, which live in `<name>_stochastic.py` — listed in `STOCHASTIC_MODULE_DECISIONS`.
Do not add yourself to that list unless you have a reason to.)

## Step 3 — Give it a configuration block

`config/decisions.yaml`:

```yaml
loyalty_choice:
  intercept: 0.5
  coefficients:
    honesty_humility: 0.1
  stochastic:
    sigma_strategy: overall
    sigma_overall: 0.25
    sigma_value: 0
    scale_factor: 1.0
    in_copula: false
```

Two conventions to respect:

* **The file is read-only at run time.** The app never writes it (ruling R11). Values
  a user can change must travel from a session key to the engine as a *patch* — see
  step 5.
* **Keep every constant here, not in the code.** The migration removed four sigma
  literals from the app for exactly this reason (ruling R12).

At this point the decision already runs from the CLI and from Python:

```bash
python scripts/run_simulation.py --agents 100 --seed 42 --decision loyalty_choice
```

## Step 4 — Make it selectable in the UI

`app/models.py` — append it to `ALL_DECISIONS` (same order as `DECISION_ORDER`; the
two lists must match):

```python
ALL_DECISIONS = [..., "final_donation_rate", "loyalty_choice"]
```

`src/contract/defaults.py` — add an entry to `DEFAULT_DECISION_VALUES` (and a line
to `DEFAULT_DECISION_DESCRIPTIONS`). This is what the decision does when the user
does **not** select it, i.e. when it runs "with default values". The registry lives
on the contract side because both the seam and the screens read it;
`app/pages/decision_execution.py` re-exports it. Pick one of the existing shapes (`random_probability`,
`checkbox_selection`, `radio_selection`, `prioritized_selection`, or a plain constant)
so `collect_decision_settings` in `app/seam/build_plan.py` already knows how to
resolve it:

```python
"loyalty_choice": {
    "type": "random_probability",
    "probability_y": 0.5,
    "options": ["Y", "N"],
    "description": "Probability of choosing the loyalty option",
},
```

If your decision reads that default at run time, read it from
`simulation_config['default_decisions']['loyalty_choice']` — the pattern used by
Decisions 11, 12 and 13 (ruling R7). Never import the UI to get it.

## Step 5 — A configuration tab (only if the decision needs one)

Decisions without their own tab fall through to the generic renderer and need no
work. If yours needs one:

1. Write `app/pages/decision_tabs/loyalty_choice.py` with a
   `render_loyalty_choice_tab()` function.
2. Add a branch for it in `render_decision_tab` in
   `app/pages/decision_tabs/__init__.py`.
3. Write the tab's values into session keys with a clear prefix
   (`lc_intercept`, `lc_sigma_enabled`, …).
4. Add a patch builder in `app/seam/build_plan.py`:

   ```python
   def build_loyalty_choice_patch(snapshot, config_repo, pop_mode, inc_mode=None):
       patch = {}
       if "lc_intercept" in snapshot:
           patch["intercept"] = snapshot["lc_intercept"]
       patch["stochastic"] = _three_way_stochastic(
           pop_mode,
           sigma_enabled=snapshot.get("lc_sigma_enabled", True),   # R18: absent = the tab's default
           sigma_in_copula=snapshot.get("lc_sigma_in_copula", False),
           sentinel=...,
       )
       return patch
   ```

   and call it from `build_decision_patches`.

Two rules the seam enforces:

* **The builder reads the snapshot and returns a patch. It writes nothing** — no
  session keys, no files (rulings R11 and R15).
* **An absent key means the tab's own default, not "off"** (ruling R18). Otherwise
  the same visible settings give different numbers depending on whether the user
  happened to open the tab.

## Step 6 — Results and exports

* **Chart / table.** Write a renderer and register it in `DECISION_VISUALIZATIONS`
  in `app/pages/results/visualizations/__init__.py`. Without an entry the results
  page falls back to `.describe()`, which is a fine place to start.
* **Anything the renderer computes** — a frame, a statistic, a downloadable file —
  belongs in `app/reports/`, taking its inputs as arguments and returning a value.
  The page keeps the `st.*` calls and the session reads. `app/reports/` must not
  import `streamlit`, and the boundary test will tell you if it does.
* **Ask `RunContext`, not the widgets.** If the renderer needs to know the run's
  income mode, population mode, or which decisions ran, use
  `RunContext.from_session()` (`app/pages/results/run_context.py`). The live Page-1
  and Page-2 keys describe the *next* run, not the one on screen (ruling R28).
* **Do not write settings from a renderer** (ruling R30).

## Step 7 — Prove nothing else moved

```bash
pytest tests/ -q                       # expect 215 passed (more, once you add yours)
pytest tests/test_import_boundary.py -q  # the new module is picked up automatically
```

The boundary test walks `src/` and `app/reports/`, so your new files are covered the
moment they exist — no registration needed.

Then run the journey harness (see §8 of [`architecture.md`](architecture.md)) before
and after your change and compare. Because you appended rather than inserted, the
expected result is: **every existing column identical, one new column added.**
Anything else means the decision order or a shared configuration value moved, and you
should find out why before going further.

## Checklist

- [ ] `src/decisions/loyalty_choice.py` with the standard signature, no UI imports
- [ ] appended to `DECISION_ORDER` in `src/engine/core.py` (and to `POP_CONTEXT_DECISIONS` if it takes noise)
- [ ] a block in `config/decisions.yaml`, holding every constant
- [ ] appended to `ALL_DECISIONS` in `app/models.py`
- [ ] an entry in `DEFAULT_DECISION_VALUES` in `src/contract/defaults.py`
- [ ] a tab + patch builder, if it is configurable
- [ ] a renderer in `DECISION_VISUALIZATIONS`, with the numbers in `app/reports/`
- [ ] `pytest tests/` green
- [ ] journeys compared before/after: existing columns identical, one column added
