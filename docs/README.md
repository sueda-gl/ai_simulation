# Start here

This repository is a Streamlit dashboard over a pure simulation engine. The engine
produces, for a population of synthetic or real agents, the thirteen decisions of the
COOPECON platform; the dashboard configures a run, launches it and shows and exports
the results. The top-level [`README.md`](../README.md) says how to install and run.
This page says how the code is organised and why.

The rules of the simulation are not in this repository's prose: they are in the
platform documentation under [`reference/`](reference/README.md). When the code and
that documentation disagree, the documentation wins.

---

## 1. The four layers

Each layer may use the one above it, never the other way round.

```
  src/engine  +  src/contract  +  src/decisions   THE ENGINE       (no Streamlit)
              |
  app/seam                                        THE SEAM         (no Streamlit)
              |
  app/state   +  app/reports                      STORES & REPORTS (reports: no Streamlit)
              |
  app/pages   +  app/components                   THE SCREENS      (all the Streamlit)
```

### The engine, `src/`

Everything needed to produce numbers, and nothing else.

* **`src/engine/core.py`** — the `Engine` class: one loop over agents × decisions. It
  loads `config/decisions.yaml` and `config/simulation.yaml`, imports one module per
  decision, sources the agents and runs the two passes described in §3. It holds the
  single decision order:

  ```python
  DECISION_ORDER = ['disclose_income', 'disclose_documents', 'donation_default',
                    'rejected_transaction_defaults', 'vendor_choice_weights',
                    'purchasing_quantity', 'purchasing_frequency', 'vendor_selection',
                    'purchase_vs_bid', 'bid_value', 'rejected_transaction_option',
                    'rejected_bid_value', 'final_donation_rate']
  ```

  A decision's *position* in this list seeds its random numbers (§3), so the order is
  part of the model, not a cosmetic choice.
* **`src/engine/profile.py`** — `ModeProfile` and `PROFILES`: the only differences
  between the three population modes (§4).
* **`src/engine/sampling.py`** — which of the 280 participants a research-mode run uses.
* **`src/engine/vendors.py`** — vendor prices, quality, sustainability and per-period
  quantities, drawn once per run.
* **`src/engine/postprocess.py`** — the global transaction IDs, assigned after a run so
  every export numbers the same request the same way.
* **`src/decisions/<name>.py`** — one module per decision. Every decision function has
  the same shape:

  ```python
  def my_decision(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
      ...
      return {"my_decision": value}
  ```

  The four *modelled* decisions (`donation_default`, `disclose_income`,
  `disclose_documents`, `rejected_transaction_defaults`) also receive `pop_context=`,
  the population mode's name, because that is what switches their noise on or off.
  Decisions 1 and 2 live in `<name>_stochastic.py`.
* **`src/contract/plan.py`** — the frozen description of one run (§2): `RunPlan`,
  `SubRun`, `RunMetadata`, `SavedExpectation`, the deep-merge rules for configuration
  patches, and the hash that proves a pinned configuration was reproduced.
* **`src/contract/defaults.py`** — what each decision does when the user does *not*
  select it. Both the seam and the screens need it, so it lives where nothing imports
  Streamlit.
* **`src/data/participants.py`** — the 280 experiment participants, read from the two
  workbooks in `data/` the first time they are asked for, then cached. Nothing happens
  at import time.
* **`src/trait_engine.py`** — samples synthetic agents from the fitted copula in
  `config/trait_model.pkl`. Retrain only with `scripts/train_copula.py`, and only if
  the experiment data or the master trait list changes.
* **`src/config.py`** — the one loader of the two YAML files (`read_yaml_file`, the
  path constants, `DecisionsConfig`); the engine and the seam both load through it.

### The seam, `app/seam/`

Turns "the user pressed a button" into "run these engine runs". Ordinary Python: no
Streamlit, no globals, no writes to session state.

* **`snapshot.py`** — `take_snapshot(st.session_state)`: a read-only copy, taken once
  per click. Everything downstream reads the copy, so nothing a run does can feed
  back into it.
* **`config_repo.py`** — `DecisionsConfig`: the two YAML files, loaded once and never
  written. `fresh_decisions_dict()` hands out a deep copy for one engine run to patch.
* **`sentinels.py`** — the sigma constants, read from the configuration file rather
  than spelled as literals in the app.
* **`build_plan.py`** — `build_run_plan(snapshot, config_repo) -> RunPlan`: resolves
  seed and agent count, decides which decisions run, which population and income
  modes apply, builds one configuration patch per decision, and produces the texts the
  screen shows before the run. A pure function: the same snapshot always yields the
  same plan.
* **`execute.py`** — `execute(plan) -> {result_key: DataFrame}`: for each sub-run,
  samples the agents, builds a fresh `Engine`, applies the patches and the Page-1
  parameters, runs it, then assigns the global transaction IDs.
* **`mc.py`** — Monte Carlo on the same plan: picks the sub-run a study repeats and
  replays it with another seed (`run_mc_repetition`); the CLI's `--plan-file` reads
  a sub-run stored with `write_mc_plan_file`.

### Stores and reports, `app/state/`, `app/reports/`

* **`app/state/saved_configs.py`** — the "Use This Config" store. Pinning a decision's
  result records the seed, the agent count, the population mode, that decision's
  parameters and a sha256 of the columns it produced.
* **`app/state/registry.py`** — every session-state key the screens use, with its
  default and initialiser; `tests/test_session_key_registry.py` keeps it honest. See
  [`session-state.md`](session-state.md).
* **`app/reports/*.py`** — every number, table, preview frame and Excel/CSV byte string
  the screens show or offer for download. These take their inputs as arguments and
  return frames or bytes; they never import Streamlit, so an export can be built and
  checked without a browser.

### The screens, `app/pages/`, `app/components.py`

* `page1_common_params.py` — Page 1: population size, seed, income distribution,
  vendors, thresholds.
* `page2_decisions.py` + `decision_tabs/` — Page 2: the Overview tab and one tab per
  configurable decision.
* `decision_execution.py` — the Run buttons (`run_individual_decision`,
  `run_combined_simulation`, the per-tab `render_simulation_buttons`) and the gate that
  disables "Run Complete Simulation" while a selected decision would produce several
  configurations and none is saved (`can_run_complete_simulation` and its warnings).
* `results/` — the results page; `results/run_context.py` is the piece to know (§5).
* `app/models.py` — `SimulationParameters` (the Page-1 settings, kept in
  `st.session_state.sim_params`), `SESSION_DEFAULTS` (the other values a fresh session
  starts with) and `initialize_session_state()`, which `app_enhanced_new.py` calls on
  every script run to seed whatever is missing.
* `app/simulation.py` — the bridge from session state to the seam: snapshot, plan,
  execute, store. It also hosts the in-process Monte-Carlo loop (one
  `run_mc_repetition` per seed, statistics from `app/reports/mc.py`).

---

## 2. What one click does

Take "🚀 Run Complete Simulation" on Page 2.

```
  Page 2 button
        |
        v
  run_combined_simulation()                     app/pages/decision_execution.py
    builds a RunRequest (which decisions run, which with custom settings, which
    with defaults), records the custom/default lists for the results page, then calls
        |
        v
  run_full_simulation(request)                  app/simulation.py
        |
        |-- take_snapshot(st.session_state)     app/seam/snapshot.py     (read ONCE)
        |-- build_run_plan(snapshot, config)    app/seam/build_plan.py   (pure)
        |-- for message in plan.messages: st.info / st.caption / st.success
        |-- execute(plan)                       app/seam/execute.py
        |        for each SubRun:
        |          sample_agents(population, n, seed)
        |          Engine(PROFILES[population])           <- fresh yaml load
        |          configure_engine(...)                  <- patches + Page-1 params
        |          engine.run_simulation(n, seed, decisions, agents_df)
        |        then assign_global_transaction_ids(...) on every result
        |-- verify_saved_expectations(...)      re-hash each pinned decision's columns;
        |                                       a mismatch shows an error, the run is dropped
        |-- st.session_state.simulation_results = results
        |   st.session_state._run_metadata      = what was actually run
        |   st.session_state.vendors            = the run's vendor list
        |   st.session_state.page = 'results'; st.rerun()
        v
  Results page                                  app/pages/results/
    RunContext.from_session() reads _run_metadata + simulation_results
```

Three things to notice.

**One snapshot, one plan.** The session state is read once, at the top. A test can
hand the builder a plain dictionary instead of a Streamlit session.

**A plan is data.** `RunPlan` is a frozen dataclass: it can be printed, stored,
compared, or built by hand. A `SubRun` is one engine run producing one key in the
results dictionary; "Compare all × Compare both" produces six sub-runs in a fixed order.

**The runner writes nothing until the results exist.** The only session writes inside
`run_full_simulation` happen after `execute` returns, so a failed run leaves the
results, the run metadata and the vendor list as they were. The *buttons* do write
before the run starts (app/pages/decision_execution.py): a per-tab "Run X Only"
clears a pinned configuration of that decision (Donation Default, Disclose Income,
Disclose Documents), and every run button records `custom_decisions` /
`default_decisions`; those writes stay even when the run fails. The buttons hand the
runner a `RunRequest` (src/contract/plan.py) saying which decisions run and which use
defaults; the plan builder reads the request, never a swapped selection.

The other Run buttons (per-tab "Run X Only", per-tab "Run Complete", the Results-page
"Run Complete") all funnel into the same `run_full_simulation()` in Single Run mode. In
Monte-Carlo mode (`sim_params.simulation_mode == "Monte-Carlo Study"`) every Run button
calls `run_monte_carlo_study(request)` (app/simulation.py) instead, which repeats one
sub-run of the same plan with a different seed per repetition.

---

## 3. Where the random numbers come from

This is what guarantees "same seed, same settings → same numbers". Reproduced from
`src/engine/core.py`:

```
rng_setup    = default_rng(seed)                    -> vendor attributes
rng_pass1    = default_rng(seed + 1_000_000)        -> one integers(1e9) per agent
                                                       = that agent's BASE SEED
income_rng   = default_rng(base_seed + 999_999)     -> the agent's income
decision_rng = default_rng(base_seed + position*1000)  -> that one decision, that one agent
```

`position` is the decision's index in `DECISION_ORDER` (0 for `disclose_income`, 12
for `final_donation_rate`). Research Specification draws its participants from a
`default_rng(seed)` of its own; the copula draws its agents from `default_rng(seed)`
inside `TraitEngine.sample`.

Two consequences:

* **Which decisions are selected does not change the numbers of the ones that run.**
  Decision 3's stream is `base_seed + 2000` whether or not Decisions 1 and 2 ran.
* **An agent's income is drawn twice, identically.** Pass 1 draws it to compute the
  population statistics some decisions need; Pass 2 re-draws it from the same generator
  so the agent keeps the same income whatever else runs.

The run is two passes over the agents:

* **Pass 1** — take a base seed per agent and draw the income; compute the population
  statistics (income median, mean and sample standard deviation, and the extra
  statistics Decisions 1, 2, 3 and 4 need, see `Engine._compute_population_stats`).
  Decision 3's statistic is the population maximum of every agent's floored draw,
  found by replaying each agent's own Decision-3 stream.
* **Pass 2** — rebuild the income from the same base seed, then run the selected
  decisions in `DECISION_ORDER`. Each decision's output dictionary is merged into the
  agent's state, so a later decision can read what an earlier one decided. **The
  mutated agent state is the output row.**

Adding a decision *at the end* of `DECISION_ORDER` leaves every existing number
untouched. Inserting one in the middle shifts every later decision's offset and changes
every later result. See [`how-to-change-the-model.md`](how-to-change-the-model.md).

---

## 4. The three population modes

They differ in exactly four fields, all in `src/engine/profile.py`:

| field | Copula | Research Specification | Research Baseline |
|---|---|---|---|
| `pop_context` | `"copula"` | `"documentation"` | `"baseline"` |
| `agent_source` | `"copula"`: `TraitEngine.sample(n, seed)` | `"research"`: the 280 participants | `"research"`: the 280 participants |
| `random_sample` | — | `True`: random, seeded | `False`: file order, cycling |
| `force_donation_sigma_zero` | `False` | `False` | `True` |

Research Baseline takes the 280 participants in file order, cycling: agent k is
participant k mod 280, so N = 280 is exactly the 280 and N = 1000 is three full cycles
and the first 160. Research Specification samples randomly, seeded by the run seed: a
subset without replacement below 280, all 280 at 280, a bootstrap above. In both modes
every agent is its own agent for the random numbers, so Research Specification noise
differs between two copies of one participant. At N = 280 with every σ off the two
research modes give the same rows (`tests/test_research_cyclic_population.py`).

`pop_context` is what the four modelled decisions use to decide whether to add noise.
The single rule is `should_use_stochastic()` in `src/utils/stochastic.py`:

* `copula` → noise iff the decision's `stochastic.in_copula` flag is on (the "Use
  Normal(anchor, σ) draw in Copula mode" tick box);
* `documentation` → noise iff `stochastic.sigma_value > 0` (the Research tick box
  writes the σ there);
* `baseline` → never.

Everything else, the decision order, the row building, the error handling, the module
chosen for each decision, is identical across modes.

---

## 5. How the results page knows what it is showing

When a run finishes, `app/simulation.py` stores `st.session_state._run_metadata`: the
population and income modes the run actually used, the result keys, the custom and
default decision lists, the seed and the agent count, per result key the income
mode each income-dependent decision actually ran with, the moment the run started,
and - since 2026-10-22 - the Page 1 parameters the run used (the whole
`SimulationParameters` snapshot as a plain dict, `RunMetadata.simulation_params`),
the Decision 4 element the run button named (`rtd_run_element`) and the donation
anchor weight the run ran with (the tab slider, or the pinned donation record's).
The per-decision income modes matter because a complete run names its result keys
after the global income mode while Decision 4 runs with its own tab's mode.

The results page builds a `RunContext` (`app/pages/results/run_context.py`) from that
metadata plus the result frames and asks *it* every question about the run: its
shape, and its Page 1 values (`ctx.param(name)`, `ctx.platform_markup`,
`ctx.periods`, ...). It deliberately does **not** read the live Page-1 / Page-2
widgets or `st.session_state.sim_params`: those describe the run being configured
next, not the one on screen. So editing Page 1 after a run and returning to the
results page via "View Results" shows the run's summary, bidding explanation and
exports unchanged (the `page1_edit_after_run` golden journey pins this). Metadata
written before these fields existed falls back to the live values, the only ones
such a session has.

"Use This Config" (`app/state/saved_configs.py`) pins from the same `RunContext`:
the run's seed, agent count, population mode and income mode, not the live widgets.
When the pin is refused because another pinned decision ran with a different seed
or agent count, the function returns the failure and the results page shows an
error naming both runs; nothing is pinned and no success message appears.

---

## 6. The import boundary

Two rules:

1. Nothing under `src/` may import `streamlit` or the `app` package.
2. Nothing under `app/reports/` may import `streamlit`.

`tests/test_import_boundary.py` enforces both for every module it finds, in two
independent ways: it imports the module in a fresh subprocess and inspects the child's
`sys.modules`, and it scans the source with `ast` for an import at any nesting depth,
including one hidden inside a function body. The walk is guarded so the suite cannot
pass by finding no modules.

---

## 7. Running without the UI

### The CLI

```bash
python scripts/run_simulation.py --agents 1000 --seed 42
python scripts/run_simulation.py --agents 5000 --decision disclose_income --format csv
python scripts/run_mc_study.py --agents 10000 --runs 500 --base-seed 1
```

`--population-mode` takes `copula`, `documentation` or `baseline`; `--decision` may be
repeated; `--plan-file F` (both scripts) replays a sub-run stored with
`app.seam.mc.write_mc_plan_file`, the same repetition the app's Monte-Carlo button
runs in process. The study's per-run statistics and summary come from
`app/reports/mc.py` in both. Note that `--income-mode` currently patches only
`donation_default`; the other modelled decisions keep the income mode of
`config/decisions.yaml`. Results land in `outputs/` (git-ignored).

### From Python: the engine directly

```python
from src.engine.core import Engine
from src.engine.profile import PROFILES

engine = Engine(PROFILES["documentation"])          # or "copula" / "baseline"
df = engine.run_simulation(n_agents=280, seed=42, single_decision="disclose_income")
print(df.shape, (df["disclose_income"] == "Y").mean())
```

`single_decision` accepts a name, a list of names (run in `DECISION_ORDER` order), or
`None` for all thirteen. Pass `agents_df=` to reuse the same agents across runs.

### From Python: a run plan

The same path the app takes: patches, Page-1 parameters, several sub-runs,
transaction IDs.

```python
from src.contract.plan import RunPlan, RunMetadata, SubRun
from app.seam.execute import execute

sub = SubRun(
    result_key="categorical", population="copula", income_mode="categorical",
    seed=42, n_agents=50, decisions_to_run=("disclose_income", "donation_default"),
    decision_config_patches={}, simulation_params={}, decision_settings={},
    default_decisions_list=(), purchasing_limits=None,
)
plan = RunPlan(
    sub_runs=(sub,), messages=(),
    metadata=RunMetadata("Copula (synthetic)", "categorical only", ("categorical",),
                         (), (), 42, 50, False),
    saved_expectations=(),
)
results = execute(plan)                      # {"categorical": DataFrame}
print(results["categorical"]["donation_default"].mean())
```

`decision_config_patches` is `{decision_name: patch}`; each patch is deep-merged onto
that decision's block of a freshly loaded `config/decisions.yaml`. Wrap a value in
`Replace(...)` to overwrite a whole subtree instead of merging into it.

Run either snippet from the repository root with the virtual environment active.

---

## 8. Tests

```bash
pip install pytest               # not in requirements.txt
pytest tests/ -q -m "not slow"   # engine, seam, boundary: about 30 seconds
pytest tests/ -q                 # everything: about two minutes; everything passes
```

`slow` (registered in `pytest.ini`) marks every test module that drives the app
through Streamlit `AppTest` or the browser model (`tests/browser_sim.py`), or runs the
simulation in a subprocess. A module is marked as a whole, so the fast run also leaves
out the fast tests of such a module (the Decision 4 reference-parity file has one
`AppTest` test, for example); run the full suite before you commit.

Four kinds of test live in `tests/`:

* **Reference parity.** The modelled decisions against frozen reference extracts in
  `data/` (`stata_d1_verification.csv`, `stata_d4_verification.csv`,
  `stata_d4_flexibility_verification.csv`, `d3_doc_reference.csv`, `stata_incomes.csv`),
  participant by participant.
* **Seam.** The run plan built from a session snapshot, Monte-Carlo parity with a
  single run, the session-key registry.
* **Screens.** Streamlit `AppTest` drives the real pages: widget keys, resets, the
  Decision 4 layouts and wording. These are the slow ones.
* **Boundary.** The import rules of §6.

**Four parity tests also read the original Stata `.dta` files and fail when they are
missing**, naming the file and the constant that points at it; with
`COOPECON_ALLOW_DTA_SKIP=1` set they skip instead. The paths are the `PROF_DTA`
constants at the top of `tests/test_disclose_income_stata.py`,
`tests/test_disclose_documents.py`, `tests/test_donation_default_doc.py` and
`tests/test_research_modes_use_reference_income.py`. Obtain the files and put them
where those constants point (or point the constants at them).

---

## 9. Design decisions worth knowing

The reasons behind behaviour that looks odd at first sight.

* **The configuration files are read-only at run time.** The app never writes
  `config/*.yaml`; user changes travel to the engine as patches on a fresh copy. One
  file is therefore the only source of every constant.
* **A pinned configuration must reproduce its run.** "Use This Config" stores a sha256
  of the decision's output columns; a later complete run recomputes it and refuses to
  show results that differ. The seed, agent count, population and income mode it
  stores are the run's (`RunContext`), so a Page 1 edit between the run and the pin
  cannot change what is pinned; a seed or agent-count conflict with an already pinned
  decision is shown as an error, never reported as a save.
* **The results page shows the run, not the widgets.** Every Page 1 value the page
  and the exports need (markup, price range, market price, periods, duration, vendor
  price bounds, the agent count) comes from `RunMetadata.simulation_params` through
  `RunContext`; the same holds for the Decision 4 element flag and the donation anchor
  mix. A Page 1 edit after a run changes the next run, never the one on screen.
* **Page 1 reports an income maximum that is not above the minimum.** The three
  "b - Maximum Value ($)" fields accept any number; a value at or below the minimum
  shows one error and the discount-threshold widget is drawn as if no maximum were
  set (the engine only clamps incomes to the maximum, so the run itself never failed -
  only the screen used to).
* **An absent tab key means the tab's default, never "off".** Otherwise the same visible
  settings would give different numbers depending on whether a tab had been opened.
* **The income standard deviation divides by N − 1**, matching Stata's `egen std()`
  used for the reference z-scores.
* **Research Baseline never adds noise.** Its profile forces the donation σ to zero and
  its `pop_context` switches the other three gates off.
* **A standalone Disclose Documents run forces Decision 1 first, with its default
  probability**, because eligibility for documents depends on income disclosure. The
  plan lists it in `default_decisions_list` and the screen says so.
* **Decision 3 rescales by the population maximum of the floored draws**, so the
  maximum is computed in Pass 1 by replaying every agent's Decision-3 stream.
* **With a single vendor, Page 1 sets both price bounds to the market price.** The
  vendor score's price component then normalises to 1.0 (the only vendor is the
  cheapest available), in every sheet.
* **Customer Price in the exports** is `(1 + price_range) × (1 + platform_markup) ×
  vendor price` for Purchase Now, the bid value for a Bid, and N/A for Fixed and
  Discount customers, whose price the platform algorithm sets. `tests/test_export_pricing.py`
  pins this.
* **Decisions 9 and 10 are transaction-level.** The per-request purchase-now-or-bid
  choice and bid value are drawn inside Decision 6; the per-agent values the two slots
  still produce are not part of the documented outputs.
* **Transaction IDs are assigned after the run**, by a stable sort on
  `timestamp_hours`, so the Decision 6 and Decision 9 exports agree.
* **Export timestamps are relative to the moment the run was started.** The run
  records it once (`RunMetadata.started_at`, written by `app/simulation.py` when the
  plan is built) and every export reads it back through `RunContext.base_time`, so
  re-exporting the same run on another day gives the same timestamp strings. The
  builders take an injected base time, so the tests pass a fixed one.

---

## 10. Known limits

* Decision 4's stochastic path is verified only from draw to bin: no reference file
  carries loyalty, risk-taking or flexibility draws.
* Decision 4 has no categorical-income σ of its own; the continuous constants are
  reused, as the documentation leaves it.
* Decision 4's five segments are equal slices between a population's lowest and
  highest score, so copula and research populations are cut on different windows. This
  follows the specification.
