# Architecture

Plain-language tour of the code after the logic / UI split. Every path below is a
real file; open them alongside this page.

---

## 1. The four layers

Read them top to bottom: each layer may use the one above it, never the other way
round.

```
  src/engine  +  src/contract  +  src/decisions   THE ENGINE       (no Streamlit)
              |
  app/seam                                        THE SEAM         (no Streamlit)
              |
  app/state   +  app/reports                      STORES & REPORTS (reports: no Streamlit)
              |
  app/pages   +  app/components                   THE SCREENS      (all the Streamlit)
```

### The engine — `src/`

Everything needed to produce numbers, and nothing else.

* **`src/engine/core.py`** — the `Engine` class: one loop over agents × decisions.
  It loads `config/decisions.yaml` and `config/simulation.yaml`, imports one module
  per decision, sources the agents, and runs the two passes described in §3. It
  holds the single decision order:

  ```python
  DECISION_ORDER = ['disclose_income', 'disclose_documents', 'donation_default',
                    'rejected_transaction_defaults', 'vendor_choice_weights',
                    'purchasing_quantity', 'purchasing_frequency', 'vendor_selection',
                    'purchase_vs_bid', 'bid_value', 'rejected_transaction_option',
                    'rejected_bid_value', 'final_donation_rate']
  ```

  Thirteen entries, the same list in every population mode. A decision's *position*
  in this list is what seeds its random numbers, so the list is not just cosmetic —
  see §3.

* **`src/engine/profile.py`** — `ModeProfile` and `PROFILES`: the only genuine
  differences between the three population modes (§4).
* **`src/engine/sampling.py`** — which participants a Research-mode run uses: the
  280 in their file order, cycling (agent k = participant k mod 280; ruling R-CYC,
  no random numbers involved).
* **`src/engine/vendors.py`** — vendor prices, quality, sustainability and per-period
  quantities, drawn once per run.
* **`src/engine/postprocess.py`** — assigns the global transaction IDs after a run,
  so Decision 6 and Decision 9 exports agree.
* **`src/engine/log.py`** — the `[Copula] …` / `[DocMode] …` / `[Baseline] …` console
  lines. Console only; nothing parses them.
* **`src/decisions/<name>.py`** — one module per decision. Every decision function
  has the same shape:

  ```python
  def my_decision(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
      ...
      return {"my_decision": value}
  ```

  The four *modelled* decisions (`donation_default`, `disclose_income`,
  `disclose_documents`, `rejected_transaction_defaults`) also receive
  `pop_context=` — the population mode's name — because that is what switches their
  stochastic component on or off.
* **`src/contract/plan.py`** — the frozen description of one run (§2). It defines
  `RunPlan`, `SubRun`, `RunMetadata`, `SavedExpectation`, the deep-merge rules for
  configuration patches, and the hash used to prove a saved configuration was
  reproduced.
* **`src/contract/defaults.py`** — `DEFAULT_DECISION_VALUES` and
  `DEFAULT_DECISION_DESCRIPTIONS`: what each decision does when the user does *not*
  select it. Both the seam and the screens need them, so they live on the contract
  side, where nothing imports Streamlit. `app/pages/decision_execution.py`
  re-exports both names, so older import paths keep working.
* **`src/data/participants.py`** — the 280 experiment participants. Reads the two
  Excel workbooks the first time it is asked and caches the result. It never does
  anything at import time (the module it replaced read Excel and could call
  `sys.exit(1)` while merely being imported).
* **`src/orchestrator.py`, `src/orchestrator_doc_mode.py`,
  `src/orchestrator_baseline.py`** — three-line subclasses of `Engine`, one per
  population mode. They exist so old code and the CLI keep working:
  `Orchestrator()` is exactly `Engine(PROFILES['copula'])`.

### The seam — `app/seam/`

The seam turns "the user pressed a button" into "run these engine runs". It is
ordinary Python: no Streamlit, no globals, no writing to session state.

* **`snapshot.py`** — `take_snapshot(st.session_state)` returns a read-only
  `SessionSnapshot`: a shallow copy taken **once** per click. Everything downstream
  reads the snapshot, so no part of a run can see a value another part of the same
  run wrote.
* **`config_repo.py`** — `DecisionsConfig`: `config/decisions.yaml` and
  `config/simulation.yaml`, loaded once and **never written**. `fresh_decisions_dict()`
  hands out a deep copy for one engine run to patch.
* **`sentinels.py`** — the sigma constants, read out of the configuration file rather
  than spelled as literals in the app.
* **`build_plan.py`** — `build_run_plan(snapshot, config_repo) -> RunPlan`. This is
  the big one: it resolves the seed and agent count, decides which decisions run,
  which population and income modes apply, builds one configuration patch per
  decision, and produces the exact `st.info` / `st.caption` / `st.success` texts the
  screen will show. It is a pure function — the same snapshot always yields the
  same plan.
* **`execute.py`** — `execute(plan) -> {result_key: DataFrame}`. For each sub-run it
  samples the agents, builds a fresh `Engine`, applies the patches and the Page-1
  parameters, runs it, and finally assigns global transaction IDs.
* **`mc.py`** — Monte Carlo on the same plan: `select_mc_sub_run` picks the sub-run a
  study repeats, `write_mc_plan_file` / `load_mc_plan_file` hand it to the subprocess,
  and `run_mc_repetition(sub_run, seed)` replays it with another seed exactly as
  `execute` would.

### Stores and reports — `app/state/`, `app/reports/`

* **`app/state/saved_configs.py`** — the "Use This Config" store. When you pin a
  decision's result, this records the seed, the agent count, the population mode,
  that decision's parameters, *and* a sha256 of the columns the decision produced.
* **`app/reports/*.py`** — every number, table, preview frame and Excel/CSV byte
  string the screens display or offer for download. These modules take their inputs
  as arguments (the results frame, the vendor list, the Page-1 prices, the base time
  and period length for timestamps) and return frames or bytes. They may import `app`, but **never
  `streamlit`** — so an export can be built and checked without a browser session.

### The screens — `app/pages/`, `app/components.py`

* `page1_common_params.py` — Page 1 (population size, seed, income distribution,
  vendors, thresholds).
* `page2_decisions.py` + `decision_tabs/` — Page 2: the Overview tab and one tab per
  configurable decision.
* `decision_execution.py` — the Run buttons, and a re-export of
  `DEFAULT_DECISION_VALUES` from `src/contract/defaults.py`.
* `results/` — the results page. `results/run_context.py` is the piece to know: see §5.
* `app/simulation.py` — **the only module that connects `st.session_state` to the
  seam.** Roughly forty lines of real logic; the rest of the file is the Monte-Carlo
  subprocess runner, which builds the same run plan and hands the subprocess its
  sub-run as a plan file (`app/seam/mc.py`), so Monte-Carlo runs use exactly a single
  run's settings.

---

## 2. What one click does

Take "🚀 Run Complete Simulation" on Page 2
(`app/pages/page2_decisions.py`, key `run_complete_simulation`).

```
  Page 2 button
        |
        v
  run_combined_simulation()                     app/pages/decision_execution.py
    records custom_decisions / default_decisions, then calls
        |
        v
  run_full_simulation()                         app/simulation.py
        |
        |-- take_snapshot(st.session_state)     app/seam/snapshot.py     (read ONCE)
        |
        |-- build_run_plan(snapshot, config)    app/seam/build_plan.py   (pure)
        |        -> RunPlan(sub_runs, messages, metadata, saved_expectations)
        |
        |-- for message in plan.messages: st.info / st.caption / st.success
        |
        |-- execute(plan)                       app/seam/execute.py
        |        for each SubRun:
        |          sample_agents(population, n, seed)
        |          Engine(PROFILES[population])           <- fresh yaml load
        |          configure_engine(...)                  <- patches + Page-1 params
        |          engine.run_simulation(n, seed, decisions, agents_df)
        |        then assign_global_transaction_ids(...) on every result
        |
        |-- verify_saved_expectations(...)      re-hash each pinned decision's
        |                                       output columns; a mismatch shows an
        |                                       error and the run is discarded
        |
        |-- st.session_state.simulation_results = results
        |   st.session_state._run_metadata      = what was actually run
        |   st.session_state.vendors            = the run's vendor list
        |   st.session_state.page = 'results'; st.rerun()
        v
  Results page                                  app/pages/results/
    RunContext.from_session() reads _run_metadata + simulation_results
```

Three things are worth noticing.

**One snapshot, one plan.** The session state is read once, at the top. The builder
cannot be influenced by anything the run itself does, and a test can hand it a plain
dictionary instead of a Streamlit session.

**A plan is data.** `RunPlan` is a frozen dataclass. You can print it, store it,
compare two of them, or build one by hand. A `SubRun` is one engine run producing one
key in the results dictionary; a "Compare all × Compare both" click produces six
sub-runs (three populations × two income modes), in a fixed order.

**Nothing is written before the results exist.** The old code wrote several session
keys while the run was in progress. Now the only writes happen after `execute`
returns, so a failed run leaves the session exactly as it was.

The other Run buttons (the per-tab "Run X Only", the per-tab "Run Complete", the
Results-page "Run Complete") all funnel into the same `run_full_simulation()`.

---

## 3. Where the random numbers come from

**Unchanged by the migration.** This is the part that guarantees "same seed, same
settings → same numbers", so it is reproduced here verbatim from
`src/engine/core.py`.

```
rng_setup    = default_rng(seed)                    -> vendor attributes (Research
                                                       participants are taken in file
                                                       order, cycling - no draw, R-CYC)
rng_pass1    = default_rng(seed + 1_000_000)        -> one integers(1e9) per agent
                                                       = that agent's BASE SEED
income_rng   = default_rng(base_seed + 999_999)     -> the agent's income
decision_rng = default_rng(base_seed + position*1000)  -> that one decision, that one agent
```

`position` is the decision's index in `DECISION_ORDER` (0 for `disclose_income`,
12 for `final_donation_rate`).

Two consequences students trip over:

* **Which decisions you select does not change the numbers of the ones you did
  select.** Decision 3's stream is `base_seed + 2000` whether or not Decisions 1 and 2
  ran. That is the point of seeding by position rather than by call order.
* **An agent's income is drawn twice, identically.** Pass 1 draws it to compute the
  population statistics (median, mean, standard deviation) that some decisions need;
  Pass 2 re-draws it from the same `default_rng(base_seed + 999_999)` so the agent
  keeps the same income no matter which decisions run.

The run is two passes over the agents:

* **Pass 1** — for each agent, take a base seed from `rng_pass1` and draw the income.
  Then compute the population statistics: income median, income mean and standard
  deviation, and the extra population-level statistics Decisions 1, 2, 3 and 4 need
  (`_compute_population_stats`). Decision 3's is the maximum of every agent's floored
  donation draw (ruling R-D3), found by replaying each agent's own Decision-3 stream.
* **Pass 2** — for each agent, rebuild the income from the same base seed, then run
  the selected decisions in `DECISION_ORDER` order. Each decision's output dictionary
  is merged into the agent's state, so a later decision can read what an earlier one
  decided. **The mutated agent state is the output row.**

Adding a decision *at the end* of `DECISION_ORDER` therefore leaves every existing
number untouched. Inserting one in the middle shifts every later decision's offset
and changes every later result. See [`how-to-add-a-decision.md`](how-to-add-a-decision.md).

---

## 4. The per-mode profile

There are three population modes. After the owner rulings they differ in exactly
four fields, all listed in `src/engine/profile.py` — three that affect the run
(`pop_context`, `agent_source`, `force_donation_sigma_zero`) and the console log
prefix:

| field | Copula | Research Specification | Research Baseline |
|---|---|---|---|
| `pop_context` | `"copula"` | `"documentation"` | `"baseline"` |
| `agent_source` | `"copula"` — `TraitEngine.sample(n, seed)` | `"research"` — the 280 participants | `"research"` — the 280 participants |
| `force_donation_sigma_zero` | `False` | `False` | `True` |
| `log_prefix` | `[Copula]` | `[DocMode]` | `[Baseline]` |

Both research modes take the agents the same way (ruling R-CYC, 2026-10-07): the
280 participants in their real file order, cycling — agent k (0-based) is
participant k mod 280. N = 280 is exactly the 280; N = 1000 is three full cycles and
then the first 160; N < 280 is the first N. Nothing is bootstrapped, resampled or
permuted, so the research population does not depend on the seed. Every repeated
agent is still its own agent for the random numbers (its base seed comes from
`rng_pass1` by agent index), so Research Specification noise differs between the
copies of one participant. A Research Specification run with every σ off therefore
equals the Research Baseline run row for row at any N (`tests/test_research_cyclic_population.py`).

`pop_context` is the important one: it is handed to the four modelled decisions and
is what decides whether noise is added. The single rule lives in
`should_use_stochastic()` in `src/utils/stochastic.py`:

* `copula` → noise iff the decision's `stochastic.in_copula` flag is on (the
  "Use Normal(anchor, σ) draw in Copula mode" tick box);
* `documentation` → noise iff `stochastic.sigma_value > 0` (the Research tick box
  writes the σ there);
* `baseline` → never.

Everything else — the decision order, the row-building, the error handling, the
module chosen for each decision — is now identical across modes. Before the
migration each mode had its own copy of the loop and they had drifted apart; see
rows Q-01 to Q-07 of [`rulings-and-quirks.md`](rulings-and-quirks.md).

---

## 5. How the results page knows what it is showing

`app/pages/results/run_context.py`.

When a run finishes, `app/simulation.py` stores `st.session_state._run_metadata`:
the population mode and income mode the run actually used, the result keys, the
custom and default decision lists, the seed and the agent count — and, per result
key, the income mode each income-dependent decision actually ran with
(`decision_income_modes`). That last one matters because a complete run names its
result keys after the global income mode while Decision 4 runs with its own tab's
mode; the page labels each decision with what it really ran
(`RunContext.decision_income_label`), and `rtd_compare_both_fallback` makes it say
so when a "Compare both" Decision 4 tab had to run continuous.

The results page builds a `RunContext` from that metadata plus the result frames and
asks *it* every question about the run's shape — is this a Compare-all grid, which
frame feeds the agent details, how many decisions ran, may this cell offer "Use This
Config". It deliberately does **not** read the live Page-1 / Page-2 widgets
(`population_mode`, `income_spec_mode`, `di_income_mode`, …), because those describe
the run you are *about to configure*, not the one on screen.

---

## 6. The import boundary, and how it is tested

Two rules:

1. **Nothing under `src/` may import `streamlit` or the `app` package.**
2. **Nothing under `app/reports/` may import `streamlit`.** (It may import `app` —
   it *is* app code.)

`tests/test_import_boundary.py` enforces both, in two independent ways, for every
module it finds by walking the two directories (106 tests in total):

* **A fresh subprocess per module.** The test imports the module in a clean
  interpreter and asks the child which modules ended up in its `sys.modules`. If
  `streamlit` or `app` appears, the test fails and names the culprit. Doing this
  in-process would prove nothing: another test in the same session may already have
  imported Streamlit.
* **A static scan of the source.** `ast.walk` finds `import streamlit` /
  `from app… import …` at *any* nesting depth, including one hidden inside a
  function body — which the subprocess check can never see, but which still breaks
  the first time that function runs outside Streamlit.

The walk itself is guarded (`test_src_modules_were_discovered`,
`test_report_modules_were_discovered`), so the suite cannot silently pass by finding
no modules at all.

Run just this group:

```bash
pytest tests/test_import_boundary.py -q      # 106 passed
```

---

## 7. Running the engine without the UI

### The CLI

```bash
python scripts/run_simulation.py --agents 1000 --seed 42
python scripts/run_simulation.py --agents 5000 --decision disclose_income --format csv
python scripts/run_mc_study.py --agents 10000 --runs 500 --base-seed 1
```

`--population-mode` takes `copula`, `documentation` or `baseline`;
`--income-mode` takes `categorical` or `continuous`; `--decision` may be repeated.
`--plan-file F` (both scripts) replays the sub-run the app stored in `F` instead —
every tab setting of the app's run, with the command line's seed and agent count;
the app's Monte-Carlo button always passes one. Results land in `outputs/` (git-ignored).

### From Python: the engine directly

Use this when you want one population mode and the configuration file's own values.

```python
from src.engine.core import Engine
from src.engine.profile import PROFILES

engine = Engine(PROFILES["documentation"])          # or "copula" / "baseline"
df = engine.run_simulation(n_agents=280, seed=42, single_decision="disclose_income")
print(df.shape, (df["disclose_income"] == "Y").mean())
```

`single_decision` accepts a name, a list of names (run in `DECISION_ORDER` order),
or `None` for all thirteen. Pass `agents_df=` to reuse the same agents across runs.

### From Python: a run plan

Use this when you want the same path the app takes — patches, Page-1 parameters,
several sub-runs, transaction IDs.

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

`decision_config_patches` is `{decision_name: patch}`; each patch is deep-merged
onto that decision's block of a freshly loaded `config/decisions.yaml`. Wrap a value
in `Replace(...)` to overwrite a whole subtree instead of merging into it — that is
how the donation coefficient set replaces the file's nested blocks.

Run either snippet from the repository root with the virtual environment active,
so `import src` and `import app` resolve.

---

## 8. The journey harness (how "identical" was proved)

Documentation is not proof, so the migration was checked mechanically.

A *journey* is a scripted trip through the real app using Streamlit's `AppTest`:
set these widget keys, press that button, on this page. Seventeen journeys cover the
default screens with no run at all (`P0`), complete runs in each population × income
mode at n=50 and n=280, each of the four "Run X Only" buttons, the four stochastic
decisions with each set of σ tick boxes, and pin-then-rerun.

Each journey runs in a **fresh subprocess** and saves everything it can observe:

* every result DataFrame — as a pickle, as parquet, plus its column names, dtypes
  and `df.attrs`;
* a normalised snapshot of every rendered page (every element, its text and its key);
* every download button — label, file name, sheet names, and a normalised sha256 of
  the file contents;
* the session-state keys the run wrote.

Capture the same seventeen journeys before and after a change and compare them
element by element. A difference is either explained by a ruling or it is a
regression; there is no third option. That is what produced the verdicts summarised
in [`rulings-and-quirks.md`](rulings-and-quirks.md), and it is how any future change
to this code should be checked.
