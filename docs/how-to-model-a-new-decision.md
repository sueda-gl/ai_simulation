# How to model one of the nine default decisions

The platform has thirteen decisions, and it will always have thirteen. Four of them are
*modelled*: Decisions 1 (Disclose Income), 2 (Disclose Documents), 3 (Donation
Default) and 4 (Rejected Transaction Defaults) compute their value from each agent's
traits, with equations the research author wrote and checked against the experiment
data. The other nine (5 Vendor Choice Weights, 6 Purchasing Quantity, 7 Purchasing
Frequency, 8 Vendor Selection, 9 Purchase Now vs Bid, 10 Bid Value, 11 Rejected
Transaction Option, 12 Rejected Bid Value, 13 Final Donation Rate) run with simple
default rules that the user edits on the Overview tab of Page 2. In this repository,
"adding a new decision" always means one thing: the author sends a methodology
document for one of those nine, and you turn that slot into a modelled decision, the
way Decisions 1 to 4 were done, without changing anything else. There will never be a
fourteenth decision. (If one were ever added after all,
[`how-to-change-the-model.md`](how-to-change-the-model.md) §9 covers it.)

The running example is **Decision 5, Vendor Choice Weights**. Every name below that
contains `vendor_choice_weights`, `vcw_` or `d5` shows what a file or key *would* be
called; none of them exists for Decision 5 yet unless the text says so. For another
decision, replace the name. Files are named, not given line numbers: find a name with
`grep -n`.

Read [`README.md`](README.md) §1 to §5 and [`session-state.md`](session-state.md) §1
and §2 first. The checklist in §10 is the short version of this page.

---

## Words used on this page

| Word | What it means here |
|---|---|
| Streamlit | The web framework of the screens. Every click runs the page's Python script again from the top. Anything that must survive that re-run lives in one dictionary, `st.session_state`. A widget drawn with `key="x"` keeps its value in `st.session_state["x"]`; Streamlit may drop that entry in a run that does not draw the widget. |
| the engine | `src/`: the code that computes numbers. It never imports Streamlit. |
| the seam | `app/seam/`: plain Python between the screens and the engine. On a Run click it copies the session state once (the *snapshot*), builds a *run plan* from the copy and runs the engine. It writes nothing back. |
| patch | A small dictionary the seam merges onto a fresh copy of a decision's block of `config/decisions.yaml` for one run. A user's setting reaches the engine this way; the file itself is never written. |
| widget key | The `key=` of a widget. Streamlit owns it and may drop it. Decision-tab widget keys contain `_tab_` (`dd_tab_income_mode`). |
| canonical key | The session-state key that holds a setting for the rest of the app and for the seam (`dd_income_mode`). The tab writes it right after the widget, on every run; it is never dropped. |
| the registry | `app/state/registry.py`: the list of every session-state key the app uses, with its default, the function that creates it, and whether it reaches the engine. A test refuses any key the list does not hold. |
| golden copy | A recorded output of the app: result tables, export tables, everything on screen, stored in `tests/golden/` and `tests/golden_screens/`. A golden test runs the app again and compares, cell by cell. |
| parity net | The three golden tests together: `tests/test_golden_runs.py`, `tests/test_golden_screens.py`, `tests/test_complete_run_gate.py`. They fail when anything moved; you decide whether the move was intended. |
| pass 1, pass 2 | The engine walks over the agents twice. Pass 1 gives each agent its random seed and its income, then computes population statistics (a mean, a maximum over all agents). Pass 2 runs the decisions, agent by agent. |
| z-scoring | Standardising a value: (value − mean) / standard deviation. The modelled decisions use the 280 participants' mean and *sample* standard deviation (divided by N − 1, like Stata's `egen std()`), stored as constants in the YAML file. |
| σ (sigma), noise | The optional random draw `Normal(score, σ)` that a modelled decision adds to its deterministic score. |
| `.dta` | A Stata data file. |
| reference extract | A small CSV file in `data/`, frozen from the `.dta` file or from the document's commands: the numbers the code must reproduce, participant by participant. |
| population modes | Copula (synthetic agents drawn from a fitted model of the 280), Research Specification (the 280 participants, noise allowed), Research Baseline (the 280 in file order, never noise). README §4. |

---

## 0. What "modelling a decision" means

### What arrives from the author

1. **A methodology document** (`.docx`). It reviews the literature, states the
   equations, gives the Stata commands that compute every variable, and shows the
   results: regression tables, frequency tables (often by allowance level), figures
   (usually histograms of the score).
2. **A Stata data file** (`.dta`), one row per experiment participant (280 rows): the
   column `participantid`, the raw survey and experiment columns, and usually the
   model's own computed columns. (Decision 2's file has `weighted_dd_categorical`,
   `dd_deterministic_cont`, `disclosedoc_categorical` and more.)
3. **Usually, new sections of the platform documentation** for the decision's tab and
   exports: "Tab Controls Reference", "Default Parameter Values", "Output",
   "Downstream Impact", "Excel Export Structure". Decisions 1 to 4 have such sections
   in `docs/reference/COOPECON AI Agent Documentation 2026-10-07.docx`. If they do not
   arrive, ask for them before you build the tab and the exports (§6, §7).

### The one rule

**The document is the source of truth for the model. The `.dta` file is how you prove
that the code reproduces the document. The platform documentation (`docs/reference/`)
says what the app must show and export.**

When the code and the document disagree, the document wins. When the document
disagrees with its own data, do not choose: ask the author. Decision 2 is the example:
two coefficients printed in its Equation 3 were transcription slips; the values the
code uses reproduce the corrected data file (`... - CORRECTED.dta`), and
`config/decisions.yaml` shows both, with a comment
(`# CORRECTED (was -0.022306825775166)`).

### Where a modelled decision lives: Decision 2, layer by layer

This is the trail you will repeat. Open each file once before you start.

| Piece | Decision 2 today | Decision 5 (example names) |
|---|---|---|
| Methodology document | not in the repository | `docs/reference/decisions/Decision 5 - Vendor Choice Weights <date>.docx` |
| Data file | `Stata_File_Decision 2_260626 - CORRECTED.dta`, outside the repository; its path is `PROF_DTA` in `tests/test_disclose_documents.py` | `data/stata/<file as received>.dta` (§1.3) |
| Reference extract | the document's tables, written into the test; `data/stata_incomes.csv` | `data/d5_doc_reference.csv`, written by `experiments/d5_doc_reference.py` |
| Constants | `config/decisions.yaml`, block `disclose_documents:` | block `vendor_choice_weights:` (today it holds only `uses_global_parameters: []`) |
| Traits it reads | `config/trait_requirements.yaml`, `disclose_documents:` | `vendor_choice_weights:` (today `placeholder_trait`) |
| Model code | `src/decisions/disclose_documents_stochastic.py` | `src/decisions/vendor_choice_weights.py`, rewritten in place |
| Engine lists (`src/engine/core.py`) | `POP_CONTEXT_DECISIONS`, `STOCHASTIC_MODULE_DECISIONS`, a block in `_compute_population_stats`, `dd_cont_stats` in `RUN_SCRATCH_KEYS` | `POP_CONTEXT_DECISIONS`; the rest only if needed |
| Default when unselected | `DEFAULT_DECISION_VALUES` and `DEFAULT_DECISION_DESCRIPTIONS`, `src/contract/defaults.py` | unchanged (§5) |
| Tab on Page 2 | `app/pages/decision_tabs/disclose_documents.py`, using `_controls.py` and `_disclosure.py` | `app/pages/decision_tabs/vendor_choice_weights.py` |
| Session keys | the `dd_*` rows of `app/state/registry.py` | `vcw_*` rows |
| Seam | `app/seam/build_plan.py`: `SCALAR_FIELDS`, `INCOME_MODE_RULES`, `_SIGMA_PREFIX`, `build_disclose_documents_patch`, `build_decision_patches`; `app/seam/sentinels.py` | `build_vendor_choice_weights_patch` and its rows |
| Results view | `app/pages/results/visualizations/disclosure_viz.py`, `render_disclose_documents` | `vendor_viz.py`, `render_vendor_choice_weights` (exists; rewrite) |
| Report builders | `app/reports/disclosure.py`; columns in `agent_level.py` and `transaction_level.py` | `app/reports/vendor.py` (Decision 5's builders already live there) |
| Tests | `tests/test_disclose_documents.py`, `tests/test_disclose_documents_gate.py`, rows in `tests/test_config_is_single_source.py`, golden scenarios and journeys | `tests/test_vendor_choice_weights_doc.py` and the same kinds of rows |

Decisions 1 to 4 also have a second group of pieces: "Use This Config" pins, the gate
on "Run Complete Simulation", the comparison grids. They are needed only if the new
tab can produce more than one configuration per click, and they are listed in §7.5.
Build §1 to §7.4 first.

### Where the nine decisions' behaviour lives today

Check this before you choose the file the model goes into: several of the nine are not
computed by their own module.

| Decision | What actually computes it today |
|---|---|
| 5 Vendor Choice Weights | `src/decisions/vendor_choice_weights.py`. It returns a dict `{price, quality, proximity, sustainability}`; Decision 6 reads it to choose the agent's preferred vendor. |
| 6 Purchasing Quantity | `src/decisions/purchasing_quantity.py`: the purchase requests, their timestamps, their vendor. |
| 7 Purchasing Frequency | `purchasing_frequency.py`: quantity / periods, from Decision 6's output. |
| 8 Vendor Selection | `vendor_selection.py` only reports the preferred vendor Decision 6 already chose. |
| 9 Purchase Now vs Bid, 10 Bid Value | Per purchase request, **inside Decision 6**: `_enrich_purchase_requests` calls `purchase_vs_bid_single` and `generate_single_bid_value` with Decision 6's random stream and an empty `params`, and Decision 9 uses the Overview probability whether it is selected or not. The slots 9 and 10 still compute one value per agent, but nothing reads those values (README §9). A model for 9 or 10 goes into those two per-request functions. It draws from Decision 6's stream, so drawing a different number of values per request moves every later draw of that agent's Decision 6 (later requests, bids). |
| 11 Rejected Transaction Option, 12 Rejected Bid Value | They return the Overview value, selected or not. |
| 13 Final Donation Rate | `final_donation_rate.py` copies Decision 3's rate when Decision 3 ran. But the rate each request carries is set inside Decision 6, which runs before Decision 13, so it comes from `donation_default` (or 0.10). Where a Decision 13 model must act is a question for the author. |

---

## 1. Before writing code

### 1.1 Read the document and make two lists

Read the whole document once without coding. Then write two lists (a spreadsheet is
fine) and keep them to the end:

* **Every number it states**, with where it is (section, page or table): every
  coefficient, intercept, mean, standard deviation, regression output, table cell,
  percentage and histogram bar you can read. Each one is recomputed in §4.
* **Every variable it uses**, with its Stata name (`extraversionbig5`,
  `totalallowance`, ...), and whether it is a raw input, a value computed from raw
  inputs, or the result.

### 1.2 Map each variable to a column the agent carries

An agent is a dictionary, `agent_state`. Before any decision runs it holds the traits
below (`src/data/participants.py` reads them from the two workbooks in `data/`, and
`config/trait_requirements.yaml` says which), plus four values the engine adds. The
second column is the name in the reference `.dta` files. In the Decision 1 and
Decision 4 files, every column below that the file has equals the agent's column,
participant by participant (checked on 2026-10-22).

| Agent column (exact spelling) | Stata name | Notes |
|---|---|---|
| `Agreeable` | `agreeable` | survey workbook |
| `Assigned Allowance Level` | `assignedallowancelevel` | 1 to 5, stored as a float; experiment workbook |
| `ConscientiousnessBig5` | `conscientiousnessbig5` | survey |
| `Education` | `education` | 1 or 2; survey |
| `ExtraversionBig5` | `extraversionbig5` | survey |
| `Group_experiment` | `group` | `HighSub`, `MidSub` or `NoSub`. Both workbooks have `Group`; the merge names the experiment one `Group_experiment` |
| `Honesty_Humility` | `honesty_humility` | survey |
| `NeuroticismBig5` | `neuroticismbig5` | survey |
| `OpennessBig5` | `opennessbig5` | survey |
| `ReligiousAffiliation` | `religiousaffiliation` | survey |
| `ReligiousService` | `religiousservice` | survey |
| `Study Program` | `studyprogram` | experiment |
| `Study Program Category` | `studyprogramcategory` | research agents only: the copula does not carry it (Decision 3 maps it from `Study Program`) |
| `TWT+Sospeso [=AW2+AX2]{Periods 1+2}` | `twtsospesoaw2ax2periods1` | Stata cut the name at 32 characters |
| `stdactions` | `stdactions` | Decision 4 file only; merged from `data/stata_stdactions.csv` |
| `income` | `income` | Research modes: the fixed reference income of `data/stata_incomes.csv`. Copula: drawn from the Page 1 distribution, inside the band of the agent's allowance level |
| `actual_allowance` | `totalallowance` | 12, 32, 72, 128 or 200 credits, from the level (`ALLOWANCE_CREDIT_MAPPING`, `src/engine/income.py`) |
| `index`, `agent_id` | – | the row (from 0) and the agent number (from 1) |

The agent also carries everything the decisions before it returned in the same run:
Decision 5 can read `customer_type`, `disclose_income`, `donation_default` and the
Decision 4 columns when those decisions ran. A decision can never read a later one,
because the order is fixed (§3.4).

If the document uses a variable that is in neither list, stop. A new trait means a
change to `config/trait_requirements.yaml`, to the data merge, and a retrained copula
(`scripts/train_copula.py`), which moves every Copula number of every decision. That
is an owner decision.

### 1.3 Put the files where the repository expects them

Today none of the decision documents is in the repository, and the `.dta` files are
outside it. Four tests point at them with a constant `PROF_DTA` that holds an absolute
path in one person's Downloads folder: `tests/test_disclose_income_stata.py`,
`tests/test_disclose_documents.py`, `tests/test_donation_default_doc.py`,
`tests/test_research_modes_use_reference_income.py`. Do better for the new decision:

* **The document** goes in `docs/reference/decisions/` (create the folder), file name
  as received, with its date. Add a row to `docs/reference/README.md`. A newer version
  replaces the old file, as for the platform documentation.
* **The data file** goes in `data/stata/` (create the folder), file name as received.
  The test's constant is then `PROF_DTA = os.path.join(REPO, "data", "stata",
  "<file>.dta")`, which works on every machine. The files are small (about 350 KB),
  and the same 280 participants' workbooks are already committed in `data/`, but ask
  the owner and the author before you commit participant data.
* **If the data file may not be committed**, keep it outside, point `PROF_DTA` at it,
  and write in `tests/README.md` the exact file name and whom to ask for it. The
  frozen extract (§1.5) is committed either way, so the participant-by-participant
  test always runs. Only the test that rebuilds the extract needs the `.dta`. It
  calls `require_reference_file` (`tests/conftest.py`): it fails naming the file when
  the file is missing, and skips instead when `COOPECON_ALLOW_DTA_SKIP=1` is set.

### 1.4 Check the data file against the repository's data

Load it with `pd.read_stata(path, convert_categoricals=False)` and check, before
anything else:

* 280 rows, and `participantid` in the engine's order. The Decision 1, 2 and 4 files
  are exactly in that order. The engine's participant IDs are
  `P.merged().loc[P.original_data().index, "Participant ID"]`
  (`from src.data import participants as P`).
* Every trait the model uses equals the agent's column (§1.2).
* `income`, if the file has it, equals `data/stata_incomes.csv`. The Research modes
  run with that one fixed income. If the new file's income differs, stop and ask the
  author: continuous-income results cannot match otherwise.

### 1.5 Freeze a reference extract

Write a script that computes the numbers the code must reproduce and saves them in
`data/`. The pattern is `experiments/d3_doc_reference.py`, which writes
`data/d3_doc_reference.csv`. For Decision 5: `experiments/d5_doc_reference.py`,
writing `data/d5_doc_reference.csv`. The script:

* takes the `.dta` path as its first argument (with a default) and the output path as
  its second;
* has a `build(dta_path)` function that returns the extract, so the test can call it;
* writes one row per participant (per income mode, if the model has two), keyed by
  `participantid`, with the model's intermediate values and its result;
* copies values from the `.dta` where the file has the model's own columns, and
  re-implements the document's Stata commands from the raw columns where it does not
  (Decision 3 had no model columns: the script re-runs the regressions with numpy);
* writes with `float_format="%.17g"`; the test reads with
  `float_precision="round_trip"`, so no digit is lost.

The Decision 1 and Decision 4 extracts (`data/stata_d1_verification.csv`,
`data/stata_d4_verification.csv`) have no script in the repository. Do not repeat
that: the script is the record of how the extract was made.

---

## 2. The constants

Every constant of the model goes into the decision's own block of
`config/decisions.yaml` (`vendor_choice_weights:`), with a comment on each value: the
document revision and section it comes from, or how it was derived. Decision 2's block
is the one to copy: the coefficients, the intercept, `z_scoring` (mean and standard
deviation per trait), `categorical_intercepts` (`level_1` to `level_5`), the
`stochastic` block. (The key `uses_global_parameters` is in every block, but no Python
code reads it.)

Rules:

* **Read every constant strictly.** In the module, use `require(params, DECISION,
  "key", "subkey")` from `src/decisions/_shared.py`. A missing key raises
  `MissingDecisionKey`, naming the decision and the dotted key. Never write
  `params.get("key", <number>)`.
* **No copy of a file value in the code.** If the app or a test needs a constant by
  name, read it once at import with `file_block("vendor_choice_weights")`, which goes
  through the one loader in `src/config.py`. The same holds for the tab: it reads every
  number it prints from the block.
* **Only switches may be absent**: `income_mode` (absent means categorical) and the
  `stochastic` block (absent means no noise).
* **Add the module to `tests/test_config_is_single_source.py`**: a row in `MODULES`,
  and a pair of tests like `test_disclose_documents_runs_on_the_full_block` and
  `test_disclose_documents_missing_key_raises`, with the list of keys the module must
  refuse to run without. The first part of that file scans each listed module for a
  literal copy of a value of its block (a `.get` fallback, a dict literal, a
  module-level number; 0, 1 and −1 are never flagged).

**How the file reaches a run.** The file is read once into `DecisionsConfig`
(`src/config.py`). For each engine run the seam takes a deep copy
(`fresh_decisions_dict()`), merges the run's patches onto it (`apply_patches` in
`src/contract/plan.py`; a value wrapped in `Replace(...)` replaces a whole subtree)
and hands the copy to the engine, which gives each decision its block as `params`.
Nothing in `app/` writes the file:
`tests/test_stochastic_widgets_no_reset.py::test_no_yaml_writer_under_app` and
`::test_no_ui_action_writes_the_config_file` hold that.

---

## 3. The decision module

### 3.1 Which file, which function

Rewrite `src/decisions/vendor_choice_weights.py` in place and keep the function name
`vendor_choice_weights`: the engine imports `src.decisions.<name>` and takes the
function of the same name (`Engine.__init__`, `src/engine/core.py`). Do not use the
`<name>_stochastic.py` spelling of Decisions 1 and 2; it needs an entry in
`STOCHASTIC_MODULE_DECISIONS` and buys nothing.

```python
DECISION = "vendor_choice_weights"

def vendor_choice_weights(agent_state: dict, params: dict, rng,
                          simulation_config: dict = None, **kwargs) -> dict:
    ...
    return {"vendor_choice_weights": weights, "vendor_choice_weights_score": score, ...}
```

**Warning.** If the module fails to import, or the function name is wrong, the engine
prints "Warning: Could not load decision module ..." and runs without the decision.
The run "works" and the column is simply missing. After every change, check that
`Engine("baseline").get_available_decisions()` lists the decision.

### 3.2 What the function receives

| Argument | What it holds |
|---|---|
| `agent_state` | the agent's columns (§1.2) plus every earlier decision's output. Read traits with `agent_state["ExtraversionBig5"]`, so a missing trait raises. (Decision 2 still reads `agent_state.get("ExtraversionBig5", 0)`; do not copy that.) |
| `params` | the decision's block of the patched configuration. It is the engine's live dictionary for the whole run: **never write into it**, or the next agent sees your change. |
| `rng` | a numpy random `Generator` for this agent and this decision only: its seed is the agent's base seed + position × 1000. Decision 5 is at position 4 (counted from 0), so its offset is 4000. **Draw only from it.** |
| `simulation_config` | the run's configuration. Page 1 values: `get_simulation_param(simulation_config, "market_price", 100.0)` (`src/engine/config_access.py`). Pass 1 results: `income_stats` (mean and sample standard deviation of the run's incomes), `income_median`, and your own pass 1 key (§3.6). Also `vendors`, `simulation_seed`, `default_decisions`, `default_decisions_list`. |
| `pop_context` (in `kwargs`) | `"copula"`, `"documentation"` (Research Specification) or `"baseline"`. Only the decisions listed in `POP_CONTEXT_DECISIONS` receive it (§3.7). |

### 3.3 What it returns

A dictionary. Each key becomes a column of the result table and is merged into
`agent_state` for the decisions after it.

* **Keep the decision's own column and its shape**, because other code reads it. For
  Decision 5 that is `vendor_choice_weights`: a dict with exactly the four keys
  `price`, `quality`, `proximity`, `sustainability`, summing to 1 (the platform
  documentation says so). Decision 6 reads it (`purchasing_quantity.py`), and so do
  the vendor score (`calculate_vendor_score_with_breakdown`,
  `src/vendor_attribute_generator.py`) and every export (`vendor_weights_or_none`,
  `app/reports/_common.py`). Always return all four keys: the readers do not agree on
  what a missing key means (0.0 in `app/reports/vendor.py`, 0.25 in
  `app/reports/agent_level.py`).
* **Prefix the model's extra columns with the decision's name**
  (`vendor_choice_weights_score`, `vendor_choice_weights_sigma_used`, ...), like
  `disclose_documents_*`. If the decision is ever made pinnable (§7.5), the pin store
  selects its columns by that prefix, as it does for Decisions 1 and 2.
* **Return the same keys on every path of the model** (Decision 2 attaches the same
  columns to every return), so the table has no holes. The default path (§3.5) may
  return only the decision's own column, as Decisions 1 to 4 do.
* **Return the intermediate values the document shows**: the z-scores, the composite,
  the deterministic score, the value after the draw, the σ used, the income mode. The
  reference test and the exports need them.

### 3.4 Rules, and what enforces each

| Rule | Enforced by |
|---|---|
| Never `import streamlit`, never import from `app`, not even inside a function | `tests/test_import_boundary.py` (it finds new modules by itself) |
| No literal copy of a YAML value; a missing key raises | `tests/test_config_is_single_source.py`, once the module is in `MODULES` |
| Draw random numbers only from `rng`; never `np.random.*` or `random.*` | convention; a run that does not repeat itself fails the golden tests |
| Raise on bad input; never substitute a placeholder and go on | convention |
| Use the shared helpers: `require`, `file_block`, `z_score_trait`, `categorical_intercept`, `normalise_income_mode` (`src/decisions/_shared.py`), and the one noise gate `should_use_stochastic` (`src/utils/stochastic.py`) | convention |
| Never reorder `DECISION_ORDER` (`src/engine/core.py`): a decision's position seeds its random numbers | `tests/test_decision_order.py` (the length and the copies); the golden tests (the numbers) |
| Do nothing at import time except `file_block` | convention |

### 3.5 The unselected case

When the user does not select the decision, it must behave exactly as it does today.
Every modelled decision starts with the same branch:

```python
if DECISION in (simulation_config or {}).get("default_decisions_list", []):
    entry = simulation_config.get("default_decisions", {}).get(DECISION)
    ...   # today's default rule, unchanged
    return {DECISION: ...}
# otherwise: the model
```

For Decision 5 the default rule is the one in today's module: return
`entry["weights"]` when the entry is a `checkbox_selection` (the seam computes the
equal weights of the ticked attributes in `collect_decision_settings`,
`app/seam/build_plan.py`). Copy the branch before you rewrite the module. Note the two
kinds of run: in a complete run an unselected decision is in `default_decisions_list`;
in a "Run X Only" run the list is empty and only X runs, so the model runs.

### 3.6 Population statistics in pass 1

If the model needs a number over all agents (a mean, a standard deviation, a minimum
or a maximum), compute it in pass 1:

1. add a block to `Engine._compute_population_stats` (`src/engine/core.py`) that runs
   only when the decision runs and is not in `default_decisions_list` (as Decisions 3
   and 4 do);
2. store the result in `self.simulation_config["vcw_population_stats"]` (example
   name);
3. add that key to `RUN_SCRATCH_KEYS`, which clears it at the start of every run.

Decision 2's continuous composite (`compute_continuous_dd_stats`) is the simple
pattern. If the statistic depends on the decision's own random draws, as Decision 3's
population maximum does, replay each agent's pass 2 draws exactly: the same agent
state, `np.random.default_rng(base_seed + decision_index * 1000)`, the same draws in
the same order. `compute_donation_population_max` (Decision 3) and
`compute_rtd_population_stats` (Decision 4, which always draws its five noise values
so the replay stays aligned) are the patterns.

### 3.7 The engine's lists, and two conventions that matter

* **`POP_CONTEXT_DECISIONS`** (`src/engine/core.py`): add the decision if it has
  noise. Without it the function never receives `pop_context`, and a module written
  like Decisions 1, 2 and 4 (`kwargs.get("pop_context", "documentation")`) treats
  every run as Research Specification, Research Baseline included. Reading
  `kwargs["pop_context"]` instead makes a missing value fail.
* **The noise gate.** One rule for every decision, `should_use_stochastic`: Copula
  draws if `stochastic.in_copula` is on; Research Specification draws if
  `stochastic.sigma_value > 0`; Research Baseline never draws. For Decisions 1, 2 and 4
  `sigma_value` is only an on/off signal: the size of σ is `sigma_overall` (or
  `sigma_quintile[level]`) × `scale_factor`, from the block. (For Decision 3
  `sigma_value` is the σ itself; do not copy that.) The YAML ships `sigma_value: 0`,
  but in the app the Research Specification tick box is on by default, so an app run
  in Research Specification adds noise unless the user clears it. An engine run from
  Python does not.
* **The income mode.** The strings are `"Categorical only"`, `"Continuous only"` and
  `"Compare both"` (`MODE_OPTIONS`, `app/pages/decision_tabs/_controls.py`).
  `normalise_income_mode(mode)` turns any spelling into `"categorical"` or
  `"continuous"`; Decisions 1 to 3 default to categorical, Decision 4 to continuous.
  Categorical mode works from the allowance level (`categorical_intercepts.level_1` to
  `level_5`); continuous mode from the agent's income, z-scored with the run's
  `income_stats`. In a complete run Decisions 1 to 3 run with the run's income mode and
  Decision 4 with its own tab's mode (README §5). Decide with the author which
  convention the new decision follows, and write it in the YAML comment.

### 3.8 Run it on the 280 before any screen exists

From Python, at the repository root:

```python
import contextlib, io
from src.data import participants as P
from src.engine.core import Engine
from src.engine.profile import PROFILES

agents = P.original_data()                       # the 280, in file order
pids = P.merged().loc[agents.index, "Participant ID"].astype(int).to_numpy()
engine = Engine(PROFILES["baseline"])            # never adds noise
engine.config["vendor_choice_weights"]["income_mode"] = "Categorical only"   # if it has one
with contextlib.redirect_stdout(io.StringIO()):
    df = engine.run_simulation(len(agents), 42, single_decision="vendor_choice_weights",
                               agents_df=agents.copy())
df.insert(0, "participantid", pids)
```

From the command line (the file lands in `--output-dir`, by default `outputs/`, which
git ignores):

```bash
python scripts/run_simulation.py --agents 280 --seed 42 --population-mode baseline \
    --decision vendor_choice_weights --format csv
```

At exactly 280 agents, Research Baseline and Research Specification use the same
agents in the same order (Specification samples randomly only below or above 280), so
with σ off they give the same rows. Compare in Research Baseline, which never adds
noise. Copula agents are synthetic and cannot be compared participant by participant.

---

## 4. The reference test

Create `tests/test_vendor_choice_weights_doc.py`, modelled on
`tests/test_donation_default_doc.py`. Do not mark it `slow` (it does not start the
app). It holds six groups of tests:

1. **The extract is what the script makes.** Load `experiments/d5_doc_reference.py`
   with `importlib.util`, call `build(PROF_DTA)` and compare with the frozen CSV
   (differences below 1e-12). The test starts with
   `require_reference_file(PROF_DTA, "PROF_DTA in tests/test_vendor_choice_weights_doc.py")`.
2. **The configuration equals the document.** Every coefficient and constant in the
   block equals the number the document prints
   (`test_config_coefficients_equal_the_doc` in the Decision 3 test).
3. **The engine reproduces the extract, participant by participant**, through the real
   engine (§3.8), in every income mode, in Research Baseline and Research
   Specification with σ off. Match rows by `participantid`, not by position; count the
   matches and assert 280. (Decision 3 allows 1e-6 on a score; Decision 2 compares
   every Y/N exactly.)
4. **The document's tables and figures.** Recompute each table from the engine's
   output, such as the counts and percentages by allowance level (`DOC_CATEGORICAL` in
   `tests/test_disclose_documents.py`). For a histogram, use Stata's default number of
   bins, min(√N, 10·log10 N) cut to an integer: 16 at N = 280. `rtd_stata_bins` in
   `app/reports/rtd.py` computes it and `tests/test_rtd_histogram_stata_parity.py`
   compares bar heights with the document's figure.
5. **The noise.** σ on in Research Specification changes the values; σ requested in
   Research Baseline changes nothing; Copula draws only with its tick box
   (`test_noise_gate_and_population_rescale` is the pattern).
6. **The default branch.** With the decision in `default_decisions_list`, it returns
   the Overview default (`tests/test_disclose_documents_gate.py` is the pattern).

**How Decision 2 was checked against its document (October 2026).** The scripts were
one-off and are not in the repository. This is what they did; do the same before you
write the test.

* *Recompute every number of the document from the raw `.dta` columns*, with numpy
  only: the meta-analytic weights, each reduced-form coefficient, the z-scoring means
  and standard deviations (against the YAML), each composite (against the `.dta`'s
  own column, as a maximum difference), the regression that gives the per-level
  intercepts (coefficients and standard errors against the document's table), every
  cell of the frequency tables, the correlations, the σ values, and the 16-bin
  histograms behind the document's two figures. The output was one table: item, where
  in the document, document value, recomputed value, difference, ok. Every "not ok"
  row is either a bug in the code or a question for the author.
* *Run the real engine on the 280* in Research Specification and Research Baseline,
  in both income modes, alone and after Decision 1. Check the alignment first (traits
  and income equal the `.dta`, row by row); then compare every intermediate column
  with the `.dta`'s (maximum absolute difference), the result per participant, the
  counts per allowance level, the eligibility gate, and that the two research modes
  give the same rows.
* *Graph by graph*: draw the engine's histogram beside the one computed from the
  `.dta`, with the same bins.

---

## 5. The default for the unselected case

Keep it. Decisions 1 to 4 kept theirs:

* their entries in `DEFAULT_DECISION_VALUES` and `DEFAULT_DECISION_DESCRIPTIONS`
  (`src/contract/defaults.py`) are still there: a probability for 1 and 2, the number
  0.10 for 3, a priority list for 4;
* the Overview tab still offers an editor for them while they are not selected:
  `app/pages/decision_tabs/default_config.py` draws every unselected decision by the
  type of its entry;
* the results page still shows the description on the "Default Behavior" line of a
  decision that ran with its default (`get_dynamic_description`,
  `app/pages/results/visualizations/viz_helpers.py`);
* the module's default branch (§3.5) still applies it.

So for Decision 5, leave its entry (`checkbox_selection`, the four attributes, equal
weights), its description and the Overview checkboxes as they are. Change them only if
the author's platform documentation changes the default behaviour. Both tables must
keep one entry per decision; `tests/test_decision_order.py` checks it.

---

## 6. The tab on Page 2

### 6.1 How a tab is chosen and built

`render_decision_tab(decision_name)` in `app/pages/decision_tabs/__init__.py` picks a
tab: a module of its own for Decisions 1 to 4 and 10, a short note for 11, and
`render_generic_decision_tab` ("This decision currently uses default values") for
every other decision. Write `app/pages/decision_tabs/vendor_choice_weights.py` with
`render_vendor_choice_weights_tab()` and
`initialize_vendor_choice_weights_session_state()`, and add one `elif` that calls the
first. Follow `disclose_documents.py`, in this order:

1. apply a pending reset (§6.5);
2. run the initialiser: it seeds every *canonical* key that is absent, from the YAML
   block (`read_yaml_file`, never a copied number);
3. draw the controls (§6.2), the formula display (every number read from the block;
   the Decision 2 tab still carries copies of its σ values and fallbacks, do not copy
   that), and the override and reset buttons (§6.3);
4. end with `render_simulation_buttons(decision_name=..., selected_decisions=...)`:
   "Run X Only" and "Run Complete Simulation".

### 6.2 The shared controls (`_controls.py`)

The four modelled tabs draw their σ controls and their income-mode radio through one
component, `app/pages/decision_tabs/_controls.py`: `render_income_mode`,
`render_stochastic_tick_boxes`, `render_sigma_controls`. Each tab describes itself in a
`TabControls` spec: its keys, its σ values for the "Effective σ" lines, its wording.
The keys come from the tab's row of `CONTROL_KEYS` in the registry:

```python
"vendor_choice_weights": ControlKeys(
    dp="vcw_", wp="vcw_tab_", sp="vcw_", suffix="", research="sigma_enabled",
    coefficient="vcw_scale_factor", income_widget="vcw_tab_income_mode",
    income="vcw_income_mode"),
```

A `CONTROL_KEYS` row commits the tab to the whole set: both tick boxes, the σ mode
radio, the coefficient slider, the five per-level sliders and the income radio.
`tests/test_session_key_registry.py` requires a `TAB_WIDGET_CANONICAL` entry for each.
If the new decision has no income mode or no σ, do not add a row; draw the controls it
has in the tab module, under the same rules (§6.5).

### 6.3 The twin spec (`_disclosure.py`)

Disclose Income and Disclose Documents share an intercept override, a whole-tab reset
and a "Reset Intercept" button, in `app/pages/decision_tabs/_disclosure.py`; each is
described by a `DisclosureTab` spec and a row of `DISCLOSURE_KEYS` in the registry. If
the new tab has the same shape (one intercept the user may override, the two reset
buttons), add a row `DisclosureKeys(dp="vcw_", wp="vcw_tab_", suffix="")` and a spec
in the tab module, like `DD_TAB`. If not, write the tab's own reset following §6.5.

### 6.4 The registry rows

For every key the tab uses, add a `KeySpec` to `app/state/registry.py` (the tables
`_DD_MIRRORS` and `_DD_WIDGETS` are the pattern), and add the new tables to
`REGISTRY`:

* **canonical keys** (`vcw_income_mode`, `vcw_sigma_enabled`, `vcw_sigma_in_copula`,
  `vcw_sigma_strategy`, `vcw_scale_factor`, `vcw_quintile_scale_factors`, one per
  model setting): category `page2-mirror`, `engine_input=True`, `initialised_by`
  `"decision_tabs.vendor_choice_weights.initialize_vendor_choice_weights_session_state"`;
* **widget keys** (`vcw_tab_*`, the override input, the buttons): category
  `page2-widget`, `is_widget=True`, default `WIDGET` for a button;
* **the widget-to-canonical table**: one `TAB_WIDGET_CANONICAL` entry per widget that
  `restore` seeds, `(canonical key, None)`, or `(canonical key, "1".."5")` for the
  per-level sliders;
* **the reset**: the prefix in `PREFIX_DELETE_PREFIXES`, and
  `_vcw_reset_to_defaults_pending` in `_RESET_PENDING_FLAGS` if you use the twin spec;
* **results-page widgets** that take a `key=` (a chart, a "Use This Config" button):
  `_RESULT_WIDGETS`.

Choose a short prefix that starts no other key; `vcw_` starts none today. Do not use
`vendor_choice_weights_`: the reset deletes every key with the prefix, and the
Overview's keys (`vendor_choice_weights_default_params`, ...) start with it.

Then run `.venv/bin/python -m pytest tests/test_session_key_registry.py -q`. It
refuses a key the code uses and the registry lacks, an entry nothing uses, a wrong
widget flag, an `engine_input` key the seam never names, a `restore()` of a widget
missing from `TAB_WIDGET_CANONICAL`, and a prefix delete missing from
`PREFIX_DELETE_PREFIXES`. It reads the code, not the screen, so it cannot tell whether
your tab really draws a control.

### 6.5 The session-state rules

[`session-state.md`](session-state.md) §2 explains them; here they are as rules.

1. **Two keys per setting.** Draw the widget with its widget key (`vcw_tab_...`) and
   write the canonical key (`vcw_...`) right after it, on every run.
2. **Seed right before the widget, and only when absent.** Call
   `restore(widget_key, default)` (`app/pages/decision_tabs/_tab_state.py`) right
   before the widget. It returns the live widget key; otherwise the canonical value;
   otherwise the default. It writes only when the key is absent.
3. **Draw every keyed widget through `stateful(st.<widget>, ..., key=...)`**
   (`app/state/widgets.py`), never with `value=`, `index=` or `default=`.
4. **Never push a value again.** Do not assign the stored value to the widget key on
   every run: Streamlit then sends it to the browser again and fast clicks are lost.
   To correct a value, use `seed_if_changed`, which writes only when it differs.
5. **Never write a widget key in a run that does not draw its widget.** The stale
   value comes back later, in front of the user's.
6. **Reset by prefix, at the top of the next run.** The button only sets
   `_vcw_reset_to_defaults_pending`. At the top of the next run, before any widget of
   the tab exists, delete every key that starts with `vcw_`, then write from the YAML
   block the canonical keys and the widget keys this run will draw (not the per-level
   sliders while the σ mode is "uniformly"). `apply_pending_reset` in `_disclosure.py`
   does exactly this.
7. **An absent key means the tab's default, never "off".** The seam must give the same
   numbers whether or not the user ever opened the tab.

### 6.6 The seam rows

The seam turns the canonical keys into a patch on the decision's block, in
`app/seam/build_plan.py`:

* **Scalar settings**: one `ScalarField` row per setting in `SCALAR_FIELDS` (session
  key, path in a saved record, path in the YAML block, optional transform), and the
  decision in `_PATCH_CONTAINERS`.
* **Income mode**: a row in `INCOME_MODE_RULES`, and the decision in
  `INCOME_MODE_DECISIONS` so the results page knows the mode it ran with.
  `_model_settings_patch` looks a decision up in `INCOME_MODE_RULES`,
  `_PATCH_CONTAINERS` and `SCALAR_FIELDS` without a check, so a decision without an
  income mode cannot use it as it is.
* **σ**: the prefix in `_SIGMA_PREFIX`; `_disclosure_stochastic` then builds the
  `stochastic` patch with the shared three-way rule: Baseline off; Copula by its tick
  box (absent means off); Research Specification by its tick box (absent means on).
  Its "on" value is the block's `sigma_overall`: add a reader to `DecisionsConfig`
  (`src/config.py`, like `disclose_documents_sigma_overall`) and a function to
  `app/seam/sentinels.py`.
* **The patch builder**: `build_vendor_choice_weights_patch(snapshot, config_repo,
  pop_mode, inc_mode=None)`, modelled on `build_disclose_documents_patch`, and one more
  entry in the dictionary `build_decision_patches` returns.

Then add the new canonical keys, with the tab's defaults, to `base_state` in
`tests/test_build_plan.py` (the session the seam tests and the golden runs start
from), and add a test of the patch to that file.

---

## 7. The results page and the exports

### 7.1 The results view

The results page draws each decision's section through `DECISION_VISUALIZATIONS` in
`app/pages/results/visualizations/__init__.py`. Decision 5 is already routed to
`render_vendor_choice_weights` in `vendor_viz.py`, which assumes equal weights. Rewrite
it (or route the name to a new module) so it shows the model's results when the model
ran (its columns are present) and today's view when the default ran. The function
receives `(df, decision_name, decision_title, decision_data)`. Ask
`RunContext.from_session()` (`app/pages/results/run_context.py`) everything about the
run: its income mode, its Page 1 values, its default settings. Never read the live
widgets; they describe the next run.

### 7.2 The report builders

Everything computed for the screen or a download belongs in `app/reports/`; Decision
5's builders are in `app/reports/vendor.py`. A builder takes its inputs as arguments
and returns a table or bytes; the view only calls it and draws. Rules:

* no Streamlit in `app/reports/` (`tests/test_import_boundary.py`);
* **a missing value is `N/A`, never a substitute** (`NA`, `value_or_na`,
  `column_or_na`, `yes_no_code` in `app/reports/_common.py`; README §9). A model column
  of an agent whose decision ran on its default is `N/A`, as the Decision 1 columns
  are in `agent_level.py`;
* **money**: compute from the exact inputs, then round once with `money()`; a Purchase
  Now price always through `purchase_now_price()`;
* re-export every public name from `app/reports/__init__.py`.

### 7.3 The export columns

The platform documentation gives the exact columns, their order and the sheet names:
the decision's "Excel Export Structure", and the Agent-Level and Transaction-Level
sections. Today it lists, for Decision 5's own workbook, `Agent ID`,
`Honesty_Humility`, `Assigned Allowance Level`, `Study Program`, `Group_experiment`,
`TWT+Sospeso [=AW2+AX2]{Periods 1+2}`, `income`, `Price`, `Quality`, `Proximity`,
`Sustainability`; and, in the Agent-Level and Transaction-Level sheets,
`weight_price`, `weight_quality`, `weight_proximity`, `weight_sustainability`.
Implement the author's new columns exactly. In the Agent-Level sheet, add the model
columns only when the row carries them, as Decision 4 does
(`if 'rtd_choice_length' in row:` in `app/reports/agent_level.py`).

Then update `tests/test_reports_interface.py::test_vendor_report_interface` with the
new columns and sheets, add a missing-value case to
`tests/test_export_missing_values.py`, and add every new table builder to
`export_frames` in `tests/test_golden_runs.py`, so the golden net covers it.

### 7.4 The export of a "Run X Only" run

`render_export_section` (`app/pages/results/components/export_section.py`) chooses
the export by the run's shape. Decisions 1 to 4 each have a focused export for their
individual run (`_render_disclose_documents_only_export` is Decision 2's); every other
run gets the two-level export (Agent-Level and Transaction-Level). If the platform
documentation describes a focused export for the decision, add a branch on
`ctx.is_individual_run("vendor_choice_weights")` and its function.

### 7.5 Pins, the complete-run gate and comparisons (only if needed)

If the tab can produce several configurations per click (an income radio with
"Compare both", or the "Compare all" population mode), the user must pin one ("Use
This Config") before a complete run. Decisions 1 to 4 have this, and it touches many
lists written by hand. Agree with the owner first. Then add the decision to:

* `app/state/saved_configs.py`: `get_current_*_params`, `calculate_*_metrics`,
  `extract_*_configuration_details`, and `get_decision_result_columns` (which picks
  the hashed columns by prefix);
* `app/pages/results/config_pin.py` (`_SELECTION_BUTTONS`) and `config_selection.py`
  (the banner);
* `_SELECTABLE_INDIVIDUAL_RUNS` in `app/pages/results/run_context.py`;
* `app/pages/results/main_results.py`: the two lists of pinnable decisions in
  `render_single_run_results`;
* `app/pages/results/overview.py` and `comparisons.py`: the analysis cell and the
  comparison grids;
* `app/pages/decision_execution.py`: `GATE_DECISIONS`, `GATE_SUCCESS_ORDER`,
  `GATE_BLOCK_TYPE_TITLES`, the gate's texts, and the list in
  `run_individual_decision` of decisions whose pin a "Run X Only" clears;
* `app/pages/page2_decisions.py`: a `render_selected_*_config_display` call in
  `render_overview_tab`, and the disabled button's help text;
* `app/seam/build_plan.py`: the saved record's precedence in the patch builder, and in
  `_resolve_modes` (a pinned population mode).

---

## 8. The nets

Work in this order.

1. **Before you start**: `.venv/bin/python -m pytest tests/ -q`. Everything passes.
2. **While you work**: the fast run, `.venv/bin/python -m pytest tests/ -q -m "not slow"`,
   plus the new reference test.
3. **Extend the nets to the new decision.**
   * `MODELLED` in `tests/test_golden_runs.py` adds the scenarios
     `single_vendor_choice_weights_copula` and `single_vendor_choice_weights_baseline`.
   * `MODELLED` in `tests/test_golden_screens.py` adds the journeys
     `single_vendor_choice_weights` and `tab_controls_vendor_choice_weights`, and a
     Page 2 step. The tab-controls journey needs a `TAB_WIDGETS` row (σ mode radio,
     per-level slider pattern, income radio, the two tick boxes, the reset button); if
     the tab has other controls, write a journey of its own and add it to `JOURNEYS`.
     If it has an income radio, add its keys to the tuple in `_complete`, which the
     continuous complete-run journey sets.
   * Other tests list the modelled tabs: `DECISIONS` and the widget lists in
     `tests/test_stochastic_widgets_no_reset.py`, the decision list in
     `tests/test_no_widget_default_warning.py`, `TABS` in `tests/test_browser_state.py`,
     and `DECISIONS` in `tests/test_complete_run_gate.py` if you added a gate row.
     These do not fail when you forget them; they just do not cover the new tab.
4. **Read every failure before you re-record.** When the change is what you intend:

   ```bash
   GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/test_golden_runs.py \
       tests/test_golden_screens.py tests/test_complete_run_gate.py -q
   git diff --stat tests/golden tests/golden_screens
   ```

5. **Run everything again**: `.venv/bin/python -m pytest tests/ -q`, and commit the new
   golden files with the change.

What each failure tells you:

| Failing test | What it means |
|---|---|
| `test_golden_runs.py` | A result table or an export moved; the message names the changed columns. For Decision 5 the expected moves are `vendor_choice_weights` and the new columns, and what depends on the weights: Decision 6's preferred vendor and request vendors, Decision 8, the vendor cells and scores of the exports, in every scenario where Decision 5 runs (the complete runs select every decision). **Not expected: any column of Decisions 1 to 4.** If one moves, something shared changed (the decision order, a shared helper, a random stream). |
| `test_golden_screens.py` | Something on screen or in a download changed; the message names the journey and the first differing element. Expected: the new tab (the complete-run journeys select every decision, so they draw it), Decision 5's results section, the download hashes. A non-empty `browser_mismatches` means a widget the browser would show with another value than the script holds: one of the rules of §6.5 is broken. |
| `test_complete_run_gate.py` | The gate's outcome or wording changed. Expected only if you added a gate row (§7.5). |
| `test_session_key_registry.py` | A key is missing from the registry, unused, or misdescribed (§6.4). |
| `test_config_is_single_source.py` | The module holds a copy of a YAML value, or runs without a key it needs (§2). |
| `test_import_boundary.py` | Streamlit or `app` imported under `src/`, Streamlit under `app/reports/`, or `app.pages` under `app/state/`. |
| `test_decision_order.py` | `DECISION_ORDER` or a default table changed. Put it back. |
| the new reference test | The code does not reproduce the document. Fix the code, or ask the author. Never edit the expected numbers to make it pass. |

---

## 9. Documentation to update

In the same change:

* **`docs/README.md`**: the module map in §1 (the four modelled decisions become five;
  the tab and visualisation lines), §3 (the pass 1 statistics, if you added a block),
  §4 (which decisions receive `pop_context`), §8 (the reference extract and the
  `PROF_DTA` list), §9 (every choice you made where the document left room, with its
  reason), §10 (known limits).
* **`docs/how-to-change-the-model.md`**: the reference table in §0, "the four modelled
  decision modules" in §1, and the lists in §2 and §3 (`SCALAR_FIELDS`, the σ
  prefixes).
* **`docs/session-state.md`**: a row in the §4 table of keys the seam reads, the new
  reset in the §7 table, and §2 or §8 if the tab uses the shared controls or the twin
  spec.
* **`tests/README.md`**: the new test module, the new extract, the `.dta` file and
  where it lives, and the new numbers of golden scenarios and journeys.
* **`docs/reference/README.md`**: the new document's row.
* **The platform documentation** (`docs/reference/COOPECON AI Agent Documentation
  <date>.docx`) belongs to the author. Ask for the updated version with the decision's
  sections, and replace the file, keeping the date in its name.

---

## 10. Checklist

The names are Decision 5's; replace them for another decision.

1. [ ] Read the document; list every number and every variable (§1.1).
2. [ ] Map every variable to an agent column; stop if one is missing (§1.2).
3. [ ] Document in `docs/reference/decisions/`; data file in `data/stata/`, or its
   location written down; `PROF_DTA` set (§1.3).
4. [ ] Data file checked: 280 rows, participant order, traits, income (§1.4).
5. [ ] `experiments/d5_doc_reference.py` written; `data/d5_doc_reference.csv` frozen (§1.5).
6. [ ] Every constant in the `vendor_choice_weights:` block, each with its source (§2).
7. [ ] Module in `MODULES` of `tests/test_config_is_single_source.py`, with a
   missing-key test (§2).
8. [ ] `src/decisions/vendor_choice_weights.py` rewritten: same function name, default
   branch kept, strict reads, `rng` only, output shape kept, prefixed model columns (§3).
9. [ ] `POP_CONTEXT_DECISIONS`; a pass 1 block and a `RUN_SCRATCH_KEYS` entry if
   needed (§3.6, §3.7).
10. [ ] `get_available_decisions()` lists the decision; it runs on the 280 from Python
    and from the command line (§3.1, §3.8).
11. [ ] Every number of the document recomputed; every difference explained or
    answered by the author (§4).
12. [ ] `tests/test_vendor_choice_weights_doc.py`: extract, constants, 280/280 per
    income mode and research mode, tables, figures, noise, default branch (§4).
13. [ ] `DEFAULT_DECISION_VALUES`, `DEFAULT_DECISION_DESCRIPTIONS` and the Overview
    editor left as they are (§5).
14. [ ] Tab module written and routed in `render_decision_tab` (§6.1).
15. [ ] `CONTROL_KEYS` / `DISCLOSURE_KEYS` rows, if the tab uses the shared parts
    (§6.2, §6.3).
16. [ ] Registry rows, `TAB_WIDGET_CANONICAL`, `PREFIX_DELETE_PREFIXES`; registry test
    green (§6.4).
17. [ ] Session-state rules checked: two keys, `restore`, `stateful`, no re-push,
    reset at the top of the next run (§6.5).
18. [ ] Seam rows; `build_vendor_choice_weights_patch` in `build_decision_patches`;
    `base_state` updated (§6.6).
19. [ ] Results view rewritten; it reads only `RunContext` (§7.1).
20. [ ] Report builders in `app/reports/`, exact columns from the platform
    documentation, `N/A` and money rules; interface and missing-value tests updated
    (§7.2, §7.3).
21. [ ] Focused "Run X Only" export, if the documentation describes one (§7.4).
22. [ ] Pins, gate and comparisons, only if agreed with the owner (§7.5).
23. [ ] Golden scenarios, screen journeys and the other test lists extended (§8).
24. [ ] Every golden failure read and explained; golden copies re-recorded; full suite
    green (§8).
25. [ ] `docs/README.md`, `how-to-change-the-model.md`, `session-state.md`,
    `tests/README.md` and `docs/reference/README.md` updated; platform documentation
    requested from the author (§9).
