# How to model one of the nine default decisions

The platform has thirteen decisions. Four of them are
*modelled*: Decisions 1 (Disclose Income), 2 (Disclose Documents), 3 (Donation
Default) and 4 (Rejected Transaction Defaults) compute their value from each agent's
traits, with equations from the decision documents, checked against the experiment data. The other nine (5 Vendor Choice Weights, 6 Purchasing Quantity, 7 Purchasing
Frequency, 8 Vendor Selection, 9 Purchase Now vs Bid, 10 Bid Value, 11 Rejected
Transaction Option, 12 Rejected Bid Value, 13 Final Donation Rate) currently only have the simple default rules that the user edits on the Overview tab of
Page 2. (A correction to an existing decision, or the unlikely fourteenth decision, is covered
by [`how-to-change-the-model.md`](how-to-change-the-model.md), §9 for the latter.)

The running example is **Decision 5, Vendor Choice Weights**. Every name below that
contains `vendor_choice_weights`, `vcw_` or `d5` shows what a file or key *would* be
called; none of them exists for Decision 5 yet. For another
decision, replace the name. Files are named, not given line numbers: find a name with
`grep -n`.

Read [`README.md`](README.md) §1 to §5 and [`session-state.md`](session-state.md) §1
and §2 first. The checklist in §10 is the short version of this page.

## The work, in order

Modelling a decision means turning the author's document into code that gives the same numbers,
then letting the user set the decision up and see its results. In order:

1. **Read the document**; list every number and variable it uses (§1).
2. **Check the Stata data file** against the app's participants, and freeze the numbers the code
   must reproduce in a small CSV, the reference extract (§1).
3. **Put the constants** in `config/decisions.yaml` (§2).
4. **Write the decision's code** in `src/decisions/`, keeping today's default for when the
   decision is not selected (§3, §5).
5. **Prove that it reproduces the document**: a test runs the real engine on the 280 participants
   and compares each one with the extract (§4).
6. **Build the screens**: the Page 2 tab and the seam rows that carry its settings to the engine
   (§6), then the results view and the exports (§7).
7. **Extend the test nets**, re-record the golden copies (§8) and **update the docs** (§9).

The model is proved before any screen exists, because every screen and export only shows what the
model computes.

## Words used on this page

* **Streamlit**: the web framework of the screens. Every click re-runs the page's script from the
  top; what must survive lives in `st.session_state`. A widget drawn with `key="x"` keeps its
  value in `st.session_state["x"]`, which Streamlit may drop in a run that does not draw the
  widget.
* **the engine**: `src/`, the code that computes numbers.
* **the seam**: `app/seam/`. On a Run click it copies the session state once (the *snapshot*),
  builds a *run plan* and runs the engine. It writes nothing back.
* **patch**: a small dictionary merged onto a fresh copy of a decision's block of
  `config/decisions.yaml` for one run (§2).
* **widget key**: a widget's `key=`, which Streamlit may drop; tab widget keys contain `_tab_`
  (`dd_tab_income_mode`). **Canonical key**: the never-dropped session key that holds the setting
  for the app and the seam (`dd_income_mode`; §6.5).
* **the registry**: `app/state/registry.py`, the list of every session key (§6.4).
* **golden copies**: recorded outputs of the app in `tests/golden/` and `tests/golden_screens/`;
  the tests that compare against them (the *parity net*, §8) fail when anything moved, and you
  decide whether the move was intended.
* **pass 1, pass 2**: pass 1 gives each agent its seed and income and computes population
  statistics; pass 2 runs the decisions.
* **z-scoring**: (value − mean) / standard deviation, with the 280 participants' mean and *sample*
  standard deviation (N − 1, like Stata's `egen std()`).
* **σ (sigma), noise**: the optional draw `Normal(score, σ)` added to the deterministic score.
* **`.dta`**: a Stata data file. **Reference extract**: a CSV in `data/` with the numbers the code
  must reproduce, participant by participant.
* **population modes**: Copula (synthetic agents from a model fitted to the 280), Research
  Specification (the 280, noise allowed), Research Baseline (the 280 in file order, never noise).
  README §4.

## 0. What "modelling a decision" means

### What arrives from the author

1. **A methodology document** (`.docx`): the equations, the Stata commands for every variable, and
   the results (regression tables, frequency tables often by allowance level, figures, usually
   histograms of the score).
2. **A Stata data file** (`.dta`), one row per participant (280): `participantid`, the raw survey
   and experiment columns, usually the model's computed columns (Decision 2's has
   `weighted_dd_categorical`, `dd_deterministic_cont`, `disclosedoc_categorical`).

### The one rule

**The document is the source of truth for the model. The `.dta` file is how you prove
that the code reproduces the document. The platform documentation (`docs/reference/`)
says what the app must show and export.**

When the code and the document disagree, the document wins. When the document
disagrees with its own data, ask the prof. 

### Where a modelled decision lives: Decision 2, layer by layer

Decision 2 is the pattern to copy. The list below gives, for each piece, Decision 2's real file
or name on the left and the name the same piece would have for Decision 5 on the right. Open the
Decision 2 files once before you start.

* **Document**: not in the repository ->
  `docs/reference/decisions/Decision 5 - Vendor Choice Weights <date>.docx`
* **Data file**: `Stata_File_Decision 2_260626 - CORRECTED.dta` in `data/stata/` (`PROF_DTA` in
  `tests/test_disclose_documents.py`) -> `data/stata/<file as received>.dta`
* **Reference extract**: the document's tables, written into the test, and
  `data/stata_incomes.csv` -> `data/d5_doc_reference.csv`
* **Constants**: block `disclose_documents:` of `config/decisions.yaml` ->
  `vendor_choice_weights:` (today only `uses_global_parameters: []`)
* **Traits**: `disclose_documents:` in `config/trait_requirements.yaml` ->
  `vendor_choice_weights:` (today `placeholder_trait`)
* **Model**: `src/decisions/disclose_documents_stochastic.py` ->
  `src/decisions/vendor_choice_weights.py`
* **Engine** (`src/engine/core.py`): `STOCHASTIC_MODULE_DECISIONS`, `_compute_population_stats`,
  `dd_cont_stats` in `RUN_SCRATCH_KEYS` -> only what the model needs (§3.6)
* **Default when unselected**: unchanged (§5)
* **Tab**: `app/pages/decision_tabs/disclose_documents.py`, with `_controls.py` and
  `_disclosure.py` -> `app/pages/decision_tabs/vendor_choice_weights.py`
* **Session keys**: the `dd_*` rows of `app/state/registry.py` -> `vcw_*` rows
* **Seam**: `build_disclose_documents_patch` and its rows (`app/seam/build_plan.py`,
  `app/seam/sentinels.py`) -> `build_vendor_choice_weights_patch`
* **Results view**: `render_disclose_documents`
  (`app/pages/results/visualizations/disclosure_viz.py`) -> `render_vendor_choice_weights`
  (`vendor_viz.py`, exists)
* **Reports**: `app/reports/disclosure.py`, columns in `agent_level.py`, `transaction_level.py` ->
  `app/reports/vendor.py` (exists)
* **Tests**: `tests/test_disclose_documents.py`, `tests/test_disclose_documents_gate.py` ->
  `tests/test_vendor_choice_weights_doc.py`, plus rows in shared tests (§2, §8)

All four modelled decisions (1, 2, 3 and 4) also have a second group of pieces: a "Use This
Config" pin, a row in the gate on "Run Complete Simulation", the comparison grids and a display of
the pinned configuration on the Overview tab. A new decision needs them only when its tab can
produce more than one configuration per click (§7.5). Build §1 to §7.4 first.

### Where the nine decisions' behaviour lives today

Several are not computed by their own module; check before you choose a file.

* **5 Vendor Choice Weights**: `src/decisions/vendor_choice_weights.py` returns a dict
  `{price, quality, proximity, sustainability}`; Decision 6 reads it to choose the agent's
  preferred vendor.
* **6 Purchasing Quantity**: `src/decisions/purchasing_quantity.py` makes the purchase requests,
  their timestamps, their vendor.
* **7 Purchasing Frequency**: `purchasing_frequency.py`, quantity / periods, from Decision 6's
  output.
* **8 Vendor Selection**: `vendor_selection.py` only reports the vendor Decision 6 chose.
* **9 Purchase Now vs Bid, 10 Bid Value**: per purchase request, **inside Decision 6**:
  `_enrich_purchase_requests` calls `purchase_vs_bid_single` and `generate_single_bid_value` with
  Decision 6's random stream and an empty `params`, and Decision 9 uses the Overview probability,
  selected or not. The slots 9 and 10 still compute one value per agent, but nothing reads it
  (README §9). A model for 9 or 10 goes into those two functions; drawing a different number of
  values per request moves every later draw of that agent's Decision 6.
* **11 Rejected Transaction Option, 12 Rejected Bid Value**: they return the Overview value,
  selected or not.
* **13 Final Donation Rate**: `final_donation_rate.py` copies Decision 3. But the rate each
  request carries is set inside Decision 6, which runs before Decision 13, so it comes from
  `donation_default` (or 0.10). Where a Decision 13 model must act is a question for the author.

## 1. Before writing code

The decision document goes in `docs/reference/decisions/`, the Stata data file in `data/stata/`,
and the reference test points at it:
`PROF_DTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "stata", "<file>.dta")`.

### 1.1 Read the document and make two lists

Read the whole document once without coding, then list, and keep to the end:

* **every number it states**, with its place (section, page, table): coefficients, intercepts,
  means, standard deviations, regression output, table cells, percentages, histogram bars. §4
  recomputes each one;
* **every variable it uses**, with its Stata name (`extraversionbig5`, `totalallowance`, ...): a
  raw input, a value computed from raw inputs, or the result.

### 1.2 Map each variable to a column the agent carries

An agent is a dictionary, `agent_state`. Before any decision runs it holds these traits, spelled
exactly so (`src/data/participants.py` reads them from the two workbooks in `data/`;
`config/trait_requirements.yaml` says which), plus four values the engine adds. In the Decision 1
and 4 `.dta` files, every column below that the file has equals the agent's, participant by
participant.

| Agent column | Stata name | Notes |
|---|---|---|
| `Agreeable` | `agreeable` | survey workbook |
| `Assigned Allowance Level` | `assignedallowancelevel` | experiment; 1 to 5, a float |
| `ConscientiousnessBig5` | `conscientiousnessbig5` | survey |
| `Education` | `education` | survey; 1 or 2 |
| `ExtraversionBig5` | `extraversionbig5` | survey |
| `Group_experiment` | `group` | experiment; see below |
| `Honesty_Humility` | `honesty_humility` | survey |
| `NeuroticismBig5` | `neuroticismbig5` | survey |
| `OpennessBig5` | `opennessbig5` | survey |
| `ReligiousAffiliation` | `religiousaffiliation` | survey |
| `ReligiousService` | `religiousservice` | survey |
| `Study Program` | `studyprogram` | experiment |
| `Study Program Category` | `studyprogramcategory` | research agents only; see below |
| `TWT+Sospeso [=AW2+AX2]{Periods 1+2}` | `twtsospesoaw2ax2periods1` | cut at 32 characters |
| `stdactions` | `stdactions` | Decision 4 file only; see below |
| `income` | `income` | added by the engine; see below |
| `actual_allowance` | `totalallowance` | added by the engine; see below |
| `index`, `agent_id` | – | row (from 0), agent number (from 1) |

* `Group_experiment`: `HighSub`, `MidSub` or `NoSub`. Both workbooks have `Group`; the merge names
  the experiment one `Group_experiment`.
* `Study Program Category`: the copula does not carry it; Decision 3 maps it from `Study Program`.
* `stdactions`: merged from `data/stata_stdactions.csv`.
* `income`: in the Research modes, the fixed reference income of `data/stata_incomes.csv`; in
  Copula, drawn from the Page 1 distribution inside the band of the allowance level.
* `actual_allowance`: 12, 32, 72, 128 or 200 credits, from the level (`ALLOWANCE_CREDIT_MAPPING`,
  `src/engine/income.py`).

The agent also carries what earlier decisions returned in the same run (Decision 5 can read
`customer_type`, `disclose_income`, `donation_default` and the Decision 4 columns when those ran),
never a later one: the order is fixed (§3.4).

A variable in neither list is a new trait. Add it to `config/trait_requirements.yaml` and to the
data merge in `src/data/participants.py`, then retrain the copula with `scripts/train_copula.py`.
Expect every Copula-mode number of every decision to move, and regenerate the golden copies
afterwards (§8).

### 1.3 Check the data file against the repository's data

Load it with `pd.read_stata(path, convert_categoricals=False)` and check:

* 280 rows, `participantid` in the engine's order (the Decision 1, 2 and 4 files are). The
  engine's IDs are `P.merged().loc[P.original_data().index, "Participant ID"]`
  (`from src.data import participants as P`);
* every trait the model uses equals the agent's column (§1.2);
* `income`, if present, equals the Research modes' fixed income (`data/stata_incomes.csv`, §1.2).
  If not, ask the author: continuous-income results cannot match otherwise.

### 1.4 Freeze a reference extract

Write `experiments/d5_doc_reference.py`, which writes `data/d5_doc_reference.csv` (pattern:
`experiments/d3_doc_reference.py` and `data/d3_doc_reference.csv`). The script is the record of
how the extract was made; `data/stata_d1_verification.csv` and `data/stata_d4_verification.csv`
have none. It:

* takes the `.dta` path (with a default) and the output path as arguments;
* has a `build(dta_path)` function that returns the extract, for the test;
* writes one row per participant (per income mode, if there are two), keyed by `participantid`,
  with the model's intermediate values and result;
* copies the `.dta`'s model columns, or re-implements the document's Stata commands from the raw
  columns where it has none (Decision 3's script re-runs the regressions with numpy);
* writes with `float_format="%.17g"`; the test reads with `float_precision="round_trip"`, so no
  digit is lost.

## 2. The constants

Every constant goes into the decision's block of `config/decisions.yaml`
(`vendor_choice_weights:`), each with a comment giving its source. The rules are those of
[`how-to-change-the-model.md`](how-to-change-the-model.md) §1: one file, strict reads with
`require(params, DECISION, "key", "subkey")` (a missing key raises `MissingDecisionKey`), never
`params.get("key", <number>)`, constants needed by name read at import with
`file_block("vendor_choice_weights")`, and only switches absent (`income_mode`, the `stochastic`
block). The tab, too, reads every number it prints from the block. Decision 2's block shows the
shape: coefficients, intercept, `z_scoring` (mean and standard deviation per trait),
`categorical_intercepts` (`level_1` to `level_5`), the `stochastic` block.
(`uses_global_parameters` is in every block; no Python code reads it.)

In `tests/test_config_is_single_source.py`, add the module to `MODULES`, and two tests like
`test_disclose_documents_runs_on_the_full_block` and `test_disclose_documents_missing_key_raises`
with the keys the module must refuse to run without. The file's scan flags a `.get` fallback, a
dict literal or a module-level number that copies a block value (0, 1 and −1 never count).

Each run gets a patched deep copy of the file (`apply_patches`, `src/contract/plan.py`; README
§7); nothing in `app/` writes the file.

## 3. The decision module

### 3.1 Which file, which function

Rewrite `src/decisions/vendor_choice_weights.py` in place and keep the function name
`vendor_choice_weights`: the engine imports `src.decisions.<name>` and calls the function of the
same name (`Engine.__init__`, `src/engine/core.py`). Avoid the `<name>_stochastic.py` spelling of
Decisions 1 and 2: it needs an entry in `STOCHASTIC_MODULE_DECISIONS` and buys nothing.

```python
DECISION = "vendor_choice_weights"

def vendor_choice_weights(agent_state: dict, params: dict, rng,
                          simulation_config: dict = None, **kwargs) -> dict:
    ...
    return {"vendor_choice_weights": weights, "vendor_choice_weights_score": score, ...}
```

A broken module fails the run at engine construction, naming the decision (`Engine.__init__`
raises an `ImportError` when the module cannot be imported or lacks the function).

### 3.2 What the function receives

The arguments are described in [`how-to-change-the-model.md`](how-to-change-the-model.md) §2. For
a modelled decision:

* read traits as `agent_state["ExtraversionBig5"]`, so a missing one raises, as Decisions 1, 2 and
  4 do;
* never write into `params`: it is the engine's live dictionary for the whole run, so the next
  agent would see the change;
* Decision 5's `rng` is seeded with offset 4000 (position 4 × 1000; README §3);
* `simulation_config` also carries the pass 1 results `income_stats` (mean and sample standard
  deviation of the run's incomes), `income_median` and your own pass 1 key (§3.6), plus `vendors`,
  `simulation_seed`, `default_decisions` and `default_decisions_list`;
* every decision receives `pop_context` in `kwargs` (its values: README §4).

### 3.3 What it returns

* **Keep the decision's own column and its shape.** For Decision 5, `vendor_choice_weights` is a
  dict with exactly `price`, `quality`, `proximity`, `sustainability`, summing to 1. Decision 6
  (`purchasing_quantity.py`), the vendor score (`calculate_vendor_score_with_breakdown`,
  `src/vendor_attribute_generator.py`) and every export (`vendor_weights_or_none`,
  `app/reports/_common.py`) read it, and they disagree on a missing key (0.0 in
  `app/reports/vendor.py`, 0.25 in `app/reports/agent_level.py`): always return all four.
* **Prefix the model's extra columns with the decision's name** (`vendor_choice_weights_score`,
  `vendor_choice_weights_sigma_used`, ...), like `disclose_documents_*`. The pin store selects a
  decision's columns by that prefix.
* **Return the same keys on every path of the model**, as Decision 2 does, so the table has no
  holes. The default path (§3.5) may return only the decision's own column, as Decisions 1 to 4
  do.
* **Return the intermediate values the document shows** (z-scores, composite, deterministic score,
  value after the draw, σ used, income mode): the reference test and the exports need them.

### 3.4 Rules for the module

The rules of [`how-to-change-the-model.md`](how-to-change-the-model.md) §2 apply. In addition: the
shared helpers are `z_score_trait`, `categorical_intercept` and `normalise_income_mode`
(`src/decisions/_shared.py`); a run that does not repeat itself fails the golden tests;
`tests/test_decision_order.py` checks only the length and the copies of `DECISION_ORDER`, and the
golden tests catch a reordering.

### 3.5 The unselected case

Unselected, the decision must behave exactly as today. Every modelled decision starts with the
same branch:

```python
if DECISION in (simulation_config or {}).get("default_decisions_list", []):
    entry = simulation_config.get("default_decisions", {}).get(DECISION)
    ...   # today's default rule, unchanged
    return {DECISION: ...}
# otherwise: the model
```

Decision 5's rule today returns `entry["weights"]` for a `checkbox_selection` entry (the seam
computes the equal weights of the ticked attributes in `collect_decision_settings`,
`app/seam/build_plan.py`); copy it before you rewrite the module. In a complete run an unselected
decision is in `default_decisions_list`; in a "Run X Only" run the list is empty and only X runs,
so the model runs.

### 3.6 Population statistics in pass 1

A number over all agents (a mean, a standard deviation, a minimum, a maximum) is computed in pass
1, as [`how-to-change-the-model.md`](how-to-change-the-model.md) §2 says. For a modelled decision,
the block in `Engine._compute_population_stats` runs only when the decision runs and is not in
`default_decisions_list`, as for Decisions 3 and 4; store the result under a key such as
`self.simulation_config["vcw_population_stats"]` and add it to `RUN_SCRATCH_KEYS`.

Decision 2's `compute_continuous_dd_stats` is the simple pattern. A statistic that depends on the
decision's own draws (Decision 3's population maximum) must replay each agent's pass 2 draws
exactly: the same agent state, `np.random.default_rng(base_seed + decision_index * 1000)`, the
same draws in the same order. See `compute_donation_population_max` (Decision 3) and
`compute_rtd_population_stats` (Decision 4, which always draws its five noise values so the replay
stays aligned).

### 3.7 Two conventions that matter: noise and income mode

* **Noise** goes through `should_use_stochastic` (README §4). For Decisions 1, 2 and 4
  `sigma_value` is only an on/off signal and σ is `sigma_overall` × `scale_factor`, or `sigma_quintile[level]` × `quintile_scale_factors[level]` in per-level mode from the block; only Decision 3 uses `sigma_value` as σ. The YAML ships
  `sigma_value: 0`, but the app's Research Specification tick box is on by default: an app run in
  Research Specification adds noise unless the user clears it; an engine run from Python does not.
* **The income mode**: `"Categorical only"`, `"Continuous only"` or `"Compare both"`
  (`MODE_OPTIONS`, `app/pages/decision_tabs/_controls.py`); `normalise_income_mode(mode)` turns
  any spelling into `"categorical"` or `"continuous"`. Categorical works from the allowance level
  (`categorical_intercepts.level_1` to `level_5`), continuous from the income z-scored with the
  run's `income_stats`. Decisions 1 to 3 default to categorical and, in a complete run, follow the
  run's income mode; Decision 4 defaults to continuous and keeps its own tab's mode (README §5).
  Decide with the author which convention the new decision follows, and write it in the YAML
  comment.

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

Or from the command line (the file lands in `--output-dir`, by default `outputs/`):

```bash
python scripts/run_simulation.py --agents 280 --seed 42 --population-mode baseline \
    --decision vendor_choice_weights --format csv
```

At 280 agents with σ off, the two research modes give the same rows (README §4); compare in
Research Baseline. Copula agents are synthetic and cannot be compared participant by participant.

## 4. The reference test

Create `tests/test_vendor_choice_weights_doc.py`, modelled on
`tests/test_donation_default_doc.py`, not marked `slow` (it does not start the app). Six groups of
tests:

1. **The extract is what the script makes**: load `experiments/d5_doc_reference.py` with
   `importlib.util`, call `build(PROF_DTA)`, compare with the frozen CSV (differences below
   1e-12). The test starts with
   `require_reference_file(PROF_DTA, "PROF_DTA in tests/test_vendor_choice_weights_doc.py")`.
2. **The configuration equals the document**, constant by constant
   (`test_config_coefficients_equal_the_doc` in the Decision 3 test).
3. **The engine reproduces the extract**, through the real engine (§3.8), in every income mode, in
   Research Baseline and in Research Specification with σ off. Match rows by `participantid`, not
   position, and assert 280 matches. (Decision 3 allows 1e-6 on a score; Decision 2 compares every
   Y/N exactly.)
4. **The document's tables and figures**, recomputed from the engine's output: counts and
   percentages by allowance level (`DOC_CATEGORICAL` in `tests/test_disclose_documents.py`);
   histograms with Stata's default number of bins, min(√N, 10·log10 N) cut to an integer, 16 at N
   = 280 (`rtd_stata_bins` in `app/reports/rtd.py`; `tests/test_rtd_histogram_stata_parity.py`
   compares bar heights with the figure).
5. **The noise**: σ on in Research Specification changes the values; σ requested in Research
   Baseline changes nothing; Copula draws only with its tick box
   (`test_noise_gate_and_population_rescale`).
6. **The default branch**: in `default_decisions_list`, the decision returns the Overview default
   (`tests/test_disclose_documents_gate.py`).

Before writing the test, check the whole document, as was done for Decision 2:

* recompute every number of the document from the raw `.dta` columns with numpy (the meta-analytic
  weights, coefficients, z-scoring constants against the YAML, each composite against the `.dta`'s
  column, the per-level intercepts and their standard errors, table cells, correlations, σ values,
  the 16-bin histograms) into one table: item, place, document value, recomputed value,
  difference, ok. Each "not ok" row is a bug in the code or a question for the author;
* run the engine also after Decision 1; check the alignment first (traits and income equal the
  `.dta`), then every intermediate column and the eligibility gate against the `.dta`;
* draw each of the engine's histograms beside the `.dta`'s, with the same bins.

## 5. The default for the unselected case

Keep it, as Decisions 1 to 4 did (a probability for 1 and 2, the number 0.10 for 3, a priority
list for 4): the entries in `DEFAULT_DECISION_VALUES` and `DEFAULT_DECISION_DESCRIPTIONS` and the
Overview editor ([`how-to-change-the-model.md`](how-to-change-the-model.md) §4), the "Default
Behavior" line on the results page (`get_dynamic_description`,
`app/pages/results/visualizations/viz_helpers.py`) and the module's default branch (§3.5). For
Decision 5, leave its entry (`checkbox_selection`, the four attributes, equal weights), its
description and the Overview checkboxes as they are, unless the author's platform documentation
changes the default behaviour. Both tables keep one entry per decision
(`tests/test_decision_order.py`).

## 6. The tab on Page 2

### 6.1 How a tab is chosen and built

`render_decision_tab(decision_name)` (`app/pages/decision_tabs/__init__.py`) gives Decisions 1 to
4 and 10 a module of their own, 11 a short note, and the others `render_generic_decision_tab`
("This decision currently uses default values"). Write
`app/pages/decision_tabs/vendor_choice_weights.py` with `render_vendor_choice_weights_tab()` and
`initialize_vendor_choice_weights_session_state()`, and add one `elif` that calls the first.
Follow `disclose_documents.py`, in this order:

1. apply a pending reset (§6.5);
2. run the initialiser: it seeds every absent *canonical* key from the YAML block
   (`read_yaml_file`, never a copied number);
3. draw the controls (§6.2), the formula display with every number read from the block (the
   Decision 2 tab still holds copies of its σ values and fallbacks), and the override and reset
   buttons (§6.3);
4. end with `render_simulation_buttons(decision_name=..., selected_decisions=...)`: "Run X Only"
   and "Run Complete Simulation".

### 6.2 The shared controls (`_controls.py`)

The four modelled tabs draw their σ controls and income-mode radio with `render_income_mode`,
`render_stochastic_tick_boxes` and `render_sigma_controls`, each tab described by a `TabControls`
spec ([`session-state.md`](session-state.md) §2). The keys are the tab's row of `CONTROL_KEYS` in
the registry:

```python
"vendor_choice_weights": ControlKeys(
    dp="vcw_", wp="vcw_tab_", sp="vcw_", suffix="", research="sigma_enabled",
    coefficient="vcw_scale_factor", income_widget="vcw_tab_income_mode",
    income="vcw_income_mode"),
```

A row commits the tab to the whole set: both tick boxes, the σ mode radio, the coefficient slider,
the five per-level sliders and the income radio; `tests/test_session_key_registry.py` requires a
`TAB_WIDGET_CANONICAL` entry for each. A decision without an income mode or σ gets no row and
draws its controls in its own module, under the rules of §6.5.

### 6.3 The twin spec (`_disclosure.py`)

Disclose Income and Disclose Documents share an intercept override, a whole-tab reset and a "Reset
Intercept" button (`app/pages/decision_tabs/_disclosure.py`, a `DisclosureTab` spec and a
`DISCLOSURE_KEYS` row each). A tab of that shape adds a row
`DisclosureKeys(dp="vcw_", wp="vcw_tab_", suffix="")` and a spec like `DD_TAB`; any other tab
writes its own reset (§6.5).

### 6.4 The registry rows

Add a `KeySpec` ([`session-state.md`](session-state.md) §1) for every key the tab uses, modelled
on the tables `_DD_MIRRORS` and `_DD_WIDGETS`, and the new tables to `REGISTRY`:

* **canonical keys** (`vcw_income_mode`, `vcw_sigma_enabled`, `vcw_sigma_in_copula`,
  `vcw_sigma_strategy`, `vcw_scale_factor`, `vcw_quintile_scale_factors`, one per model setting):
  category `page2-mirror`, `engine_input=True`, `initialised_by`
  `"decision_tabs.vendor_choice_weights.initialize_vendor_choice_weights_session_state"`;
* **widget keys** (`vcw_tab_*`, the override input, the buttons): category `page2-widget`,
  `is_widget=True`, default `WIDGET` for a button;
* **`TAB_WIDGET_CANONICAL`**: one entry per widget that `restore` seeds, `(canonical key, None)`,
  or `(canonical key, "1".."5")` for the per-level sliders;
* **the reset**: the prefix in `PREFIX_DELETE_PREFIXES`, and `_vcw_reset_to_defaults_pending` in
  `_RESET_PENDING_FLAGS` if you use the twin spec;
* **results-page widgets** that take a `key=` (a chart, a "Use This Config" button):
  `_RESULT_WIDGETS`.

Choose a short prefix that starts no other key (`vcw_` starts none today), not
`vendor_choice_weights_`: the reset deletes every key with the prefix, and the Overview's keys
(`vendor_choice_weights_default_params`, ...) start with it.

Then run `.venv/bin/python -m pytest tests/test_session_key_registry.py -q` (what it checks:
[`session-state.md`](session-state.md) §1). It reads the code, not the screen, so it cannot tell
whether the tab really draws a control.

### 6.5 The session-state rules

[`session-state.md`](session-state.md) §2 and §7 explain why.

1. Two keys per setting: the widget key (`vcw_tab_...`), and the canonical key (`vcw_...`) written
   right after the widget on every run.
2. Seed the widget key right before the widget, only when absent: `restore(widget_key, default)`
   (`app/pages/decision_tabs/_tab_state.py`).
3. Draw every keyed widget through `stateful(st.<widget>, ..., key=...)` (`app/state/widgets.py`),
   never with `value=`, `index=` or `default=`.
4. Never re-push the stored value into the widget key on every run; correct a value with
   `seed_if_changed`.
5. Never write a widget key in a run that does not draw its widget.
6. Reset by prefix at the top of the next run: the button only sets
   `_vcw_reset_to_defaults_pending`; the next run, before any widget of the tab exists, deletes
   every key starting with `vcw_` and writes the canonical keys and the widget keys it will draw
   from the YAML block (not the per-level sliders while the σ mode is "uniformly"), as
   `apply_pending_reset` in `_disclosure.py` does. Apply it from
   `page2_decisions.apply_pending_tab_resets` (while the decision is selected), not only at the
   top of the tab: the Overview is drawn before the tabs and reads their keys (the run gate), so a
   reset applied inside the tab reaches the Overview one rerun late.
7. An absent key means the tab's default, never "off": the seam must give the same numbers whether
   or not the user ever opened the tab.

### 6.6 The seam rows

The seam turns the canonical keys into a patch on the decision's block, in
`app/seam/build_plan.py`:

* **scalar settings**: one `ScalarField` row per setting in `SCALAR_FIELDS`
  ([`how-to-change-the-model.md`](how-to-change-the-model.md) §3), and the decision in
  `_PATCH_CONTAINERS`;
* **income mode**: a row in `INCOME_MODE_RULES`, and the decision in `INCOME_MODE_DECISIONS` so
  the results page knows the mode it ran with. `_model_settings_patch` looks a decision up in
  `INCOME_MODE_RULES`, `_PATCH_CONTAINERS` and `SCALAR_FIELDS` without a check, so a decision
  without an income mode cannot use it as it is;
* **σ**: the prefix in `_SIGMA_PREFIX`; `_disclosure_stochastic` then builds the `stochastic`
  patch with the shared three-way rule (Baseline off; Copula by its tick box; Research
  Specification by its tick box, and an absent tick box counts as on). Its "on" value is the
  block's `sigma_overall`: add a reader to `DecisionsConfig` (`src/config.py`, like
  `disclose_documents_sigma_overall`) and a function to `app/seam/sentinels.py`;
* **the patch builder**:
  `build_vendor_choice_weights_patch(snapshot, config_repo, pop_mode, inc_mode=None)`, modelled on
  `build_disclose_documents_patch`, and one more entry in the dictionary `build_decision_patches`
  returns.

Then add the new canonical keys, with the tab's defaults, to `base_state` in
`tests/helpers/seam.py` (the session the golden runs start from), and a test of the patch.

## 7. The results page and the exports

### 7.1 The results view

`DECISION_VISUALIZATIONS` (`app/pages/results/visualizations/__init__.py`) routes Decision 5 to
`render_vendor_choice_weights` in `vendor_viz.py`, which assumes equal weights. Rewrite it (or
route the name to a new module) to show the model's results when its columns are present and
today's view when the default ran. It receives
`(df, decision_name, decision_title, decision_data)`. Ask `RunContext.from_session()`
(`app/pages/results/run_context.py`) everything about the run, never the live widgets (README §5).

### 7.2 The report builders

Everything computed for the screen or a download belongs in `app/reports/` (Decision 5's builders:
`app/reports/vendor.py`). A builder takes its inputs as arguments and returns a table or bytes;
the view only calls it and draws. No Streamlit there (`tests/test_import_boundary.py`); re-export
every public name from `app/reports/__init__.py`. The export rules are in README §9:

* **a missing value is `N/A`, never a substitute** (`NA`, `value_or_na`, `column_or_na`,
  `yes_no_code` in `app/reports/_common.py`): a model column of an agent whose decision ran on its
  default is `N/A`, like the Decision 1 columns in `agent_level.py`;
* **money**: compute from the exact inputs, round once with `money()`; a Purchase Now price always
  through `purchase_now_price()`.

### 7.3 The export columns

The platform documentation gives the exact columns, their order and the sheet names (the
decision's "Excel Export Structure", and the Agent-Level and Transaction-Level sections). For
Decision 5's own workbook it lists today `Agent ID`, `Honesty_Humility`,
`Assigned Allowance Level`, `Study Program`, `Group_experiment`,
`TWT+Sospeso [=AW2+AX2]{Periods 1+2}`, `income`, `Price`, `Quality`, `Proximity`,
`Sustainability`; in the Agent-Level and Transaction-Level sheets, `weight_price`,
`weight_quality`, `weight_proximity`, `weight_sustainability`. Implement the author's new columns
exactly; in the Agent-Level sheet, only when the row carries them, as Decision 4 does
(`if 'rtd_choice_length' in row:` in `app/reports/agent_level.py`).

Then add a missing-value case to `tests/test_export_missing_values.py`, and add every new table
builder to `export_frames` in `tests/test_golden_runs.py`, so the golden net covers it.

### 7.4 The export of a "Run X Only" run

`render_export_section` (`app/pages/results/components/export_section.py`) chooses the export by
the run's shape: Decisions 1 to 4 each have a focused export for their individual run (Decision
2's is `_render_disclose_documents_only_export`), every other run gets the two-level export
(Agent-Level and Transaction-Level). If the platform documentation describes a focused export for
the decision, add a branch on `ctx.is_individual_run("vendor_choice_weights")` and its function.

### 7.5 Pins, the complete-run gate and comparisons (only if needed)

If the tab can produce several configurations per click (an income radio with "Compare both", or
the "Compare all" population mode), the user must pin one ("Use This Config") before a complete
run. All four modelled decisions have this; it touches many lists written by hand, so agree with
the owner first. Then add the decision to:

* `app/state/saved_configs.py`: `get_current_*_params`, `calculate_*_metrics`,
  `extract_*_configuration_details`, and `get_decision_result_columns` (which picks the hashed
  columns by prefix);
* `app/pages/results/config_pin.py` (`_SELECTION_BUTTONS`) and `config_selection.py` (the banner);
* `_SELECTABLE_INDIVIDUAL_RUNS` in `app/pages/results/run_context.py`;
* `app/pages/results/main_results.py`: the two lists of pinnable decisions in
  `render_single_run_results`;
* `app/pages/results/overview.py` and `comparisons.py`: the analysis cell and the comparison
  grids;
* `app/pages/decision_execution.py`: `GATE_DECISIONS`, `GATE_SUCCESS_ORDER`,
  `GATE_BLOCK_TYPE_TITLES`, the gate's texts, and the list in `run_individual_decision` of
  decisions whose pin a "Run X Only" clears;
* `app/pages/page2_decisions.py`: a `render_selected_*_config_display` call in
  `render_overview_tab`, and the disabled button's help text;
* `app/seam/build_plan.py`: the saved record's precedence in the patch builder, and in
  `_resolve_modes` (a pinned population mode).

## 8. The nets

1. **Before you start**, `.venv/bin/python -m pytest tests/ -q` passes. **While you work**, run
   `.venv/bin/python -m pytest tests/ -q -m "not slow"` and the new reference test.
2. **Extend the nets to the new decision.**
   * `MODELLED` in `tests/test_golden_runs.py`: the scenarios
     `single_vendor_choice_weights_copula` and `single_vendor_choice_weights_baseline`.
   * `MODELLED` in `tests/test_golden_screens.py`: the journeys `single_vendor_choice_weights` and
     `tab_controls_vendor_choice_weights`, and a Page 2 step. The tab-controls journey needs a
     `TAB_WIDGETS` row (σ mode radio, per-level slider pattern, income radio, the two tick boxes,
     the reset button); a tab with other controls needs its own journey in `JOURNEYS`. An income
     radio's keys go in the tuple in `_complete`, which the continuous complete-run journey sets.
   * `DECISIONS` in `tests/test_complete_run_gate.py`, if you added a gate row: a list that does
     not fail when forgotten, but then misses the new decision.
3. **Read every failure before you re-record.** When the change is what you intend:

   ```bash
   GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/test_golden_runs.py \
       tests/test_golden_screens.py tests/test_complete_run_gate.py -q
   git diff --stat tests/golden tests/golden_screens
   ```

4. **Run everything again** and commit the new golden files with the change.

What a failure means:

* `test_golden_runs.py`: a result table or an export moved (the message names the columns).
  Expected for Decision 5: `vendor_choice_weights`, the new columns, and what depends on the
  weights (Decision 6's preferred vendor and request vendors, Decision 8, the vendor cells and
  scores of the exports), in every scenario where Decision 5 runs, complete runs included. **Not
  expected: any column of Decisions 1 to 4**; if one moves, something shared changed (the decision
  order, a shared helper, a random stream).
* `test_golden_screens.py`: something on screen or in a download changed (the message names the
  journey and the first differing element). Expected: the new tab (the complete-run journeys draw
  it), Decision 5's results section, the download hashes. A non-empty `browser_mismatches` means a
  rule of §6.5 is broken.
* `test_complete_run_gate.py`: expected to fail only if you added a gate row (§7.5).
* `test_session_key_registry.py`, `test_config_is_single_source.py`, `test_import_boundary.py`,
  `test_decision_order.py`: see `tests/README.md`; for the last, put `DECISION_ORDER` or the
  default table back.
* The new reference test: the code does not reproduce the document. Fix the code, or ask the
  author; never edit the expected numbers.

## 9. Documentation to update

In the same change:

* **`docs/README.md`**: the module map in §1 (four modelled decisions become five; the tab and
  visualisation lines), §3 (pass 1 statistics, if you added a block), §8 (the reference extract
  and the `PROF_DTA` list), §9 (every choice you made where the document left room, with its
  reason), §10 (known limits).
* **`docs/how-to-change-the-model.md`**: the reference table in §0, "the four modelled decision
  modules" in §1, the lists in §2 and §3 (`SCALAR_FIELDS`, the σ prefixes).
* **`docs/session-state.md`**: a row in the §4 table of keys the seam reads, the new reset in the
  §7 table, and §2 or §8 if the tab uses the shared controls or the twin spec.
* **`tests/README.md`**: the new test module, the new extract, the `.dta` file and where it lives,
  the new numbers of golden scenarios and journeys.
* **`docs/reference/README.md`**: the new document's row.
* **The platform documentation** (`docs/reference/COOPECON AI Agent Documentation <date>.docx`)
  belongs to the author: ask for the updated version with the decision's sections, and replace the
  file, keeping the date in its name.

## 10. Checklist

Names are Decision 5's; replace them for another decision.

1. [ ] Document read; every number and every variable listed (§1.1).
2. [ ] Every variable mapped to an agent column; a new trait added, copula retrained (§1.2).
3. [ ] Document in `docs/reference/decisions/`, data file in `data/stata/`, `PROF_DTA` set (§1).
4. [ ] Data file checked: 280 rows, participant order, traits, income (§1.3).
5. [ ] `experiments/d5_doc_reference.py` written; `data/d5_doc_reference.csv` frozen (§1.4).
6. [ ] Every constant in the `vendor_choice_weights:` block, with its source (§2).
7. [ ] Module in `MODULES` of `tests/test_config_is_single_source.py`, with a missing-key test
   (§2).
8. [ ] `src/decisions/vendor_choice_weights.py` rewritten: same function name, default branch
   kept, strict reads, `rng` only, output shape kept, prefixed model columns (§3).
9. [ ] A pass 1 block and a `RUN_SCRATCH_KEYS` entry, if the model needs a population statistic
   (§3.6).
10. [ ] The engine builds without an `ImportError`; the decision runs on the 280 from Python and
    the command line (§3.1, §3.8).
11. [ ] Every number of the document recomputed; each difference explained or answered (§4).
12. [ ] `tests/test_vendor_choice_weights_doc.py`: extract, constants, 280/280 per income and
    research mode, tables, figures, noise, default branch (§4).
13. [ ] `DEFAULT_DECISION_VALUES`, `DEFAULT_DECISION_DESCRIPTIONS`, Overview editor unchanged
    (§5).
14. [ ] Tab module written and routed in `render_decision_tab` (§6.1).
15. [ ] `CONTROL_KEYS` / `DISCLOSURE_KEYS` rows, if the tab uses the shared parts (§6.2, §6.3).
16. [ ] Registry rows, `TAB_WIDGET_CANONICAL`, `PREFIX_DELETE_PREFIXES`; registry test green
    (§6.4).
17. [ ] Session-state rules kept: two keys, `restore`, `stateful`, no re-push, reset at the top of
    the next run (§6.5).
18. [ ] Seam rows; `build_vendor_choice_weights_patch` in `build_decision_patches`; `base_state`
    updated (§6.6).
19. [ ] Results view rewritten; it reads only `RunContext` (§7.1).
20. [ ] Report builders in `app/reports/`: exact columns, `N/A` and money rules; missing-value
    test updated (§7.2, §7.3).
21. [ ] Focused "Run X Only" export, if the documentation describes one (§7.4).
22. [ ] Pins, gate and comparisons, only if agreed with the owner (§7.5).
23. [ ] Golden scenarios, screen journeys and the other test lists extended (§8).
24. [ ] Every golden failure read and explained; golden copies re-recorded; full suite green (§8).
25. [ ] `docs/README.md`, `how-to-change-the-model.md`, `session-state.md`, `tests/README.md`,
    `docs/reference/README.md` updated; platform documentation requested from the author (§9).
