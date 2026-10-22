# How to change the model

The thirteen decisions are fixed; what changes over time is their constants, their
equations, the controls the user can adjust, the noise model, and what the exports
contain. This page is the recipe for each kind of change, and the routine that proves
nothing else moved. It assumes you have read [`README.md`](README.md), in particular
§3 (where the random numbers come from) and §9 (the design decisions).

The rules of the model are in the platform documentation under
[`reference/`](reference/README.md). A change to the model starts from a change to
that document, or from a correction the document already contains and the code
does not.

---

## 0. The routine, before and after every change

```bash
pytest tests/ -q                       # everything passes before you start
```

Two tests pin the behaviour cell by cell and must be read as "did anything move":

| Test | Pins | Takes |
|---|---|---|
| `tests/test_golden_runs.py` | fifteen fixed runs: every result frame, the vendor list, and every export builder's output | 4 s |
| `tests/test_golden_screens.py` | eleven journeys through the real app: every element on screen and every download after every click | 15 s |

Both compare against the copies in `tests/golden/` and `tests/golden_screens/`.

* A **refactor** must leave them green. If one fails, the refactor changed behaviour:
  find out why before going on.
* An **intended change** to the model or the screens makes them fail by design. Then,
  once the change is verified (below), regenerate the copies and commit them in the
  same commit as the change, so the diff of the hash files shows exactly which
  columns moved:

  ```bash
  GOLDEN_UPDATE=1 pytest tests/test_golden_runs.py tests/test_golden_screens.py -q
  git add tests/golden tests/golden_screens
  ```

The four modelled decisions are also pinned, participant by participant, to the
reference files in `data/`:

| Decision | Test | Reference |
|---|---|---|
| 1 Disclose income | `tests/test_disclose_income_stata.py` | `data/stata_d1_verification.csv` |
| 2 Disclose documents | `tests/test_disclose_documents.py` | the document's tables and `data/stata_incomes.csv` |
| 3 Donation default | `tests/test_donation_default_doc.py` | `data/d3_doc_reference.csv` |
| 4 Rejected-transaction defaults | `tests/test_rejected_transaction_defaults.py`, `tests/test_rtd_flexibility.py` | `data/stata_d4_verification.csv`, `data/stata_d4_flexibility_verification.csv` |

A change to one of these decisions must come with a new reference extract, or with
an explicit, documented reason why the old one no longer applies. Changing the
expected numbers in the test to make it pass is not a verification.

---

## 1. Change a constant or coefficient

Every constant lives in `config/decisions.yaml`, under the decision's block, and that
file is the only source of truth: the app never writes it, and a user's change travels
to the engine as a patch on a fresh copy. The four modelled decision modules read every
constant strictly from that block (`require` in `src/decisions/_shared.py`): a key the
block lacks raises a `KeyError` naming the decision and the dotted key path, never a
module fallback, and `tests/test_config_is_single_source.py` fails if a module
re-introduces a literal copy of a file value. The few module-level names the app and the
tests import by name (`PRIORITY_SEQUENCES`, `SIGMA_OVERALL`, `DEFAULT_Z_SCORING`,
`FLEX_COEFFS` in `src/decisions/rejected_transaction_defaults.py`, `DOC_COEFFICIENTS`
in `donation_default.py`, ...) are read from the file at import through the one loader
in `src/config.py`; there is nothing to keep in step. Change the YAML; cite the
document revision the value comes from in the YAML comment. (The only reads that
tolerate an absent key are two Decision 4 blocks whose meaning existing tests pin:
`intercepts`, where an absent element is 0, and `aggregation`, where a partial block is
merged over the file's.)

Then: the decision's reference test tells you whether the new value reproduces the
reference file (if the reference moved too, regenerate the extract first), and the
routine of §0 tells you what else moved. A constant that the user can adjust on a tab
has a default on the tab as well: see §3.

## 2. Change a decision's equation or rule

The code of a decision is one module, `src/decisions/<name>.py` (Decisions 1 and 2:
`<name>_stochastic.py`), with one function of the decision's name:

```python
def my_decision(agent_state: dict, params: dict, rng, simulation_config: dict = None, **kwargs) -> dict:
```

| argument | what it holds |
|---|---|
| `agent_state` | the agent's traits **plus everything the earlier decisions returned**; a decision reads a previous one's output as `agent_state['disclose_income']` |
| `params` | this decision's block of `config/decisions.yaml`, already patched with the user's settings |
| `rng` | a `numpy.random.Generator` seeded for this agent × this decision. **Draw only from this.** Never `random.random()`, never `np.random.*` |
| `simulation_config` | the whole run configuration; read Page-1 parameters with `get_simulation_param(simulation_config, 'market_price', 100.0)` from `src/engine/config_access.py`, never with a top-level `.get` |
| `pop_context` (in `kwargs`) | the population mode, for the four modelled decisions; it drives the noise gate (§5) |

What it returns becomes columns of the results frame and is merged into
`agent_state` for the decisions after it.

Rules the tests enforce:

* **Never `import streamlit`, never `from app… import …`**, not even inside a
  function. A value that lives in the UI reaches the engine through `simulation_config`.
* Do nothing at import time.
* Raise on bad input; never substitute a placeholder and continue.
* Use the shared helpers in `src/decisions/_shared.py` (z-scoring, the categorical
  intercept lookup, income-mode normalisation) and the one noise gate
  `should_use_stochastic` in `src/utils/stochastic.py`, rather than a private copy.
* If the rule needs a **population-level** statistic (a median, a maximum over all
  agents), compute it in Pass 1: add a hook in `Engine._compute_population_stats`
  (`src/engine/core.py`) and store the result on `simulation_config`. Decision 3's
  population maximum and Decision 4's score ranges are the patterns to copy.
* **Do not change the order of `DECISION_ORDER`** in `src/engine/core.py`. A
  decision's position seeds its random numbers; moving one changes every later
  decision's draws in every mode. Retiring a decision is covered in §8.

Then the routine of §0, and a new reference extract for a modelled decision.

## 3. Change a user-adjustable parameter

A setting the user can change on a decision tab travels through four places, in this
order. Add or change all four.

1. **The widget**, in `app/pages/decision_tabs/<decision>.py`, written through
   `stateful(st.<widget>, …, key=…)` with a `<prefix>_tab_` key, and mirrored every
   render into the canonical key the seam reads (`di_intercept`, `dd_intercept`,
   `rtd_intercept_<mechanism>`, …). [`session-state.md`](session-state.md) explains
   the layers and the naming.
2. **The registry**, `app/state/registry.py`: one `KeySpec` per key, with
   `engine_input=True` for the canonical key. `tests/test_session_key_registry.py`
   refuses an unregistered key and a registered key nothing uses.
3. **The seam**, `app/seam/build_plan.py`: the canonical key becomes a patch on the
   decision's YAML block. For a scalar, add one row to `SCALAR_FIELDS` (session key,
   path in a saved configuration, path in the YAML block, optional transform). The
   donation coefficient set and the Decision 4 per-mechanism sigma are the two
   explicit special cases next to it. An absent key must mean the tab's own default,
   never "off": the same visible settings must give the same numbers whether or not
   the tab was opened.
4. **The YAML default**, `config/decisions.yaml`, which is also what the tab's reset
   button restores.

Then the routine of §0. The screen net will fail on the new widget by design;
regenerate it once the screen is what you intend.

## 4. Change what an unselected decision does

When a decision is not selected, it runs with the default in
`src/contract/defaults.py` (`DEFAULT_DECISION_VALUES`, and the one-line
`DEFAULT_DECISION_DESCRIPTIONS` the Overview tab shows). A decision reads its default
from `simulation_config['default_decisions'][<name>]`; Decisions 11, 12 and 13 are
the pattern. The Overview tab's editor for these defaults is
`app/pages/decision_tabs/default_config.py`, and the shapes it knows are
`random_probability`, `checkbox_selection`, `radio_selection`,
`prioritized_selection` and a plain constant.

## 5. Change the noise (σ) model

Each modelled decision has a `stochastic:` block in its YAML:

```yaml
stochastic:
  sigma_strategy: overall     # or quintile
  sigma_overall: 9.899547     # the base σ, from the reference data
  sigma_quintile: {'1': …, '2': …, '3': …, '4': …, '5': …}
  sigma_value: 0              # the Research tick box writes σ here
  scale_factor: 1.0           # the σ coefficient slider
  in_copula: false            # the "draw in Copula mode" tick box
```

The one rule that decides whether noise is added is `should_use_stochastic()` in
`src/utils/stochastic.py`: Copula draws iff `in_copula`, Research Specification
draws iff `sigma_value > 0`, Baseline never. Do not add a second rule. The base σ
constants are derived from the raw experiment files; Decision 2's derivation is
reproducible with `python scripts/build_dd_sigma.py`. A new base σ goes into the
YAML with its derivation named in the comment.

## 6. Change an export or a chart

* **Anything computed** (a frame, a statistic, the bytes of a workbook) lives in
  `app/reports/`, takes its inputs as arguments and returns a value. Nothing there
  imports Streamlit; `tests/test_import_boundary.py` enforces it.
* **The page** keeps the `st.*` calls and the session reads, and asks `RunContext`
  (`app/pages/results/run_context.py`) about the run on screen, never the live
  Page-1 / Page-2 widgets, which describe the next run.
* The exact columns of every export are specified in the agent documentation under
  `reference/` (section "Excel Export Structure" and the per-decision export
  paragraphs). Customer Price and the vendor score follow the rules recorded in
  [`README.md`](README.md) §9 and pinned by `tests/test_export_pricing.py`.
* A chart for a decision is registered in `DECISION_VISUALIZATIONS` in
  `app/pages/results/visualizations/__init__.py`.

Then the routine of §0: the golden exports and the golden screens will name the
changed columns and elements.

## 7. Change the population or the Page-1 parameters

* Page-1 parameters are the `SimulationParameters` dataclass in `app/models.py`,
  mirrored by the Page-1 widgets, copied into the run by `build_simulation_params`
  (`app/seam/build_plan.py`), and read by decisions through `get_simulation_param`.
  `config/simulation.yaml` holds the engine's own defaults for the same keys; keep
  the two in step.
* The 280 participants come from the two workbooks in `data/` through
  `src/data/participants.py`; the synthetic population from the copula in
  `config/trait_model.pkl`, trained by `scripts/train_copula.py` on the traits listed
  in `config/trait_requirements.yaml`. Retrain only if the experiment data or the
  trait list changes, and expect every copula number to move.
* The research sampling rule (Specification random and seeded, Baseline in file
  order cycling) is `src/engine/sampling.py` and is pinned by
  `tests/test_research_cyclic_population.py`.

## 8. Retire a decision

Decisions 9 and 10 are the live example: the documentation calls them
transaction-level, the per-request choice and bid value are drawn inside Decision 6,
and the per-agent values the two slots still produce are read by nothing. Retiring a
slot is a behaviour change, not a refactor, because of the random-number scheme:

* removing an entry from `DECISION_ORDER` shifts the position, and so the seed, of
  every decision after it. Check which of those decisions actually draw from `rng`
  (Decisions 11, 12 and 13 do not), and expect the others to move;
* the results frame loses the slot's columns, so every pinned configuration whose
  hash covered them stops reproducing;
* the same name must leave `DEFAULT_DECISION_VALUES`, `DECISION_VISUALIZATIONS`, the
  Overview tab and the registry; `tests/test_decision_order.py` ties them together.

Get the change ruled from the documentation, make it, regenerate the golden copies,
and say in the commit which columns and which decisions moved.

## 9. If a decision is ever added

Append it at the **end** of `DECISION_ORDER`, so every existing stream keeps its
offset; give it a module (§2), a YAML block (§1), an entry in
`DEFAULT_DECISION_VALUES` (§4), a tab and a `SCALAR_FIELDS` row if it is configurable
(§3), and a renderer plus report builders (§6). The import-boundary test picks the
new module up by itself. Expected result on a fixed-seed run before and after:
every existing column identical, the new columns added.
