# Tests

The tests are the specification: each module's docstring says what it pins and against
which reference. This page is the map.

```bash
pip install pytest                                   # not in requirements.txt
.venv/bin/python -m pytest tests/ -q -m "not slow"   # 407 tests, about 35 seconds
.venv/bin/python -m pytest tests/ -q                 # 474 tests, about 80 seconds
```

Run from the repository root: `pytest.ini` puts the root on `sys.path`
(`pythonpath = .`), so `tests/conftest.py` can import `app` and `src` whether you
call `pytest` or `python -m pytest`.

`slow` (registered in `pytest.ini`) is set per module (`pytestmark = pytest.mark.slow`)
on every module that drives the real app through Streamlit `AppTest` or
`tests/browser_sim.py`, or runs the simulation in a subprocess. The fast run leaves out
every test of such a module, including the module's own quick unit tests. Three modules
are slow: `test_rejected_transaction_defaults.py`, `test_golden_screens.py` and
`test_reset_reaches_overview.py`.

The owner keeps an additional suite outside the repository; nothing in `tests/`
depends on it.

## References the tests pin

* **Frozen reference extracts** (in git): `data/stata_d1_verification.csv` (Decision 1),
  `data/stata_incomes.csv` (the reference income the parity tests inject), `data/d3_doc_reference.csv`
  (Decision 3, written by `experiments/d3_doc_reference.py`),
  `data/stata_d4_verification.csv` and `data/stata_d4_flexibility_verification.csv`
  (Decision 4), plus the survey and experiment workbooks in `data/` that the research
  populations are built from.
* **The original Stata `.dta` files**, in `data/stata/`. Four modules compare against
  them: `test_disclose_income_stata.py`, `test_disclose_documents.py`,
  `test_donation_default_doc.py` and `test_research_modes_income.py`. Each
  test that reads one fails when the file is missing, naming the file and the `PROF_DTA`
  constant at the top of its module; with `COOPECON_ALLOW_DTA_SKIP=1` it skips instead.
* **Golden copies** recorded from the app itself, the parity net for refactoring:

  | Golden | Written by | Regenerate (only after an intended change) |
  |---|---|---|
  | `tests/golden/<scenario>/` (15 scenarios) | `test_golden_runs.py` | `GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/test_golden_runs.py -q` |
  | `tests/golden/complete_run_gate.json.gz` | `test_complete_run_gate.py` | `GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/test_complete_run_gate.py -q` |
  | `tests/golden_screens/<journey>.json.gz` (24 journeys) | `test_golden_screens.py` | `GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/test_golden_screens.py -q` |

  All three read the same `GOLDEN_UPDATE` variable, so `GOLDEN_UPDATE=1` on a wider run
  re-records every golden it reaches. Afterwards review
  `git diff --stat tests/golden tests/golden_screens` and commit the new files with the
  change that caused them.

## Shared code (not tests)

| File | What it holds |
|---|---|
| `conftest.py` | Repository paths (`REPO`, `ROOT`, `APP_FILE`, ...), `load_decisions_yaml`, `start_app`, `require_reference_file`, and the module-scoped fixtures `gold` / `sim_config` (the Decision 4 reference extract) and `repo` / `default_values` (the run seam). |
| `helpers/apptest.py` | `all_markdown`, `all_captions` (AppTest page readers) and `rtd_tab_and_results_script` (the Decision 4 tab plus the results page, Research Baseline, 60 agents). |
| `helpers/d4_reference.py` | `agent_state`: a row of a Decision 4 reference extract as the agent state the decision functions take. |
| `helpers/seam.py` | `base_state` (a session snapshot with the app's defaults for every key the seam reads), `plan_for` (its run plan) and `ALL` (the decision order): the session the golden runs start from. |
| `browser_sim.py` | `BrowserSim`: AppTest plus a model of the browser's widget-state manager, which catches a widget the browser would draw with another value than the script holds. |
| `ui_strings.py` | The on-screen wording the tests pin, in one place. |

No test module imports another.

## The test modules

### Reference parity

| Module | What it covers | Pinned against |
|---|---|---|
| `test_disclose_income_stata.py` | Decision 1: continuous 280/280, categorical 279/280 (participant 848), the document's constants. | `data/stata_d1_verification.csv`; the `.dta` at `PROF_DTA`. |
| `test_disclose_documents.py` | Decision 2: 76/280 categorical and 63/280 continuous disclosure, every allowance cell, 280/280 participant match. | Corrected Stata Decision 2 results; `data/stata_incomes.csv`; the `.dta` at `PROF_DTA`. |
| `test_disclose_documents_gate.py` | Decision 2's platform gate (a 50 % draw when Decision 1 has not run) and default short-circuit, on the shipped configuration. | `config/decisions.yaml`; the platform documentation's Eligibility Gate. |
| `test_donation_default_doc.py` | Decision 3 participant by participant at shift 0 / sigma 0, in both income modes; noise gate and rescale. | The methodology document, re-implemented in `data/d3_doc_reference.csv`; the `.dta` at `PROF_DTA`. |
| `test_rejected_transaction_defaults.py` (slow) | Decision 4: every deterministic mechanism 280/280, the WTP stochastic pipeline, intercepts, categorical income, one AppTest run. | `data/stata_d4_verification.csv`; Decision 4 document rev 040926-2. |
| `test_rtd_flexibility.py` | Decision 4 Flexibility: the deterministic pipeline 280/280, beta4, fallback, stochastic layer, copula trait. | `data/stata_d4_flexibility_verification.csv`. |
| `test_rtd_rank_aggregation.py` | Decision 4 Section 6: Kemeny-Young with the Schulze / Copeland / footrule / random cascade, and its integration. | Decision 4 document rev 040926-2 Section 6; "Ranking Cascade Results (V1)" stage shares. |
| `test_rtd_histogram_stata_parity.py` | Decision 4 score histograms use Stata's default bin rule (16 bins at N = 280) and draw the document's figure. | `data/stata_d4_verification.csv`; the design document's figure. |
| `test_research_modes_income.py` | Every population mode draws its income from Page 1 (seeded; moves with the Page 1 settings); with the reference income injected, the Decision 4 Research Baseline run reproduces the file. | `data/stata_incomes.csv`, the Decision 4 extracts; the `.dta` at `PROF_DTA`. |

### Golden nets

| Module | What it covers | Pinned against |
|---|---|---|
| `test_golden_runs.py` | 15 runs through snapshot, plan and execute: result frames, vendor lists, 13 export builders. | Golden `tests/golden/<scenario>/`. |
| `test_golden_screens.py` (slow) | 24 journeys through the real app: every element, table and download after every click. | Golden `tests/golden_screens/<journey>.json.gz`. |
| `test_complete_run_gate.py` | Every outcome and warning text of `can_run_complete_simulation`. | Golden `tests/golden/complete_run_gate.json.gz`. |

### Rules

| Module | What it covers | Pinned against |
|---|---|---|
| `test_config_is_single_source.py` | No decision module carries a literal copy of a YAML constant; a missing key raises naming the key. | `config/decisions.yaml`. |
| `test_decision_order.py` | One decision order for engine, screens, seam and default registries. | `src/engine/core.py` `DECISION_ORDER`. |
| `test_import_boundary.py` | `src/` never imports `streamlit` or `app`; `app/reports/` never imports `streamlit`; `app/state/` never imports `app.pages` (subprocess and AST). | The import rules (`docs/README.md` §6). |
| `test_session_key_registry.py` | Every `st.session_state` key the app names has a registry entry and every entry is used (AST scan of `app/`). | `app/state/registry.py`. |
| `test_reset_reaches_overview.py` (slow) | A whole-tab "Reset Config to Defaults" (Disclose Income, Disclose Documents, Donation Default) out of "Compare both" unblocks the Page-2 Overview's "Run Complete Simulation" in the run the reset causes. | The Overview is drawn before the tabs; Page 2 applies a pending reset first (`page2_decisions.apply_pending_tab_resets`). |

### Exports

| Module | What it covers | Pinned against |
|---|---|---|
| `test_export_pricing.py` | Customer Price per transaction type; one vendor score across the Agent- and Transaction-Level exports. | `docs/reference/COOPECON AI Agent Documentation 2026-10-07.docx`. |
| `test_export_missing_values.py` | The one rule for a missing value in every sheet: `N/A`, nothing substituted (group, income, vendor, weights, Y/N labels, product id, averages). | `docs/README.md` §9. |
