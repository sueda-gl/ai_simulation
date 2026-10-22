# Tests

The tests are the specification: each module's docstring says what it pins and against
which reference. This page is the map.

```bash
pip install pytest                                   # not in requirements.txt
.venv/bin/python -m pytest tests/ -q -m "not slow"   # 493 tests, about 40 seconds
.venv/bin/python -m pytest tests/ -q                 # 767 tests, about 3 minutes
```

Run from the repository root: `pytest.ini` puts the root on `sys.path`
(`pythonpath = .`), so `tests/conftest.py` can import `app` and `src` whether you
call `pytest` or `python -m pytest`.

`slow` (registered in `pytest.ini`) is set per module (`pytestmark = pytest.mark.slow`)
on every module that drives the real app through Streamlit `AppTest` or
`tests/browser_sim.py`, or runs the simulation in a subprocess. The fast run leaves out
every test of such a module, including the module's own quick unit tests.

## References the tests pin

* **Frozen reference extracts** (in git): `data/stata_d1_verification.csv` (Decision 1),
  `data/stata_incomes.csv` (the reference income), `data/d3_doc_reference.csv`
  (Decision 3, written by `experiments/d3_doc_reference.py`),
  `data/stata_d4_verification.csv` and `data/stata_d4_flexibility_verification.csv`
  (Decision 4), plus the survey and experiment workbooks in `data/` that the research
  populations are built from.
* **The original Stata `.dta (in `data/stata/`)` files**, outside the repo. Four modules compare against
  them: `test_disclose_income_stata.py`, `test_disclose_documents.py`,
  `test_donation_default_doc.py` and `test_research_modes_use_reference_income.py`. Each
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
| `browser_sim.py` | `BrowserSim`: AppTest plus a model of the browser's widget-state manager, which catches a widget the browser would draw with another value than the script holds. |
| `ui_strings.py` | The on-screen wording the tests pin, in one place. |

Some modules also import from each other: `base_state` / `plan_for` / `ALL` from
`test_build_plan.py` (used by `test_golden_runs.py`, `test_mc_plan_parity.py`,
`test_decision_income_labels.py` and `test_browser_state.py`), `_assert_same_rows` from
`test_mc_plan_parity.py` (used by `test_mc_pinned_config.py`), `_first_draw` /
`_reset_consistent` from `test_browser_state.py` (used by
`test_rtd_sigma_coefficient_default.py`), and the `offenders` fixture from
`test_no_widget_default_warning.py` (used by `test_browser_state.py`).

## The test modules

### Fast (run by `-m "not slow"`)

| Module | What it covers | Pinned against |
|---|---|---|
| `test_build_plan.py` | The run-plan contract and `app/seam/build_plan.py`: snapshots in, sub-runs, messages and patches out; a pinned run reproduced through `execute`. | The original `app/simulation.py` mapping (read with `git show` at 80ee344 / e973a99) plus the specified deltas. |
| `test_category_first_income.py` | Income drawn inside the quintile that the Assigned Allowance Level names (`src/engine/income.py`). | The specification's rule. |
| `test_complete_run_gate.py` | Every outcome and warning text of `can_run_complete_simulation`. | Golden `tests/golden/complete_run_gate.json.gz`. |
| `test_config_is_single_source.py` | No decision module carries a literal copy of a YAML constant; a missing key raises naming the key. | `config/decisions.yaml`. |
| `test_decision_order.py` | One decision order for engine, screens, seam and default registries. | `src/engine/core.py` `DECISION_ORDER`. |
| `test_disclose_documents.py` | Decision 2: 76/280 categorical and 63/280 continuous disclosure, every allowance cell, 280/280 participant match. | Corrected Stata Decision 2 results; `data/stata_incomes.csv`; the `.dta` at `PROF_DTA`. |
| `test_disclose_documents_gate.py` | Decision 2's platform gate and default short-circuit on the shipped configuration. | `config/decisions.yaml`. |
| `test_disclose_income_stata.py` | Decision 1: continuous 280/280, categorical 279/280 (participant 848), the document's constants. | `data/stata_d1_verification.csv`; the `.dta` at `PROF_DTA`. |
| `test_donation_default_doc.py` | Decision 3 participant by participant at shift 0 / sigma 0, in both income modes; noise gate and rescale. | The methodology document, re-implemented in `data/d3_doc_reference.csv`; the `.dta` at `PROF_DTA`. |
| `test_engine_helpers.py` | `should_use_stochastic`, `assign_global_transaction_ids`, `build_simulation_params`. | Their contracts. |
| `test_export_pricing.py` | Customer Price per transaction type; one vendor score across the Agent- and Transaction-Level exports. | `docs/reference/COOPECON AI Agent Documentation 2026-10-07.docx`. |
| `test_export_missing_values.py` | The one rule for a missing value in every sheet: `N/A`, nothing substituted (group, income, vendor, weights, Y/N labels, product id, averages). | `docs/README.md` §9. |
| `test_golden_runs.py` | 15 runs through snapshot, plan and execute: result frames, vendor lists, 13 export builders. | Golden `tests/golden/<scenario>/`. |
| `test_import_boundary.py` | `src/` never imports `streamlit` or `app`; `app/reports/` never imports `streamlit`; `app/state/` never imports `app.pages` (subprocess and AST). | The import rules (`docs/README.md` §6). |
| `test_no_consensus_in_outputs.py` | Exports and screens say "integrated", never "consensus". | The output wording rule. |
| `test_reports_interface.py` | What each `app/reports` builder returns, its sheet names and promised columns. | The builders' documented shape. |
| `test_research_cyclic_population.py` | Research Baseline cycles through the 280 in file order; Research Specification samples randomly by seed. | The specification's sampling rule. |
| `test_research_modes_use_reference_income.py` | Research populations carry the reference income; the Decision 4 Research Baseline run reproduces the file. | `data/stata_incomes.csv`, the Decision 4 extracts; the `.dta` at `PROF_DTA`. |
| `test_results_module_split.py` | The results-components shim removed in 0ca3a29 stays gone; `viz_helpers.yes_no_pie` draws the figure the four old copies drew. | The old `px.pie` call on a golden Disclose Income frame. |
| `test_rtd_flexibility.py` | Decision 4 Flexibility: the deterministic pipeline 280/280, beta4, fallback, stochastic layer, copula trait. | `data/stata_d4_flexibility_verification.csv`. |
| `test_rtd_histogram_stata_parity.py` | Decision 4 score histograms use Stata's default bin rule (16 bins at N = 280) and draw the document's figure. | `data/stata_d4_verification.csv`; the design document's figure. |
| `test_rtd_rank_aggregation.py` | Decision 4 Section 6: Kemeny-Young with the Schulze / Copeland / footrule / random cascade, and its integration. | Decision 4 document rev 040926-2 Section 6; "Ranking Cascade Results (V1)" stage shares. |
| `test_session_key_registry.py` | Every `st.session_state` key the app names has a registry entry and every entry is used (AST scan of `app/`). | `app/state/registry.py`. |

### Slow (`pytestmark = pytest.mark.slow`)

| Module | What it covers | Pinned against |
|---|---|---|
| `test_browser_state.py` | Widgets show in the browser the value the script runs with; Page 1 and tab resets; sigma coefficient 0; the gate's headings. | `config/decisions.yaml` and a fresh session's first draw, through `browser_sim.py`. |
| `test_continuous_only_income_ui.py` | Decision 1 / 2 runs in "Continuous only" render results, without a duplicate overview. | Two reported bugs. |
| `test_decision_income_labels.py` | A complete run records Decision 4's own income mode, and the results page labels it. | The run metadata. |
| `test_golden_screens.py` | 24 journeys through the real app: every element, table and download after every click. | Golden `tests/golden_screens/<journey>.json.gz`. |
| `test_mc_pinned_config.py` | Monte Carlo repeats a pinned ("Use This Config") configuration with its own agents and population. | The pinned complete run, agent by agent. |
| `test_mc_plan_parity.py` | Monte Carlo (in process and the `scripts/run_mc_study.py` subprocess) repeats the single run's plan. | A single run with the same seed, agent by agent. |
| `test_no_widget_default_warning.py` | No "created with a default value ... Session State API" warning on Page 1 or Page 2. | Streamlit's own warning rule. |
| `test_override_rerun_cost.py` | Override and Donation intercept clicks: the parsed YAML is cached, the Override is a fragment, nothing stale is pushed back. | The reported freeze and lost clicks. |
| `test_page1_income_range.py` | A maximum at or below the minimum is reported on Page 1, not a crash. | A reproduced browser crash. |
| `test_pin_seed_mismatch.py` | A configuration that cannot be pinned (other seed or agent count) shows an error and pins nothing. | `validate_seed_consistency`. |
| `test_rejected_transaction_defaults.py` | Decision 4: every deterministic mechanism 280/280, the WTP stochastic pipeline, intercepts, categorical income, one AppTest run. | `data/stata_d4_verification.csv`; Decision 4 document rev 040926-2. |
| `test_rtd_batch4_ui.py` | Decision 4 results display and exports: histograms, allocation charts, run shapes, workbooks. | The Decision 4 display specification. |
| `test_rtd_results_ui.py` | The Decision 4 results page: subtitles, configuration summary, score charts, complete run, "Use This Config", comparison layouts. | The Decision 4 display specification. |
| `test_rtd_sigma_coefficient_default.py` | Decision 4's sigma coefficient defaults to 0.5, in the tab, on reset and in a run. | `config/decisions.yaml`; the updated user manual. |
| `test_rtd_tab_text_ui.py` | The Decision 4 configuration tab's wording and structure. | The review round that settled the tab's wording. |
| `test_run_context_run_values.py` | The results page reads the run's Page 1 values and default-decision settings, not the live widgets. | `RunMetadata`. |
| `test_stochastic_widgets_no_reset.py` | No stochastic control on Page 2 resets the session; no UI action writes `config/decisions.yaml`. | The reported reset glitch. |
