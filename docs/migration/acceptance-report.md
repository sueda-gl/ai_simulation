# Final acceptance report — logic / UI split

**Tree under test.** `/Users/suedagul/coopecon-migration`, branch `migration`, HEAD `2860584`
("Migration step 4: export builders moved out of the screens") plus the uncommitted C1 (code clean-up)
and C3 (session-key registry) work: 19 modified files, `src/decisions/donation_default_stochastic.py`
deleted, and four untracked additions (`app/state/registry.py`, `src/contract/defaults.py`,
`tests/test_session_key_registry.py`, `docs/`). `git status --short` was the same 24-line list before
and after this acceptance pass, apart from the documentation edits listed in §7 (which live inside that
list: `README.md` and the untracked `docs/` folder). Nothing was committed and no branch was switched.

**Interpreter.** `/Users/suedagul/coopecon-migration/.venv/bin/python` (the `.venv` symlink resolves to
the original project's virtual environment: Python 3.13.3, streamlit 1.49.0), run from the migration
repository with `PYTHONPATH=/Users/suedagul/coopecon-migration`.

**The original repository** `/Users/suedagul/<sdg` was opened read-only twice (`git status --short`,
`git log -1`): clean, at `80ee344`, branch `perf`, before and after this pass. It was never edited.

**Baselines.** `step1` = the screens and numbers of the ORIGINAL behaviour (captured before step 2);
`step4` = the capture of the committed step-4 tree (the state before C1/C3). Both captures, the new
`final` capture, the comparison outputs and every script named below live outside the repository under
`scratchpad/reference/{step1,step4,final}` and `scratchpad/verify/final/`
(`/private/tmp/claude-501/-Users-suedagul--sdg/7e5dc018-2435-4c04-abe0-987fa357d237/scratchpad`).

**Verdict: pass.** Every difference against the original behaviour carries an owner ruling. Nothing is
unattributed. The one ruled change that was still outstanding when this report was first written (R8)
was applied on 2026-09-04 after the owner's ruling — see §5; the journeys did not move. Later the same day
the owner ruled on three of the deferred items (§10): the shared income SD now divides by N−1 (R-SD; one
line, no decision in any journey flipped), and the Decision 1 scale factor (R-DI01) and the Period-2 merge
(R-P2) were confirmed correct as they stand.

---

## 1. Tests

| command | result |
|---|---|
| `.venv/bin/python -m pytest tests/ -q` | **215 passed**, 0 failed, 34.5 s |
| `.venv/bin/python -m pytest tests/test_import_boundary.py -q` | **106 passed**, 23.6 s |
| `.venv/bin/python -m pytest tests/ -q` — re-run after R8 was applied (2026-09-04) | **215 passed**, 0 failed |

The headline moved from 204 to 215 because C3 added `tests/test_session_key_registry.py` (11 tests).
Per file: `test_import_boundary.py` 106, `test_rejected_transaction_defaults.py` 32, `test_build_plan.py` 31,
`test_category_first_income.py` 16, `test_session_key_registry.py` 11, `test_rtd_batch4_ui.py` 6,
`test_continuous_only_income_ui.py` 5, `test_disclose_documents.py` 5, `test_rtd_comparison_layout_ui.py` 3.
The four registry failures C1 reported while C3 was still writing the registry are gone: the registry is
complete against the tree (220 key specs) and all 11 tests pass.

## 2. Boundary

| check | evidence |
|---|---|
| no `streamlit` under `src/` | `grep -rn streamlit src/ --include='*.py'` → 0 hits (the word appears once, in a docstring of `src/contract/plan.py:184` describing where `RunMetadata` is stored — no import, no attribute access) |
| no `streamlit` under `app/reports/` | `grep -rn streamlit app/reports/ --include='*.py'` → 0 hits |
| no `st.session_state` read under `src/` | `grep -rn session_state src/` → only the docstring above |
| no YAML write under `app/` | `grep -rn 'yaml.dump\|safe_dump\|.dump(' app/ src/` → 0 hits |
| no file opened for writing under `app/` | every `open(` under `app/` is mode `'r'` (`app/models.py:252,355,374`, `app/seam/config_repo.py:42,44`, five tab modules reading `config/decisions.yaml`); no `write_text`, `makedirs`, `shutil`, `to_csv(path)`, `to_excel(path)` — `app/reports/xlsx.py:42` writes into a `BytesIO`, `app/reports/mc.py:150,159` return CSV strings |
| `pd.ExcelWriter` / `BytesIO` in the screens | `grep -rn "ExcelWriter\|BytesIO" app/pages/` → 0 hits (the six inline blocks step 4 left in `export_section.py` are gone; all workbooks come from `app.reports.xlsx.to_xlsx_bytes`) |
| the boundary test | `tests/test_import_boundary.py`: a fresh subprocess per module under `src/` and `app/reports/` plus an AST scan; 106/106 green |

## 3. Journeys — final vs step4 (must be identical except the ruled title fix)

`run_journeys.py --repo /Users/suedagul/coopecon-migration --out reference/final`: 17/17 journeys `rc=0`,
`FAILED: []` (P0 2.2 s, n=50 runs 4.8–5.2 s, n=280 runs 18.1–20.1 s).
`compare_step3_closeout.py step4 final` → **ROLLUP frames_identical=True downloads_identical=True;
identical snapshots 34, differing 6.**

* 17 frames (16 journeys + the J16 run-only phase): values, `columns.json`, `attrs.json` and every manifest
  sha256 identical.
* Downloads identical in all 20 snapshots that carry any (same labels, file names, sheet lists, normalised
  workbook sha256).
* Session dumps identical for all 17 journeys.
* The 6 differing snapshots are `J2`, `J4`, `J6` `results_page` (ids kept and ids dropped), each with exactly
  **2 changed records** — the same markdown element on both sides:
  `**📈 Raw Disclose Documents Distribution (Categorical)**` → `**📈 Raw Disclose Documents Distribution (Continuous)**`.
  These are the three "continuous only" complete runs; the title now names the run's real income mode.
  **Ruling R28** (owner accepted the `(Continuous)` title). No other record differs anywhere.

The Disclose Income title site uses the same helper (`_income_mode_suffix()` in `disclosure_viz.py`) but is
not exercised by a continuous journey (no journey selects Decision 1 in continuous mode), so it does not
appear in the diff.

## 4. Journeys — final vs step1 (the original behaviour): complete attribution

Method: `compare_step3_closeout.py step1 final` (full-record page diff after auto-id masking, per-column
frame diff, download sha diff, session dict diff) followed by `attribution.py` over its JSON. **16 snapshots
identical** (P0 page 1 and page 2; J9, J10, J11, J12 results pages; J16 run-only and after-pin pages — each
with ids kept and dropped). Every changed record on the 12 differing results pages was classified; **zero
records fall outside the ruled classes.**

Rulings referenced: **R9** vendor migration block deleted (single vendor now runs at the typed price 100 and
100 products instead of `uniform(50,150)` = 127.3956 / 93 — changes bid values and everything computed
from them); **R12** one sigma constant (donation σ 9.8995 → 9.899547 from the file); **R13** no auto-implied
saved configurations; **R14** explicit pin + reproduction hash; **R28** results page reads run metadata, the
run writes no mode keys, the DD/DI distribution titles show the run's real income mode; **R27** dead post-run
blocks removed (the `- Decisions: N` line — reads exactly as step 1 for every journey, so it is not in the
diff); **R18** absent sigma key = tab default (control journey J15).

### 4.1 Per journey

| journey | frames (column → ruling) | attrs | downloads changed | page records changed | session keys | Decisions 1–4 columns |
|---|---|---|---|---|---|---|
| P0 | — | — | — (0 files) | page1_default, page2_default **identical** | `sigma_value_ui` 9.8995 → 9.899547 (R12) | — |
| J1 Copula×cat n=50 | `categorical` [50×53]: `purchase_requests` 23/50 rows — `bid_value` inside the request dicts (R9) | `vendors` price 127.3956→100, quantity 93→100; `simulation.vendor_price_min/max` 50/150→100, `vendor_products_min/max` 50/150→100 (R9) | 7/14: purchasing_transactions, purchase_requests_detailed, purchase_now_vs_bid_decisions, bid_values, donation_transactions, simulation_agent_level, simulation_transaction_level (R9) | Mean/Min/Max Bid metrics (3), `✅ N unique bid values` caption, the `Distribution of N Bid Values Across All Requests` histogram, the 7 download buttons, 4 bid-dependent preview dataframes (`📥 Export Transaction Data`, `📊 Statistics`, purchase-request download caption, `📋 Data Preview`) — all R9 | `sigma_value_ui` (R12); `_run_metadata` new (R28) | **identical** (`disclose_income*`, `disclose_documents*`, `donation_default`, `rtd_*`, `final_donation_rate`) |
| J2 Copula×cont n=50 | as J1 (`continuous`) | as J1 (R9) | as J1 (R9) | as J1 (R9) **+** `Raw Disclose Documents Distribution (Categorical)` → `(Continuous)` (R28) | as J1 | **identical** |
| J3 ResSpec×cat n=280 | `purchase_requests` 132/280 (R9) | R9 | as J1 (R9) | as J1 (R9) | as J1 | **identical** |
| J4 ResSpec×cont n=280 | `purchase_requests` 132/280 (R9) | R9 | as J1 (R9) | as J1 (R9) + title (R28) | as J1 | **identical** |
| J5 ResBase×cat n=280 | `purchase_requests` 132/280 (R9) | R9 | as J1 (R9) | as J1 (R9) | as J1 | **identical** |
| J6 ResBase×cont n=280 | `purchase_requests` 132/280 (R9) | R9 | as J1 (R9) | as J1 (R9) + title (R28) | as J1 | **identical** |
| J7 Copula×cat, donation/DI/DD copula ticks (no decision selected) | `purchase_requests` 23/50 (R9) | R9 | as J1 (R9) | as J1 (R9) | as J1 | **identical** (ticks have no effect on default-value runs, same as step 1) |
| J8 ResSpec×cat, all research ticks (no decision selected) | `purchase_requests` 132/280 (R9) | R9 | as J1 (R9) | as J1 (R9) | as J1 | **identical** |
| J9 Run Disclose Income Only | `categorical` [50×28] **value-identical** | R9 (vendor attrs only) | identical (2 files) | **identical** (77 records) | `sigma_value_ui` (R12); `_run_metadata` new (R28); `income_spec_mode` no longer rewritten to "Categorical only" by the run — stays "categorical only" (R28) | **identical** |
| J10 Run Disclose Documents Only | `categorical` [50×36] **value-identical** | R9 | identical (3 files) | **identical** (116) | as J9 (R12, R28) | **identical** |
| J11 Run Donation Default Only | `categorical` [50×18] **value-identical** | R9 | identical (2 files) | **identical** (89) | `sigma_value_ui` (R12); `_run_metadata` (R28) | **identical** |
| J12 Run RTD Only | `continuous` [50×52] **value-identical** | R9 | identical (6 files) | **identical** (144) | as J9; `income_spec_mode` no longer rewritten to "Continuous only" (R28) | **identical** |
| J13 Copula×cat, 4 decisions, all copula ticks | `categorical` [50×101]: `donation_default` and `final_donation_rate` 43/50 rows, max abs 7.7e-7 (R12: σ 9.8995 vs 9.899547); `purchase_requests` 43/50 (R9, and the donation rate carried into the request dicts, R12) | R9 | 8/19: the 7 of J1 (R9) + `donation_default_results` (R12) | R9 set (3 preview dataframes here) + `Distribution of 3. Donation Default` and `Distribution of Donation Rates Across Agents` charts (R12) + the Donation Excel download (R12) | `sigma_value_ui` (R12); `_run_metadata` (R28); `selected_decision_configs.donation_default` auto-implied record absent (R13) | `disclose_income*`, `disclose_documents*`, `rtd_*` **identical**; `donation_default` / `final_donation_rate` differ by ≤ 7.7e-7 — **R12 only** |
| J14 ResSpec×cat, 4 decisions, all research ticks | as J13 with 233/280 rows, max abs 1.2e-6 (R12); `purchase_requests` 232/280 (R9/R12) | R9 | as J13 (R9 + R12) | as J13 (4 previews) | as J13 (R12, R28, R13) | as J13 — **R12 only** |
| J15 ResSpec×cat, 4 decisions, **no sigma key set** (R18 control) | as J14 (R12) | R9 | as J13 | as J13 | as J13 | as J13 — **R12 only**. R18 changes nothing here: the four tabs render when the decisions are selected and write `di_/dd_/rtd_sigma_enabled = True` themselves, in step 1 as now; DI/DD/RTD columns are identical |
| J16 donation-only → pin → complete run | phase 1 `categorical` [50×18] **value-identical**; phase 2 `categorical` [50×53] `purchase_requests` 23/50 (R9) | R9 | run-only page identical (2), after-pin page identical (2); complete-run page 7/15 (R9) | run-only and after-pin pages **identical** (89 each); complete-run page as J1 (R9) | `sigma_value_ui` (R12); `_run_metadata` (R28); pinned record gains `result_columns` + `result_sha256` (R14) and its `sigma_value` 9.8995 → 9.899547 (R12) | **identical** (the pinned donation run reproduced: no R14 error) |

### 4.2 Rollup

* **Decisions 1–4 columns** (`disclose_income*`, `disclose_documents*`, `donation_default`, `final_donation_rate`,
  `rtd_*` / `rejected_transaction_defaults*`): identical to the original in every journey except
  `donation_default` / `final_donation_rate` in J13, J14, J15, where the only cause is R12 (max abs 1.2e-6).
  R18 produces no difference in any journey (see J15).
* Frames: the only other differing column anywhere is `purchase_requests` (R9 bid values); every column set,
  order and dtype identical; no row-count change.
* Downloads: only the 7 bid-dependent workbooks (R9) and, in J13–J15, `donation_default_results` (R12);
  all 12 other files, and every file on the run-only and after-pin pages, identical.
* Pages: 16 identical; on the 12 differing results pages every changed record is R9, R12 or the R28 title.
  The `- Decisions: N` line (R27) and the final_donation_rate block (R13) read exactly as step 1.
* Session: `sigma_value_ui` (R12), `_run_metadata` (R28), `income_spec_mode` not rewritten (R28),
  `result_columns`/`result_sha256`/pinned `sigma_value` (R14/R12), no auto-implied record (R13).
* **Unattributed: none.**

## 5. R8 — IMPLEMENTED 2026-09-04 (was "report only")

Ruling R8: vendor scoring uses the Page-1 price bounds and the "Average Price per Vendor" as the reference
price. When this report was first written the engine half was missing, confirmed exactly as C1 and C3 found:

```python
# src/decisions/purchasing_quantity.py:85-86, _calculate_preferred_vendor  (BEFORE)
price_min_config = simulation_config.get('vendor_price_min', 50.0)
price_max_config = simulation_config.get('vendor_price_max', 150.0)
```

`simulation_config` here is the engine's top-level dictionary (`src/engine/core.py:266` writes
`simulation_config['simulation_seed']` at that level and the same function reads it at line 71). The Page-1
bounds are packed one level down — `build_simulation_params` in `app/seam/build_plan.py` puts
`vendor_price_min/max` into `simulation_config['simulation']`, which is where `src/engine/vendors.py:43-44`
reads them. No writer of a top-level `vendor_price_min` existed anywhere in `src/` or `app/`, so the engine
always normalised on `[50, 150]` while the reports (`app/reports/agent_level.py:49`,
`transaction_level.py:71`, `vendor_viz.py:220,457`) normalised on the Page-1 bounds.

**Owner ruling (recorded verbatim).** "the engine's vendor scoring is a mistake because it normalises vendor
prices with a fixed 50-150 range read from top-level simulation_config keys that nothing sets, while the
results page and exports normalise with the Page-1 bounds; the engine must use the Page-1 min/max bounds like
the reports do, and the Average Price per Vendor (sim_params.market_price) is the reference price wherever
'the price' is needed — bounds only shape the random vendor prices. Everything else stays identical."

**Change applied (2026-09-04).** The two reads now go through the standard Page-1 accessor:

```python
# src/decisions/purchasing_quantity.py, _calculate_preferred_vendor  (AFTER, commented R8 at the site)
price_min_config = get_simulation_param(simulation_config, 'vendor_price_min', 50.0)
price_max_config = get_simulation_param(simulation_config, 'vendor_price_max', 150.0)
```

`get_simulation_param` (`src/decisions/income_utils.py`) reads `simulation_config['simulation']` — the same
mapping `src/engine/vendors.py` draws the vendor prices from and the reports normalise with. The 50 / 150
fallbacks stay for runs without Page-1 parameters; `config/simulation.yaml` carries the same 50 / 150 under
`simulation:`, so a CLI run reads the values it always did.

**Audit of every other site** (`grep vendor_price_min|vendor_price_max|market_price` under `src/decisions`
and `src/engine`; nothing else changed, per "everything else stays identical"):

| site | what it does | verdict |
|---|---|---|
| `src/engine/vendors.py:43-44, 63, 68` | reads the bounds and `market_price` from `simulation_config['simulation']`; the bounds only shape the random vendor prices, `market_price` fills the explicit-price list | already per the ruling — unchanged |
| `src/decisions/bid_value.py:46` | `get_simulation_param(simulation_config, 'market_price', 100.0)` is the reference price whenever no vendor price is passed in | already per the ruling — unchanged |
| `src/decisions/vendor_selection.py` | echoes the `preferred_vendor` Decision 6 stored; scores nothing (it imports `select_best_vendor` but never calls it, and nothing in `src/` or `app/` does) | unchanged |
| `src/vendor_attribute_generator.py`, `calculate_vendor_score_with_breakdown` | with `max_price <= min_price` returns `norm_price = 1.0` (the single vendor at 100/100); with both bounds `None` it falls back to the vendor-list min/max, a branch the engine no longer reaches because it always passes the Page-1 bounds | unchanged |
| `app/reports/agent_level.py` inline `avg_vendor_score` (Q-45, `0.5` at equal bounds) | reports side; deferred by the owner | out of scope — unchanged |

**Effect on the default run.** One vendor at price 100, Page-1 bounds 100 / 100: the engine's `norm_price`
was `1 - (100 - 50) / (150 - 50) = 0.5` and is now `1.0` (equal bounds), the value the reports compute. With
one vendor the choice cannot change. Runs with several vendors and bounds other than 50 / 150 now rank
vendors on the same scale the results page reports; no journey covers that case.

**Verification.** `.venv/bin/python -m pytest tests/ -q`: **215 passed**. The 17 journeys were re-captured
from the patched tree (`scratchpad/r8/final`, every worker rc=0) and compared with `scratchpad/reference/final`
by `scratchpad/verify/compare_step3_closeout.py` (`scratchpad/r8/cmp/r8_vs_final.txt`): every frame
identical (values, columns, dtypes, attrs, manifest hashes), every download identical, 40 / 40 page snapshots
identical, session state identical. **No number moved**, exactly as predicted above. Documented in
`rulings-and-quirks.md` (Q-10, now `fixed`), `docs/migration/README.md` ("Known gap": none open) and
`open-questions-for-professor.md` §4 (resolved).

## 6. End-to-end drive (fresh process, AppTest)

`scratchpad/verify/final/apptest_e2e_final.py` (log `apptest_e2e_final.log`, exit 0, `E2E_FINAL OK`).
Both phases: no `at.exception`, no `st.error`, on any page at any step.

**Phase 1 — the sequence as specified.** Page 1 (n=50, seed 42, Copula) → Page 2 → select the four
stochastic decisions → tick the four "Add Normal(anchor, σ) draw to Copula runs" checkboxes through the real
widgets (all eight canonical/widget keys read `True`) → Run Complete Simulation → results
(`_run_metadata`: custom = the 4 decisions, 9 defaults, seed 42, n 50, key `categorical`, frame 50×101).
**"Save This Configuration" is not on this page** — by design: `render_config_selection_ui` renders it only
for an individual donation run (`ctx.is_individual_run('donation_default')`), and the "Use This Config"
buttons only on individual donation/DI/DD runs; the page offers Clear Results / Back to Decision Parameters /
Back to Common Parameters. → Back to Decision Parameters (the four decisions still selected) → Run Complete
Simulation again: same metadata, Decisions 1–4 columns identical to the first run → Clear Results: no
exception, `simulation_results` is `None`, the session re-initialised (`n_agents` back to 1000), page stays
`results`.

**Phase 2 — the save path.** Same Page 1/Page 2 setup (4 decisions, copula ticks) → Run Donation Default
Only → results (custom = `['donation_default']`, no defaults) → "Save This Configuration"
(`save_single_config`) → pinned record with `result_columns=['donation_default']`, a `result_sha256`, source
`individual_donation_default_run` → Back to Decision Parameters → Run Complete Simulation: no error; the page
shows `🎯 Donation Default used saved configuration: Copula (synthetic) + categorical only` and `📊 Using
Distribution from Selected Donation Configuration`; the R14 reproduction check passed → Clear Results: no
exception.

(Streamlit logs an internal `ArrowTypeError … column disclose_documents` while serialising a mixed-type
preview for display; it is a logged fallback, not an app exception, and the same line is in the step-1
captures — `reference/step1/J1/run.log`, `J13/run.log`.)

## 7. Documentation corrections made in place

Every factual claim in `docs/migration/*.md` and `README.md` was checked against the code (function and
constant names, file lists, line-level behaviour, the configuration values quoted, the data-file shapes and
counts in `open-questions-for-professor.md` §2 — 280×32/165, 280×32/252, 280×12/232, `participant_id`
0…279, the allowance-level bands of `stata_incomes.csv` incl. the level-2 maximum 20 262.87 and the
57/61/52/56/54 counts, the deleted `income_reference_breaks.json` contents at `80ee344`). Wrong or stale
statements, fixed:

| file | was | now |
|---|---|---|
| `README.md` | test suite "204 tests" / `# 204 passed` / "What those 204 tests are" and a table of 8 files | 215; `tests/test_session_key_registry.py` (11) added to the table |
| `README.md` | "Runs from the Streamlit app also save an `enhanced_params_*.json` alongside the results" — no such write exists any more (R11; `grep enhanced_params app/ src/ scripts/` → nothing) | single runs write nothing to disk; only the Monte-Carlo subprocess writes to `outputs/` |
| `README.md` | Page 1 described as holding the "income mode (categorical … vs. continuous)" | Page 1 = population mode, income distribution, vendors/prices, thresholds; the categorical/continuous income mode is the Page-2 radio on the Donation Default tab |
| `README.md` | repository layout omitted `app/state/registry.py` | line added |
| `docs/migration/README.md` | "The four documents" — five exist; `session-state.md` missing from the table | "The five documents", row added |
| `docs/migration/how-to-add-a-decision.md` | `# expect 204 passed` | 215 |
| `docs/migration/architecture.md` §4 | "differ in exactly four things" above a five-row table | five fields — four that affect the run plus the log prefix |
| `docs/migration/rulings-and-quirks.md` Q-43 | "The only session write left under `app/pages/results/` is the navigation key `page`" — the Clear Results wipe (`export_section.py:721`) is also there | qualified: apart from the Clear Results wipe (Q-50); the pin buttons write through `save_selected_configuration` |
| `docs/migration/session-state.md` §2 | quoted mirror line `st.session_state.di_sigma_enabled = st.session_state.di_tab_sigma_enabled` is not the code | the real two statements (`sigma_enabled = st.checkbox(..., key="di_tab_sigma_enabled")`, `st.session_state.di_sigma_enabled = sigma_enabled`) |
| `docs/migration/session-state.md` §4 | "Keys that look like engine inputs and are not: `n_runs`, `base_seed`, `anchor_observed_weight`, `population_mode`" — three of the four ARE seam inputs (`resolve_seed_and_n` reads `base_seed`; the registry marks all three `engine_input=True`) | rewritten: only `n_runs` is not a seam input; the Monte-Carlo command line is the separate path |

Everything else checked out, including: `DECISION_ORDER`, `POP_CONTEXT_DECISIONS`, `STOCHASTIC_MODULE_DECISIONS`,
the RNG offsets (+1 000 000, +999 999, position×1000), `ModeProfile` fields, `should_use_stochastic`'s
three-way rule, every seam/store/report function name cited, the `SubRun`/`RunMetadata`/`RunPlan` field lists,
the CLI options, the deleted-file list (all nine gone; `get_actual_default_value` with its `random.random()`
branch still present, unused; `experiments/` git-ignored), the sigma/scale-factor values in
`config/decisions.yaml`, the `ddof` sites, the three customer-price conventions, `norm_price = 0.5` vs `1.0`,
the 16 `donation_coeff_*_input` keys, the registry's fields, ten categories, `PREFIX_DELETE_PREFIXES`,
`DYNAMIC_KEY_VARIABLES`, the four write-only candidates and the read-only fallbacks.

## 8. Deferred (kept as-is, documented for the professor)

Monte-Carlo subprocess ignoring tab settings and its lowercase income-mode map; Page-1 reset-button exception; widget snap-back; once-per-process
policy warning; export price conventions, `round(…, 2)` on stored values and the inline vendor-score formula
in the Agent-Level sheet; transaction timestamps anchored to today's midnight and `datetime.now()` in file
names; the unseeded example-bids caption; the duplicated "Vendor Selection Breakdown by Period" header; a
saved DI config ignoring the per-column income mode in Compare-both runs; a saved donation config's
coefficients not applied (the tab set is used; the R14 hash catches drift); `custom_coefficients` written and
never cleared; `dd_intercept = -0.5` fallback; the Baseline `force_donation_sigma_zero`; the twice-built
`TraitEngine`; the ~150 console prints; live-dict sharing in `df.attrs`.

Two items that stood in this list when the report was first written are no longer deferred: the ddof mix was
ruled `fixed` (R-SD) and the DI `scale_factor: 0.1` was confirmed correct (R-DI01) — both on 2026-09-04, see §10.

## 9. Notes for the owner (no action taken)

1. **R8** (§5) — applied on 2026-09-04 after the owner's ruling; the entry is kept for the record.
2. **"Compare both" raw-distribution title**: `_income_mode_suffix()` keeps the historical three-way test, so a
   Compare-both run still shows no suffix. No journey covers Compare both; the ruled `(Continuous)` /
   `(Categorical)` correction is only proven for single-mode runs.
3. **Disclose Income reset** writes `di_tab_sigma_*` keys whose live widgets carry a `_stochastic` suffix
   (dead writes, documented in the function docstring); the Disclose Documents twin sets the right keys.
   Adding the suffix would change screens — needs a ruling.
4. `app/reports/purchase.py` still takes an injected `ts_converter` where the other six report modules build
   one from `base_time/duration_hours/periods`; `app/models.py:234` still seeds the never-read
   `individual_results`; `save_results`, `simulation_running`, `_default_params_initialized` are written and
   never read; `clear_input_field_cache()` deletes 16 keys no widget binds. All harmless; removal candidates.
5. Pre-existing `SyntaxWarning: invalid escape sequence '\_'` at `app/pages/decision_tabs/disclose_income.py:600`
   (a `st.latex` docstring); the registry scanner suppresses it locally.

## 10. Addendum 2026-09-04 — R-SD applied; R-DI01 and R-P2 confirmed

**Rulings.** (R-SD) The income standard deviation used for the continuous-income z-scores of Decisions 1 and
2 must divide by N−1 like Stata's `egen std()`, which every z-score in `stata/*.do` uses. Decision 4's own income
SD (already N−1) must stay unchanged. (R-DI01) `disclose_income.stochastic.scale_factor: 0.1` in
`config/decisions.yaml` is correct and intended — the tab's "σ Coefficient (multiplier) 0.10" — no longer open.
(R-P2) The Period-2 row-position merge in `src/build_dd_sigma.py` is exactly the professor's Stata procedure and
is correct; only the shifted `Participant ID` column of the Period-2 export file is kept as a data-file note.

**Every `std` in `src/`, before the change** (`grep -rn 'np\.std(\|\.std(' src/`):

| site | quantity standardised | ddof before | after |
|---|---|---|---|
| `src/engine/core.py` end of Pass 1, `income_stats['sd']` | the population income (z-income for D1/D2 continuous) | **0** | **1** (changed) |
| `src/decisions/disclose_income_stochastic.py` `compute_continuous_de_stats` | D1 continuous direct-effect composite | 1 | 1 |
| `src/decisions/disclose_documents_stochastic.py` `compute_continuous_dd_stats` | D2 continuous `weighted_dd_cont` composite | 1 | 1 |
| `src/decisions/rejected_transaction_defaults.py` `compute_rtd_population_stats` (×2) | D4's own income SD; each mechanism's raw score SD | 1 | 1 (untouched) |
| `src/build_dd_sigma.py` (×2) | D2 sigma constants, overall and per level | 1 | 1 |

**The change.** One statement in `src/engine/core.py`: `'sd': float(np.std(all_incomes))` became
`float(np.std(all_incomes, ddof=1)) if len(all_incomes) > 1 else float(np.std(all_incomes))` — the same
`len > 1` guard `compute_rtd_population_stats` uses. Nothing else in `src/`, `app/` or `tests/` changed.

**Proof on the professor's data** (`scratchpad/ddof/prof_table_ddof_variants.py`, a scratch re-run of the body of
`tests/test_disclose_documents.py::test_continuous_reproduces_professor_table_with_frozen_income` — the test
itself was not edited). With the frozen income (`data/stata_incomes.csv`) and with his own `income` column from
the CORRECTED Decision 2 `.dta` (identical values): income SD 17 679.947640 (`ddof=0`) vs 17 711.603793 (`ddof=1`).
Under **both** formulas the continuous table is 63 / 280 = 22.50 % with every per-allowance-level cell
(78.95 / 27.87 / 1.92 / 0 / 0) and `disclosedoc_cont` matches 280 / 280; 0 agents flip; the largest shift of
`dd_deterministic` is 1.57 × 10⁻³ and the nearest score to the threshold is 0.011583. So the old engine value
was **not** off on the 280 — the test could not tell the formulas apart. What settles it is the `.dta` itself:
its `z_net_income` equals `(income − mean) / sd` with the **sample** SD to 1.2 × 10⁻⁷ and `z_picont` equals its
negative to 2.7 × 10⁻⁷; with the population SD the error is ≈ 5 × 10⁻³. Stata standardised with N−1.

**Tests.** `.venv/bin/python -m pytest tests/ -q`: **215 passed** (43 s).

**Journeys** (`scratchpad/reference/ddof`, all 17 workers rc=0, compared with `scratchpad/reference/r8` = the
23dae63 tree, by `scratchpad/ddof/compare_runs.py`; details in `scratchpad/ddof/refined_r8_vs_ddof.txt`).
Raw `.xlsx` bytes and the raw page hash are never stable between two runs of the same tree
(`reference/_determinism_check`: 0 / 14 raw hashes equal, 14 / 14 normalised equal), so the comparison is on
frame values, `df.attrs`, normalised workbook content and the v2 / v2-noids page snapshots.

| journey | what moved |
|---|---|
| P0, J1, J3, J5, J7, J8, J9, J10, J11, J12, J13, J14, J15, J16 | **nothing** — every column identical, every page snapshot identical, every normalised export identical. The only difference is the statistic itself, carried as metadata in `df.attrs['simulation_config']['income_stats']['sd']` (n = 50 Copula: 11 173.90 → 11 287.35; n = 280 Research: 11 316.50 → 11 336.77). |
| J2, J4, J6 (continuous complete runs; D1 and D2 run as **defaults**) | only the five `disclose_documents_*` analytic columns (`_raw`, `_score`, `_z_picont`, `_weighted_dd`, `_z_weighted_dd`), which the default path still computes from the model score, in the third decimal (all 50 / 280 / 280 rows); `attrs` `dd_cont_stats.sd` (J2 0.152868 → 0.151414; J4/J6 0.151556 → 0.151299) and `di_cont_de_stats.sd`. The `disclose_documents` value, `customer_type`, `disclose_income` and every other column are identical. Page snapshot: 10 / 12 / 12 changed lines, all in the "DD Raw Values" histogram and the download hashes; one export with changed content, `agent_disclose_documents_data.xlsx`. |

`rtd_*` (Decision 4) and `donation_default` are identical in all 17 journeys, including J12 (D4 only,
continuous) and J13–J15 (D4 model with sigma). Nothing is unattributed.

**Extra journey JX** — Research Specification × continuous only, n = 280, seed 42, Disclose Income **and**
Disclose Documents selected (model path; tab defaults, i.e. `di_sigma_enabled` / `dd_sigma_enabled` True and no
sigma key set by the harness), run at HEAD 23dae63 in a temporary worktree and on the patched tree
(`scratchpad/ddof/jx_head`, `jx_patched`): **0 of 280 `disclose_income` and 0 of 280 `disclose_documents`
values change** (Y/N counts 241/39 and NA/Y/N 247/31/2 in both); `customer_type` identical. Seven analytic
columns move: `disclose_income_raw` (max 9.7 × 10⁻⁴), `disclose_income_di`, `disclose_documents_raw`
(max 1.6 × 10⁻³), `_score`, `_z_picont` (max 8.2 × 10⁻³), `_weighted_dd`, `_z_weighted_dd`; three exports with
changed content (the two disclose data workbooks and `simulation_agent_level`); page changes only in the "DI Raw
Values" / "DD Raw Values" histograms and download hashes.

**Documents.** `rulings-and-quirks.md`: Q-31 → `fixed (R-SD)`, Q-14 qualified, Q-59 (R-DI01) and Q-60 (R-P2)
added. `open-questions-for-professor.md`: items 1, 2 and 5 removed as open questions and kept as resolved records
(now items 5–7), the remaining items renumbered 1–4, the shifted-ID data-file note kept at the end.
`docs/migration/README.md`: a paragraph on the three rulings. `README.md` and `architecture.md` do not state the
formula and were not touched. Nothing committed; the original repository was not opened for writing.
