# Rulings and quirks

Before anything was moved, the old code was read line by line and every surprising
behaviour was written down — 58 of them. Each one was then put to the owner, who
either **fixed** it (the behaviour changes, deliberately) or **deferred** it (the
behaviour is kept exactly as it was, and is documented instead of silently carried).
Two rows were added on 2026-09-04 (Q-59, Q-60) to record owner rulings on questions
that had been referred to the professor.

That register is reproduced below. It is the answer to "why does the code do *that*?"

**How to read a row.** *Behaviour before* is what the old code did. *Ruling* says
`fixed` or `deferred` and what the code does now. *Where in the new code* is where to
look. The `Rnn` codes are the owner's ruling numbers; you will find the same codes in
comments in the source.

> Every `fixed` row below was re-checked against the code in this repository while
> this document was written. The one row that did **not** check out at the time was
> Q-10 — the engine half of R8 was missing. It was applied on 2026-09-04 and
> re-checked; no row is outstanding. On the same day the owner ruled on three of the
> deferred rows: Q-31 became `fixed` (R-SD, applied and re-checked), and Q-59 / Q-60
> record that the Decision 1 scale factor (R-DI01) and the Period-2 merge (R-P2) are
> correct as they stand.

---

## The engine: three orchestrators became one

| id | behaviour before | ruling | where in the new code |
|---|---|---|---|
| Q-01 | A fourteenth, non-existent decision slot (`enrich_purchase_requests`) sat in the Research modes' decision list, shifting decisions 7–13 by one position — and therefore giving them different random streams than in Copula mode. | **fixed (R1)** — one 13-entry decision order for every mode. Changes `purchase_vs_bid` and `bid_value` in the two Research modes. | `DECISION_ORDER` in `src/engine/core.py` |
| Q-02 | Research Baseline emitted a column `enrich_purchase_requests` whose value was always the string `'NA'`. | **fixed (R2)** — the column is gone; there is no such decision. | `src/engine/core.py` (no fallback column is written) |
| Q-03 | When a Baseline decision raised, its value silently became the string `'NA'`. | **fixed (R4)** — see Q-04. | `src/engine/core.py` |
| Q-04 | Research Baseline caught every decision exception, printed it and continued; Copula and Research Specification let it propagate. | **fixed (R4)** — errors raise in every mode. A broken decision stops the run instead of producing a column of `'NA'`. | `src/engine/core.py`, Pass 2 (no `try/except` around the decision call) |
| Q-05 | Baseline built its output row from a *copy of the agent taken before the decisions ran*, so its frames had a different column set and order (no `vendor_proximity_scores`). | **fixed (R3)** — one row mechanism: the mutated agent state is the row, in every mode. | `results.append(agent_state)` in `src/engine/core.py` |
| Q-06 | Baseline forced the donation decision's `sigma_value` to 0 on a shallow copy of its parameters. | **deferred** — kept; this is the intended meaning of "Baseline". | `force_donation_sigma_zero` in `src/engine/profile.py`; applied in `Engine._decision_params` |
| Q-07 | Research Specification loaded `donation_default_stochastic`, the other modes `donation_default` — two different modules with different noise gates. | **fixed (R6)** — one donation module in every mode, with one gate: the mode's tick box **and** a resolved σ greater than zero. `src/decisions/donation_default_stochastic.py` is deleted. | `STOCHASTIC_MODULE_DECISIONS` in `src/engine/core.py` (only D1/D2); `should_use_stochastic()` in `src/utils/stochastic.py`; the `use_stochastic and sigma_0_100_scaled > 0` test in `src/decisions/donation_default.py` |
| Q-30 | The CLI sampled Research participants by rules that differ from the app's (Research Specification permuted even at n=280; Baseline bootstrapped above 280 where the app wrapped around). | **deferred** — both rules kept, side by side and documented, so CLI output does not move. | `load_original_participants` (app path) and `sample_participants_internal` (CLI path) in `src/engine/sampling.py` |
| Q-32 | Agent values are plain Python floats/ints/strings, because rows are read with `iterrows()` and `to_dict()`. | **deferred** — kept; changing it would change dtypes in every export. | `src/engine/core.py`, Pass 2 |
| Q-34 | Transaction IDs are written *into the dictionaries stored inside DataFrame cells*, after a stable sort on `timestamp_hours` (missing timestamps sort as 0). | **deferred** — kept, but moved out of the app layer. | `assign_global_transaction_ids` in `src/engine/postprocess.py` |
| Q-35 | Two of the orchestrators created an `rng_global` generator that nothing ever used. | **fixed** — dropped with the unified loop; a random-number trace proved it consumed nothing. | not present in `src/engine/core.py` |
| Q-36 | The copula `TraitEngine` was built twice per run. | **deferred** — still built twice (the app samples the agents, the engine validates the traits). Constructing it draws no random numbers, so no number changes. | `sample_agents` in `app/seam/execute.py`; the `trait_engine` property in `src/engine/core.py` |
| Q-37 | About 150 `print` statements, including one per agent in the donation decision. | **deferred** — kept; console only, nothing parses them. | `src/decisions/*`, `src/engine/log.py` |
| Q-38 | `src/validate_traits.py` read two Excel workbooks *at import time* and could call `sys.exit(1)`. | **fixed (R37)** — the module is deleted. Participant loading is lazy, cached, and raises instead of exiting. | `src/data/participants.py` |
| Q-57 | Research Baseline had no `get_available_decisions()`, so the CLI skipped validating `--decision` in that mode. | **fixed** — the method is on `Engine`, so all three modes have it. | `Engine.get_available_decisions` in `src/engine/core.py` |

## Decision defaults (Decisions 11, 12, 13)

| id | behaviour before | ruling | where in the new code |
|---|---|---|---|
| Q-08 | Decisions 11 and 13 read their default value out of live Streamlit session keys **from inside the engine** (`from app.pages.decision_execution import get_actual_default_value`), ignoring the values collected everywhere else; Decision 12 was the constant `'NA'`. | **fixed (R7)** — all three read `simulation_config['default_decisions'][<decision>]`, the settings dictionary the seam collects once per click, with a module-level fallback constant. No engine module imports the app any more. | `src/decisions/rejected_transaction_option.py`, `rejected_bid_value.py`, `final_donation_rate.py`; the settings are built by `collect_decision_settings` in `app/seam/build_plan.py` |
| Q-09 | The default resolver contained an **unseeded** `random.random()` branch (not NumPy, not seeded), unreachable from the engine. | **deferred** — the branch is carried verbatim, inside `get_actual_default_value`, which is now called by nothing at all. | `app/pages/decision_execution.py` (`get_actual_default_value`, the `if random.random() < probability_y` branch) |
| Q-52 | The settings dictionary already carried entries for Decisions 11–13 that nothing read. | **fixed (R7)** — they are exactly what the three decisions now read. | as Q-08 |

## Vendors and prices

| id | behaviour before | ruling | where in the new code |
|---|---|---|---|
| Q-10 | Decision 6 normalised vendor price on the hard-coded range `[50, 150]`, while the results page and the exports recomputed the same score with the Page-1 bounds — so the two could disagree. | **fixed (R8)** — vendor scoring uses the Page-1 price bounds everywhere, with the Page-1 "Average Price per Vendor" (`market_price`) as the reference price wherever a single price is needed; the bounds only shape the random vendor prices. `_calculate_preferred_vendor` now reads `vendor_price_min` / `vendor_price_max` from `simulation_config['simulation']` through `get_simulation_param()` (fallbacks 50 / 150 when no Page-1 parameters exist) — the same place `src/engine/vendors.py` draws the prices from and the reports normalise with. It used to read them as top-level keys nothing sets. Applied 2026-09-04; for the default single vendor the engine's normalised price goes from 0.5 to 1.0 (equal bounds), the same as the reports, and with one vendor the choice is unchanged, so no journey number moved (`acceptance-report.md` §5). | engine: `src/decisions/purchasing_quantity.py`, `_calculate_preferred_vendor` (commented `R8` at the site). Reports: `app/reports/agent_level.py`, `transaction_level.py`, `vendor.py`, fed from `sim_params` by `app/pages/results/components/export_section.py` |
| Q-11 | On every rerun after the first, a "value migration" block silently rewrote the Page-1 vendor settings whenever they happened to equal an old default (`vendor_price_min` 100 → 50, `max` 100 → 150, `market_price` 10 → 100, products 100 → 50/150). A single vendor therefore priced at `uniform(50, 150)` — 127.40 at seed 42 — instead of the 100 shown on screen. | **fixed (R9)** — the block is deleted. A single vendor now runs at exactly the Average Price per Vendor and the Products Offered you typed. | `app/models.py` (the removal is commented at the site) |
| Q-45 | The Agent-Level export sheet computes `avg_vendor_score` with an inline formula that is **not** `calculate_vendor_score_with_breakdown`: when the price bounds are equal it uses `norm_price = 0.5`, where the shared function uses `1.0`. With the default single vendor the bounds *are* equal, so the two disagree. | **deferred** — the inline copy is kept verbatim; see [`open-questions-for-professor.md`](open-questions-for-professor.md). | inline: `app/reports/agent_level.py`; shared: `calculate_vendor_score_with_breakdown` in `src/vendor_attribute_generator.py` |
| Q-54 | Page 1 forces `vendor_price_min = vendor_price_max = market_price` while a single vendor is configured, and the "number of fixed income categories" follows `price_grid - 1`. | **deferred** — kept; visible on screen. | `app/pages/page1_common_params.py` |
| Q-55 | Page 1's income-distribution preview uses its own sampler (fixed seed 42, rejection sampling) which is deliberately not the engine's. | **deferred** — kept, and moved out of the page into a testable module. | `app/reports/preview.py` |

## The configuration file, sigma and coefficients

| id | behaviour before | ruling | where in the new code |
|---|---|---|---|
| Q-12 | Three different "donation coefficients" existed in the UI and none of them reached the model: only the nested blocks in `config/decisions.yaml` mattered, and an edited intercept reached the engine only because the UI rewrote that file. | **fixed (R10)** — the flat coefficient set the Donation tab shows for the run's income mode **is** what the engine runs. The file's nested blocks are *replaced*, not merged into. | `session_donation_coefficient_set` and `build_donation_patch` in `app/seam/build_plan.py`; the `Replace` marker in `src/contract/plan.py` |
| Q-13 | Editing a donation intercept, or pressing a reset button, made the app **rewrite `config/decisions.yaml`** — a whole-file dump that also dropped 44 comment lines. Each run then re-read the file. | **fixed (R11)** — the app never writes a configuration file. Tab values live in session keys and travel to the engine as patches. | `app/seam/config_repo.py` (read-only); no `yaml.dump` remains anywhere under `app/` or `src/` |
| Q-14 | The tracked `config/decisions.yaml` carries `disclose_income.stochastic.scale_factor: 0.1` and `donation_default.adjustment.shift_value: -4.0`. | **deferred** — the file is unchanged, and it is now the only source of these values. The `scale_factor: 0.1` half was confirmed correct by the owner on 2026-09-04 (R-DI01, see Q-59); the `shift_value: -4.0` half is kept as it was. | `config/decisions.yaml` |
| Q-59 | Decision 1's Research Specification draw is `Normal(anchored_pb, sigma_overall × scale_factor)` with `scale_factor: 0.1`: one tenth of the raw-units TWT+Sospeso standard deviation (9.899547) applied to a z-scale variable, with nothing in the code or the file saying where `0.1` came from. It was referred to the professor as open question 2. | **resolved — kept (R-DI01, 2026-09-04)** — the owner ruled that `0.1` is correct and intended: it is the Disclose Income tab's "σ Coefficient (multiplier)" default of 0.10. No longer an open question; nothing changed. | `config/decisions.yaml`; `sigma_scaled = sigma_raw * scale_factor` in `src/decisions/disclose_income_stochastic.py`; the σ coefficient slider in `app/pages/decision_tabs/disclose_income.py` |
| Q-60 | The Decision 2 sigma constants are derived from a two-period consumption count in which Period 2 (269 rows) is merged onto the 280 master rows by **row position**, not by `Participant ID` — the Period-2 export's ID column is offset relative to its data rows, so an ID join mismatches 89 of 280. It was referred to the professor as open question 5. | **resolved — kept (R-P2, 2026-09-04)** — the owner ruled that the positional merge is exactly the professor's Stata procedure and is correct; the sigma constants stand. The shifted ID column of the Period-2 export file remains a data-file note for the professor (`open-questions-for-professor.md`, last section). Nothing changed. | `reconstruct_consumed_2periods` in `src/build_dd_sigma.py` |
| Q-15 | Four sigma constants were spelled as literals in the app, one of them (`9.8995`) differing from the file's value (`9.899547`) in the fifth decimal. | **fixed (R12)** — one constant, read from the configuration file. The donation σ handed to the engine is `sigma_overall × the coefficient slider`. | `app/seam/sentinels.py`; `DONATION_SIGMA_OVERALL` in `app/models.py`; `BASE_SIGMA_OVERALL` in `app/pages/decision_tabs/disclose_income.py` |
| Q-53 | A rarely-reached branch could call a YAML loader that wrote 49 session keys. | **fixed (R10/R11)** — the coefficient sets come from `DecisionsConfig`, in memory. | `donation_coefficient_set` in `app/seam/config_repo.py` |

## Saved configurations ("Use This Config")

| id | behaviour before | ruling | where in the new code |
|---|---|---|---|
| Q-16 | A saved donation snapshot could be created *implicitly*, and once it existed, moving the σ slider no longer affected a complete run. | **fixed (R13)** — implicit ("auto-implied") saved configurations do not exist. Only an explicit "Use This Config" creates one; a stale implicit record from an old session is ignored. | `explicit_saved_configs` in `app/seam/build_plan.py` |
| Q-17 | Applying a saved donation configuration **overwrote session keys in the middle of a run**, so sub-run 1 used the live tick boxes and sub-runs 2–6 used the saved ones. | **fixed (R15)** — the session is read once, into an immutable snapshot, before anything runs. Every sub-run sees the same settings. | `app/seam/snapshot.py`; `build_plan_from_session` in `app/simulation.py` |
| Q-18 | The seed and agent count were pinned by the first saved configuration *including the implicit ones*, so later Page-1 edits were silently ignored. | **fixed (R13 + R14)** — only explicit saved configurations pin them, and the caption states the pinned seed. | `resolve_seed_and_n` in `app/seam/build_plan.py`; `get_simulation_seed_from_configs` in `app/state/saved_configs.py` |
| — | *(new behaviour, R14)* A saved configuration now also records **which columns the decision produced and their sha256**. After a complete run those columns are re-hashed; if they differ the run is rejected with an explicit error instead of quietly showing different numbers. | **fixed (R14)** | `get_decision_result_columns` / `hash_result_columns` in `app/state/saved_configs.py`; `SavedExpectation` in `src/contract/plan.py`; `verify_saved_expectations` in `app/seam/execute.py` |
| Q-19 | Every individual donation run wrote `custom_coefficients['donation_default']`, which nothing ever cleared. | **deferred** — still written, still read by nothing (the seam does not consult it). Visible in a session dump; numerically inert. | `app/pages/decision_execution.py` |
| Q-20 | With a saved Disclose Income configuration, a "Compare both" run used the saved income mode for *both* columns instead of each column's own. | **deferred** — kept. See [`open-questions-for-professor.md`](open-questions-for-professor.md). | `build_disclose_income_patch` in `app/seam/build_plan.py` |
| Q-21 | A saved Disclose Documents configuration never reached the model at all: it only enabled the Run button and pinned the seed. | **fixed (R17)** — it is applied exactly like a saved Disclose Income configuration (same precedence: saved income mode and intercept, live σ tick boxes). | `_saved_disclose_documents_patch` in `app/seam/build_plan.py` |
| Q-56 | A helper falls back to `dd_intercept = -0.5` while the configuration file and the tab both default to `-0.75`; reachable only if the Disclose Documents tab was never opened. | **deferred** — kept. | `get_current_disclose_documents_params` in `app/state/saved_configs.py` |

## Run flow and session state

| id | behaviour before | ruling | where in the new code |
|---|---|---|---|
| Q-22 | The Decision 4 configuration always set `model_enabled = True`, so the flag meant nothing; gating actually happened elsewhere. | **fixed (R23)** — the flag is removed. The trait model runs whenever the decision is selected. | `src/decisions/rejected_transaction_defaults.py` (the ruling is commented at the site) |
| Q-23 | The σ tick boxes for Decisions 1 and 2 defaulted to **off** if you had never opened their tab and **on** once you had — the same visible settings could give σ = 0 or σ > 0 depending on your navigation history. | **fixed (R18)** — an absent key means the tab's own default, which is on. History no longer changes the numbers. | `snapshot.get("di_sigma_enabled", True)` / `dd_sigma_enabled` / `rtd_sigma_enabled` in `app/seam/build_plan.py`; the same default in `app/state/saved_configs.py` |
| Q-24 | A standalone Disclose Documents run silently forced Decision 1 to run first with its default probability, to work out who is eligible. | **deferred** — kept, and now visible: it is a `default_decisions_list` entry in the plan plus a caption on screen, not a hidden flag. | the `single_decision == ["disclose_documents"]` branch of `build_run_plan` in `app/seam/build_plan.py` |
| Q-25 | "Is this a single-decision run?" was decided by `len(selected) == 13`. | **deferred** — kept, in the builder. | `build_run_plan` in `app/seam/build_plan.py` |
| Q-26 | The run path **wrote back** `income_spec_mode` (in a different capitalisation), so a run could silently change the mode shown on Page 2. | **fixed (R28)** — the run writes no mode keys. What the run actually used travels in `RunMetadata` / `_run_metadata`. | `RunMetadata` in `src/contract/plan.py`; `_run_metadata_dict` in `app/simulation.py` |
| Q-27 | A successful run ended in `st.rerun()`, which raises — so the "✅ … complete!" messages and metrics after it were unreachable, and a combined run left `selected_decisions` set to all thirteen. | **fixed (R27)** — the unreachable blocks are deleted and the restoration is handed over explicitly through `_pending_decisions_restore`. The results page's "- Decisions: N" line now reports the decisions this run executed. | `run_combined_simulation` / `run_individual_decision` in `app/pages/decision_execution.py`; `RunContext.num_decisions` in `app/pages/results/run_context.py` |
| Q-33 | `df.attrs['simulation_config']` is the engine's **live** dictionary; `st.session_state.vendors` is the same list object as `df.attrs['vendors']`; `random_decisions` and `default_decisions` are the same object. | **deferred** — the sharing is preserved (no deep copies), because copying would change what the exports see. | `Engine.run_simulation` in `src/engine/core.py`; `configure_engine` in `app/seam/execute.py` |
| Q-42 | The Overview tab renders before the decision tabs, so keys written only at tab-render time are one rerun old for a run launched from Overview. | **deferred** — render order kept; no user-reachable divergence was found. | `app/pages/page2_decisions.py` |
| Q-43 | Rendering the **results** page wrote Page-2 default keys if they were missing — so looking at results changed the next run's inputs. | **fixed (R30)** — the results renderers write no settings. Apart from the "Clear Results" wipe (Q-50), the only session write left under `app/pages/results/` is the navigation key `page`; the "Save This Configuration" / "Use This Config" buttons write through `save_selected_configuration` in `app/pages/decision_execution.py` and `app/state/saved_configs.py`. | `app/pages/results/` |
| Q-44 | The reset buttons are inconsistent: the donation "Reset Config to Defaults" leaves the live σ sliders alone; the Disclose Income / Documents resets switch the Research tick box off; "Reset Adjustment" gives 0.0 while "Reset Config" gives −4.0. | **deferred** — kept. | `app/pages/decision_tabs/donation_default.py`, `disclose_income.py`, `disclose_documents.py` |
| Q-50 | "🔄 Clear Results" deletes **every** session key and re-initialises. | **deferred** — kept; nothing of value lives outside session state. | `app/pages/results/components/export_section.py` |
| Q-51 | A long list of dead code: `OrchestratorDepVar`, `vendor_price_generator.py`, the legacy `disclose_income.py` / `disclose_documents.py` decision modules, `validate_traits.py`, `build_master_traits.py`, uncalled renderers, an unused configuration file. | **fixed (R37)** — deleted: `src/orchestrator_depvar.py`, `src/vendor_price_generator.py`, `src/decisions/disclose_income.py`, `src/decisions/disclose_documents.py`, `src/decisions/donation_default_stochastic.py`, `src/validate_traits.py`, `src/build_master_traits.py`, `config/income_reference_breaks.json`, `app/pages/results/components/parameter_summary.py`. Still present and still dead: `get_actual_default_value` (Q-09) and the git-ignored `experiments/` folder. | `git diff --name-status 80ee344 HEAD` lists the deletions |

## Page 1 and Streamlit itself

| id | behaviour before | ruling | where in the new code |
|---|---|---|---|
| Q-39 | Page 1's "🔄 Reset to Default Values" writes widget keys *after* the widgets were created, which raises `StreamlitAPIException` — after half-resetting the parameters to a set of values that is **not** the dataclass default set. | **deferred** — reproduced as-is, pending a decision on which default set is the real one. | `reset_all_page1_defaults` in `app/pages/page1_common_params.py`; the defaults it disagrees with are in `SimulationParameters` in `app/models.py` |
| Q-40 | Widget keys pre-seeded before the widget is drawn "snap back" for one rerun. A browser heals this on the next interaction; the automated test harness never does. | **deferred** — kept; journeys avoid those widgets. | `app/pages/page1_common_params.py` |
| Q-41 | Streamlit prints a once-per-process policy warning for `n_agents_input`. | **deferred** — kept and recorded by the harness so it is compared like any other page element. | Streamlit internals; triggered from `app/pages/page1_common_params.py` |

## Exports

| id | behaviour before | ruling | where in the new code |
|---|---|---|---|
| Q-46 | Three different customer-price conventions across the exports. | **deferred** — moved verbatim, unchanged. See [`open-questions-for-professor.md`](open-questions-for-professor.md). | `app/reports/transaction_level.py`, `app/reports/donation.py`, `app/reports/purchase.py` |
| Q-47 | `round(…, 2)` baked into stored (not merely displayed) export values, and six near-duplicate price-formatting helpers. | **deferred** — kept, but the six helpers now sit together in one module. | `app/reports/xlsx.py` (`apply_export_/disclosure_/transaction_/bid_/vendor_/donation_price_formatting`) |
| Q-48 | `datetime.now()` in 34 places; transaction timestamps are relative to **today's midnight**, so re-exporting the same run on a different day changes every timestamp string. | **deferred** — kept. File names still use `datetime.now()` inside the pages; the builders take an injected base time so they can be tested under a frozen clock. | `midnight_of` and `TimestampConverter` in `app/reports/timestamps.py` |
| Q-49 | An "example bids" caption uses the unseeded `random.uniform`, so the displayed text changes on every rerun. | **deferred** — kept. | `app/pages/results/visualizations/bidding_viz.py` |
| Q-58 | The Monte-Carlo screen injects a `running_mean` column into the results frame in place, so the exported Detailed CSV carries it. | **deferred** — kept. | `app/reports/mc.py` |
| — | The "📅 Vendor Selection Breakdown by Period" header is rendered twice, one line apart. | **deferred** — kept (it is visible on screen; removing it would change the screen). | `app/pages/results/visualizations/vendor_viz.py` |

## Monte Carlo

| id | behaviour before | ruling | where in the new code |
|---|---|---|---|
| Q-28 | The Monte-Carlo income-mode lookup only has lower-case keys. It works today only because nothing writes the capitalised form any more (see Q-26). | **deferred** — kept verbatim. | `run_monte_carlo_study` in `app/simulation.py` |
| Q-29 | Monte Carlo runs as a **subprocess** and forwards none of the Page-2 tab settings; its summary CSV is only written when the donation decision ran; the screen shows absolute paths. | **deferred** — kept. See [`open-questions-for-professor.md`](open-questions-for-professor.md). | `run_monte_carlo_study` in `app/simulation.py`; `scripts/run_mc_study.py` |

## Statistics

| id | behaviour before | ruling | where in the new code |
|---|---|---|---|
| Q-31 | Standard deviations were computed inconsistently: the shared income statistics used the **population** formula (`ddof=0`), while the composite statistics and Decision 4's income statistics use the **sample** formula (`ddof=1`). | **fixed (R-SD, 2026-09-04)** — the shared income SD is now the sample SD (`ddof=1`), like every `egen std()` in `stata/*.do`; with fewer than two incomes it falls back to the population formula, the same guard Decision 4 uses. Decision 4's own income SD (already `ddof=1`) is untouched. Verified: the professor's Decision 2 `.dta` carries `z_net_income` / `z_picont` equal to the `ddof=1` z-score to 3e-7; his 280-row tables are reproduced under both formulas (largest score shift 1.6e-3, nearest score to the threshold 0.0116), so no earlier result was wrong; in the 17 journeys only the `disclose_documents_*` analytic columns of the three continuous complete runs moved, no decision flipped. See item 5 of [`open-questions-for-professor.md`](open-questions-for-professor.md) and `acceptance-report.md` §10. | `np.std(all_incomes, ddof=1)` in `src/engine/core.py` (end of Pass 1); `ddof=1` in `src/decisions/disclose_income_stochastic.py`, `disclose_documents_stochastic.py`, `rejected_transaction_defaults.py` |
