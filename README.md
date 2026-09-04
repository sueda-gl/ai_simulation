# COOPECON AI Agent Simulation

A Streamlit dashboard that simulates how a population of synthetic agents makes
decisions in COOPECON platform.

---

## 1. Prerequisites

- **Python 3.11 or newer** (the project is developed on 3.13). Check with:
  ```bash
  python3 --version
  ```
- **Git** (to clone the repository).

No external services, databases, or API keys are needed. All input data
(`data/`) and configuration (`config/`, including the pre-trained copula model
`config/trait_model.pkl`) are included in the repository.

## 2. Setup (step by step)

**Step 1 — Clone the repository and enter it:**

```bash
git clone <repository-url>
cd <repository-folder>
```

**Step 2 — Create a virtual environment** (keeps dependencies isolated from your system Python):

```bash
python3 -m venv .venv
```

**Step 3 — Activate the virtual environment:**

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. You must activate the environment
in every new terminal before running the app.

**Step 4 — Install the dependencies:**

```bash
pip install -r requirements.txt
```

**Step 5 — Launch the app:**

```bash
streamlit run app_enhanced_new.py
```

Streamlit prints a local URL (normally <http://localhost:8501>) and opens it in
your browser. Stop the app with `Ctrl+C` in the terminal.

## 3. Using the app

The dashboard is a three-page flow:

1. **Page 1 — Common parameters.** Population size, random seed, population
   mode (Copula / Research Specification / Research Baseline / Compare all), the
   income distribution and its parameters, vendors and prices, discount income
   threshold, and other settings shared by all decisions.
2. **Page 2 — Decisions.** Enable and configure the individual decisions
   (disclose income, disclose documents, donation default, purchasing, vendor
   selection, bidding, …), including their stochastic components and the income
   mode (categorical allowance levels vs. continuous income, set on the Donation
   Default tab), then run the simulation.
3. **Results.** Summary tables, visualizations, comparisons across population
   modes, and Excel/CSV export of per-agent results.


### Reproducibility

Every random element (agent sampling, stochastic decision draws) flows through a
seeded NumPy random generator. Running twice with the same seed and settings
produces identical results; change the seed on Page 1 to get a new draw.

## 4. Running without the UI (command line)

For scripted or batch runs, use the CLI entry points in `scripts/`:

```bash
# Single run: 1,000 copula agents, seed 42, all decisions, parquet output in outputs/
python scripts/run_simulation.py --agents 1000 --seed 42

# Options: --decision <name> (repeatable), --population-mode copula|documentation|baseline,
#          --income-mode categorical|continuous, --format parquet|csv, --output-dir <dir>
python scripts/run_simulation.py --agents 5000 --decision disclose_income --format csv

# Monte-Carlo study: many repetitions with incrementing seeds
python scripts/run_mc_study.py --agents 10000 --runs 500 --base-seed 1
```

Results land in `outputs/` (git-ignored) as timestamped parquet/CSV files, e.g.
`simulation_seed42_agents1000_all_20260805_120000.parquet`. (Single runs from
the Streamlit app write nothing to disk: their results live in the session and
are offered as Excel/CSV downloads on the results page. Only the Monte-Carlo
study, which runs `scripts/run_mc_study.py` as a subprocess, writes to `outputs/`.)

## 5. Repository layout

The simulation logic and the Streamlit screens are separate. Everything under `src/`
runs without a browser, and a test enforces that it never imports Streamlit. See
[`docs/migration/architecture.md`](docs/migration/architecture.md) for the full tour.

```
app_enhanced_new.py   Streamlit entry point (run this)

src/                  THE ENGINE - pure simulation, never imports Streamlit or app/
  engine/core.py         the one agent x decision loop, shared by all population modes
  engine/profile.py      the per-mode differences (agent source, noise gate, log prefix)
  engine/sampling.py     Research-mode participant selection
  engine/vendors.py      vendor attributes, drawn once per run
  engine/postprocess.py  global transaction IDs
  contract/plan.py       RunPlan / SubRun / RunMetadata - one run, described as data
  contract/defaults.py   what each decision does when it is NOT selected
  decisions/             one module per decision (+ *_stochastic for Decisions 1 and 2)
  data/participants.py   the 280 experiment participants (read lazily, cached)
  trait_engine.py        samples synthetic agents from the copula model
  vendor_attribute_generator.py   vendor attributes and the shared vendor score
  utils/stochastic.py    shared sigma helpers and the one stochastic gate
  build_dd_sigma.py      re-derives the Decision 2 sigma from the raw experiment data
  orchestrator*.py       three-line wrappers: Orchestrator() == Engine(PROFILES['copula'])

app/                  THE STREAMLIT APP
  simulation.py          the ONLY module connecting st.session_state to the engine
  seam/                  session snapshot -> run plan -> execute (no Streamlit)
  state/saved_configs.py the "Use This Config" store
  state/registry.py      every session-state key: its layer, default, initialiser (see docs/migration/session-state.md)
  reports/               every exported number, frame and workbook (no Streamlit)
  pages/                 the screens: page1, page2 + decision_tabs/, results/
  models.py              SimulationParameters and session-state initialisation
  components.py          CSS and the Monte-Carlo screen

config/               decisions.yaml, simulation.yaml, trait_model.pkl (copula), seeds
data/                 experiment/survey workbooks + frozen Stata verification CSVs
docs/migration/       what changed when logic and UI were split, and why
scripts/              CLI runners (run_simulation, run_mc_study) and copula training
tests/                pytest suite (215 tests)
stata/                the professor's .do reference scripts
outputs/              simulation results, created at runtime (git-ignored)
experiments/          one-off analysis and debugging scripts (git-ignored)
```

**Note:** `config/trait_model.pkl` is committed, so you do **not** need to train
anything before running. Only re-run `python scripts/train_copula.py` if the
underlying experiment data or the master trait list changes (it overwrites the
pickle in `config/`).

**New here?** After this README, read
[`docs/migration/README.md`](docs/migration/README.md). It is the guide to how the
code is organised, why some of it looks the way it does, and how to extend it.

## 6. Running the tests

The suite checks the decision models against the professor's Stata reference results,
the run plan the UI builds, and the import boundary between the engine and the UI.
`pytest` is not in `requirements.txt`, so install it once:

```bash
pip install pytest
pytest tests/          # 215 passed
```

What those 215 tests are:

| file | tests | what it checks |
|---|---|---|
| `tests/test_import_boundary.py` | 106 | no module under `src/` imports Streamlit or `app/`; no module under `app/reports/` imports Streamlit |
| `tests/test_rejected_transaction_defaults.py` | 32 | Decision 4's four mechanisms against the Stata reference |
| `tests/test_build_plan.py` | 31 | the run plan the seam builds from a session snapshot |
| `tests/test_category_first_income.py` | 16 | income generation (allowance level to income) |
| `tests/test_session_key_registry.py` | 11 | every session-state key the screens touch has an entry in `app/state/registry.py`, and every entry is still used |
| `tests/test_rtd_batch4_ui.py` | 6 | Decision 4 driven through the real UI (Streamlit AppTest) |
| `tests/test_continuous_only_income_ui.py` | 5 | continuous income mode through the real UI |
| `tests/test_disclose_documents.py` | 5 | Decision 2 against the professor's frozen income column |
| `tests/test_rtd_comparison_layout_ui.py` | 3 | the Decision 4 comparison layout on the results page |

To run one group on its own:

```bash
pytest tests/test_import_boundary.py -q     # 106 passed
```

The boundary tests import each module in a fresh subprocess, so that group takes
longer to run than its count suggests.

## 7. Troubleshooting

- **`ModuleNotFoundError: No module named 'src'` / `'app'`** — run commands from
  the repository root (the folder containing `app_enhanced_new.py`), not from a
  subdirectory.
- **`streamlit: command not found`** — the virtual environment is not activated
  (step 3), or dependencies were installed into a different environment.
- **Port 8501 already in use** — another Streamlit instance is running. Stop it,
  or launch on a different port: `streamlit run app_enhanced_new.py --server.port 8502`.
- **Excel read errors** — make sure the workbooks in `data/` are not open in
  Excel (Office creates `~$` lock files) and that `openpyxl` installed correctly.
- **Results change between runs** — expected when the stochastic component is on
  and the seed changes; fix the seed on Page 1 for reproducible output.
