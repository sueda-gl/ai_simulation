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

1. **Page 1 — Common parameters.** Population size, random seed, income mode
   (categorical allowance levels vs. continuous income), discount income
   threshold, and other settings shared by all decisions.
2. **Page 2 — Decisions.** Enable and configure the individual decisions
   (disclose income, disclose documents, donation default, purchasing, vendor
   selection, bidding, …), including their stochastic components, then run the
   simulation.
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

# Options: --decision <name> (repeatable), --population-mode copula|research spec|research baseline|,
#          --income-mode categorical|continuous, --format parquet|csv, --output-dir <dir>
python scripts/run_simulation.py --agents 5000 --decision disclose_income --format csv

# Monte-Carlo study: many repetitions with incrementing seeds
python scripts/run_mc_study.py --agents 10000 --runs 500 --base-seed 1
```

Results land in `outputs/` (git-ignored) as timestamped parquet/CSV files, e.g.
`simulation_seed42_agents1000_all_20260805_120000.parquet`. (Runs from the
Streamlit app also save an `enhanced_params_*.json` alongside the results,
recording the parameters used.)

## 5. Repository layout

```
app_enhanced_new.py   Streamlit entry point (run this)
app/                  UI layer: pages, tabs, results views, session state
src/                  Simulation engine
  orchestrator*.py    One orchestrator per population mode
  trait_engine.py     Samples synthetic agents from the copula model
  decisions/          One module per decision (plus *_stochastic variants)
  utils/stochastic.py Shared stochastic-component helpers (sigma logic)
  build_dd_sigma.py   Derives the Decision 2 sigma from the raw experiment data
config/               decisions.yaml, simulation.yaml, trait_model.pkl (pre-trained copula), seeds
data/                 Raw experiment/survey Excel files + frozen Stata verification CSVs
scripts/              CLI runners (run_simulation, run_mc_study) and copula training
tests/                Pytest validation suite (checks model output against Stata references)
outputs/              Simulation results (created at runtime, git-ignored)
stata/                Stata reference material
```

**Note:** `config/trait_model.pkl` is committed, so you do **not** need to train
anything before running. Only re-run `python scripts/train_copula.py` if the
underlying experiment data or the master trait list changes (it overwrites the
pickle in `config/`).

## 6. Running the tests

The tests validate the decision models against the professor's Stata reference
results. `pytest` is not in `requirements.txt`, so install it once:

```bash
pip install pytest
pytest tests/
```

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
