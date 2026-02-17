# Technical Documentation: AI Agent Simulation Platform

## 1. High-Level Architecture

This project is an **AI Agent Simulation platform** built in Python, designed to model the behavior of agents in an economic environment. The architecture is logically divided into three main components:

1.  **Frontend (UI):** A multi-page web application built with **Streamlit**, located in the `app/` directory. This is the user's primary interface for configuring simulation parameters, initiating runs, and visualizing the results.
2.  **Backend (Simulation Engine):** The core simulation logic, located in the `src/` directory. This engine is responsible for generating synthetic agents with specific traits and running them through a sequence of 13 predefined decisions.
3.  **Configuration:** The behavior of the simulation is defined by **YAML** files in the `config/` directory. This separation of configuration from code allows for easy modification of parameters without altering the underlying logic.

The simulation can be executed in various modes, including single runs for specific scenarios, comprehensive Monte Carlo studies for statistical analysis, and direct comparisons between different population models.

### Architectural Flow Diagram

The following diagram illustrates the interaction between the main components of the application:

```
[ User ] --> [ Streamlit Frontend (app/) ]
   |                  |
   | (UI Events)      | (Simulation Parameters)
   |                  v
   |         [ Simulation Runner (app/simulation.py) ]
   |                  |
   |                  | (Instantiates & Configures)
   |                  v
   |        [ Orchestrator (src/orchestrator.py) ]---------+
   |                  |                                    |
   | (Agent States)   | (Calls)                            | (Reads)
   |                  v                                    |
   |         [ Decision Modules (src/decisions/) ]          |
   |                                                       |
   +------------------------------------------------------[ Configuration (config/*.yaml) ]
```

## 2. Directory & File Structure

### 2.1. Root Directory

- `app_enhanced_new.py`: This is the main entry point for the Streamlit application. It sets up the page configuration, applies custom CSS, initializes the session state, and handles the routing between the three main pages (`page1`, `page2`, and `results`).
- `requirements.txt`: This file lists all the Python dependencies required to run the project, such as `streamlit`, `pandas`, `numpy`, and `scikit-learn`. It ensures that the project can be set up in a consistent environment.
- `README.md`: This file provides a general overview of the project, setup instructions, and usage guidelines.

### 2.2. `app/` - Frontend & UI Logic

- `app/pages.py`: This file acts as a central module for the application's pages. It imports and re-exports rendering functions from the various page modules in `app/pages/`. This modular design keeps the main application file clean and makes the page structure easy to manage.
- `app/simulation.py`: This is a critical file that bridges the UI and the backend simulation engine. It contains the logic for running both single simulations and Monte Carlo studies. It takes the parameters configured by the user in the UI, selects the appropriate `Orchestrator` from `src/`, and executes the simulation. For long-running Monte Carlo studies, it calls `scripts/run_mc_study.py` as a subprocess to keep the UI from freezing.
- `app/components.py`: This file contains reusable UI components and visualization functions used across the application, primarily on the results page. This includes functions for displaying simulation overviews, plotting histograms of donation rates, and rendering CSS styles. Centralizing these components promotes consistency and code reuse.
- `app/models.py`: This file defines the data structures and session state management for the application. It includes the `SimulationParameters` dataclass, which holds all the parameters from Page 1, and the `initialize_session_state` function, which sets up the default values for the application's state. It also contains logic for loading simulation parameters from the YAML configuration files.

#### `app/pages/`

- `page1_common_params.py`: This file is responsible for rendering the first page of the UI. This page allows users to configure global simulation parameters that affect all agents and decisions, such as the number of agents, simulation mode (single run vs. Monte Carlo), market conditions, and income distribution settings.
- `page2_decisions.py`: This file renders the second page of the UI, which is focused on configuring the 13 specific agent decisions. It features a multi-select box to choose which decisions to customize and then creates a tabbed interface for those selections. It also includes an "Overview" tab that provides a summary and a button to run the complete simulation.
- `decision_execution.py`: This file contains the logic for executing simulations based on user actions on Page 2. It defines the default values for decisions that are not explicitly configured by the user, and it includes functions to run a simulation for a single decision or for all decisions combined.
- `navigation.py`: This file manages the navigation flow between the different pages of the application. It contains functions like `go_to_page1`, `go_to_page2`, and `render_navigation` which display the appropriate "Back" and "Next" buttons based on the current page.

#### `app/pages/decision_tabs/`

This directory is responsible for rendering the configuration tabs on "Page 2" of the application. Each file corresponds to a specific agent decision, providing a UI for the user to adjust its parameters.

- `bid_value_tab.py`: Renders an informational tab for the `bid_value` decision. It does not contain user-configurable parameters but instead serves to explain how the decision works. It displays the mathematical formulas used to calculate an agent's bidding range, shows the current global market parameters it depends on (from Page 1), and provides a step-by-step example calculation.
- `default_config.py`: This file renders the UI for configuring the default behavior of all decisions that were *not* explicitly selected by the user for customization. It dynamically creates UI elements (sliders, radio buttons, checkboxes) for each unselected decision, allowing the user to set their fallback behaviors from the "Overview" tab on Page 2.
- `donation_default.py`: This is the most complex tab, providing a detailed interface for the `donation_default` decision. Its key features include:
    - **Dynamic Controls**: The UI for configuring the stochastic (random) component changes based on the "population mode" selected on Page 1, showing different options for "Copula" vs. "Research" modes.
    - **Formula Display**: It interactively displays the mathematical regression formulas used in the model, updating the coefficients in the formula as the user changes them.
    - **Parameter Overrides**: It provides input fields for the user to override the default regression coefficients and saves these changes back to the `decisions.yaml` file.
- `generic_decision.py`: This is a fallback UI for any decision that does not have a dedicated tab file. It renders a simple placeholder message informing the user that the decision will run with its default behavior.
- `global_parameters.py`: This module renders a read-only, multi-column summary of all the global parameters that were configured on Page 1. This is displayed on the "Overview" tab to give the user context without needing to navigate back.
- `rejected_transaction.py`: Renders the UI for the `rejected_transaction_defaults` and `rejected_transaction_option` decisions. It primarily consists of a set of radio buttons that allow the user to choose the default strategy an agent should follow when a transaction fails (e.g., "forgo the transaction", "place a bid", etc.).

#### `app/pages/results/`

This directory and its subdirectories are responsible for rendering the entire "Results" page, from high-level summaries to detailed visualizations.

- `main_results.py`: This file acts as the main router for the results page. It checks the session state to determine what kind of results are available (single run, Monte Carlo, or none) and then calls the appropriate functions from other modules to render the page content. It also handles the display of the decision configuration summary, showing which decisions were run with custom parameters versus defaults.
- `comparisons.py`: This module is responsible for generating the comparison views when the simulation is run in a mode that produces multiple results (e.g., "Compare all" population modes). It creates side-by-side layouts using `st.columns` to display the overview and charts for each result set, making it easy to compare their outcomes.
- `decision_visualizations.py`: This file acts as a sub-router for visualizing the results of individual decisions. It uses a dictionary (`DECISION_VISUALIZATIONS`) to map a decision's name to the correct rendering function in the `visualizations/` subdirectory. This modular approach keeps the code clean and makes it easy to add new visualizations.
- `config_selection.py`: This module renders the UI for the "configuration selection" feature. When a user runs a comparison for the `donation_default` decision, this module displays interactive "cards" for each resulting configuration. Each card shows key metrics and a "Use This Config" button, allowing the user to promote one set of parameters to be used in a subsequent full simulation run.
- `details.py`: A simple module that groups together and re-exports the functions from the `components/` subdirectory. This helps to organize the imports for the main results page.

#### `app/pages/results/components/`

This directory contains the reusable UI components that make up the building blocks of the results page.

- `agent_details.py`: Renders the "Individual Agent Details" section. It uses an `st.selectbox` to create a dropdown menu of all agents, and when an agent is selected, it displays their specific traits and decision outcomes in two separate pandas DataFrames.
- `export_section.py`: Renders the "Export Results" section. Its primary feature is the `st.download_button` for downloading the results as an Excel file. It contains logic to handle two export scenarios: for a single run, it creates a simple Excel file; for comparison runs, it intelligently combines the results into a single file, with the differing columns highlighted in green for easy analysis. It also contains the "Clear Results" button, which resets the application's session state.
- `parameter_summary.py`: This file contains the logic to render an analysis of which global parameters were used in a given simulation run. It compares the parameters required by the selected decisions against the full list of available global parameters and displays metrics on "parameter efficiency".

#### `app/pages/results/visualizations/`

This directory holds the specific plotting and visualization logic for each agent decision.

- `bidding_viz.py`: Creates the visualizations for the `bid_value` decision. It displays the mathematical formula for the bidding range, shows the current market parameters, and plots a histogram of the actual bid values from the simulation, overlaid with vertical lines indicating the theoretical minimum and maximum bids.
- `consumption_viz.py`: Renders visualizations for `consumption_quantity` and `consumption_frequency`. This includes histograms of the number of items purchased, box plots showing consumption by income category, and scatter plots visualizing the timing of individual agent purchases over the simulation period.
- `disclosure_viz.py`: Creates visualizations for the `disclose_income` and `disclose_documents` decisions. Since these are binary (Yes/No) decisions, it primarily uses `st.metric` to show the counts and percentages of each choice and `plotly.express.pie` to create pie charts of the distributions.
- `donation_viz.py`: Renders the visualizations for the donation-related decisions. It uses `plotly.express.histogram` to display the distribution of donation rates across the agent population and provides a statistical summary table. It contains special logic to change its display based on whether a specific donation configuration has been selected by the user.
- `transaction_viz.py`: Handles visualizations for transaction-related decisions like `purchase_vs_bid`. It analyzes the `purchase_requests` data to show the breakdown of how many requests resulted in a "Purchase Now", "Bid", "Fixed", or "Discount" outcome, and displays this information in metrics and pie charts.
- `vendor_viz.py`: Creates the visualizations for `vendor_choice_weights` and `vendor_selection`. It shows the distribution of agent selections across the different vendors and displays the underlying vendor attributes (price, quality, etc.) in a table and bar charts to help explain *why* certain vendors were more popular than others.
- `viz_helpers.py`: A utility module that contains helper functions used by the other visualization modules. For example, it has a function to create a consistent set of UI controls for decisions that are based on a configurable probability.

### 2.3. `src/` - Backend Simulation Engine

- `src/orchestrator.py`: This is the primary orchestrator for the "Copula (synthetic)" mode. Its main responsibilities are to initialize the `TraitEngine` to sample synthetic agents, load the decision modules, and then iterate through each agent to execute the decisions in a predefined sequence.
- `src/orchestrator_doc_mode.py`: This orchestrator is used for the "Research Specification" mode. Instead of generating synthetic agents, it uses the original 280 participants from the study's dataset. If the requested number of agents is larger than 280, it uses bootstrap sampling.
- `src/orchestrator_baseline.py`: This orchestrator is for the "Research Baseline" mode. It is similar to the `OrchestratorDocMode` in that it uses the original 280 participants, but it explicitly disables the stochastic (random) components of the decisions to provide a deterministic baseline.
- `src/orchestrator_depvar.py`: This is a specialized orchestrator for the "Dependent variable resampling" mode. It pre-computes the `donation_default` rates for the original 280 participants and then generates new populations by simply resampling from this distribution of outcomes, without preserving any of the agents' underlying traits.
- `src/trait_engine.py`: This is a crucial component of the backend. It is responsible for generating the synthetic population of agents using a Gaussian copula model, which is loaded from `config/trait_model.pkl`. This approach allows the simulation to create an unlimited number of agents that retain the statistical properties and correlations of the original study participants.
- `src/build_master_traits.py`: A utility script that reads the `config/trait_requirements.yaml` file to determine the complete set of agent traits that are required by all the different decision modules.
- `src/validate_traits.py`: A script used for data validation. It loads the original survey and experiment data, merges them, and checks if all the master traits required for the simulation are present.
- `src/vendor_attribute_generator.py`: This module contains functions to generate the attributes of the vendors in the simulation, such as their product quality and sustainability scores.
- `src/vendor_price_generator.py`: A module for generating vendor prices based on the parameters set on Page 1, such as the minimum, maximum, and average price.

#### `src/decisions/`

This directory contains a Python module for each of the 13 agent decisions. The orchestrator calls these modules in sequence for each agent.

- `bid_value.py`: **(Decision 10)** Determines the monetary value of a bid for agents who choose to bid rather than purchase now.
- `consumption_frequency.py`: **(Decision 7)** A calculated decision that determines how frequently an agent makes purchases based on their total consumption quantity and the duration of the simulation.
- `consumption_quantity.py`: **(Decision 6)** Determines the total number of items an agent wishes to purchase during the simulation period.
- `disclose_documents.py`: **(Decision 2)** A probabilistic decision for low-income agents on whether to disclose documents to qualify for a discount.
- `disclose_income.py`: **(Decision 1)** A probabilistic decision on whether an agent discloses their income to get a fixed price.
- `donation_default.py`: **(Decision 3)** A core decision module that calculates an agent's default donation rate based on a regression model using their traits.
- `donation_default_stochastic.py`: A variant of `donation_default` that is used in the "Research Specification" mode to add a random component to the donation calculation.
- `enrich_purchase_requests.py`: **(Decision 6b)** An intermediate step that enriches the purchase requests generated by `consumption_quantity` with more detailed transaction-level information.
- `final_donation_rate.py`: **(Decision 13)** Determines the final donation rate, which typically defaults to the value calculated in `donation_default`.
- `income_utils.py`: A helper module providing utility functions for income-related calculations, centralizing the logic for income generation and customer type determination.
- `purchase_vs_bid.py`: **(Decision 9)** A probabilistic decision for agents who are not on a fixed or discounted price, determining whether they will "purchase now" at a set price or place a bid.
- `rejected_bid_value.py`: **(Decision 12)** Determines the value of a subsequent bid if an agent's initial bid is rejected.
- `rejected_transaction_defaults.py`: **(Decision 4)** Sets the default behavior for how an agent should react to a rejected transaction.
- `rejected_transaction_option.py`: **(Decision 11)** The specific action an agent takes after a transaction is rejected, based on the defaults set in Decision 4.
- `vendor_choice_weights.py`: **(Decision 5)** Sets the weights an agent assigns to different vendor attributes (like price, quality, etc.), which influences their choice of vendor.
- `vendor_selection.py`: **(Decision 8)** The process by which an agent selects a vendor, based on the weights from Decision 5 and the vendor attributes.

### 2.4. `config/` - Simulation Configuration

- `simulation.yaml`: This file defines all the global parameters for the simulation that are configurable on Page 1 of the UI. This includes settings for the market (e.g., `num_vendors`, `market_price`), time parameters (`periods`, `duration_hours`), and the parameters for the income distribution models.
- `decisions.yaml`: This file contains the detailed parameters for each of the 13 agent decisions. For decisions that are based on a regression model, like `donation_default`, this file stores the coefficients. For other decisions, it may specify default behaviors or formulas. It also lists which global parameters from `simulation.yaml` each decision depends on.
- `trait_requirements.yaml`: This file explicitly lists which agent traits from the original dataset are required by each decision module. This is used by `src/build_master_traits.py` to create a master list of all necessary traits for the simulation.

### 2.5. `scripts/` - Command-Line Runners

- `run_simulation.py`: This script provides a command-line interface for running a single simulation. It allows for the configuration of the number of agents, the random seed, and the selection of specific decisions to run. It's a useful tool for running simulations in a server environment or for automated testing.
- `run_mc_study.py`: This script is used to run a Monte Carlo study from the command line. It repeatedly calls `run_simulation.py` with different seeds and then aggregates the results to provide statistical summaries. This is the script that is called by the UI when a user initiates a Monte Carlo study, allowing the computationally intensive work to be done in a separate process.

## 3. Execution Lifecycle & Engine Internals

### 3.1. Determinism, Seeding, and RNG Strategy

- **Global reproducibility**: Each simulation run is anchored by a single user-provided seed. The engine derives independent RNG streams for separate concerns to avoid cross-coupling:
  - Setup RNG: initializes vendors once per run.
  - Global agent RNG: derived from the seed; each agent gets an independent child RNG via `rng_global.integers(1e9)` ensuring stable per-agent randomness regardless of population size.
  - Decision modules receive the agent RNG and must not create their own global RNGs.
- **Mode-specific stochasticity**: Some decisions (notably `donation_default`) can enable or disable stochastic components depending on population mode (e.g., `copula` vs `documentation`) and UI flags like `sigma_in_copula` or `sigma_in_research`.

### 3.2. Orchestrator Contract

- Entry point: `src/orchestrator.py` class `Orchestrator`.
- Loads configuration:
  - `config/decisions.yaml` into `self.config` (decision parameters, coefficients, default behaviors)
  - `config/simulation.yaml` into `self.simulation_config` (Page 1 globals)
- Initializes `TraitEngine` and sets `pop_context`.
- Defines canonical decision order (chronological, with 6b for per-request enrichment):
  1. `disclose_income`
  2. `disclose_documents`
  3. `donation_default`
  4. `rejected_transaction_defaults`
  5. `vendor_choice_weights`
  6. `consumption_quantity`
  6b. `enrich_purchase_requests`
  7. `consumption_frequency`
  8. `vendor_selection`
  9. `purchase_vs_bid` (deprecated; retained for backward compatibility)
  10. `bid_value` (deprecated; retained for backward compatibility)
  11. `rejected_transaction_option`
  12. `rejected_bid_value`
  13. `final_donation_rate`
- Dynamic module loading: Each decision is imported via `importlib` using the name above and invoked as a function with signature: `decision(agent_state, params, rng, simulation_config=..., **kwargs)`.
- Agent loop:
  - Samples a row of immutable traits from `TraitEngine` (copula-sampled dataframe row converted to dict).
  - Injects identifiers: `index` and `agent_id` (1-indexed).
  - Creates an agent-scoped RNG.
  - Runs decisions in order; merges each decision’s returned dictionary into the `agent_state`.
- Vendors: `_initialize_vendors(rng_setup)` creates a vendor list with attributes and stores it in `self.simulation_config['vendors']`. The returned DataFrame of agents attaches vendor data to `results_df.attrs['vendors']` for UI visualization.

### 3.3. Agent State: Schema and Key Invariants

The `agent_state` dictionary evolves as decisions run. Common keys:
- Core identifiers and traits:
  - `index` (int), `agent_id` (int)
  - Copula traits (e.g., `Honesty_Humility`, `Assigned Allowance Level`, `Study Program`, `Group_experiment`, `TWT+Sospeso [=AW2+AX2]{Periods 1+2}`)
- Income system:
  - `income` (float, dollar-scale), `actual_allowance` (float, 12–200 scale)
  - `customer_type` ∈ {`discount`, `fixed`, `regular`}
  - `income_category` (int)
- Consumption:
  - `consumption_quantity` (int), `consumption_frequency` (float)
  - `purchase_requests` (list of request dicts; see schema below)
- Vendor selection:
  - `vendor_choice_weights` (dict of weights)
  - `vendor_proximity_scores` (dict[str vendor_id → float 0–100])
  - `vendor_selection` (int vendor_id), `vendor_rank` (int), `preferred_vendor` (int), `got_preferred` (bool), `allocation_failed` (bool)
- Donation:
  - `donation_default` (float in [0,1]) and `final_donation_rate` (float)
- Purchase vs bid:
  - `purchase_vs_bid` (legacy agent-level string when used)
  - On requests: `platformPrice` and `bid_value` per purchase request

Purchase request schema (per element of `purchase_requests`):
```json
{
  "request_id": int,               // 1..N within agent
  "quantity": 1,                   // default for snapshot mode
  "timestamp_hours": float,        // uniform in [0, periods*duration_hours]
  "customer_id": int,              // mirrors agent_id
  "customer_type": "discount" | "fixed" | "regular",
  "vendorID": int,                 // set by vendor_selection
  "platformPrice": "DISCOUNT" | "FIXED" | "PN" | "BID" | null,
  "bid_value": number | "N/A" | null
}
```

### 3.4. Configuration Precedence and Propagation

- YAML provides defaults, the UI overrides them at runtime. The UI layer copies all Page 1 values (market, time, income distributions, category counts, etc.) into `orchestrator.simulation_config['simulation']` before execution.
- Decision defaults and UI-configured overrides for unselected decisions are passed via `simulation_config`:
  - `default_decisions_list`: names of decisions not explicitly customized; modules use this to branch to default behavior.
  - `default_decisions` and `random_decisions`: per-decision structures for radio/checkbox/numeric/random-probability defaults; request-level decisions respect these at enrichment time.
- `donation_default` coefficients are loaded from YAML and may be overridden by selection of a saved configuration or session-state custom coefficients.

## 4. Trait Engine (Copula) Details

- Model file: `config/trait_model.pkl` stores the correlation matrix `Sigma`, trait names, and per-trait CDF decoders.
- Sampling pipeline per agent: draw standard normals → apply Cholesky factor → transform via normal CDF to uniforms → decode to original trait scales using stored inverse CDF approximations.
- Only the traits required by decisions are included, preserving the empirical correlation structure across traits.

## 5. Income Architecture (Category-First)

- Source of truth is `Assigned Allowance Level` in {1..5} coming from traits.
- Two derived values are generated once per agent and cached:
  - `actual_allowance` (deterministic 12–200 mapping: {1:12, 2:32, 3:72, 4:128, 5:200}) used by `donation_default` regression when running with continuous income mode.
  - `income` (dollar-scale) sampled by mapping the level to its percentile bucket and applying the inverse CDF (PPF) of the selected income distribution with UI-provided parameters and optional max clipping.
- Percentile buckets: [0–20), [20–40), [40–60), [60–80), [80–100].
- Customer type derivation (after disclosure decisions):
  - `discount`: income ≤ threshold AND documents disclosed
  - `fixed`: disclosed income
  - `regular`: otherwise
- All helpers live in `src/decisions/income_utils.py` and are the single source of truth for income-related logic and parameter access.

## 6. Adding a New Decision Module

- Create `src/decisions/<name>.py` exposing a function `<name>(agent_state, params, rng, simulation_config=None, **kwargs) -> dict`.
- Add the decision name to the orchestrator’s `decision_order` and to `app.models.ALL_DECISIONS` to preserve chronological UI ordering.
- Declare its Page 1 parameter usage in `config/decisions.yaml` under `uses_global_parameters` so it appears in the parameter efficiency summary.
- If it needs defaults when not selected on Page 2, ensure it respects `simulation_config['default_decisions_list']` and reads any structured defaults under `simulation_config['default_decisions'][<name>]`.
- To visualize the decision on the Results page, add an entry in the visualization router (`DECISION_VISUALIZATIONS`) and implement a renderer.

## 7. Error Handling & Edge Cases

- No purchase requests: vendor selection returns `NaN` selection and leaves requests unchanged.
- No vendors configured: selection falls back to vendor 1.
- Vendor capacity exhausted: `allocation_failed=True` and selection `NaN`.
- Missing or malformed defaults: decisions fall back conservatively and prefer numeric-safe `NaN`/`None` to keep DataFrame operations stable.
- Distribution max bounds: income sampling clips to configured maximums where applicable.


## 8. Glossary

- **Agent state**: The evolving dictionary for a single agent; starts with traits and accumulates decision outputs.
- **Decision order**: The fixed chronological sequence in which decisions run within the orchestrator.
- **Population mode**: Controls the source of agents and stochastic components: `Copula (synthetic)`, `Research Specification`, `Research Baseline`, `Dependent variable resampling`.
- **Category-first**: Architecture where categorical levels from traits drive both deterministic `actual_allowance` and percentile-bucket sampling for dollar `income`.
- **Platform price labels**: `DISCOUNT`, `FIXED`, `PN` (Purchase Now), `BID`.

## 9. Orchestrator Variants

### 9.1. Documentation/Research Spec (`src/orchestrator_doc_mode.py`)

- Uses original 280 participants (`src.validate_traits.merged` filtered to the required traits) instead of copula sampling.
- Loads `*_stochastic` decision modules when available, falling back to regular implementations; passes `pop_context='documentation'` to modules that support it (e.g., `donation_default`).
- Supports `outcome_draws`: repeats the dependent-variable draw per agent, emitting `draw_id` and appending each draw as a separate row.
- When only `donation_default_raw_pos` exists, performs a global-max rescale to produce `donation_default` in [0,1].
- Vendor initialization mirrors the main orchestrator, with optional explicit `vendor_prices` vs randomized within `[vendor_price_min, vendor_price_max]`.

### 9.2. Research Baseline (`src/orchestrator_baseline.py`)

- Uses original participants (bootstrap if `n_agents>280`) with `pop_context='baseline'`.
- Disables stochasticity by forcing `donation_default.stochastic.sigma_value=0.0` at call time; loads regular (non-stochastic) decision modules.
- Catches decision exceptions and substitutes `get_actual_default_value(decision_name)` output, preserving the pipeline.
- Shares the same vendor generation and cross-agent capacity handling.

### 9.3. Dependent Variable Resampling (`src/orchestrator_depvar.py`)

- Pre-computes an empirical donation distribution for the original participants using `donation_default_stochastic` if present, else deterministic `donation_default`; stores both processed and RAW (pre-truncation) variants.
- Generates populations by bootstrap resampling only the donation outcome; traits and other decisions are not executed.
- Exposes `set_raw_output(True|False)` to toggle which column name is emitted: `donation_default_raw` vs `donation_default`.
- Only `donation_default` is supported in this mode.

## 10. Disclosure and Vendor Choice Decisions

### 10.1. Decision 1 - `disclose_income` (`src/decisions/disclose_income.py`)

- Inputs: `simulation_config.random_decisions.disclose_income` when configured with `type='random_probability'` (`probability_y`, `options`).
- Behavior: Weighted random draw via injected `rng` if configured; otherwise 50/50 fallback.
- Output: `{"disclose_income": "Y"|"N"}`.

### 10.2. Decision 2 - `disclose_documents` (`src/decisions/disclose_documents.py`)

- Inputs: Uses `get_agent_income(...)` and compares to `discount_income_threshold` (Page 1). Also reads `simulation_config.random_decisions.disclose_documents` when eligible.
- Behavior:
  - If income ≥ threshold: sets `disclose_documents="NA"`, derives and returns `customer_type` immediately (`fixed` or `regular`).
  - If income < threshold: performs a probability-based Y/N draw; then calls `get_customer_type(...)` and returns both fields.
- Outputs: `{"disclose_documents": "Y"|"N"|"NA", "customer_type": "discount"|"fixed"|"regular"}`.

### 10.3. Decision 5 - `vendor_choice_weights` (`src/decisions/vendor_choice_weights.py`)

- Inputs: When unselected, reads Overview configuration from `simulation_config['default_decisions']['vendor_choice_weights']` and uses its precomputed `weights`.
- Behavior: If not configured, returns equal weights across `price`, `quality`, `proximity`, `sustainability` (0.25 each).
- Output: `{"vendor_choice_weights": {price: float, quality: float, proximity: float, sustainability: float}}`.

## 11. Default Decisions Plumbing (Overview → Execution)

- `app/pages/decision_execution.py` defines `DEFAULT_DECISION_VALUES` and human-readable `DEFAULT_DECISION_DESCRIPTIONS` for all decisions.
- `get_actual_default_value(decision_name)` resolves runtime defaults with this priority:
  1) Page 2 Overview pre-configured session state (e.g., `{decision}_default_probability_y`, `{decision}_default_params`, `{decision}_default_selection`, `{decision}_default_value`)
  2) Post-simulation adjustments on the Results page (e.g., `{decision}_probability_y`, decision-specific selection keys)
  3) Static `DEFAULT_DECISION_VALUES`
- Checkbox selections produce equal-share weights for included parameters and 0 for excluded ones; random-probability decisions draw with configured `probability_y`.
- Complete Simulation is gated by `can_run_complete_simulation()` to ensure only a single donation configuration is active (or an explicit selection) when UI settings would otherwise produce multiple variants.

## 12. Results Keys, Visualization Router, and Config Selection

- Result dict keys by mode:
  - Compare-all population: `copula_{income}`, `research_spec_{income}`, `research_baseline_{income}` where `{income}` ∈ {`categorical`,`continuous`}.
  - Single population: `categorical` or `continuous` (or a single-mode key like `copula`).
- Visualization router: `app/pages/results/visualizations/__init__.py` registers `DECISION_VISUALIZATIONS` and routes decisions to specialized renderers; unknown decisions fall back to a generic preview.
- Configuration selection flow:
  - A donation config is saved via `save_selected_configuration` (coefficients, stochastic params, metrics, `result_key`).
  - Subsequent full simulations apply this selection (`apply_selected_donation_config`) and update UI state; only the selected variant is generated.

## 13. Export Behavior

- The export component (`export_section.py`) filters out internal columns (`raw`, `index`, `consumption_frequency`, `actual_allowance`, `income`, `customer_type`, `enriched_requests_count`).
- Donation-only runs retain only donation-related columns plus core traits and `agent_id`.
- Multi-config export builds a combined sheet: shared trait columns and non-donation decisions are included once; each configuration’s donation column is appended with a suffixed name and highlighted in green using OpenPyXL.
- “Clear Results” purges session state and reinitializes defaults while staying on the Results page.

## 14. Decision I/O Cheat Sheet

- `disclose_income`
  - Inputs: `random_decisions.disclose_income`
  - Outputs: `disclose_income` ∈ {`Y`,`N`}
- `disclose_documents`
  - Inputs: `income` (via `get_agent_income`), `discount_income_threshold`, optional random config
  - Outputs: `disclose_documents` ∈ {`Y`,`N`,`NA`}, `customer_type` ∈ {`discount`,`fixed`,`regular`}
- `donation_default`
  - Inputs: traits (`Honesty_Humility`, `Group_experiment`, `Study Program`, `Assigned Allowance Level`, observed prosocial), regression coefficients, anchor weights, optional stochastic σ; `pop_context`
  - Outputs: `donation_default` in [0,1] (and raw intermediates in stochastic variants)
- `rejected_transaction_defaults`
  - Inputs: default radio selection from Overview
  - Outputs: `rejected_transaction_defaults` (string key)
- `vendor_choice_weights`
  - Inputs: checkbox selection and computed weights (if unselected)
  - Outputs: `vendor_choice_weights` dict with weights for `price`,`quality`,`proximity`,`sustainability`
- `consumption_quantity`
  - Inputs: `income`, distribution parameters, `num_fixed_categories`, `consumption_limits` or `max_purchases_per_term`, time params
  - Outputs: `consumption_quantity` (int), `purchase_requests` list, `income_category` (int), echoes `income`
- `enrich_purchase_requests` (6b)
  - Inputs: `purchase_requests`, `customer_type`, random decision config for `purchase_vs_bid`
  - Outputs: `purchase_requests` with `platformPrice` and `bid_value` set per request; `enriched_requests_count`
- `consumption_frequency`
  - Inputs: `consumption_quantity`, `periods`, `duration_hours`
  - Outputs: `consumption_frequency` (float)
- `vendor_selection`
  - Inputs: `vendor_choice_weights`, `vendors`, generated `vendor_proximity_scores`, capacity map
  - Outputs: `vendor_selection` (int), `vendor_rank`, `preferred_vendor`, `got_preferred`, `allocation_failed`; updates `purchase_requests[].vendorID`
- `purchase_vs_bid`
  - Inputs: `customer_type`, random decision config
  - Outputs: `purchase_vs_bid` ∈ {`Purchase Now`,`bid`,`NA_discount`,`NA_fixed`} (agent-level legacy); request-level decisions are handled in 6b
- `bid_value`
  - Inputs: pricing params (`market_price`, `platform_markup`, `price_range`)
  - Outputs: `bid_value` (float) or `NaN` when not applicable
- `rejected_transaction_option`
  - Inputs: default radio selection
  - Outputs: `rejected_transaction_option` (string key)
- `rejected_bid_value`
  - Inputs: default placeholder/value
  - Outputs: `rejected_bid_value`
- `final_donation_rate`
  - Inputs: `donation_default` (if present) else default value
  - Outputs: `final_donation_rate`

## 15. Engineering Architecture Patterns & Decisions

- **Functional core, imperative shell**: Decision modules are pure functions over `(agent_state, params, rng, simulation_config)` and return dict diffs; the orchestrators manage sequencing and mutation of `agent_state`.
- **Config-driven pipeline**: YAML + UI session state drive all parameterization; decisions should avoid hard-coded constants and read through `simulation_config` or their `params` block.
- **Category-first invariants**: `Assigned Allowance Level → actual_allowance (12–200) → income (dollar PPF)` is generated once and cached, ensuring consistency across decisions and avoiding logical contradictions.
- **Request-level vs agent-level**: Transactional decisions operate per request via Decision 6b to avoid agent-level coupling; legacy agent-level endpoints remain for backward compatibility.
- **Order dependence**: Vendor capacity enforcement is intentionally order-dependent (first-come, first-served). Keep this in mind for fairness analyses and when parallelizing.
- **Serialization contracts**: `vendor_proximity_scores` uses string keys; `purchase_requests` is nested. Downstream consumers should JSON-encode these before formats like Parquet/CSV lacking nested support.
- **Dynamic module loading**: Missing modules log warnings and the baseline orchestrator substitutes defaults to keep runs healthy.

## 16. Performance & Complexity Notes

- Trait sampling is vectorized across agents; per-agent decision execution is scalar and runs in Python.
- Vendor selection is `O(V)` per agent due to composite scoring over all vendors; proximity generation is also `O(V)`. With capacity constraints, a best-available scan may occur when preferred is sold out.
- Documentation mode with `outcome_draws>1` multiplies the effective number of rows by draws per agent.
- RNG derivation per agent avoids cross-talk; long runs remain reproducible across `n_agents` changes as long as the seed and decision order are unchanged (except for capacity-induced order effects).

## 17. Determinism & Reproducibility Guarantees

- Single global seed → distinct streams for setup vs agent processing; each agent receives a child RNG.
- Decisions must consume randomness only from the injected RNG to preserve reproducibility.
- OrchestratorDocMode bootstrapping uses the setup RNG and thus is reproducible for the same seed.
- Capacity constraints introduce run-order dependence by design; otherwise the system is deterministic given seed, parameters, and code.

## 18. Extensibility & Contracts

- New decisions must expose `<name>(agent_state, params, rng, simulation_config=None, **kwargs) -> dict`, declare `uses_global_parameters` in `config/decisions.yaml`, and be added to `decision_order` and `ALL_DECISIONS`.
- Respect `simulation_config['default_decisions_list']` to provide default-mode behavior when the decision is unselected; read any structured defaults from `simulation_config['default_decisions'][<name>]`.
- If a decision introduces nested structures or large arrays, document its schema here and ensure Result export can either flatten or exclude as appropriate.
- To surface results, register a renderer in `DECISION_VISUALIZATIONS` or the system will display a generic preview.



