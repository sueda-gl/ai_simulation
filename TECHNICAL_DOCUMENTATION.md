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

*(This section will detail the purpose of each directory and file within the project.)*

### 2.1. Root Directory

- `app_enhanced_new.py`: This is the main entry point for the Streamlit application. It sets up the page configuration, applies custom CSS, initializes the session state, and handles the routing between the three main pages (`page1`, `page2`, and `results`).
- `requirements.txt`: This file lists all the Python dependencies required to run the project, such as `streamlit`, `pandas`, `numpy`, and `scikit-learn`. It ensures that the project can be set up in a consistent environment.
- `README.md`: *(Assuming standard use)* This file provides a general overview of the project, setup instructions, and usage guidelines.
- `... and other root files ...`

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

## 3. Core Concepts & Workflow

This section explains the key concepts of the simulation and the typical workflow for a user.

### 3.1. The Agent State

The central data structure in the simulation is the **Agent State**. Each agent is represented by a Python dictionary. The simulation begins by initializing this dictionary with the agent's immutable traits generated by the `TraitEngine`.

As the Orchestrator runs the agent through the sequence of 13 decisions, the dictionary is progressively enriched. Each decision module:
1.  **Reads** necessary information from the current agent state (e.g., traits, outputs of previous decisions).
2.  **Executes** its logic.
3.  **Writes** its output back into the agent state dictionary as one or more new keys.

This creates a rich log of each agent's journey through the simulation, with the final state containing all traits and decision outcomes.

### 3.2. Simulation Flow

The simulation progresses through a clear and logical sequence of steps:

1.  **Configuration**: The user starts on **Page 1** of the application to set global parameters, such as the number of agents, the simulation mode, and market conditions. On **Page 2**, they can select specific decisions to customize or leave them to their default behaviors.
2.  **Initiation**: When the user starts a simulation, the frontend in the `app/` directory gathers all the configured parameters.
3.  **Orchestration**: The `app/simulation.py` module selects the appropriate **Orchestrator** from the `src/` directory based on the chosen "population mode".
4.  **Agent Generation**: The Orchestrator calls the `TraitEngine` to generate a population of synthetic agents. Each agent is created with a set of traits (e.g., honesty, income level) that are statistically consistent with the original study data.
5.  **Decision Execution**: The Orchestrator iterates through each agent and executes the 13 decision modules from `src/decisions/` in a fixed order. The state of each agent is updated as it progresses through the decisions.
6.  **Results Aggregation**: Once all agents have completed all decisions, the results are compiled into a pandas DataFrame.
7.  **Visualization**: The results are passed back to the frontend, where the modules in `app/pages/results/` render the data through tables, charts, and visualizations.

### 3.3. Simulation & Population Modes

The platform supports several simulation modes, selectable on Page 1, each serving a different analytical purpose:

- **Copula (synthetic)**: The default mode. It uses the `TraitEngine` to generate a new, synthetic population of agents for each run. This is the most flexible mode, allowing for an unlimited number of agents that still preserve the statistical properties of the original dataset.
- **Research Specification**: This mode uses the original 280 study participants as the agent pool. If more than 280 agents are requested, it performs bootstrap sampling (sampling with replacement). This mode is designed to closely replicate the methodology of the original research.
- **Research Baseline**: Similar to the Research Specification, this mode uses the original 280 participants. However, it disables all stochastic (random) components in the decision models, making the simulation deterministic. This is useful for establishing a baseline or for debugging.
- **Dependent Variable Resampling**: A specialized mode that focuses only on the `donation_default` outcome. It pre-calculates the distribution of donation rates from the original 280 participants and then generates a new population by simply sampling from this outcome distribution, ignoring all other agent traits.

### 3.4. The Orchestrator

The **Orchestrator** is the central component of the simulation engine. There are several orchestrators, each corresponding to a different "population mode" that can be selected in the UI:
*   `Orchestrator`: The default, for generating a synthetic population using the `TraitEngine`.
*   `OrchestratorDocMode`: For the "Research Specification" mode, using the original 280 study participants.
*   `OrchestratorBaseline`: For the "Research Baseline" mode, which is deterministic.
*   `OrchestratorDepVar`: A specialized mode for resampling the outcome variable directly.

The orchestrator's primary job is to manage the simulation run, from agent creation to the sequential execution of the decision modules.

### 3.5. The Trait Engine

The `TraitEngine` is responsible for creating the population of synthetic agents. It uses a **Gaussian copula model** that has been pre-trained on the data of the original 280 study participants. A copula is a statistical tool that allows for modeling the dependence structure between multiple variables separately from their individual distributions. In this project, it captures the complex correlations between different agent traits (e.g., the relationship between honesty, income, and study program), ensuring that the synthetic agents are realistic and statistically sound.

### 3.6. Decision Modules

Each of the 13 decisions that an agent makes is encapsulated in its own Python module within the `src/decisions/` directory. This modular design is a key strength of the architecture, as it makes it easy to understand, modify, or add new decisions without affecting the rest of the system. Each decision function takes the current state of an agent as input and returns a dictionary with the outcomes of that decision, which are then used to update the agent's state for the subsequent decisions.
