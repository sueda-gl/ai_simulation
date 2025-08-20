# AI Agent Simulation Framework

A modular simulation system for generating synthetic agents that make decisions based on personality traits and behavioral models fitted from survey and experimental data.

## Overview

This framework uses a **Gaussian copula** to generate unlimited synthetic agents with realistic trait combinations, then applies **13 decision modules** to simulate various choices (donation rates, vendor selection, consumption patterns, etc.).

## Architecture

```
project_root/
├── data/                          # Raw survey and experiment data
├── config/                        # Configuration files
│   ├── trait_requirements.yaml   # Which traits each decision needs
│   ├── decisions.yaml            # Regression coefficients and parameters
│   └── trait_model.pkl           # Fitted copula model
├── src/                          # Core simulation code
│   ├── trait_engine.py           # Generates synthetic agents via copula
│   ├── orchestrator.py           # Coordinates trait sampling and decisions
│   ├── build_master_traits.py    # Extracts trait requirements from YAML
│   ├── validate_traits.py        # Validates trait availability
│   └── decisions/                # Individual decision modules
│       ├── donation_default.py   # Decision 3 (fully implemented)
│       ├── disclose_income.py    # Decision 1 (placeholder)
│       └── ...                   # Decisions 2, 4-13 (placeholders)
├── scripts/                      # Utilities and entry points
│   ├── train_copula.py          # Fits copula model on trait data
│   ├── run_simulation.py        # CLI interface for single simulations
│   └── run_mc_study.py           # Monte-Carlo driver for uncertainty analysis
├── app.py                        # Streamlit web dashboard
├── launch_dashboard.py           # Quick launcher for web interface
└── outputs/                      # Simulation results (auto-generated)
```

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment (if not already active)
source prosocial_analysis_env/bin/activate

# Install required packages
pip install pandas numpy scipy pyyaml scikit-learn pyarrow openpyxl
```

### 2. Train the Copula Model

```bash
# This reads trait requirements and fits a Gaussian copula
python scripts/train_copula.py
```

### 3. Run Simulations

#### 🌐 Web Interface (Recommended)
```bash
# Launch interactive dashboard
python launch_dashboard.py

# Or directly with Streamlit
streamlit run app.py
```
**Features:**
- Interactive parameter control (agents, seeds, decisions)
- Real-time visualization of results
- Monte-Carlo studies with convergence plots
- Individual agent inspection
- One-click CSV export
- Academic-ready statistics

#### 📝 Command Line Interface
```bash
# Run all 13 decisions on 1000 agents
python scripts/run_simulation.py --agents 1000 --seed 42

# Run only donation decision on 5000 agents
python scripts/run_simulation.py --agents 5000 --seed 123 --decision donation_default

# Save as CSV instead of Parquet
python scripts/run_simulation.py --agents 100 --format csv
```

#### 🎲 Monte-Carlo Studies
```bash
# Run 500 repetitions with 10,000 agents each (academic standard)
python scripts/run_mc_study.py --agents 10000 --runs 500 --base-seed 1

# Quick test with 50 runs for donation decision only
python scripts/run_mc_study.py --agents 1000 --runs 50 --decision donation_default

# Keep individual run files for detailed analysis
python scripts/run_mc_study.py --agents 5000 --runs 100 --keep-individual
```

## Current Status

### ✅ Fully Implemented
- **TraitEngine**: Generates synthetic agents using fitted Gaussian copula
- **Decision 3 (donation_default)**: Complete with regression model and 6-step anchor formula
- **Orchestrator**: Coordinates trait sampling and decision execution  
- **CLI Interface**: Full command-line control with multiple options
- **Configuration System**: YAML-based parameters for easy modification

### 🚧 Placeholder Implementations
- **Decisions 1-2, 4-13**: Return default values, ready for future model implementation

## Key Features

### Copula-Based Trait Generation
- Preserves correlation structure from original 280 participants
- Generates unlimited novel personality combinations
- Only includes traits actually needed by decision modules

### Modular Decision Architecture
- Each decision is an independent module
- Pure functions: `(agent_state, params, rng) -> dict`
- Easy to add new decisions or modify existing ones

### Reproducible Simulation
- Fixed random seeds ensure identical results
- Configuration files version-controlled separately from code
- Monte-Carlo wrapper ready for uncertainty quantification

## Decision 3: Donation Default

The donation_default module implements your exact methodology:

1. **Predicted Prosocial Score**: Uses regression with coefficients for group, income quintile, study program, and honesty-humility
2. **Anchor Computation**: `0.75 * observed_prosocial + 0.25 * predicted_prosocial`
3. **Stochastic Draw**: Normal distribution centered on anchor
4. **Truncation**: Floor at 0, clip at 99th percentile, rescale to [0,1]

### Current Output Statistics
- Mean donation rate: ~0.28 (28%)
- Range: [0, 1] as expected
- Realistic variation based on personality traits

## Adding New Decisions

To add a new decision model:

1. **Add trait requirements** to `config/trait_requirements.yaml`
2. **Rerun copula training**: `python scripts/train_copula.py`
3. **Create decision module** in `src/decisions/new_decision.py`
4. **Add parameters** to `config/decisions.yaml`
5. **Test**: `python scripts/run_simulation.py --decision new_decision`

## Output Format

Each simulation produces a DataFrame with:
- **Trait columns**: All traits used by any decision
- **Decision columns**: One output per decision module
- **Agent ID**: Implicit via row index

Results are saved as Parquet (default) or CSV files with timestamped filenames for traceability.

## Monte-Carlo Analysis

The framework includes a Monte-Carlo driver for uncertainty quantification:

### Usage
```bash
# Standard academic study: 500 runs × 10,000 agents
python scripts/run_mc_study.py --agents 10000 --runs 500

# Results include:
# - mc_summary.csv: Mean, std, 95% confidence intervals
# - mc_detailed.csv: All individual run statistics  
# - mc_config.json: Complete reproducibility information
```

### Output Format
- **Summary Statistics**: Cross-run means, standard deviations, 95% confidence intervals
- **Detailed Results**: Individual statistics from each Monte-Carlo run
- **Reproducibility**: Exact seeds and configuration for replication

### Academic Reporting
Use the 95% confidence intervals from `mc_summary.csv` in your Results section:
> "Across 500 Monte-Carlo repetitions the average donation rate was 0.284 with a 95% interval of [0.273, 0.295]."

## Next Steps

1. **Implement remaining decisions** as regression models become available
2. **Create web interface** using Streamlit or Dash
3. **Add validation plots** comparing synthetic vs. real distributions
4. **Extend Monte-Carlo** to include parameter uncertainty (Bayesian posterior draws)