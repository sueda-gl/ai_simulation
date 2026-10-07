# src/engine/core.py
"""
The single simulation engine.

One two-pass loop (the body of the former src/orchestrator.py:69-272) serves every
population mode; the per-mode differences that survived the owner rulings live in
:class:`src.engine.profile.ModeProfile`.

RNG scheme (unchanged, parity-critical):
    rng_setup    = default_rng(seed)                       -> vendors (research participants:
                                                              Specification default_rng(seed)
                                                              of their own, Baseline cyclic)
    rng_pass1    = default_rng(seed + 1000000)             -> one integers(1e9) per agent = base seed
    income_rng   = default_rng(base + 999999)              -> drawn in Pass 1 and again in Pass 2
    decision_rng = default_rng(base + decision_index * 1000)
Agents are iterated with ``agents_df.iterrows()`` and ``row.to_dict()``.
"""

import importlib
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from src.engine.log import EngineLog
from src.engine.profile import ModeProfile, PROFILES
from src.engine.sampling import sample_participants_internal
from src.engine.vendors import initialize_vendors
from src.trait_engine import TraitEngine

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "decisions.yaml"
SIMULATION_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "simulation.yaml"

# The ONE decision order for every mode (== app.models.ALL_DECISIONS).
# Position in this list * 1000 is the decision's RNG offset from the agent base seed.
DECISION_ORDER: List[str] = [
    'disclose_income',                # 1
    'disclose_documents',             # 2
    'donation_default',               # 3
    'rejected_transaction_defaults',  # 4
    'vendor_choice_weights',          # 5
    'purchasing_quantity',            # 6
    'purchasing_frequency',           # 7
    'vendor_selection',               # 8
    'purchase_vs_bid',                # 9 (deprecated - kept for backward compatibility)
    'bid_value',                      # 10 (deprecated - kept for backward compatibility)
    'rejected_transaction_option',    # 11
    'rejected_bid_value',             # 12
    'final_donation_rate',            # 13
]

# Decisions whose module lives in src/decisions/<name>_stochastic.py (function
# <name>_stochastic). Every other decision, donation_default included, is the plain
# src/decisions/<name>.py module in every mode.
STOCHASTIC_MODULE_DECISIONS = ('disclose_income', 'disclose_documents')

# The four modelled decisions that receive pop_context=.
POP_CONTEXT_DECISIONS = ('donation_default', 'disclose_income', 'disclose_documents',
                         'rejected_transaction_defaults')


class Engine:
    """
    Coordinates agent sourcing, vendor setup and decision execution for one
    population mode.

    Supports both full-run (all 13 decisions) and single/multi-decision modes.
    Each agent maintains state that accumulates across decisions; the mutated
    agent state is the output row.
    """

    def __init__(self, profile: Union[ModeProfile, str]):
        if isinstance(profile, str):
            profile = PROFILES[profile]
        self.profile: ModeProfile = profile
        self.log = EngineLog(profile.log_prefix)

        # Load decision configuration
        with open(CONFIG_PATH, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load global simulation configuration (Page 1 parameters)
        with open(SIMULATION_CONFIG_PATH, 'r') as f:
            self.simulation_config = yaml.safe_load(f)

        # Population context for decision modules
        self.pop_context = profile.pop_context

        # Decision order (1-13) - identical in every mode
        self.decision_order = list(DECISION_ORDER)

        # Load decision modules
        self.decision_modules = {}
        for decision_name in self.decision_order:
            try:
                if decision_name in STOCHASTIC_MODULE_DECISIONS:
                    module = importlib.import_module(f'src.decisions.{decision_name}_stochastic')
                    self.decision_modules[decision_name] = getattr(module, f'{decision_name}_stochastic')
                else:
                    module = importlib.import_module(f'src.decisions.{decision_name}')
                    self.decision_modules[decision_name] = getattr(module, decision_name)
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load decision module {decision_name}: {e}")

        # Lazily built agent sources
        self._trait_engine: Optional[TraitEngine] = None
        self._original_data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ agents

    @property
    def trait_engine(self) -> TraitEngine:
        """The fitted copula (loaded on first use)."""
        if self._trait_engine is None:
            self._trait_engine = TraitEngine()
        return self._trait_engine

    @property
    def original_data(self) -> pd.DataFrame:
        """The 280 original participants (research profiles only; loaded on first use)."""
        if not self.profile.is_research:
            raise AttributeError(
                f"original_data is only available for research profiles, not {self.profile.name!r}")
        if self._original_data is None:
            from src.data.participants import original_data
            self._original_data = original_data()
            self.log(f"Using {len(self._original_data)} original participants")
        return self._original_data

    def _resolve_agents(self, agents_df: Optional[pd.DataFrame], n_agents: int, seed: int,
                        rng_setup: np.random.Generator) -> pd.DataFrame:
        """Source the agents when none are supplied; validate copula traits otherwise."""
        if self.profile.agent_source == 'copula':
            if agents_df is None:
                # Copula sampling uses its own default_rng(seed), independent of rng_setup
                return self.trait_engine.sample(n_agents, seed)
            # Validate that provided agents have the required traits
            required_traits = set(self.trait_engine.get_available_traits())
            provided_traits = set(agents_df.columns)
            if not required_traits.issubset(provided_traits):
                missing = required_traits - provided_traits
                raise ValueError(f"Provided agents_df missing required traits: {missing}")
            return agents_df

        if agents_df is None:
            # Research modes (ruling R-CYC, owner clarification 2026-10-07): Specification
            # random from default_rng(seed), Baseline file order cycling - the app's rule
            return sample_participants_internal(self.original_data, n_agents, seed,
                                                self.profile.random_sample)
        return agents_df

    # --------------------------------------------------------------- decisions

    def _resolve_decisions(self, single_decision: Optional[Union[str, List[str]]]) -> List[str]:
        """
        str  -> that one decision; list -> those decisions in decision_order order;
        None / empty -> every decision. Unknown names raise ValueError.
        """
        if single_decision:
            if isinstance(single_decision, str):
                if single_decision not in self.decision_order:
                    raise ValueError(f"Unknown decision: {single_decision}")
                return [single_decision]
            if isinstance(single_decision, list):
                for decision in single_decision:
                    if decision not in self.decision_order:
                        raise ValueError(f"Unknown decision: {decision}")
                # Run decisions in the order they appear in decision_order
                return [d for d in self.decision_order if d in single_decision]
            raise ValueError("single_decision must be a string or list of strings")
        return self.decision_order

    def _decision_params(self, decision_name: str) -> dict:
        """
        The live config dict for the decision. Research Baseline never adds noise:
        donation_default gets a shallow copy with stochastic.sigma_value forced to 0.
        """
        params = self.config.get(decision_name, {})
        if decision_name == 'donation_default' and self.profile.force_donation_sigma_zero:
            params = params.copy()
            if 'stochastic' in params:
                params['stochastic'] = params['stochastic'].copy()
                params['stochastic']['sigma_value'] = 0.0  # Force no stochastic component
            else:
                params['stochastic'] = {'sigma_value': 0.0}
        return params

    def _compute_population_stats(self, decisions_to_run: List[str], decision_index: dict,
                                  agents_df: pd.DataFrame, all_incomes: list,
                                  agent_base_seeds: list) -> None:
        """Pass-1 hooks: population statistics the modelled decisions need in Pass 2."""
        log = self.log

        # disclose_income: continuous DE stats
        if 'disclose_income' in decisions_to_run and 'disclose_income' in self.decision_modules:
            disclose_income_params = self.config.get('disclose_income', {})
            di_income_mode = disclose_income_params.get('income_mode', 'categorical')
            log(f"disclose_income income_mode: {di_income_mode}")

            if 'continuous' in str(di_income_mode).lower():
                from src.decisions.disclose_income_stochastic import compute_continuous_de_stats
                di_cont_stats = compute_continuous_de_stats(
                    agents_df, all_incomes, disclose_income_params, self.simulation_config)
                self.simulation_config['di_cont_de_stats'] = di_cont_stats
                log(f"Computed continuous DE stats: mean={di_cont_stats['mean']:.6f}, sd={di_cont_stats['sd']:.6f}")

        # disclose_documents: continuous DD composite stats
        if 'disclose_documents' in decisions_to_run and 'disclose_documents' in self.decision_modules:
            dd_params = self.config.get('disclose_documents', {})
            dd_income_mode = dd_params.get('income_mode', 'categorical')
            if 'continuous' in str(dd_income_mode).lower():
                from src.decisions.disclose_documents_stochastic import compute_continuous_dd_stats
                dd_cont_stats = compute_continuous_dd_stats(
                    agents_df, all_incomes, dd_params, self.simulation_config)
                self.simulation_config['dd_cont_stats'] = dd_cont_stats
                log(f"Computed continuous DD stats: mean={dd_cont_stats['mean']:.6f}, sd={dd_cont_stats['sd']:.6f}")

        # Decision 3 population maximum (ruling R-D3): the methodology doc's section 6
        # step 4 rescales every agent's floored draw by the population maximum,
        # score_k = max(draw_k, 0) / max_j max(draw_j, 0), in every mode. The maximum must
        # be known before Pass 2, because Decisions 6 and 13 read donation_default inside
        # the agent loop, so - like Decision 4 below - every agent's Pass-2 draw is replayed
        # here with its own decision RNG stream (base seed + decision_index * 1000).
        self.simulation_config.pop('donation_population_max', None)
        from src.decisions.donation_default import (compute_donation_population_max,
                                                    configured_default_value)
        if ('donation_default' in decisions_to_run
                and 'donation_default' in self.decision_modules
                and configured_default_value(self.simulation_config) is None):
            donation_max = compute_donation_population_max(
                agents_df, self._decision_params('donation_default'), self.simulation_config,
                pop_context=self.pop_context,
                agent_base_seeds=[int(s) for s in agent_base_seeds],
                decision_offset=decision_index['donation_default'] * 1000)
            self.simulation_config['donation_population_max'] = donation_max
            log(f"Computed Decision 3 population max of floored draws: {donation_max:.6f} (0-100 scale)")

        # Decision 4 population stats (min/max/std of the four mechanism scores;
        # egen min/max/std are population-level operations). The stochastic aggregates
        # replicate each agent's decision RNG stream, so pass the Pass-1 base seeds and
        # this decision's RNG offset (decision_index * 1000, matching Pass 2).
        # The model runs unless Decision 4 is in default_decisions_list.
        if ('rejected_transaction_defaults' in decisions_to_run
                and 'rejected_transaction_defaults' in self.decision_modules
                and 'rejected_transaction_defaults' not in self.simulation_config.get('default_decisions_list', [])):
            from src.decisions.rejected_transaction_defaults import compute_rtd_population_stats
            rtd_params = self.config.get('rejected_transaction_defaults', {})
            rtd_stats = compute_rtd_population_stats(
                agents_df, all_incomes, rtd_params, self.simulation_config,
                pop_context=self.pop_context,
                agent_base_seeds=[int(s) for s in agent_base_seeds],
                decision_offset=decision_index['rejected_transaction_defaults'] * 1000)
            self.simulation_config['rtd_population_stats'] = rtd_stats
            log(f"Computed Decision 4 population stats: "
                f"wtp range [{rtd_stats['wtp']['min']:.6f}, {rtd_stats['wtp']['max']:.6f}]")

    # --------------------------------------------------------------------- run

    def run_simulation(self, n_agents: int, seed: int,
                       single_decision: Optional[Union[str, List[str]]] = None,
                       agents_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run the simulation for n_agents with the given seed.

        If single_decision is provided:
        - a string runs only that decision
        - a list runs those decisions (in decision_order order)
        Otherwise all 13 decisions run in order.

        Args:
            agents_df: Optional pre-sampled agents DataFrame. If provided, agent
                       sourcing is skipped so the same agents can be reused across
                       configurations.
        """
        # Determine which decisions to run (validation only - no RNG involved)
        decisions_to_run = self._resolve_decisions(single_decision)

        # Decision index mapping for deterministic RNG seeding: each decision gets the
        # same RNG regardless of which other decisions run
        decision_index = {name: i for i, name in enumerate(self.decision_order)}

        # Setup RNG (vendors)
        rng_setup = np.random.default_rng(seed)

        # Store seed in simulation_config for access by decision modules
        self.simulation_config['simulation_seed'] = seed

        self._initialize_vendors(rng_setup)

        # Reset vendor capacity tracking for this simulation run
        if 'vendor_remaining_capacity' in self.simulation_config:
            del self.simulation_config['vendor_remaining_capacity']

        # Source the agents (copula draw / research sampling) unless supplied
        agents_df = self._resolve_agents(agents_df, n_agents, seed, rng_setup)

        from src.decisions.income_utils import get_agent_income

        # ====================================================================
        # PASS 1: generate every agent's income and compute population statistics
        # ====================================================================
        rng_pass1 = np.random.default_rng(seed + 1000000)
        all_incomes = []
        agent_base_seeds = []  # reused in Pass 2

        for idx, row in agents_df.iterrows():
            # The same base seed is used again in Pass 2
            agent_base_seed = rng_pass1.integers(1e9)
            agent_base_seeds.append(agent_base_seed)

            temp_state = row.to_dict()
            income_rng = np.random.default_rng(agent_base_seed + 999999)
            income = get_agent_income(temp_state, self.simulation_config, income_rng)
            all_incomes.append(income)

        self.simulation_config['income_median'] = float(np.median(all_incomes))
        # Income SD is the SAMPLE SD (ddof=1), matching Stata `egen z_net_income = std(income)`
        # used for the continuous-income z-scores of Decisions 1 and 2 (owner ruling R-SD,
        # 2026-09-04). Guarded like compute_rtd_population_stats: with fewer than two
        # incomes fall back to the population formula (0.0 for a single agent).
        self.simulation_config['income_stats'] = {
            'mean': float(np.mean(all_incomes)),
            'sd': (float(np.std(all_incomes, ddof=1)) if len(all_incomes) > 1
                   else float(np.std(all_incomes)))
        }
        self.log(f"Computed income median: ${self.simulation_config['income_median']:,.2f}")
        self.log(f"Computed income stats: mean=${self.simulation_config['income_stats']['mean']:,.2f}, "
                 f"sd=${self.simulation_config['income_stats']['sd']:,.2f}")

        self._compute_population_stats(decisions_to_run, decision_index, agents_df,
                                       all_incomes, agent_base_seeds)

        # ====================================================================
        # PASS 2: run the decisions per agent with the Pass-1 base seeds
        # ====================================================================
        results = []

        for list_idx, (idx, row) in enumerate(agents_df.iterrows()):
            agent_state = row.to_dict()

            # Agent ID and index (customer_id in purchase_requests depends on these)
            agent_state['index'] = idx
            agent_state['agent_id'] = idx + 1  # Agent IDs start at 1

            agent_base_seed = agent_base_seeds[list_idx]

            # Same income RNG as Pass 1 -> identical income regardless of which decisions run
            income_rng = np.random.default_rng(agent_base_seed + 999999)
            get_agent_income(agent_state, self.simulation_config, income_rng)

            for decision_name in decisions_to_run:
                if decision_name not in self.decision_modules:
                    print(f"Warning: No module found for decision {decision_name}")
                    continue

                params = self._decision_params(decision_name)

                # Decision-specific RNG: independent of which other decisions run
                decision_seed = agent_base_seed + decision_index[decision_name] * 1000
                decision_rng = np.random.default_rng(decision_seed)

                if decision_name in POP_CONTEXT_DECISIONS:
                    decision_output = self.decision_modules[decision_name](
                        agent_state, params, decision_rng,
                        pop_context=self.pop_context, simulation_config=self.simulation_config
                    )
                else:
                    decision_output = self.decision_modules[decision_name](
                        agent_state, params, decision_rng, simulation_config=self.simulation_config
                    )

                agent_state.update(decision_output)

            # The mutated agent state IS the output row
            results.append(agent_state)

        results_df = pd.DataFrame(results)

        # Vendor data for the results visualisations
        if 'vendors' in self.simulation_config:
            results_df.attrs['vendors'] = self.simulation_config['vendors']

        # The resolved sim config (incl. Page-1 discount_income_threshold) so the Excel
        # export's eligibility re-check uses the same threshold the model gated on.
        results_df.attrs['simulation_config'] = self.simulation_config

        return results_df

    def _initialize_vendors(self, rng: np.random.Generator) -> None:
        """Generate vendor attributes once per simulation (see src.engine.vendors)."""
        initialize_vendors(self.simulation_config, rng, self.log)

    def get_available_decisions(self) -> List[str]:
        """Return list of available decision modules."""
        return list(self.decision_modules.keys())


__all__ = ["Engine", "DECISION_ORDER", "STOCHASTIC_MODULE_DECISIONS", "POP_CONTEXT_DECISIONS",
           "CONFIG_PATH", "SIMULATION_CONFIG_PATH"]
