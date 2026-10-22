# app/state/saved_configs.py
"""Saved decision-configuration store.

A saved configuration is created when the user clicks "Use This Config" on a
decision result.  It pins the seed, the agent count, the population mode and
that decision's parameters so a later complete simulation reproduces the run
the user looked at.

``app/pages/decision_execution.py`` re-exports the functions here that a page
or a test still imports from it, so those callers keep working.
"""

import streamlit as st
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.contract.plan import hash_result_columns  # noqa: F401 - the writer and the checker share it
from app.seam.build_plan import get_population_mode_from_result_key
from app.seam.config_repo import get_config_repo
from app.models import DONATION_SIGMA_OVERALL, load_donation_coefficients_from_yaml


@dataclass(frozen=True)
class RunOnScreen:
    """The run whose result "Use This Config" pins: the seed, agent count,
    population and income mode it ran with.  The results page hands it in (from
    its ``RunContext`` over ``_run_metadata``, ``RunContext.run_on_screen``), so
    a pin never takes these from the live Page 1 widgets, which may have moved
    on since the run - and the store never imports the page (stores sit below
    the screens, docs/README.md section 1)."""
    seed: Optional[int]
    n_agents: Optional[int]
    population_mode: str
    income_mode: str


def _run_seed_and_agents(run):
    """(seed, n_agents) of the run on screen.  A session without run metadata (a
    foreign session) falls back to the live Page 1 values, as the store always
    did for it."""
    if run.seed is not None and run.n_agents is not None:
        return run.seed, run.n_agents
    if hasattr(st.session_state, 'sim_params') and st.session_state.sim_params.simulation_mode == "Single Run":
        seed = st.session_state.get('seed_input', st.session_state.get('seed', 42))
    else:
        seed = st.session_state.get('base_seed_input', st.session_state.get('base_seed', 42))
    return seed, st.session_state.get('n_agents', 1000)


def _run_population_mode(run):
    """The population mode of the run on screen; the live Page 1 value only for a
    session whose run metadata does not record it."""
    return run.population_mode or st.session_state.get('population_mode', 'Copula (synthetic)')


def _file_value(decision, *path):
    """The value at ``path`` in ``decision``'s block of config/decisions.yaml: what
    a run uses for a tab value the session does not hold (the seam then leaves the
    file's value in place, app/seam/build_plan._apply_scalar_fields), so a pin
    records the value the run used. A key the file lacks raises."""
    node = get_config_repo().decision(decision)
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"{decision}.{'.'.join(path)} missing from config/decisions.yaml")
        node = node[key]
    return node


# ==================== DONATION DEFAULT (Decision 3) CONFIG SELECTION ====================
# The donation triplet (details / params / metrics) plus the donation-only
# wrappers over the unified store below.

def save_selected_configuration(result_key, result_df, run):
    """
    Save the selected donation configuration for later use in combined simulations.
    ``run`` is the run on screen (``RunOnScreen``, from the results page).

    Returns ``save_decision_config``'s ``(success, config, error_info)`` triple:
    a seed / agent-count mismatch with an already pinned decision is NOT saved,
    and ``error_info`` says which decision conflicts and with what values, so the
    screen can say so instead of reporting a save that never happened.

    DEPRECATED: This is a wrapper for backwards compatibility.
    New code should use: save_decision_config('donation_default', result_key, result_df, params, metrics, extra_data, run=run)
    """
    # Extract configuration details from the result key
    config_details = extract_configuration_details(result_key, run)

    # Get current coefficient values from session state
    coefficients = get_current_coefficients()

    # Get current stochastic parameters
    stochastic_params = get_current_stochastic_params()

    # Calculate key metrics from the result
    metrics = calculate_result_metrics(result_df)

    # Build params dict for unified system
    params = {
        'coefficients': coefficients,
        'stochastic_params': stochastic_params,
        'income_mode': config_details['income_spec_mode']
    }

    extra_data = {
        'population_mode': config_details['population_mode'],
        'income_spec_mode': config_details['income_spec_mode']
    }

    # Use unified save function; the failure travels back to the screen
    return save_decision_config(
        'donation_default', result_key, result_df, params, metrics, extra_data, run=run
    )


def extract_configuration_details(result_key, run):
    """Extract population and income mode from result key (``run``: the run on
    screen, for a single-mode key)"""

    # Population mode detection (the seam's parser of a Compare-all result key)
    population_mode = get_population_mode_from_result_key(result_key)
    if population_mode is None:
        # For single-mode results, the mode the RUN used (not the live widget)
        population_mode = _run_population_mode(run)

    # Income mode detection
    if 'categorical' in result_key:
        income_spec_mode = 'categorical only'
    elif 'continuous' in result_key:
        income_spec_mode = 'continuous only'
    else:
        # For single-mode results, the mode the RUN used (not the live widget)
        income_spec_mode = run.income_mode

    return {
        'population_mode': population_mode,
        'income_spec_mode': income_spec_mode
    }


def get_current_coefficients():
    """Collect all current coefficient values from session state

    IMPORTANT: Ensures coefficients are loaded from YAML first.
    YAML is the SINGLE source of truth - no fallback values.
    """
    # Ensure coefficients are loaded from YAML
    if 'donation_coeff_intercept' not in st.session_state:
        load_donation_coefficients_from_yaml()

    # Return coefficients from session state - NO FALLBACK VALUES
    return {
        'intercept': st.session_state.donation_coeff_intercept,
        'beta_group': {
            'MidSub': st.session_state.donation_coeff_midsub,
            'NoSub': st.session_state.donation_coeff_nosub,
            'FullSub': st.session_state.donation_coeff_fullsub
        },
        'beta_income_q': {
            'Q1': st.session_state.donation_coeff_q1,
            'Q2': st.session_state.donation_coeff_q2,
            'Q3': st.session_state.donation_coeff_q3,
            'Q4': st.session_state.get('donation_coeff_q4', 0.0),
            'Q5': st.session_state.get('donation_coeff_q5', st.session_state.get('donation_coeff_q45', 0.0))  # Support both Q5 and the older Q4_Q5 name
        },
        'beta_income_linear': st.session_state.donation_coeff_linear,
        'beta_study': {
            'Incoming': st.session_state.donation_coeff_incoming,
            'Law5yr': st.session_state.donation_coeff_law,
            'UG3yr': st.session_state.donation_coeff_ug,
            'Grad2yr': st.session_state.donation_coeff_grad
        },
        'beta_hh': st.session_state.donation_coeff_hh
    }


def get_current_stochastic_params():
    """Collect current stochastic parameters from session state"""
    observed_weight = st.session_state.get(
        'anchor_observed_weight', _file_value('donation_default', 'anchor_weights', 'observed'))
    return {
        'stochastic': {
            'sigma_value': st.session_state.get('sigma_value_ui', DONATION_SIGMA_OVERALL),
            'sigma_coefficient': st.session_state.get('sigma_coefficient', 1.0),
            'sigma_in_copula': st.session_state.get('sigma_in_copula', False),
            'sigma_in_research': st.session_state.get('sigma_in_research', True)
        },
        'anchor_weights': {
            'observed': observed_weight,
            'predicted': 1 - observed_weight
        }
    }


def calculate_result_metrics(result_df):
    """Calculate key metrics from result DataFrame - always uses truncated donation_default"""

    # Always use truncated donation_default for consistency
    donation_col = 'donation_default'

    metrics = {
        'mean_donation': result_df[donation_col].mean(),
        'std_donation': result_df[donation_col].std(),
        'median_donation': result_df[donation_col].median(),
        'min_donation': result_df[donation_col].min(),
        'max_donation': result_df[donation_col].max(),
        'q25_donation': result_df[donation_col].quantile(0.25),
        'q75_donation': result_df[donation_col].quantile(0.75),
        'donation_column_used': donation_col
    }

    return metrics


def format_result_name(result_key):
    """Format result key into human-readable name"""

    name_mapping = {
        'copula_categorical': '🔬 Copula + Categorical Income',
        'copula_continuous': '🔬 Copula + Continuous Income',
        'research_spec_categorical': '📊 Research Spec + Categorical Income',
        'research_spec_continuous': '📊 Research Spec + Continuous Income',
        'research_baseline_categorical': '📈 Research Baseline + Categorical Income',
        'research_baseline_continuous': '📈 Research Baseline + Continuous Income',
        'categorical': '💰 Categorical Income Mode',
        'continuous': '📈 Continuous Income Mode',
        'copula': '🔬 Copula Population',
        'documentation': '📊 Research Specification',
        'baseline': '📈 Research Baseline'
    }

    return name_mapping.get(result_key, f"📊 {result_key.replace('_', ' ').title()}")


def clear_selected_configuration():
    """Clear the currently selected donation_default configuration."""
    clear_decision_config('donation_default')


# ==================== DISCLOSE INCOME (Decision 1) CONFIG SELECTION ====================

def extract_disclose_income_configuration_details(result_key, run):
    """Extract income mode from result key for disclose income (``run``: the run
    on screen, for a single-mode key)"""
    
    # Income mode detection
    if 'categorical' in result_key.lower():
        income_mode = 'Categorical only'
    elif 'continuous' in result_key.lower():
        income_mode = 'Continuous only'
    else:
        # the mode the RUN used (not the live tab value)
        income_mode = run.income_mode
    
    return {
        'income_mode': income_mode
    }


def get_current_disclose_income_params():
    """Collect current disclose income parameters from session state"""
    return {
        'intercept': st.session_state.get('di_intercept', _file_value('disclose_income', 'intercept')),
        'income_mode': st.session_state.get('di_income_mode', 'Categorical only'),
        'anchor_weights': {
            'observed_prosocial': st.session_state.get(
                'di_wopb', _file_value('disclose_income', 'anchor_weights', 'observed_prosocial')),
            'prosocial_weight': st.session_state.get(
                'di_wpb', _file_value('disclose_income', 'anchor_weights', 'prosocial_weight'))
        },
        'stochastic': {
            # The Disclose Income tab's own default is ON, so an unopened tab
            # must report the same sigma the run would use.
            'sigma_enabled': st.session_state.get('di_sigma_enabled', True),
            'sigma_in_copula': st.session_state.get('di_sigma_in_copula', False),
            'scale_factor': st.session_state.get('di_scale_factor', 1.0),
            'sigma_strategy': st.session_state.get('di_sigma_strategy', 'overall'),
            'quintile_scale_factors': st.session_state.get('di_quintile_scale_factors', {})
        }
    }


def calculate_disclose_income_metrics(result_df):
    """Calculate key metrics from disclose income result DataFrame"""
    
    metrics = {}
    
    # Calculate Y/N rates
    if 'disclose_income' in result_df.columns:
        total = len(result_df)
        y_count = (result_df['disclose_income'] == 'Y').sum()
        n_count = (result_df['disclose_income'] == 'N').sum()
        metrics['y_rate'] = y_count / total if total > 0 else 0
        metrics['n_rate'] = n_count / total if total > 0 else 0
        metrics['y_count'] = int(y_count)
        metrics['n_count'] = int(n_count)
    
    # Calculate raw value statistics if available
    if 'disclose_income_raw' in result_df.columns:
        raw_values = result_df['disclose_income_raw'].dropna()
        metrics['raw_mean'] = float(raw_values.mean())
        metrics['raw_std'] = float(raw_values.std())
        metrics['raw_median'] = float(raw_values.median())
        metrics['raw_min'] = float(raw_values.min())
        metrics['raw_max'] = float(raw_values.max())
        metrics['raw_q25'] = float(raw_values.quantile(0.25))
        metrics['raw_q75'] = float(raw_values.quantile(0.75))
    
    # Calculate DI_i statistics if available (pre-stochastic value)
    if 'disclose_income_di' in result_df.columns:
        di_values = result_df['disclose_income_di'].dropna()
        metrics['di_mean'] = float(di_values.mean())
        metrics['di_std'] = float(di_values.std())
    
    return metrics


def extract_disclose_documents_configuration_details(result_key, run):
    """Extract income mode from result key for disclose documents (``run``: the
    run on screen, for a single-mode key)"""
    if 'categorical' in result_key.lower():
        income_mode = 'Categorical only'
    elif 'continuous' in result_key.lower():
        income_mode = 'Continuous only'
    else:
        income_mode = run.income_mode   # the run's, not the live tab's
    return {'income_mode': income_mode}


def get_current_disclose_documents_params():
    """Collect current disclose documents parameters from session state.

    NOTE: disclose_documents has NO prosocial anchoring, so (unlike disclose_income)
    there is no anchor_weights block.
    """
    return {
        'intercept': st.session_state.get('dd_intercept', _file_value('disclose_documents', 'intercept')),
        'income_mode': st.session_state.get('dd_income_mode', 'Categorical only'),
        'stochastic': {
            # The Disclose Documents tab's own default is ON, so an unopened tab
            # must report the same sigma the run would use.
            'sigma_enabled': st.session_state.get('dd_sigma_enabled', True),
            'sigma_in_copula': st.session_state.get('dd_sigma_in_copula', False),
            'scale_factor': st.session_state.get('dd_scale_factor', 1.0),
            'sigma_strategy': st.session_state.get('dd_sigma_strategy', 'overall'),
            'quintile_scale_factors': st.session_state.get('dd_quintile_scale_factors', {})
        }
    }


def calculate_disclose_documents_metrics(result_df):
    """Calculate key metrics from a disclose documents result DataFrame.

    Y/N rates are computed over QUALIFIED (non-NA) agents only; NA agents are tracked
    separately so they do not dilute the disclosure rate.
    """
    metrics = {}
    if 'disclose_documents' in result_df.columns:
        qualified = result_df.loc[result_df['disclose_documents'] != 'NA', 'disclose_documents']
        total_q = len(qualified)
        y_count = int((qualified == 'Y').sum())
        n_count = int((qualified == 'N').sum())
        metrics['y_rate'] = y_count / total_q if total_q > 0 else 0
        metrics['n_rate'] = n_count / total_q if total_q > 0 else 0
        metrics['y_count'] = y_count
        metrics['n_count'] = n_count
        metrics['na_count'] = int((result_df['disclose_documents'] == 'NA').sum())
        metrics['qualified_count'] = total_q

    if 'disclose_documents_raw' in result_df.columns:
        # Qualified subgroup only (decision != NA). The model now emits a raw score for every
        # agent (incl. gated NAs) for export, so a bare .dropna() would include ineligibles.
        if 'disclose_documents' in result_df.columns:
            raw_values = result_df.loc[result_df['disclose_documents'] != 'NA', 'disclose_documents_raw'].dropna()
        else:
            raw_values = result_df['disclose_documents_raw'].dropna()
        if len(raw_values) > 0:
            metrics['raw_mean'] = float(raw_values.mean())
            metrics['raw_std'] = float(raw_values.std())
            metrics['raw_median'] = float(raw_values.median())
            metrics['raw_min'] = float(raw_values.min())
            metrics['raw_max'] = float(raw_values.max())
    return metrics


# ==================== REJECTED TRANSACTION DEFAULTS (Decision 4) CONFIG SELECTION ====================
# Mirrors the disclose_income pattern: a Decision 4 result cell can be selected with
# "Use This Config"; the tab settings at save time (income mode, per-element intercepts,
# Flexibility anchor mix, rank-aggregation settings, stochastic UI) are stored in the unified
# selected_decision_configs store and applied by the seam
# (app.seam.build_plan.build_rejected_transaction_patch) in combined/complete simulations
# (individual Decision 4 runs keep reflecting the tab).

RTD_CONFIG_MECHANISMS = ('ttp', 'loyalty', 'wtp', 'risk_taking', 'flexibility')


def extract_rejected_transaction_configuration_details(result_key, run):
    """Income mode and population mode of a Decision 4 result cell from its result key
    ('categorical'/'continuous' single-mode keys, 'copula_continuous' etc. Compare-all keys;
    ``run``: the run on screen, for a single-mode key)."""
    key = str(result_key or '').lower()
    if 'categorical' in key:
        income_mode = 'Categorical only'
    elif 'continuous' in key:
        income_mode = 'Continuous only'
    else:
        income_mode = str(run.income_mode)   # the run's, not the live tab's
        if 'compare' in income_mode.lower() or 'both' in income_mode.lower():
            income_mode = 'Continuous only'   # a single cell always has one mode
    population_mode = extract_configuration_details(result_key, run)['population_mode']
    return {'income_mode': income_mode, 'population_mode': population_mode}


def get_current_rejected_transaction_params():
    """Collect the current Decision 4 tab settings from session state (the model
    coefficients and sigma constants are fixed in config/decisions.yaml)."""
    flex_observed = float(st.session_state.get(
        'rtd_flex_observed_weight',
        _file_value('rejected_transaction_defaults', 'flexibility_anchor', 'observed_weight')))
    return {
        'income_mode': st.session_state.get('rtd_income_mode', 'Continuous only'),
        'intercepts': {m: float(st.session_state.get(f'rtd_intercept_{m}', 0.0) or 0.0)
                       for m in RTD_CONFIG_MECHANISMS},
        'flexibility_anchor': {
            'observed_weight': flex_observed,
            'calculated_weight': 1.0 - flex_observed,
        },
        'aggregation': {
            'enabled': bool(st.session_state.get('rtd_aggregation_enabled', True)),
        },
        'stochastic': {
            # The Decision 4 tab's own default is ON, so an unopened tab must
            # report the same sigma the run would use.
            'sigma_enabled': st.session_state.get('rtd_sigma_enabled', True),
            'sigma_in_copula': st.session_state.get('rtd_sigma_in_copula', False),
            # σ coefficient defaults from config/decisions.yaml (0.5, per the platform documentation)
            'scale_factor': st.session_state.get(
                'rtd_scale_factor', get_config_repo().rtd_sigma_coefficient_defaults()[0]),
            'sigma_strategy': st.session_state.get('rtd_sigma_strategy', 'overall'),
            'quintile_scale_factors': st.session_state.get('rtd_quintile_scale_factors', {}),
        },
    }


def calculate_rejected_transaction_metrics(result_df):
    """Key metrics of a Decision 4 result frame: options list length, integrated default
    list, first integrated option shares, per-element mean segments."""
    metrics = {'total_agents': int(len(result_df))}
    n = len(result_df)
    if 'rtd_choice_length' in result_df.columns:
        lengths = result_df['rtd_choice_length'].astype(int)
        metrics['mean_choice_length'] = float(lengths.mean()) if n else 0.0
        metrics['choice_length_distribution'] = {int(k): int(v) for k, v in lengths.value_counts().sort_index().items()}
    if 'rtd_default_list_length' in result_df.columns and n:
        metrics['mean_default_list_length'] = float(result_df['rtd_default_list_length'].mean())
    if 'rtd_default_list' in result_df.columns and n:
        firsts = result_df['rtd_default_list'].apply(lambda l: l[0] if isinstance(l, list) and len(l) else 0)
        metrics['first_option_shares'] = {int(o): float((firsts == o).sum() / n) for o in range(1, 6)}
        metrics['empty_list_rate'] = float((firsts == 0).mean())
    for mech, col in (('loyalty', 'rtd_loyalty_segment'), ('wtp', 'rtd_wtp_segment'),
                      ('risk_taking', 'rtd_rt_segment'), ('flexibility', 'rtd_flex_segment')):
        if col in result_df.columns and n:
            metrics[f'mean_{mech}_segment'] = float(result_df[col].astype(float).mean())
    if 'rtd_consensus_is_kemeny_optimal' in result_df.columns and n:
        metrics['kemeny_optimal_rate'] = float(result_df['rtd_consensus_is_kemeny_optimal'].astype(bool).mean())
    return metrics


# ==================== UNIFIED DECISION CONFIGURATION SYSTEM ====================
# This unified system replaces the fragmented per-decision config storage.
# All decision configs are now stored in a single dict: selected_decision_configs
# This enables seed matching validation and easy extensibility for new decisions.

def get_selected_decision_configs():
    """Get the unified decision configs dictionary, initializing if needed."""
    if 'selected_decision_configs' not in st.session_state:
        st.session_state.selected_decision_configs = {}
    return st.session_state.selected_decision_configs


def validate_seed_consistency(new_decision_name, new_seed, new_n_agents):
    """
    Check if new config's seed/n_agents matches existing configs.

    Every stored config is an explicit "Use This Config" selection (auto-implied
    ones do not exist), so all of them take part in the check.

    Args:
        new_decision_name: Name of the decision being saved
        new_seed: Seed used for the new config
        new_n_agents: Number of agents used for the new config
    
    Returns:
        tuple: (is_valid, existing_seed, existing_n_agents, conflicting_decision)
            - is_valid: True if seed matches or no existing configs
            - existing_seed: The seed from existing configs (if any)
            - existing_n_agents: The n_agents from existing configs (if any)
            - conflicting_decision: Name of the decision with mismatched seed (if any)
    """
    configs = get_selected_decision_configs()
    
    for decision_name, config in configs.items():
        # Skip if checking against itself (re-saving same decision)
        if decision_name == new_decision_name:
            continue

        existing_seed = config.get('original_seed')
        existing_n_agents = config.get('original_n_agents')
        
        if existing_seed != new_seed or existing_n_agents != new_n_agents:
            return (False, existing_seed, existing_n_agents, decision_name)
    
    return (True, new_seed, new_n_agents, None)


def get_decision_result_columns(decision_name, result_df):
    """The output columns a decision owns in a result DataFrame.

    The decision's own column plus its model columns, exactly as the
    visualisations define them.  Column order follows the DataFrame so the hash
    below is reproducible.

    Args:
        decision_name: Name of the decision
        result_df: DataFrame the columns are taken from (may be None)

    Returns:
        list of str: the column names present in result_df, in DataFrame order
    """
    if result_df is None:
        return []

    columns = list(result_df.columns)

    if decision_name == 'donation_default':
        return [c for c in columns if c == 'donation_default']
    if decision_name == 'disclose_income':
        return [c for c in columns if c.startswith('disclose_income')]
    if decision_name == 'disclose_documents':
        return [c for c in columns if c.startswith('disclose_documents')]
    if decision_name == 'rejected_transaction_defaults':
        return [c for c in columns
                if c.startswith('rtd_') or c == 'rejected_transaction_defaults']

    # Any other decision: just its own column, when the result carries one.
    return [c for c in columns if c == decision_name]


def save_decision_config(decision_name, result_key, result_df, params, metrics=None, extra_data=None, *, run):
    """
    Unified function to save a decision configuration.
    
    This is the standard way to save any decision config for use in combined simulations.
    All configs are validated for seed consistency before saving.
    
    Args:
        decision_name: Name of the decision (e.g., 'donation_default', 'disclose_income')
        result_key: Key identifying the result configuration
        result_df: DataFrame containing the simulation results
        params: Dict of decision-specific parameters (coefficients, weights, etc.)
        metrics: Optional dict of calculated metrics. If None, will use basic metrics.
        extra_data: Optional dict of additional data to store (e.g., for backwards compat)
        run: The run on screen (``RunOnScreen``), handed in by the results page:
            the seed, agent count and population mode the pin records
    
    Returns:
        tuple: (success: bool, config: dict or None, error_info: dict or None)
            - success: True if config was saved successfully
            - config: The saved config dict (if successful)
            - error_info: Dict with seed mismatch details (if failed)
    """
    # The seed and agent count the run on screen used (its metadata), so the pin
    # reproduces that run even after Page 1 has been edited
    original_seed, original_n_agents = _run_seed_and_agents(run)

    # Record what this decision produced, so a later complete simulation can
    # prove it reproduced the run the user pinned.
    result_columns = get_decision_result_columns(decision_name, result_df)
    result_sha256 = hash_result_columns(result_df, result_columns)

    # Validate seed consistency with existing configs
    is_valid, existing_seed, existing_n_agents, conflicting_decision = validate_seed_consistency(
        decision_name, original_seed, original_n_agents
    )
    
    if not is_valid:
        return (False, None, {
            'new_seed': original_seed,
            'new_n_agents': original_n_agents,
            'existing_seed': existing_seed,
            'existing_n_agents': existing_n_agents,
            'conflicting_decision': conflicting_decision
        })
    
    # Build the config object
    config = {
        'result_key': result_key,
        'params': params,
        'metrics': metrics or {},
        'selected_timestamp': datetime.now(),
        'total_agents': len(result_df) if result_df is not None else 0,
        'source': f'individual_{decision_name}_run',
        'original_seed': original_seed,
        'original_n_agents': original_n_agents,
        'result_columns': result_columns,
        'result_sha256': result_sha256
    }

    # For donation_default, flatten params to top level for consumer compatibility.
    # Consumers read config['coefficients'], config['stochastic_params'], etc.
    if decision_name == 'donation_default':
        if 'coefficients' in params:
            config['coefficients'] = params['coefficients']
        if 'stochastic_params' in params:
            config['stochastic_params'] = params['stochastic_params']
        if 'income_mode' in params:
            config['donation_income_mode'] = params['income_mode']
            if not extra_data or 'income_spec_mode' not in extra_data:
                config['income_spec_mode'] = params['income_mode']
    
    # Merge in any extra data
    if extra_data:
        config.update(extra_data)
    
    # For disclose_income / disclose_documents / rejected_transaction_defaults, ensure
    # population_mode is stored
    if decision_name in ('disclose_income', 'disclose_documents', 'rejected_transaction_defaults') \
            and 'population_mode' not in config:
        config_details = extract_configuration_details(result_key, run)
        config['population_mode'] = config_details['population_mode']

    # Store in unified configs dict (SINGLE source of truth)
    configs = get_selected_decision_configs()
    configs[decision_name] = config
    
    # Sync final_donation_rate_default_value when saving donation_default
    if decision_name == 'donation_default' and 'mean_donation' in (metrics or {}):
        mean_donation = metrics['mean_donation']
        st.session_state.final_donation_rate_default_value = mean_donation
        if '_persistent_defaults' not in st.session_state:
            st.session_state._persistent_defaults = {}
        st.session_state._persistent_defaults['final_donation_rate_default_value'] = mean_donation
    
    return (True, config, None)


def get_decision_config(decision_name):
    """
    Get the saved configuration for a specific decision.
    
    Args:
        decision_name: Name of the decision
    
    Returns:
        dict or None: The config dict if exists, None otherwise
    """
    configs = get_selected_decision_configs()
    return configs.get(decision_name)


def has_decision_config(decision_name):
    """
    Check if a decision has a saved configuration.
    
    Args:
        decision_name: Name of the decision
    
    Returns:
        bool: True if config exists
    """
    return get_decision_config(decision_name) is not None


def is_decision_config_selected(decision_name, result_key):
    """
    Check if a specific result_key is the currently selected config for a decision.
    
    Args:
        decision_name: Name of the decision
        result_key: The result key to check
    
    Returns:
        bool: True if this result_key is currently selected
    """
    config = get_decision_config(decision_name)
    if config is None:
        return False
    return config.get('result_key') == result_key


def clear_decision_config(decision_name):
    """
    Clear the saved configuration for a specific decision.
    
    This is the SINGLE function for clearing any decision config.
    
    Args:
        decision_name: Name of the decision to clear
    """
    configs = get_selected_decision_configs()
    if decision_name in configs:
        del configs[decision_name]
    
    if decision_name == 'donation_default':
        # Reset final_donation_rate_default_value (wrapped in try/except because
        # Streamlit prevents modification after the bound widget is instantiated)
        try:
            st.session_state.final_donation_rate_default_value = 0.10
        except Exception:
            pass
        if '_persistent_defaults' in st.session_state:
            st.session_state._persistent_defaults['final_donation_rate_default_value'] = 0.10


def get_simulation_seed_from_configs():
    """
    Get the seed to use for simulation from saved configs.

    Only explicit "Use This Config" selections are stored (auto-implied ones do
    not exist), so the pinned seed always comes from a run the user
    actually looked at.  If configs exist, returns their seed (all configs have
    matching seed).  Otherwise returns the current session state seed.

    Returns:
        tuple: (seed, n_agents, source) where source is 'configs' or 'session_state'
    """
    configs = get_selected_decision_configs()
    
    if configs:
        # All configs have matching seed, so just get from first one
        first_config = next(iter(configs.values()))
        return (
            first_config['original_seed'],
            first_config['original_n_agents'],
            'configs'
        )
    
    # No saved configs, use session state
    if hasattr(st.session_state, 'sim_params') and st.session_state.sim_params.simulation_mode == "Single Run":
        seed = st.session_state.get('seed_input', st.session_state.get('seed', 42))
    else:
        seed = st.session_state.get('base_seed_input', st.session_state.get('base_seed', 42))
    
    n_agents = st.session_state.get('n_agents', 1000)

    return (seed, n_agents, 'session_state')
