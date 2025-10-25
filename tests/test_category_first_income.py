"""
Unit tests for Category-First income generation architecture.

Tests verify:
1. Correct percentile boundaries for each distribution type
2. One-time income generation per agent (caching)
3. Logical consistency between Assigned Allowance Level and generated income
4. Donation regression stability on 12-200 scale
5. Dual income variable generation (income + actual_allowance)
"""

import pytest
import numpy as np
from src.decisions.income_utils import (
    get_agent_income,
    get_actual_allowance,
    get_percentile_boundaries,
    ALLOWANCE_CREDIT_MAPPING,
    _get_percentile_range_for_level
)
from src.decisions.donation_default import donation_default


class TestPercentileBoundaries:
    """Test that percentile boundaries are calculated correctly for each distribution."""
    
    def test_lognormal_boundaries(self):
        """Test percentile boundaries for lognormal distribution."""
        sim_params = {
            'income_distribution': 'lognormal',
            'lognormal_mu': 10.0,
            'lognormal_sigma': 0.5,
            'lognormal_min': 0.0,
            'lognormal_max': None
        }
        
        boundaries = get_percentile_boundaries(sim_params)
        
        # Should return 4 boundaries
        assert len(boundaries) == 4
        
        # Boundaries should be monotonically increasing
        assert boundaries[0] < boundaries[1] < boundaries[2] < boundaries[3]
        
        # All boundaries should be non-negative (given min=0)
        assert all(b >= 0 for b in boundaries)
    
    def test_generalised_gamma_boundaries(self):
        """Test percentile boundaries for generalised gamma distribution."""
        sim_params = {
            'income_distribution': 'generalised_gamma',
            'gg_k': 1.5,
            'gg_c': 2.0,
            'gg_lambda': 20000.0,
            'gg_min': 0.0,
            'gg_max': None
        }
        
        boundaries = get_percentile_boundaries(sim_params)
        
        assert len(boundaries) == 4
        assert boundaries[0] < boundaries[1] < boundaries[2] < boundaries[3]
        assert all(b >= 0 for b in boundaries)
    
    def test_dagum_boundaries(self):
        """Test percentile boundaries for Dagum distribution."""
        sim_params = {
            'income_distribution': 'dagum',
            'dagum_a': 2.0,
            'dagum_p': 1.5,
            'dagum_b': 25000.0,
            'dagum_min': 0.0,
            'dagum_max': None
        }
        
        boundaries = get_percentile_boundaries(sim_params)
        
        assert len(boundaries) == 4
        assert boundaries[0] < boundaries[1] < boundaries[2] < boundaries[3]
        assert all(b >= 0 for b in boundaries)
    
    def test_boundaries_respect_max(self):
        """Test that boundaries respect maximum value when specified."""
        sim_params = {
            'income_distribution': 'lognormal',
            'lognormal_mu': 10.0,
            'lognormal_sigma': 0.5,
            'lognormal_min': 0.0,
            'lognormal_max': 50000.0
        }
        
        boundaries = get_percentile_boundaries(sim_params)
        
        # All boundaries should be <= max
        assert all(b <= 50000.0 for b in boundaries)


class TestOneTimeGeneration:
    """Test that income is generated only once per agent and cached."""
    
    def test_income_cached_on_first_call(self):
        """Test that first call generates income and caches it."""
        agent_state = {'Assigned Allowance Level': 3}
        simulation_config = {
            'simulation': {
                'income_distribution': 'lognormal',
                'lognormal_mu': 10.0,
                'lognormal_sigma': 0.5,
                'lognormal_min': 0.0
            }
        }
        rng = np.random.default_rng(42)
        
        # First call should generate and cache
        income1 = get_agent_income(agent_state, simulation_config, rng)
        
        # Verify income was stored in agent_state
        assert 'income' in agent_state
        assert agent_state['income'] == income1
        
        # Verify actual_allowance was also stored
        assert 'actual_allowance' in agent_state
        assert agent_state['actual_allowance'] == 72.0  # Level 3 -> 72
    
    def test_subsequent_calls_return_cached_value(self):
        """Test that subsequent calls return the same cached value."""
        agent_state = {'Assigned Allowance Level': 4}
        simulation_config = {
            'simulation': {
                'income_distribution': 'lognormal',
                'lognormal_mu': 10.0,
                'lognormal_sigma': 0.5,
                'lognormal_min': 0.0
            }
        }
        rng = np.random.default_rng(42)
        
        # First call
        income1 = get_agent_income(agent_state, simulation_config, rng)
        
        # Second call (should return cached value, not generate new)
        income2 = get_agent_income(agent_state, simulation_config, rng)
        
        # Should be exactly the same
        assert income1 == income2
        
        # Third call
        income3 = get_agent_income(agent_state, simulation_config, rng)
        assert income1 == income3


class TestLogicalConsistency:
    """Test that income is logically consistent with Assigned Allowance Level."""
    
    def test_level_1_agents_in_bottom_quintile(self):
        """Test that Level 1 agents get income in bottom 20%."""
        simulation_config = {
            'simulation': {
                'income_distribution': 'lognormal',
                'lognormal_mu': 10.0,
                'lognormal_sigma': 0.5,
                'lognormal_min': 0.0
            }
        }
        
        # Get the boundary for 20th percentile
        boundaries = get_percentile_boundaries(simulation_config['simulation'])
        p20_boundary = boundaries[0]
        
        # Generate 100 Level 1 agents
        incomes = []
        for i in range(100):
            agent_state = {'Assigned Allowance Level': 1}
            rng = np.random.default_rng(1000 + i)
            income = get_agent_income(agent_state, simulation_config, rng)
            incomes.append(income)
        
        # All should be below the 20th percentile boundary
        assert all(income <= p20_boundary * 1.01 for income in incomes), \
            f"Some Level 1 incomes exceed 20th percentile: max={max(incomes)}, boundary={p20_boundary}"
    
    def test_level_5_agents_in_top_quintile(self):
        """Test that Level 5 agents get income in top 20%."""
        simulation_config = {
            'simulation': {
                'income_distribution': 'lognormal',
                'lognormal_mu': 10.0,
                'lognormal_sigma': 0.5,
                'lognormal_min': 0.0
            }
        }
        
        # Get the boundary for 80th percentile
        boundaries = get_percentile_boundaries(simulation_config['simulation'])
        p80_boundary = boundaries[3]
        
        # Generate 100 Level 5 agents
        incomes = []
        for i in range(100):
            agent_state = {'Assigned Allowance Level': 5}
            rng = np.random.default_rng(2000 + i)
            income = get_agent_income(agent_state, simulation_config, rng)
            incomes.append(income)
        
        # All should be above the 80th percentile boundary
        assert all(income >= p80_boundary * 0.99 for income in incomes), \
            f"Some Level 5 incomes below 80th percentile: min={min(incomes)}, boundary={p80_boundary}"
    
    def test_all_levels_produce_correct_distribution(self):
        """Test that generating many agents produces correct quintile distribution."""
        simulation_config = {
            'simulation': {
                'income_distribution': 'lognormal',
                'lognormal_mu': 10.0,
                'lognormal_sigma': 0.5,
                'lognormal_min': 0.0
            }
        }
        
        # Generate 1000 agents evenly distributed across levels
        incomes_by_level = {1: [], 2: [], 3: [], 4: [], 5: []}
        
        for level in range(1, 6):
            for i in range(200):  # 200 agents per level
                agent_state = {'Assigned Allowance Level': level}
                rng = np.random.default_rng(level * 10000 + i)
                income = get_agent_income(agent_state, simulation_config, rng)
                incomes_by_level[level].append(income)
        
        # Verify ordering: mean(Level 1) < mean(Level 2) < ... < mean(Level 5)
        means = [np.mean(incomes_by_level[level]) for level in range(1, 6)]
        
        for i in range(4):
            assert means[i] < means[i+1], \
                f"Mean income for Level {i+1} ({means[i]}) should be < Level {i+2} ({means[i+1]})"


class TestDualIncomeVariables:
    """Test that both 'income' and 'actual_allowance' are generated correctly."""
    
    def test_actual_allowance_mapping(self):
        """Test that actual_allowance follows the correct mapping."""
        expected_mapping = {1: 16, 2: 32, 3: 72, 4: 128, 5: 200}
        
        simulation_config = {
            'simulation': {
                'income_distribution': 'lognormal',
                'lognormal_mu': 10.0,
                'lognormal_sigma': 0.5,
                'lognormal_min': 0.0
            }
        }
        
        for level, expected_allowance in expected_mapping.items():
            agent_state = {'Assigned Allowance Level': level}
            rng = np.random.default_rng(3000 + level)
            
            # Trigger income generation
            get_agent_income(agent_state, simulation_config, rng)
            
            # Verify actual_allowance
            assert 'actual_allowance' in agent_state
            assert agent_state['actual_allowance'] == float(expected_allowance), \
                f"Level {level} should map to {expected_allowance}, got {agent_state['actual_allowance']}"
    
    def test_get_actual_allowance_function(self):
        """Test the get_actual_allowance convenience function."""
        simulation_config = {
            'simulation': {
                'income_distribution': 'lognormal',
                'lognormal_mu': 10.0,
                'lognormal_sigma': 0.5,
                'lognormal_min': 0.0
            }
        }
        
        agent_state = {'Assigned Allowance Level': 4}
        rng = np.random.default_rng(42)
        
        # Get actual allowance
        allowance = get_actual_allowance(agent_state, simulation_config, rng)
        
        # Should be 128 for Level 4
        assert allowance == 128.0
        
        # Should also have generated income
        assert 'income' in agent_state
        assert agent_state['income'] > 0


class TestDonationRegressionStability:
    """Test that donation regression produces stable results on 12-200 scale."""
    
    def test_regression_uses_correct_scale(self):
        """Test that continuous mode regression uses 12-200 scale values."""
        # Create agent in Level 3 (should map to 72)
        agent_state = {
            'Assigned Allowance Level': 3,
            'Honesty_Humility': 3.5,
            'Study Program': 'CLEAM',
            'Group_experiment': 'NoSub',
            'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': 20.0
        }
        
        simulation_config = {
            'simulation': {
                'income_distribution': 'lognormal',
                'lognormal_mu': 10.0,
                'lognormal_sigma': 0.5,
                'lognormal_min': 0.0
            }
        }
        
        params = {
            'regression_coefficients': {
                'income_mode': 'continuous',
                'continuous': {
                    'intercept': 1.23,
                    'beta_income_linear': 0.0256,
                    'beta_group': {},
                    'beta_study': {},
                    'beta_hh': 0.634
                }
            },
            'anchor_weights': {'observed': 0.75, 'predicted': 0.25},
            'adjustment': {'shift_value': 0.0},
            'stochastic': {'in_copula': False, 'sigma_value': 0.0}
        }
        
        rng = np.random.default_rng(42)
        
        # Run donation_default
        result = donation_default(agent_state, params, rng, simulation_config)
        
        # Verify actual_allowance was used (should be 72 for Level 3)
        assert 'actual_allowance' in agent_state
        assert agent_state['actual_allowance'] == 72.0
        
        # Verify donation rate is reasonable (0-1)
        assert 0.0 <= result['donation_default'] <= 1.0
    
    def test_regression_stability_across_levels(self):
        """Test that regression produces stable, ordered results across levels."""
        simulation_config = {
            'simulation': {
                'income_distribution': 'lognormal',
                'lognormal_mu': 10.0,
                'lognormal_sigma': 0.5,
                'lognormal_min': 0.0
            }
        }
        
        params = {
            'regression_coefficients': {
                'income_mode': 'continuous',
                'continuous': {
                    'intercept': 1.23,
                    'beta_income_linear': 0.01,  # Positive coefficient
                    'beta_group': {},
                    'beta_study': {},
                    'beta_hh': 0.634
                }
            },
            'anchor_weights': {'observed': 0.75, 'predicted': 0.25},
            'adjustment': {'shift_value': 0.0},
            'stochastic': {'in_copula': False, 'sigma_value': 0.0}
        }
        
        rng = np.random.default_rng(42)
        
        # Generate agents at each level with same other traits
        donation_rates = {}
        for level in range(1, 6):
            agent_state = {
                'Assigned Allowance Level': level,
                'Honesty_Humility': 3.5,
                'Study Program': 'CLEAM',
                'Group_experiment': 'NoSub',
                'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': 20.0
            }
            
            result = donation_default(agent_state, params, rng, simulation_config)
            donation_rates[level] = result['donation_default']
        
        # With positive income coefficient, higher levels should have higher donation rates
        # (all else being equal)
        for i in range(1, 5):
            assert donation_rates[i] <= donation_rates[i+1], \
                f"Level {i} donation ({donation_rates[i]}) should be <= Level {i+1} ({donation_rates[i+1]})"


class TestPercentileRangeMapping:
    """Test the mapping from allowance level to percentile range."""
    
    def test_level_to_percentile_ranges(self):
        """Test that each level maps to the correct percentile range."""
        expected_ranges = {
            1: (0.00, 0.20),
            2: (0.20, 0.40),
            3: (0.40, 0.60),
            4: (0.60, 0.80),
            5: (0.80, 1.00)
        }
        
        for level, expected_range in expected_ranges.items():
            actual_range = _get_percentile_range_for_level(level)
            assert actual_range == expected_range, \
                f"Level {level} should map to {expected_range}, got {actual_range}"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_allowance_level_raises_error(self):
        """Test that missing Assigned Allowance Level raises ValueError."""
        agent_state = {}  # Missing 'Assigned Allowance Level'
        simulation_config = {
            'simulation': {
                'income_distribution': 'lognormal',
                'lognormal_mu': 10.0,
                'lognormal_sigma': 0.5,
                'lognormal_min': 0.0
            }
        }
        rng = np.random.default_rng(42)
        
        with pytest.raises(ValueError, match="missing 'Assigned Allowance Level'"):
            get_agent_income(agent_state, simulation_config, rng)
    
    def test_all_distribution_types_work(self):
        """Test that all distribution types produce valid income."""
        distribution_configs = [
            {
                'income_distribution': 'lognormal',
                'lognormal_mu': 10.0,
                'lognormal_sigma': 0.5,
                'lognormal_min': 0.0
            },
            {
                'income_distribution': 'generalised_gamma',
                'gg_k': 1.5,
                'gg_c': 2.0,
                'gg_lambda': 20000.0,
                'gg_min': 0.0
            },
            {
                'income_distribution': 'dagum',
                'dagum_a': 2.0,
                'dagum_p': 1.5,
                'dagum_b': 25000.0,
                'dagum_min': 0.0
            }
        ]
        
        for dist_config in distribution_configs:
            agent_state = {'Assigned Allowance Level': 3}
            simulation_config = {'simulation': dist_config}
            rng = np.random.default_rng(42)
            
            income = get_agent_income(agent_state, simulation_config, rng)
            
            # Should be a positive number
            assert income > 0
            assert isinstance(income, float)
            
            # Should have also generated actual_allowance
            assert agent_state['actual_allowance'] == 72.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

