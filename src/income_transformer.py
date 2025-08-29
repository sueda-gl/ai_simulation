# src/income_transformer.py
"""
Income Transformer: Converts experimental income levels (1-5) to monetary incomes
using specified distributions while preserving the relative ordering and correlations.

Key Design Principles:
1. Preserve correlations from the copula/original data
2. Support both categorical (quintile) and continuous income modes
3. Generate realistic monetary incomes using three distribution families
4. Maintain backward compatibility with the regression coefficients
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar, brentq
from typing import Dict, Any, Tuple, Optional, Union, List
import warnings


class IncomeTransformer:
    """
    Transforms experimental income levels (1-5) to monetary incomes using specified distributions.
    
    The transformer ensures that:
    1. Original ordinal relationships are preserved (level 5 → higher income than level 1)
    2. Generated incomes follow the specified distribution (lognormal, pareto, weibull)
    3. The transformation is consistent within a simulation run
    """
    
    def __init__(self, simulation_config: Dict[str, Any]):
        """
        Initialize the income transformer with simulation parameters.
        
        Args:
            simulation_config: Global simulation configuration containing income parameters
        """
        self.sim_params = simulation_config.get('simulation', {})
        
        # Extract income parameters
        self.income_min = self.sim_params.get('income_min', 1000.0)
        self.income_max = self.sim_params.get('income_max', 10000.0)
        self.income_avg = self.sim_params.get('income_avg', 5000.0)
        self.income_distribution = self.sim_params.get('income_distribution', 'lognormal')
        
        # Validate parameters
        self._validate_parameters()
        
        # Calibrate distribution parameters
        self.dist_params = self._calibrate_distribution()
        
        # Store for later use in population-wide operations
        self.generated_incomes = []
        
    def _validate_parameters(self):
        """Validate that income parameters are sensible."""
        if self.income_min <= 0:
            raise ValueError("income_min must be positive")
        if self.income_max <= self.income_min:
            raise ValueError("income_max must be greater than income_min")
        if not (self.income_min <= self.income_avg <= self.income_max):
            raise ValueError("income_avg must be between income_min and income_max")
        if self.income_distribution not in ['lognormal', 'pareto', 'weibull']:
            raise ValueError(f"Unknown distribution: {self.income_distribution}")
    
    def _calibrate_distribution(self) -> Dict[str, float]:
        """
        Calibrate distribution parameters to match target mean and constraints.
        
        Returns:
            Dictionary of distribution-specific parameters
        """
        if self.income_distribution == 'lognormal':
            return self._calibrate_lognormal()
        elif self.income_distribution == 'pareto':
            return self._calibrate_pareto()
        elif self.income_distribution == 'weibull':
            return self._calibrate_weibull()
    
    def _calibrate_lognormal(self) -> Dict[str, float]:
        """
        Calibrate lognormal parameters (mu, sigma) to match target mean.
        
        For lognormal: mean = exp(mu + sigma²/2)
        We'll use a heuristic approach to set sigma based on the range.
        """
        # Estimate sigma from the ratio of max to min
        # Larger ratio → larger sigma (more spread)
        ratio = self.income_max / self.income_min
        sigma = np.sqrt(np.log(ratio) / 4)  # Heuristic: 4 sigmas cover most of range
        
        # Solve for mu given the mean constraint
        # mean = exp(mu + sigma²/2) → mu = log(mean) - sigma²/2
        mu = np.log(self.income_avg) - (sigma**2) / 2
        
        # Verify that most of the distribution falls within [min, max]
        # Adjust if needed
        test_dist = stats.lognorm(s=sigma, scale=np.exp(mu))
        p_min = test_dist.cdf(self.income_min)
        p_max = test_dist.cdf(self.income_max)
        
        # We want at least 95% of the distribution within bounds
        if p_max - p_min < 0.95:
            # Reduce sigma to tighten the distribution
            sigma *= 0.8
            mu = np.log(self.income_avg) - (sigma**2) / 2
        
        return {'mu': mu, 'sigma': sigma}
    
    def _calibrate_pareto(self) -> Dict[str, float]:
        """
        Calibrate Pareto parameters (alpha, scale) to match target mean.
        
        Pareto Type I: scale = income_min
        Mean exists only if alpha > 1: mean = scale * alpha / (alpha - 1)
        """
        scale = self.income_min
        
        # Solve for alpha: mean = scale * alpha / (alpha - 1)
        # mean * (alpha - 1) = scale * alpha
        # mean * alpha - mean = scale * alpha
        # alpha * (mean - scale) = mean
        # alpha = mean / (mean - scale)
        
        if self.income_avg <= scale:
            raise ValueError("For Pareto distribution, income_avg must be > income_min")
        
        alpha = self.income_avg / (self.income_avg - scale)
        
        # Ensure alpha > 1 for finite mean
        if alpha <= 1:
            warnings.warn("Pareto alpha <= 1 would give infinite mean; setting alpha = 1.5")
            alpha = 1.5
        
        # Check if 99th percentile is reasonable
        # For Pareto: F(x) = 1 - (scale/x)^alpha
        # x_p = scale / (1-p)^(1/alpha)
        x99 = scale / (0.01**(1/alpha))
        
        if x99 > self.income_max * 2:
            # Alpha too small, increase it to reduce tail
            alpha = min(alpha * 1.5, 3.0)
        
        return {'alpha': alpha, 'scale': scale}
    
    def _calibrate_weibull(self) -> Dict[str, float]:
        """
        Calibrate Weibull parameters (k, lambda) to match target mean.
        
        For Weibull: mean = lambda * Gamma(1 + 1/k)
        We use numerical optimization to find k, then solve for lambda.
        """
        from scipy.special import gamma as gamma_fn
        
        # Shift the distribution to start at income_min
        shift = self.income_min
        mean_shifted = self.income_avg - shift
        max_shifted = self.income_max - shift
        
        # Function to find k such that the 99th percentile ≈ max_shifted
        def objective(k):
            # For Weibull: F(x) = 1 - exp(-(x/lambda)^k)
            # At 99th percentile: 0.99 = 1 - exp(-(x99/lambda)^k)
            # Given mean = lambda * Gamma(1 + 1/k), we can solve for lambda
            lambda_param = mean_shifted / gamma_fn(1 + 1/k)
            
            # Calculate 99th percentile
            x99 = lambda_param * (-np.log(0.01))**(1/k)
            
            # We want x99 ≈ max_shifted
            return abs(x99 - max_shifted)
        
        # Find optimal k
        result = minimize_scalar(objective, bounds=(0.5, 5.0), method='bounded')
        k = result.x
        
        # Calculate corresponding lambda
        lambda_param = mean_shifted / gamma_fn(1 + 1/k)
        
        return {'k': k, 'lambda': lambda_param, 'shift': shift}
    
    def _generate_income(self, percentile: float) -> float:
        """
        Generate a single income value at the given percentile of the distribution.
        
        Args:
            percentile: Value between 0 and 1 indicating position in distribution
            
        Returns:
            Monetary income value
        """
        # Clip to avoid numerical issues at extremes
        percentile = np.clip(percentile, 0.001, 0.999)
        
        if self.income_distribution == 'lognormal':
            # Lognormal inverse CDF
            mu = self.dist_params['mu']
            sigma = self.dist_params['sigma']
            z_score = stats.norm.ppf(percentile)
            income = np.exp(mu + sigma * z_score)
            
        elif self.income_distribution == 'pareto':
            # Pareto inverse CDF: x = scale / (1-F)^(1/alpha)
            alpha = self.dist_params['alpha']
            scale = self.dist_params['scale']
            income = scale / ((1 - percentile)**(1/alpha))
            
        elif self.income_distribution == 'weibull':
            # Weibull inverse CDF: x = lambda * (-ln(1-F))^(1/k)
            k = self.dist_params['k']
            lambda_param = self.dist_params['lambda']
            shift = self.dist_params['shift']
            income = shift + lambda_param * ((-np.log(1 - percentile))**(1/k))
        
        # Ensure income is within bounds (can happen at extreme percentiles)
        return np.clip(income, self.income_min, self.income_max)
    
    def transform_agent(self, assigned_level: int, income_mode: str = 'categorical') -> Dict[str, Any]:
        """
        Transform a single agent's income level to monetary income and mode-specific value.
        
        Args:
            assigned_level: Original assigned allowance level (1-5)
            income_mode: Either 'categorical' or 'continuous'
            
        Returns:
            Dictionary with:
                - monetary_income: Actual income in currency units
                - income_level_mod: Modified level for use in regression
        """
        # Convert discrete level to percentile
        # Use midpoint of the level's range to avoid edge effects
        # Level 1 → 0.1, Level 2 → 0.3, ..., Level 5 → 0.9
        percentile = (assigned_level - 0.5) / 5.0
        
        # Generate monetary income
        monetary_income = self._generate_income(percentile)
        
        # Store for population-wide operations
        self.generated_incomes.append(monetary_income)
        
        # For categorical mode, we'll need population-wide quintiles
        # For now, return the monetary income and original level
        # The orchestrator will handle the final mapping after all agents are generated
        
        return {
            'monetary_income': monetary_income,
            'income_level_original': assigned_level,
            '_percentile': percentile  # Store for debugging
        }
    
    def compute_population_quintiles(self, all_incomes: List[float]) -> np.ndarray:
        """
        Compute quintile breakpoints for the population.
        
        Args:
            all_incomes: List of all monetary incomes in the population
            
        Returns:
            Array of 4 breakpoints defining the 5 quintiles
        """
        return np.percentile(all_incomes, [20, 40, 60, 80])
    
    def map_to_quintile(self, income: float, quintile_breaks: np.ndarray) -> str:
        """
        Map a monetary income to its quintile.
        
        Args:
            income: Monetary income value
            quintile_breaks: Array of quintile breakpoints
            
        Returns:
            Quintile label (Q1, Q2, Q3, Q4_Q5)
        """
        if income <= quintile_breaks[0]:
            return 'Q1'
        elif income <= quintile_breaks[1]:
            return 'Q2'
        elif income <= quintile_breaks[2]:
            return 'Q3'
        else:
            return 'Q4_Q5'
    
    def normalize_continuous(self, income: float) -> float:
        """
        Normalize monetary income to 1-5 scale for continuous mode.
        
        Args:
            income: Monetary income value
            
        Returns:
            Normalized value between 1 and 5
        """
        # Linear mapping from [min, max] to [1, 5]
        if self.income_max == self.income_min:
            return 3.0  # Middle value if no variation
        
        normalized = 1.0 + 4.0 * (income - self.income_min) / (self.income_max - self.income_min)
        return np.clip(normalized, 1.0, 5.0)
