# src/vendor_price_generator.py
"""
Vendor Price Generation Module

This module handles the generation of individual vendor prices that satisfy
the constraints specified in the UI (min, max, target average).

The UI collects these parameters but the actual generation logic was missing.
This module fills that gap by providing multiple algorithms to generate
vendor prices that achieve the specified target average.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VendorPriceGenerator:
    """
    Generates individual vendor prices that satisfy min/max bounds and target average.
    
    Supports multiple generation strategies:
    1. Constrained Uniform - Guarantees exact target average
    2. Beta Distribution - More natural distribution shape
    3. Normal Distribution - Bell curve around target
    4. Custom Distribution - User-defined approach
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator with optional seed for reproducibility."""
        self.rng = np.random.default_rng(seed)
    
    def generate_prices(self, 
                       num_vendors: int,
                       price_min: float,
                       price_max: float, 
                       target_avg: float,
                       method: str = "constrained_uniform",
                       **kwargs) -> List[float]:
        """
        Generate vendor prices that satisfy constraints.
        
        Args:
            num_vendors: Number of vendors to generate prices for
            price_min: Minimum price any vendor can have
            price_max: Maximum price any vendor can have
            target_avg: Target average price across all vendors
            method: Generation method ("constrained_uniform", "beta", "normal", "custom")
            **kwargs: Additional parameters for specific methods
            
        Returns:
            List of vendor prices that satisfy all constraints
            
        Raises:
            ValueError: If target_avg is not achievable within bounds
        """
        # Validate inputs
        self._validate_inputs(num_vendors, price_min, price_max, target_avg)
        
        # Route to appropriate generation method
        if method == "constrained_uniform":
            return self._generate_constrained_uniform(num_vendors, price_min, price_max, target_avg)
        elif method == "beta":
            return self._generate_beta_distribution(num_vendors, price_min, price_max, target_avg, **kwargs)
        elif method == "normal":
            return self._generate_normal_distribution(num_vendors, price_min, price_max, target_avg, **kwargs)
        elif method == "custom":
            return self._generate_custom_distribution(num_vendors, price_min, price_max, target_avg, **kwargs)
        else:
            raise ValueError(f"Unknown generation method: {method}")
    
    def _validate_inputs(self, num_vendors: int, price_min: float, price_max: float, target_avg: float):
        """Validate that the inputs are mathematically feasible."""
        if num_vendors <= 0:
            raise ValueError("Number of vendors must be positive")
        
        if price_min >= price_max:
            raise ValueError("price_min must be less than price_max")
        
        if target_avg < price_min or target_avg > price_max:
            raise ValueError(f"Target average {target_avg} must be between {price_min} and {price_max}")
        
        logger.info(f"Generating prices for {num_vendors} vendors: "
                   f"range=[{price_min:.2f}, {price_max:.2f}], target_avg={target_avg:.2f}")
    
    def _generate_constrained_uniform(self, num_vendors: int, price_min: float, 
                                    price_max: float, target_avg: float) -> List[float]:
        """
        RECOMMENDED METHOD: Generates prices that achieve EXACT target average.
        
        Algorithm:
        1. Generate (n-1) random prices uniformly between min and max
        2. Calculate what the last price must be to achieve target average
        3. If last price is out of bounds, redistribute across all prices
        4. Guarantee: Final average will be exactly target_avg (within floating point precision)
        
        Pros: Exact target achievement, simple algorithm
        Cons: Last vendor might have less random price (but still within bounds)
        """
        if num_vendors == 1:
            # Special case: single vendor gets exactly the target price
            return [target_avg]
        
        # Step 1: initial random draw (all vendors)
        prices = self.rng.uniform(price_min, price_max, num_vendors)
        delta = target_avg - prices.mean()
        eps = 1e-10  # numerical tolerance
        if abs(delta) < eps:
            return prices.tolist()
        # Step 2: sort indices in the direction we need to move
        ascend = delta > 0  # True ⇒ need to raise mean ⇒ push upward towards max
        idx = np.argsort(prices)  # ascending
        if not ascend:
            idx = idx[::-1]       # descending when lowering mean
        # Step 3: water-filling / redistribution loop
        for i in idx:
            capacity = (price_max - prices[i]) if ascend else (price_min - prices[i])
            step = np.sign(delta) * min(abs(delta) * num_vendors, abs(capacity))
            prices[i] += step
            delta -= step / num_vendors
            if abs(delta) < eps:
                break
        # Step 4: final numeric correction to hit the exact average
        prices = np.clip(prices, price_min, price_max)
        correction = target_avg - prices.mean()
        if abs(correction) > eps:
            # Apply tiny correction to the first adjustable element
            prices[idx[0]] += correction * num_vendors
        return prices.tolist()
    
    # NOTE: previous helper _redistribute_excess is no longer needed and has been removed.
    
    def _generate_beta_distribution(self, num_vendors: int, price_min: float,
                                  price_max: float, target_avg: float, 
                                  shape_param: float = 2.0) -> List[float]:
        """
        Generate prices using Beta distribution for more natural spread.
        
        Algorithm:
        1. Convert target average to [0,1] scale
        2. Find Beta(a,b) parameters that give this mean
        3. Generate Beta-distributed values and scale to price range
        4. Apply post-correction to achieve closer to target average
        
        Pros: More natural distribution shape, no "forced" last price
        Cons: Won't achieve exact target average, requires parameter tuning
        """
        # Convert target to normalized scale [0,1]
        range_size = price_max - price_min
        normalized_target = (target_avg - price_min) / range_size
        
        # For Beta(a,b), mean = a/(a+b)
        # Set b = shape_param and solve for a
        b = shape_param
        a = (b * normalized_target) / (1 - normalized_target)
        
        # Handle edge cases
        a = max(0.1, min(a, 100))  # Keep parameters reasonable
        
        # Generate beta-distributed values
        beta_values = self.rng.beta(a, b, num_vendors)
        prices = price_min + beta_values * range_size
        
        # Optional: Apply correction to get closer to target
        current_avg = np.mean(prices)
        correction = target_avg - current_avg
        prices = [p + correction for p in prices]
        
        # Ensure all prices are within bounds after correction
        prices = [max(price_min, min(p, price_max)) for p in prices]
        
        return prices
    
    def _generate_normal_distribution(self, num_vendors: int, price_min: float,
                                    price_max: float, target_avg: float,
                                    std_fraction: float = 0.2) -> List[float]:
        """
        Generate prices using Normal distribution centered on target.
        
        Algorithm:
        1. Use target_avg as mean
        2. Set std_dev as fraction of price range
        3. Generate normal values and clip to bounds
        4. Apply correction to achieve target average
        
        Pros: Natural bell curve distribution
        Cons: Clipping at boundaries can distort distribution
        """
        # Set standard deviation as fraction of price range
        price_range = price_max - price_min
        std_dev = price_range * std_fraction
        
        # Generate normal-distributed prices
        prices = self.rng.normal(target_avg, std_dev, num_vendors)
        
        # Clip to bounds
        prices = np.clip(prices, price_min, price_max)
        
        # Apply correction to achieve target average
        current_avg = np.mean(prices)
        correction = target_avg - current_avg
        prices = prices + correction
        
        # Final clipping after correction
        prices = np.clip(prices, price_min, price_max)
        
        return prices.tolist()
    
    def _generate_custom_distribution(self, num_vendors: int, price_min: float,
                                    price_max: float, target_avg: float,
                                    distribution_weights: Optional[List[float]] = None) -> List[float]:
        """
        Generate prices using custom distribution weights.
        
        This allows for specific price distributions like:
        - More vendors at low prices, few at high prices
        - Bimodal distribution (budget and premium vendors)
        - Custom business logic
        """
        if distribution_weights is None:
            # Default: uniform distribution
            return self._generate_constrained_uniform(num_vendors, price_min, price_max, target_avg)
        
        # Implementation would depend on specific requirements
        # This is a placeholder for custom logic
        raise NotImplementedError("Custom distribution not yet implemented")
    
    def generate_vendor_config(self, sim_params) -> List[Dict]:
        """
        Generate complete vendor configuration from simulation parameters.
        
        This is the main interface that the simulation orchestrator would call.
        Takes the UI parameters and returns a complete vendor configuration.
        
        Args:
            sim_params: SimulationParameters object from the UI
            
        Returns:
            List of vendor configuration dictionaries
        """
        # Extract parameters from sim_params
        num_vendors = sim_params.num_vendors
        price_min = sim_params.vendor_price_min
        price_max = sim_params.vendor_price_max
        target_avg = sim_params.market_price
        
        # Generate prices
        prices = self.generate_prices(
            num_vendors=num_vendors,
            price_min=price_min,
            price_max=price_max,
            target_avg=target_avg,
            method="constrained_uniform"  # Use the most reliable method
        )
        
        # Generate products per vendor (similar logic could be applied)
        products = self._generate_vendor_products(sim_params)
        
        # Generate carryover settings
        carryover_settings = self._generate_carryover_settings(sim_params)
        
        # Create vendor configuration list
        vendor_configs = []
        for i in range(num_vendors):
            vendor_config = {
                'vendor_id': f'V{i+1}',
                'price': prices[i],
                'products_per_period': products[i],
                'carryover': carryover_settings[i]
            }
            vendor_configs.append(vendor_config)
        
        logger.info(f"Generated {num_vendors} vendor configurations")
        logger.info(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")
        logger.info(f"Actual average: ${np.mean(prices):.2f} (target: ${target_avg:.2f})")
        
        return vendor_configs
    
    def _generate_vendor_products(self, sim_params) -> List[int]:
        """Generate products per vendor using similar logic to prices."""
        # Similar implementation for products as we did for prices
        if hasattr(sim_params, 'vendor_products_min'):
            return self.generate_integer_values(
                num_vendors=sim_params.num_vendors,
                min_val=sim_params.vendor_products_min,
                max_val=sim_params.vendor_products_max,
                target_avg=sim_params.vendor_products_avg
            )
        else:
            # Fallback to equal distribution
            return [sim_params.products_per_vendor] * sim_params.num_vendors
    
    def _generate_carryover_settings(self, sim_params) -> List[bool]:
        """Generate carryover settings based on simulation parameters."""
        if sim_params.override_carryover:
            # All vendors have the same carryover setting
            return [sim_params.global_carryover] * sim_params.num_vendors
        else:
            # Use probability-based assignment
            carryover_prob = sim_params.vendor_carryover_probability
            return self.rng.random(sim_params.num_vendors) < carryover_prob
    
    def generate_integer_values(self, num_vendors: int, min_val: int, 
                               max_val: int, target_avg: float) -> List[int]:
        """
        Generate integer values (like products per vendor) that achieve target average.
        Similar to price generation but for integers.
        """
        # Generate continuous values first
        continuous_values = self.generate_prices(
            num_vendors=num_vendors,
            price_min=float(min_val),
            price_max=float(max_val),
            target_avg=target_avg,
            method="constrained_uniform"
        )
        
        # Round to integers
        integer_values = [round(v) for v in continuous_values]
        
        # Adjust to ensure bounds and target average
        integer_values = [max(min_val, min(v, max_val)) for v in integer_values]
        
        # Fine-tune to get closer to target average
        current_avg = np.mean(integer_values)
        diff = target_avg - current_avg
        
        if abs(diff) > 0.5:  # If significant difference
            # Adjust some values up or down
            adjustment = 1 if diff > 0 else -1
            num_to_adjust = min(abs(int(diff * num_vendors)), num_vendors)
            
            for i in range(num_to_adjust):
                new_val = integer_values[i] + adjustment
                if min_val <= new_val <= max_val:
                    integer_values[i] = new_val
        
        return integer_values


# Convenience function for direct use
def generate_vendor_prices(num_vendors: int, price_min: float, price_max: float, 
                          target_avg: float, seed: Optional[int] = None) -> List[float]:
    """
    Convenience function to generate vendor prices.
    
    This is the main function that external code should call.
    """
    generator = VendorPriceGenerator(seed=seed)
    return generator.generate_prices(num_vendors, price_min, price_max, target_avg)


# Integration point for the orchestrator
def create_vendor_configuration(sim_params, seed: Optional[int] = None) -> List[Dict]:
    """
    Main integration function for the simulation orchestrator.
    
    This function should be called from the orchestrator to convert
    UI parameters into actual vendor configurations.
    
    Args:
        sim_params: SimulationParameters object from the UI
        seed: Random seed for reproducible results
        
    Returns:
        List of complete vendor configuration dictionaries
    """
    generator = VendorPriceGenerator(seed=seed)
    return generator.generate_vendor_config(sim_params)


if __name__ == "__main__":
    # Example usage and testing
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the generator
    generator = VendorPriceGenerator(seed=42)
    
    # Test case 1: 5 vendors, prices $8-$12, target average $10
    prices = generator.generate_prices(
        num_vendors=5,
        price_min=8.0,
        price_max=12.0,
        target_avg=10.0
    )
    
    print("Test Case 1:")
    print(f"Prices: {[f'${p:.2f}' for p in prices]}")
    print(f"Average: ${np.mean(prices):.4f} (target: $10.00)")
    print(f"Range: ${min(prices):.2f} - ${max(prices):.2f}")
    print()
    
    # Test case 2: Edge case - target near minimum
    prices2 = generator.generate_prices(
        num_vendors=10,
        price_min=50.0,
        price_max=150.0,
        target_avg=60.0
    )
    
    print("Test Case 2 (Edge case):")
    print(f"Prices: {[f'${p:.2f}' for p in prices2]}")
    print(f"Average: ${np.mean(prices2):.4f} (target: $60.00)")
    print(f"Range: ${min(prices2):.2f} - ${max(prices2):.2f}")
