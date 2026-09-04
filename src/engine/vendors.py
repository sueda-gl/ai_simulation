# src/engine/vendors.py
"""
Vendor attribute generation - the single ``_initialize_vendors`` body that the
three legacy orchestrators carried verbatim (src/orchestrator.py:274-341).
"""

from typing import Callable, Optional

import numpy as np


def initialize_vendors(simulation_config: dict, rng: np.random.Generator,
                       log: Optional[Callable[[str], None]] = None) -> None:
    """
    Generate vendor attributes once per simulation and store them in
    ``simulation_config['vendors']``.

    Creates vendors with:
    - vendor_id: Sequential ID (1, 2, 3, ...)
    - price: Randomized within [vendor_price_min, vendor_price_max]
    - quality: Random integer in [1, 5]
    - sustainability: Random integer in [1, 5]
    - quantity_offered: Random integer in [vendor_products_min, vendor_products_max]

    Proximity is NOT generated here - it's customer-vendor specific
    and generated per agent in vendor_selection decision.

    Args:
        simulation_config: the engine's live simulation config (mutated in place)
        rng: the setup RNG (``default_rng(seed)``); consumed exactly as before
        log: optional console logger for the "Generated N vendors" line
    """
    from src.vendor_attribute_generator import generate_vendor_attributes

    # Get vendor configuration from simulation_config
    if 'simulation' not in simulation_config:
        return  # No simulation config available

    sim_config = simulation_config['simulation']
    num_vendors = sim_config.get('num_vendors', 1)

    # Get price range for randomization
    price_min = sim_config.get('vendor_price_min', 50.0)
    price_max = sim_config.get('vendor_price_max', 150.0)

    # Get quantity range for randomization
    quantity_min = sim_config.get('vendor_products_min', 50)
    quantity_max = sim_config.get('vendor_products_max', 150)

    # Get number of periods for per-period quantity generation
    num_periods = sim_config.get('periods', 1)

    # Get vendor prices (for backward compatibility if specified)
    # If explicit vendor_prices are provided, use those instead of randomizing
    vendor_prices = []
    use_explicit_prices = False

    if 'vendor_prices' in sim_config and sim_config['vendor_prices']:
        vendor_prices = sim_config['vendor_prices']
        use_explicit_prices = True
    else:
        # Will randomize prices, but need a list for function signature
        market_price = sim_config.get('market_price', 100.0)
        vendor_prices = [market_price] * num_vendors

    # Ensure we have enough prices
    while len(vendor_prices) < num_vendors:
        vendor_prices.append(sim_config.get('market_price', 100.0))

    # Generate vendor attributes with randomization
    vendors = generate_vendor_attributes(
        num_vendors=num_vendors,
        vendor_prices=vendor_prices,
        rng=rng,
        price_min=None if use_explicit_prices else price_min,  # Only randomize if not using explicit prices
        price_max=None if use_explicit_prices else price_max,
        quantity_min=quantity_min,
        quantity_max=quantity_max,
        num_periods=num_periods  # Pass number of periods for per-period quantity generation
    )

    # Store in simulation_config for access by decision modules
    simulation_config['vendors'] = vendors

    if log is not None:
        log(f"Generated {len(vendors)} vendors with randomized attributes")


__all__ = ["initialize_vendors"]
