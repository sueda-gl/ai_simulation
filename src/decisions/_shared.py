# src/decisions/_shared.py
"""
Helpers shared by the modelled decisions (Decisions 1, 2, 3 and 4).

Each helper takes the defaults a caller used to hard-code as arguments, so every
decision keeps exactly its former fallbacks.
"""

from typing import Any, Dict, Optional


def z_score_trait(value: float, trait_name: str, z_params: Dict,
                  fallback: Optional[Dict] = None) -> float:
    """(value - mean) / sd with the trait's mean/sd taken from ``z_params[trait_name]``.

    ``fallback`` is the stats dict used when ``z_params`` lacks the trait (default: an
    empty dict, i.e. mean 0 / sd 1). An sd of 0 gives 0.0.
    """
    p = z_params.get(trait_name, fallback if fallback is not None else {})
    mean = p.get('mean', 0)
    sd = p.get('sd', 1)
    if sd == 0:
        return 0.0
    return (value - mean) / sd


def categorical_intercept(params: Dict, level: int, defaults: Dict[int, float],
                          fallback: float) -> float:
    """Per-level intercept for categorical income.

    Reads ``params['categorical_intercepts']`` accepting ``'level_N'``, ``'N'`` or ``N``
    keys; a missing key falls back to ``defaults[level]``, then to ``fallback``. When
    the block is absent or not a dict, ``defaults[level]`` / ``fallback`` is returned as is.
    """
    cfg = params.get('categorical_intercepts')
    if isinstance(cfg, dict):
        return float(
            cfg.get(f'level_{level}',
            cfg.get(str(level),
            cfg.get(level,
            defaults.get(level, fallback))))
        )
    return defaults.get(level, fallback)


def normalise_income_mode(mode: Any, default: str = 'categorical') -> str:
    """'continuous' or 'categorical' from a UI / YAML spelling ('Continuous only', ...).

    The string is searched for the mode that is NOT ``default``; any other string gives
    ``default``. Decisions 1, 2 and 3 default to 'categorical' (the string must contain
    'continuous'); Decision 4 defaults to 'continuous' (it must contain 'categorical').
    """
    other = 'continuous' if default == 'categorical' else 'categorical'
    return other if other in str(mode).lower() else default
