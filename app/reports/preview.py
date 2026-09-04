"""Pure numerics behind the Page-1 income-distribution preview.

Extracted verbatim (T3) from `SimulationParameters.sample_income_distribution`
and `SimulationParameters.get_discount_qualification_rate` in `app/models.py`,
which now delegate here. Nothing in this module imports Streamlit or reads
session state: every input is passed in explicitly, so the preview numbers can
be computed (and tested) outside a Streamlit script run.

These are *preview* numerics only -- the agent incomes an actual run uses come
from the engine, not from here.
"""
from typing import Optional

import numpy as np
from scipy import stats

# The preview's own fixed seed. `discount_qualification_rate` uses it
# unconditionally, ignoring whatever seed the histogram was drawn with -- a
# long-standing quirk of the pre-extraction code, kept deliberately so the
# rate on screen does not move.
DEFAULT_PREVIEW_SEED = 42

# The income-distribution fields the two functions below read. `distribution_kwargs`
# lifts them off the UI's `SimulationParameters` so its two delegating methods stay
# one call each; the functions themselves never see that object.
DISTRIBUTION_PARAM_NAMES = (
    "income_distribution",
    "lognormal_mu",
    "lognormal_sigma",
    "lognormal_min",
    "lognormal_max",
    "gg_k",
    "gg_c",
    "gg_lambda",
    "gg_min",
    "gg_max",
    "dagum_a",
    "dagum_p",
    "dagum_b",
    "dagum_min",
    "dagum_max",
    "income_min",
    "income_max",
)


def distribution_kwargs(params) -> dict:
    """Read DISTRIBUTION_PARAM_NAMES off any object that carries them."""
    return {name: getattr(params, name) for name in DISTRIBUTION_PARAM_NAMES}


def sample_income_distribution(
    *,
    income_distribution: str,
    lognormal_mu: float,
    lognormal_sigma: float,
    lognormal_min: float,
    lognormal_max: Optional[float],
    gg_k: float,
    gg_c: float,
    gg_lambda: float,
    gg_min: float,
    gg_max: Optional[float],
    dagum_a: float,
    dagum_p: float,
    dagum_b: float,
    dagum_min: float,
    dagum_max: Optional[float],
    income_min: float,
    income_max: float,
    n_samples: int = 1000,
    seed: int = DEFAULT_PREVIEW_SEED,
) -> np.ndarray:
    """Sample from the configured income distribution"""
    rng = np.random.default_rng(seed)

    if income_distribution == "lognormal":
        # Use user-specified mu and sigma parameters
        mu = lognormal_mu
        sigma = lognormal_sigma

        # Sample from lognormal distribution
        Y = stats.lognorm.rvs(s=sigma, scale=np.exp(mu), size=n_samples, random_state=rng)

        # Apply linear shift (X = a + Y)
        samples = lognormal_min + Y

        # Apply rejection sampling if maximum is set
        if lognormal_max is not None:
            # Keep resampling values that exceed the maximum
            max_iterations = 1000  # Prevent infinite loops
            for _ in range(max_iterations):
                mask = samples > lognormal_max
                if not np.any(mask):
                    break
                # Resample values that are too high
                n_resample = np.sum(mask)
                Y_new = stats.lognorm.rvs(s=sigma, scale=np.exp(mu), size=n_resample, random_state=rng)
                samples[mask] = lognormal_min + Y_new

            # Final clip to ensure no values exceed max
            samples = np.clip(samples, lognormal_min, lognormal_max)

    elif income_distribution == "generalised_gamma":
        # Use user-specified k, c, and lambda parameters
        k = gg_k
        c = gg_c
        lambda_param = gg_lambda

        # Sample from Generalised Gamma distribution
        # scipy.stats.gengamma uses (a, c, scale) parameterization
        # where a=c (shape2), c=k (shape1), scale=lambda
        Y = stats.gengamma.rvs(a=c, c=k, scale=lambda_param, size=n_samples, random_state=rng)

        # Apply linear shift (X = a + Y)
        samples = gg_min + Y

        # Apply rejection sampling if maximum is set
        if gg_max is not None:
            # Keep resampling values that exceed the maximum
            max_iterations = 1000  # Prevent infinite loops
            for _ in range(max_iterations):
                mask = samples > gg_max
                if not np.any(mask):
                    break
                # Resample values that are too high
                n_resample = np.sum(mask)
                Y_new = stats.gengamma.rvs(a=c, c=k, scale=lambda_param, size=n_resample, random_state=rng)
                samples[mask] = gg_min + Y_new

            # Final clip to ensure no values exceed max
            samples = np.clip(samples, gg_min, gg_max)

    elif income_distribution == "dagum":
        # Use user-specified a (tail), p (body), and b (scale) parameters
        a = dagum_a
        p = dagum_p
        b = dagum_b

        # Sample from Dagum distribution using inverse CDF method
        # Dagum CDF: F(x) = (1 + (x/b)^(-a))^(-p)
        # Inverse CDF: x = b * ((U^(-1/p) - 1)^(-1/a))
        U = rng.random(n_samples)
        samples = b * np.power(np.power(U, -1/p) - 1, -1/a)

        # Apply linear shift
        samples = dagum_min + samples

        # Apply rejection sampling if maximum is set
        if dagum_max is not None:
            # Keep resampling values that exceed the maximum
            max_iterations = 1000  # Prevent infinite loops
            for _ in range(max_iterations):
                mask = samples > dagum_max
                if not np.any(mask):
                    break
                # Resample values that are too high
                n_resample = np.sum(mask)
                U_new = rng.random(n_resample)
                new_values = b * np.power(np.power(U_new, -1/p) - 1, -1/a)
                samples[mask] = dagum_min + new_values

            # Final clip to ensure no values exceed max
            samples = np.clip(samples, dagum_min, dagum_max)

    else:
        # Fallback to uniform distribution
        samples = rng.uniform(income_min, income_max, n_samples)

    return samples


def discount_qualification_rate(
    *,
    discount_income_threshold: float,
    n_samples: int = 1000,
    seed: int = DEFAULT_PREVIEW_SEED,
    **distribution_params,
) -> float:
    """Calculate the percentage of agents that would qualify for discounts

    `seed` deliberately defaults to the preview's fixed seed instead of
    inheriting the histogram's seed: the pre-extraction code called
    `sample_income_distribution(n_samples)` with no seed argument, so the rate
    was always computed off seed 42. Callers keep passing no seed.
    """
    samples = sample_income_distribution(
        n_samples=n_samples, seed=seed, **distribution_params
    )
    qualified = np.sum(samples <= discount_income_threshold)
    return qualified / len(samples)
