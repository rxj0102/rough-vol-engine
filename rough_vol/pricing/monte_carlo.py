"""
Monte Carlo option pricing engine.

Wraps any callable payoff function together with a simulated path matrix to
produce discounted expected payoff estimates and standard errors.

The main entry point is `mc_price`, which accepts a payoff callable and an
array of asset price paths.  Convenience wrappers (`mc_call`, `mc_put`, etc.)
are provided for the common vanilla and path-dependent payoffs defined in
`rough_vol.pricing.payoffs`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MCResult:
    """Result of a Monte Carlo pricing calculation.

    Attributes
    ----------
    price : float
        Discounted expected payoff estimate.
    stderr : float
        Standard error of the estimate (= std / sqrt(n_paths)).
    n_paths : int
        Number of simulation paths used.
    ci_lower : float
        Lower bound of the 95 % confidence interval.
    ci_upper : float
        Upper bound of the 95 % confidence interval.
    """
    price: float
    stderr: float
    n_paths: int
    ci_lower: float
    ci_upper: float

    def __repr__(self) -> str:
        return (
            f"MCResult(price={self.price:.6f}, stderr={self.stderr:.6f}, "
            f"n_paths={self.n_paths}, "
            f"CI=[{self.ci_lower:.6f}, {self.ci_upper:.6f}])"
        )


# ---------------------------------------------------------------------------
# Core pricer
# ---------------------------------------------------------------------------

def mc_price(
    paths: ndarray,
    payoff_fn: Callable[[ndarray], ndarray],
    T: float,
    r: float = 0.0,
) -> MCResult:
    """Compute a Monte Carlo option price from simulated paths.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, n_steps+1)
        Asset price paths from ``simulate_paths`` (each row is one path).
        Alternatively an array of shape (n_paths,) for terminal-only payoffs;
        in that case ``payoff_fn`` should accept S_T directly.
    payoff_fn : callable
        Callable that maps the paths array to a 1-D array of per-path payoffs.
        For terminal payoffs use the terminal slice ``paths[:, -1]`` inside the
        callable; for path-dependent payoffs use the full matrix.
    T : float
        Maturity (used for discounting).
    r : float
        Continuously compounded risk-free rate.

    Returns
    -------
    MCResult
        Price, standard error, path count, and 95 % confidence interval.
    """
    paths = np.asarray(paths, dtype=float)
    payoffs = np.asarray(payoff_fn(paths), dtype=float)
    discount = math.exp(-r * T)
    discounted = discount * payoffs
    n = len(discounted)
    mean = discounted.mean()
    stderr = discounted.std(ddof=1) / math.sqrt(n)
    z = 1.959964  # 1.96 ≈ z_{0.975}
    return MCResult(
        price=float(mean),
        stderr=float(stderr),
        n_paths=n,
        ci_lower=float(mean - z * stderr),
        ci_upper=float(mean + z * stderr),
    )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def mc_call(paths: ndarray, K: float, T: float, r: float = 0.0) -> MCResult:
    """Monte Carlo European call price."""
    from rough_vol.pricing.payoffs import call_payoff
    return mc_price(paths, lambda p: call_payoff(p[:, -1], K), T, r)


def mc_put(paths: ndarray, K: float, T: float, r: float = 0.0) -> MCResult:
    """Monte Carlo European put price."""
    from rough_vol.pricing.payoffs import put_payoff
    return mc_price(paths, lambda p: put_payoff(p[:, -1], K), T, r)


def mc_asian_call(paths: ndarray, K: float, T: float, r: float = 0.0) -> MCResult:
    """Monte Carlo arithmetic Asian call price."""
    from rough_vol.pricing.payoffs import asian_call_payoff
    return mc_price(paths, lambda p: asian_call_payoff(p, K), T, r)


def mc_digital_call(paths: ndarray, K: float, T: float, r: float = 0.0) -> MCResult:
    """Monte Carlo digital (cash-or-nothing) call price."""
    from rough_vol.pricing.payoffs import digital_call_payoff
    return mc_price(paths, lambda p: digital_call_payoff(p[:, -1], K), T, r)
