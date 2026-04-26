"""
Payoff functions for path-dependent and vanilla options.

Vectorised payoff callables that operate on full path arrays returned by the
Monte Carlo engine.  Each function accepts the full path matrix and returns
a 1-D array of per-path payoffs so they can be passed directly to
`mc_price`.

Conventions
-----------
- S_T  : ndarray, shape (n_paths,) — terminal asset prices
- paths : ndarray, shape (n_paths, n_steps+1) — full asset price matrix
- K     : float — strike
- barrier : float — barrier level
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Vanilla
# ---------------------------------------------------------------------------

def call_payoff(S_T: ndarray, K: float) -> ndarray:
    """European call: max(S_T - K, 0)."""
    return np.maximum(np.asarray(S_T, dtype=float) - K, 0.0)


def put_payoff(S_T: ndarray, K: float) -> ndarray:
    """European put: max(K - S_T, 0)."""
    return np.maximum(K - np.asarray(S_T, dtype=float), 0.0)


def digital_call_payoff(S_T: ndarray, K: float) -> ndarray:
    """Cash-or-nothing digital call: 1_{S_T > K}."""
    return (np.asarray(S_T, dtype=float) > K).astype(float)


def digital_put_payoff(S_T: ndarray, K: float) -> ndarray:
    """Cash-or-nothing digital put: 1_{S_T < K}."""
    return (np.asarray(S_T, dtype=float) < K).astype(float)


# ---------------------------------------------------------------------------
# Asian (arithmetic average)
# ---------------------------------------------------------------------------

def asian_call_payoff(paths: ndarray, K: float) -> ndarray:
    """Arithmetic Asian call: max(mean(S) - K, 0), average over all time steps."""
    avg = np.asarray(paths, dtype=float).mean(axis=1)
    return np.maximum(avg - K, 0.0)


def asian_put_payoff(paths: ndarray, K: float) -> ndarray:
    """Arithmetic Asian put: max(K - mean(S), 0)."""
    avg = np.asarray(paths, dtype=float).mean(axis=1)
    return np.maximum(K - avg, 0.0)


# ---------------------------------------------------------------------------
# Barrier (single barrier, continuous monitoring approximated by path grid)
# ---------------------------------------------------------------------------

def barrier_call_payoff(
    paths: ndarray,
    K: float,
    barrier: float,
    barrier_type: str = "up-out",
) -> ndarray:
    """Barrier call payoff.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, n_steps+1)
        Full asset price paths.
    K : float
        Strike.
    barrier : float
        Barrier level.
    barrier_type : str
        One of ``'up-out'``, ``'up-in'``, ``'down-out'``, ``'down-in'``.

    Returns
    -------
    ndarray, shape (n_paths,)
        Per-path payoffs.
    """
    paths = np.asarray(paths, dtype=float)
    S_T = paths[:, -1]
    vanilla = np.maximum(S_T - K, 0.0)

    if barrier_type == "up-out":
        knocked = np.any(paths >= barrier, axis=1)
        return np.where(knocked, 0.0, vanilla)
    elif barrier_type == "up-in":
        knocked = np.any(paths >= barrier, axis=1)
        return np.where(knocked, vanilla, 0.0)
    elif barrier_type == "down-out":
        knocked = np.any(paths <= barrier, axis=1)
        return np.where(knocked, 0.0, vanilla)
    elif barrier_type == "down-in":
        knocked = np.any(paths <= barrier, axis=1)
        return np.where(knocked, vanilla, 0.0)
    else:
        raise ValueError(
            f"barrier_type must be one of 'up-out', 'up-in', 'down-out', 'down-in'; got {barrier_type!r}"
        )
