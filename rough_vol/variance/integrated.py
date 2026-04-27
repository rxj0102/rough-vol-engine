"""
Integrated and realized variance from simulated paths.

These functions operate on numpy arrays produced by path simulators;
they carry no model-specific imports so they work with any model that
exposes ``(V_paths, S_paths)`` arrays.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray


def integrated_variance(var_paths: ndarray, dt: float) -> ndarray:
    """Trapezoidal estimate of integrated variance from simulated paths.

    Computes IV(ω) = ∫₀ᵀ V_t(ω) dt for each Monte Carlo path using the
    composite trapezoidal rule (second-order accurate in dt).

    Parameters
    ----------
    var_paths : ndarray, shape (n_paths, n_steps + 1)
        Instantaneous variance V(t) at times 0, dt, 2dt, …, T.
    dt : float
        Uniform time step (T / n_steps).

    Returns
    -------
    ndarray, shape (n_paths,)
        Per-path integrated variance ∫₀ᵀ V_t dt.
    """
    return np.trapezoid(var_paths, dx=dt, axis=1)


def annualized_integrated_variance(var_paths: ndarray, dt: float) -> ndarray:
    """Annualized integrated variance = (1/T) · ∫₀ᵀ V_t dt.

    For rBergomi, E[annualized_IV] = ξ₀ by construction.

    Parameters
    ----------
    var_paths : ndarray, shape (n_paths, n_steps + 1)
        Instantaneous variance paths.
    dt : float
        Time step.

    Returns
    -------
    ndarray, shape (n_paths,)
        Annualized integrated variance per path.
    """
    T = (var_paths.shape[1] - 1) * dt
    return integrated_variance(var_paths, dt) / T


def realized_variance(price_paths: ndarray) -> ndarray:
    """Realized quadratic variation from price paths.

    Computes the sum of squared log-returns as an estimate of the
    quadratic variation (Itô integral) of log price:

        RV = Σ_{i=0}^{N-1} (log S_{t_{i+1}} − log S_{t_i})²

    Parameters
    ----------
    price_paths : ndarray, shape (n_paths, n_steps + 1)
        Asset price paths, S(0) = 1 assumed (no log needed if already log).

    Returns
    -------
    ndarray, shape (n_paths,)
        Realized variance per path (sum of squared log-returns, not annualized).
    """
    log_returns = np.diff(np.log(price_paths), axis=1)   # (n_paths, n_steps)
    return (log_returns ** 2).sum(axis=1)


def annualized_realized_variance(price_paths: ndarray, T: float) -> ndarray:
    """Annualized realized variance RV / T.

    Parameters
    ----------
    price_paths : ndarray, shape (n_paths, n_steps + 1)
        Asset price paths.
    T : float
        Total simulation horizon.

    Returns
    -------
    ndarray, shape (n_paths,)
        Annualized RV per path.
    """
    return realized_variance(price_paths) / T


def variance_risk_premium(
    integrated_var: ndarray,
    realized_var: ndarray,
) -> tuple[float, float]:
    """Estimate the variance risk premium E[RV − IV].

    Parameters
    ----------
    integrated_var : ndarray, shape (n_paths,)
        Annualized integrated variance per path.
    realized_var : ndarray, shape (n_paths,)
        Annualized realized variance per path.

    Returns
    -------
    (vrp, std_err) : (float, float)
        Sample mean and standard error of RV − IV.
    """
    diff = realized_var - integrated_var
    n = len(diff)
    return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(n))


def variance_path_stats(
    var_paths: ndarray, dt: float
) -> dict[str, float]:
    """Summary statistics for a collection of variance paths.

    Parameters
    ----------
    var_paths : ndarray, shape (n_paths, n_steps + 1)
    dt : float

    Returns
    -------
    dict with keys: mean_iv, std_iv, skew_iv, kurt_iv, mean_spot_var
    """
    iv = annualized_integrated_variance(var_paths, dt)
    from scipy.stats import skew, kurtosis  # type: ignore[import]
    return {
        "mean_iv": float(iv.mean()),
        "std_iv": float(iv.std()),
        "skew_iv": float(skew(iv)),
        "excess_kurtosis_iv": float(kurtosis(iv)),
        "mean_spot_var": float(var_paths[:, 0].mean()),
    }
