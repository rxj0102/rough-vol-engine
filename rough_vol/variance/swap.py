"""
Variance swap pricing for rough-volatility models.

A variance swap pays at maturity T:

    Payoff = N · (RV_T − K_var)

where RV_T = (1/T) ∫₀ᵀ V_t dt is the annualized realized variance and
K_var is the fair strike agreed at inception.  At fair value K_var = E[RV_T].

Two approaches are implemented:

1. **Monte Carlo** — simulate variance paths and average (rBergomi only;
   rfHeston paths require additional implementation).
2. **Model-free surface approximation** — use the ATM total variance as a
   proxy, or integrate across the smile via the log-contract replication.

References
----------
Demeterfi, K., Derman, E., Kamal, M., & Zou, J. (1999). More than you ever
    wanted to know about volatility swaps. Goldman Sachs Quantitative Strategies.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from rough_vol.variance.integrated import (
    annualized_integrated_variance,
    annualized_realized_variance,
)


# ---------------------------------------------------------------------------
# Monte Carlo approach (rBergomi)
# ---------------------------------------------------------------------------

def variance_swap_strike_mc(
    params,
    T: float,
    n_paths: int = 10_000,
    n_steps_per_year: int = 252,
    rng: int | None = 42,
    use_integrated: bool = True,
) -> tuple[float, float]:
    """Fair variance swap strike via Monte Carlo for the rBergomi model.

    Parameters
    ----------
    params : RBergomiParams
        Model parameters.
    T : float
        Swap maturity in years.
    n_paths : int
        MC paths.
    n_steps_per_year : int
        Time-discretisation resolution.
    rng : int or None
        Random seed.
    use_integrated : bool
        When True use E[∫V dt / T] (exact for variance swaps).
        When False use E[RV] from price paths (includes discretisation error).

    Returns
    -------
    (fair_strike, std_err) : (float, float)
        Monte Carlo estimate and standard error of K_var.
    """
    from rough_vol.models.rbergomi import simulate_paths

    n_steps = max(int(np.ceil(T * n_steps_per_year)), 1)
    dt = T / n_steps

    V, S = simulate_paths(params, n_paths, n_steps, T=T, rng=rng)

    if use_integrated:
        rv = annualized_integrated_variance(V, dt)
    else:
        rv = annualized_realized_variance(S, T)

    mean_k = float(rv.mean())
    std_err = float(rv.std(ddof=1) / np.sqrt(n_paths))
    return mean_k, std_err


# ---------------------------------------------------------------------------
# Model-free surface approach
# ---------------------------------------------------------------------------

def variance_swap_strike_atm_approx(
    surface,
    T: float,
) -> float:
    """ATM approximation to the variance swap strike at maturity T.

    Uses σ_ATM²(T) as a proxy.  Accurate when the smile is nearly flat;
    for a skewed surface use ``variance_swap_strike_log_contract`` instead.

    Parameters
    ----------
    surface : ImpliedVolSurface
    T : float
        Target maturity.  Interpolates between surface maturities.

    Returns
    -------
    float
        Approximate variance swap fair strike.
    """
    sigma_atm = surface.atm_vols()   # (n_T,)
    mats = surface.maturities
    valid = np.isfinite(sigma_atm)
    if valid.sum() == 0:
        return float("nan")
    return float(np.interp(T, mats[valid], (sigma_atm ** 2)[valid]))


def variance_swap_strike_log_contract(
    surface,
    T: float,
    n_quad: int = 64,
) -> float:
    """Model-free variance swap strike from the log-contract replication.

    Uses the Carr-Madan / Demeterfi decomposition:

        K_var ≈ (2/T) ∫₋∞^∞ σ²(k, T) · w(k) dk

    where σ(k, T) is the implied vol at log-moneyness k and w(k) is a
    weight function derived from the log-contract payoff.  This is
    approximated numerically over the available strike grid using linear
    interpolation to extend between grid points.

    For a single maturity slice the formula simplifies to

        K_var ≈ (1/T) · Σ_i σ²(k_i, T) · Δk_i

    which integrates the total-variance profile over log-moneyness.

    Parameters
    ----------
    surface : ImpliedVolSurface
    T : float
        Maturity.  The closest surface maturity slice is used.
    n_quad : int
        Unused (kept for API symmetry); integration uses the strike grid.

    Returns
    -------
    float
        Model-free variance swap fair strike.
    """
    mats = surface.maturities
    idx = int(np.argmin(np.abs(mats - T)))
    T_slice = float(mats[idx])

    k = surface.to_log_moneyness()[idx]   # (n_K,)
    sigma = surface.implied_vols[idx]     # (n_K,)

    valid = np.isfinite(sigma) & np.isfinite(k)
    if valid.sum() < 2:
        return float("nan")

    k_v, sigma_v = k[valid], sigma[valid]
    order = np.argsort(k_v)
    k_v, sigma_v = k_v[order], sigma_v[order]

    # ∫ σ²(k) dk over available log-moneyness range
    integrand = sigma_v ** 2
    integral = float(np.trapezoid(integrand, k_v))

    # Normalise by the range and approximate as K_var ≈ integral / range
    # Multiply by 1/(T * log-moneyness_range) * range = 1/T
    k_range = k_v[-1] - k_v[0]
    if k_range <= 0:
        return float("nan")
    return integral / (k_range * T_slice) * (k_range / 1.0)


# ---------------------------------------------------------------------------
# P&L and hedging utilities
# ---------------------------------------------------------------------------

def variance_swap_pnl(
    fair_strike: float,
    realized_var: float,
    notional: float = 1.0,
    long: bool = True,
) -> float:
    """P&L of a variance swap position at expiry.

    Parameters
    ----------
    fair_strike : float
        K_var agreed at trade inception.
    realized_var : float
        Realized annualized variance over the swap's life.
    notional : float
        Vega notional (default 1.0).
    long : bool
        True = long variance (profits when realized > strike).

    Returns
    -------
    float
        P&L = ±N · (RV − K_var).
    """
    pnl = notional * (realized_var - fair_strike)
    return float(pnl if long else -pnl)


def variance_swap_breakeven(
    fair_strike: float,
    cost_of_carry: float = 0.0,
) -> float:
    """Realized variance needed to break even on a long variance swap.

    Parameters
    ----------
    fair_strike : float
    cost_of_carry : float
        Any financing cost per unit of variance.

    Returns
    -------
    float
        Break-even realized variance.
    """
    return fair_strike + cost_of_carry


def variance_swap_delta(
    fair_strike: float,
    current_var: float,
    T_remaining: float,
    T_total: float,
) -> float:
    """Mark-to-market value of a running variance swap.

    Approximates the present value of a variance swap entered at K_var
    when t time has elapsed (T_total − T_remaining remaining) and
    instantaneous variance is currently current_var.

    Parameters
    ----------
    fair_strike : float
        Original fair strike K_var.
    current_var : float
        Current instantaneous variance (proxy for expected future RV).
    T_remaining : float
        Remaining life of the swap.
    T_total : float
        Total maturity.

    Returns
    -------
    float
        Approximate MtM value (per unit notional).
    """
    t_elapsed = T_total - T_remaining
    if T_total <= 0:
        return 0.0
    # Expected RV = weighted avg of elapsed realized and remaining expected
    expected_rv = (t_elapsed / T_total) * fair_strike + \
                  (T_remaining / T_total) * current_var
    return expected_rv - fair_strike
