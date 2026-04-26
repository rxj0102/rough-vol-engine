"""
Black-Scholes implied volatility utilities.

Provides vectorised Black-Scholes pricing formulas (call, put, vega) and
implied volatility inversion using Brent's method (scipy.optimize.brentq).
All functions operate on scalar or array inputs via numpy broadcasting.

References
----------
Brent, R. P. (1973). Algorithms for minimization without derivatives.
    Prentice-Hall.  Used via scipy.optimize.brentq.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray
from scipy.optimize import brentq
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes pricing
# ---------------------------------------------------------------------------

def bs_call(
    F: float | ndarray,
    K: float | ndarray,
    sigma: float | ndarray,
    T: float | ndarray,
    r: float | ndarray = 0.0,
) -> ndarray:
    """Black-Scholes call price.

    Parameters
    ----------
    F : float or array
        Forward price (= S * exp(r*T) for continuous dividends).
    K : float or array
        Strike.
    sigma : float or array
        Implied volatility (> 0).
    T : float or array
        Time to maturity (> 0).
    r : float or array
        Continuously compounded risk-free rate.

    Returns
    -------
    ndarray
        Call prices with shape determined by broadcasting (F, K, sigma, T, r).
    """
    F, K, sigma, T, r = (np.asarray(x, dtype=float) for x in (F, K, sigma, T, r))
    discount = np.exp(-r * T)
    intrinsic = np.maximum(discount * (F - K), 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        sqrtT = np.sqrt(T)
        d1 = np.where(
            (sigma > 0) & (T > 0),
            (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT),
            0.0,
        )
        d2 = d1 - sigma * sqrtT
        price = np.where(
            (sigma > 0) & (T > 0),
            discount * (F * norm.cdf(d1) - K * norm.cdf(d2)),
            intrinsic,
        )
    return np.asarray(price)


def bs_put(
    F: float | ndarray,
    K: float | ndarray,
    sigma: float | ndarray,
    T: float | ndarray,
    r: float | ndarray = 0.0,
) -> ndarray:
    """Black-Scholes put price via put-call parity."""
    F, K, T, r = (np.asarray(x, dtype=float) for x in (F, K, T, r))
    return bs_call(F, K, sigma, T, r) - np.exp(-r * T) * (F - K)


def bs_vega(
    F: float | ndarray,
    K: float | ndarray,
    sigma: float | ndarray,
    T: float | ndarray,
    r: float | ndarray = 0.0,
) -> ndarray:
    """Black-Scholes vega: d(call)/d(sigma) = d(put)/d(sigma)."""
    F, K, sigma, T, r = (np.asarray(x, dtype=float) for x in (F, K, sigma, T, r))
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrtT = np.sqrt(T)
        d1 = np.where(
            (sigma > 0) & (T > 0),
            (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT),
            0.0,
        )
        vega = np.where(
            (sigma > 0) & (T > 0),
            np.exp(-r * T) * F * norm.pdf(d1) * sqrtT,
            0.0,
        )
    return np.asarray(vega)


# ---------------------------------------------------------------------------
# Scalar implied volatility (Brent)
# ---------------------------------------------------------------------------

def _implied_vol_scalar(
    price: float,
    F: float,
    K: float,
    T: float,
    r: float = 0.0,
    flag: str = "call",
    tol: float = 1e-10,
    sigma_lo: float = 1e-7,
    sigma_hi: float = 10.0,
) -> float:
    """Invert BS price → implied vol for a single option using Brent's method.

    Returns NaN when the price is outside no-arbitrage bounds or Brent fails.
    """
    discount = np.exp(-r * T)
    if flag == "call":
        price_fn = lambda s: float(bs_call(F, K, s, T, r))
        lo_bound = float(np.maximum(discount * (F - K), 0.0))
        hi_bound = float(discount * F)
    elif flag == "put":
        price_fn = lambda s: float(bs_put(F, K, s, T, r))
        lo_bound = float(np.maximum(discount * (K - F), 0.0))
        hi_bound = float(discount * K)
    else:
        raise ValueError(f"flag must be 'call' or 'put', got {flag!r}")

    # Outside no-arbitrage bounds
    if price <= lo_bound or price >= hi_bound:
        return float("nan")

    # Expand upper bracket if needed
    hi = sigma_hi
    for _ in range(20):
        if price_fn(hi) >= price:
            break
        hi *= 2.0
        if hi > 1000.0:
            return float("nan")

    lo = sigma_lo
    if price_fn(lo) >= price:
        return float("nan")

    try:
        iv = brentq(lambda s: price_fn(s) - price, lo, hi, xtol=tol, rtol=tol)
    except (ValueError, RuntimeError):
        return float("nan")

    return float(iv)


# ---------------------------------------------------------------------------
# Vectorised implied volatility
# ---------------------------------------------------------------------------

def implied_vol(
    price: float | ndarray,
    F: float | ndarray,
    K: float | ndarray,
    T: float | ndarray,
    r: float | ndarray = 0.0,
    flag: str = "call",
    tol: float = 1e-10,
) -> ndarray:
    """Black-Scholes implied volatility using Brent's method.

    Parameters
    ----------
    price : float or array
        Observed option price.
    F : float or array
        Forward price.
    K : float or array
        Strike.
    T : float or array
        Time to maturity.
    r : float or array
        Risk-free rate.
    flag : {'call', 'put'}
        Option type.
    tol : float
        Convergence tolerance for Brent's method.

    Returns
    -------
    ndarray
        Implied volatilities.  NaN where inversion fails.
    """
    price, F, K, T, r = (np.asarray(x, dtype=float) for x in (price, F, K, T, r))
    broadcast = np.broadcast(price, F, K, T, r)
    result = np.empty(broadcast.shape)
    for idx, (p, f, k, t, ri) in zip(np.ndindex(broadcast.shape), broadcast):
        result[idx] = _implied_vol_scalar(float(p), float(f), float(k), float(t), float(ri), flag, tol)
    return result


def implied_vol_surface(
    prices: ndarray,
    forwards: float | ndarray,
    strikes: ndarray,
    maturities: ndarray,
    r: float | ndarray = 0.0,
    flag: str = "call",
    tol: float = 1e-10,
) -> ndarray:
    """Compute an implied vol surface from a price grid.

    Parameters
    ----------
    prices : ndarray, shape (n_T, n_K)
        Option prices.
    forwards : float or array, shape (n_T,) or scalar
        Forward prices per maturity.
    strikes : ndarray, shape (n_K,)
        Strike grid.
    maturities : ndarray, shape (n_T,)
        Maturity grid.
    r : float or array
        Risk-free rate (scalar or shape (n_T,)).
    flag : {'call', 'put'}
        Option type.
    tol : float
        Brent convergence tolerance.

    Returns
    -------
    ndarray, shape (n_T, n_K)
        Implied volatility surface. NaN where inversion fails.
    """
    prices = np.asarray(prices, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    forwards = np.asarray(forwards, dtype=float)
    r_arr = np.broadcast_to(np.asarray(r, dtype=float), maturities.shape)
    fwd_arr = np.broadcast_to(forwards, maturities.shape)

    n_T, n_K = len(maturities), len(strikes)
    ivol = np.full((n_T, n_K), np.nan)

    for i in range(n_T):
        for j in range(n_K):
            ivol[i, j] = _implied_vol_scalar(
                prices[i, j], float(fwd_arr[i]), float(strikes[j]),
                float(maturities[i]), float(r_arr[i]), flag, tol,
            )
    return ivol
