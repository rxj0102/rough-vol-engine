"""
Short-maturity asymptotics for rough volatility models.

Implements the Fukasawa (2011) formula for the ATM implied volatility level
and skew under the rough Bergomi model (and any rough vol model with the same
leading-order SDE structure).

Theory
------
For the rough Bergomi model

    V(t) = ξ₀ · exp(η · W^H(t) − ½η²t^{2H})
    dS/S = √V(t) dB_t,    dB = ρ dW + √(1−ρ²) dW⊥

the ATM implied volatility σ(T) and its log-moneyness derivative (skew)
satisfy as T → 0:

    σ_ATM(T)  →  √ξ₀
    ∂σ/∂k|_{k=0}  ≈  ρ · η · C_H · T^{H − ½}

where k = log(K/F) is log-moneyness and

    C_H = 2 / [(2H+1)² · Γ(H + ½)²]

This formula is exact to leading order in T and derives from the Fukasawa
(2011) / Alòs–León–Vives (2007) representation of the ATM skew in terms of
the covariance between the log-price and the integrated variance.

References
----------
Fukasawa, M. (2011). Asymptotic analysis for stochastic volatility: martingale
    expansion. Finance and Stochastics, 15(4), 635–654.
Bayer, C., Friz, P., & Gatheral, J. (2016). Pricing under rough volatility.
    Quantitative Finance, 16(6), 887–904.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gamma

from rough_vol.models.rbergomi import RBergomiParams


# ---------------------------------------------------------------------------
# Coefficient
# ---------------------------------------------------------------------------

def fukasawa_coefficient(H: float) -> float:
    """Compute the Fukasawa kernel coefficient C_H.

    C_H = 2 / [(2H+1)² · Γ(H + ½)²]

    This is the constant that appears in the leading-order ATM skew formula:

        ∂σ/∂k|₀ ≈ ρ · η · C_H · T^{H − ½}

    Parameters
    ----------
    H : float
        Hurst exponent in (0, 0.5).

    Returns
    -------
    float
        Coefficient C_H > 0.
    """
    if not (0.0 < H < 0.5):
        raise ValueError(f"H must be in (0, 0.5), got {H}")
    return 2.0 / ((2.0 * H + 1.0) ** 2 * gamma(H + 0.5) ** 2)


# ---------------------------------------------------------------------------
# ATM level and skew
# ---------------------------------------------------------------------------

def atm_implied_vol(params: RBergomiParams, T: float) -> float:
    """Leading-order ATM implied volatility.

    To leading order as T → 0:

        σ_ATM(T) ≈ √ξ₀

    Parameters
    ----------
    params : RBergomiParams
        Model parameters.
    T : float
        Time to maturity (used only for type consistency; result is T-independent).

    Returns
    -------
    float
        ATM implied volatility estimate √ξ₀.
    """
    return math.sqrt(params.xi0)


def atm_skew(params: RBergomiParams, T: float) -> float:
    """Fukasawa ATM implied volatility skew.

    Computes the leading-order log-moneyness derivative of the ATM implied vol:

        ∂σ_imp/∂k|_{k=0} ≈ ρ · η · C_H · T^{H − ½}

    where k = log(K/F) and C_H = 2 / [(2H+1)² · Γ(H + ½)²].

    Parameters
    ----------
    params : RBergomiParams
        Model parameters (H, eta, rho, xi0).
    T : float
        Time to maturity (> 0).

    Returns
    -------
    float
        ATM skew (negative for rho < 0, i.e. the usual equity skew).
    """
    if T <= 0.0:
        raise ValueError(f"T must be > 0, got {T}")
    C_H = fukasawa_coefficient(params.H)
    return params.rho * params.eta * C_H * T ** (params.H - 0.5)


def atm_skew_coefficient(params: RBergomiParams) -> float:
    """Time-independent part of the ATM skew: ρ · η · C_H.

    The full skew equals ``atm_skew_coefficient(params) * T^{H - 0.5}``.

    Parameters
    ----------
    params : RBergomiParams
        Model parameters.

    Returns
    -------
    float
        Coefficient c = ρ · η · C_H.
    """
    return params.rho * params.eta * fukasawa_coefficient(params.H)


# ---------------------------------------------------------------------------
# Term structure
# ---------------------------------------------------------------------------

def rough_vol_term_structure(
    params: RBergomiParams,
    maturities: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute ATM vol and skew term structure.

    Parameters
    ----------
    params : RBergomiParams
        Model parameters.
    maturities : array-like
        Array of maturities at which to evaluate the term structure.

    Returns
    -------
    dict with keys:
        ``'maturities'`` : ndarray of shape (n_T,)
        ``'atm_vol'``    : ndarray of shape (n_T,) — √ξ₀ at each maturity
        ``'atm_skew'``   : ndarray of shape (n_T,) — Fukasawa skew at each T
    """
    maturities = np.asarray(maturities, dtype=float)
    atm_vol_arr = np.full(maturities.shape, math.sqrt(params.xi0))
    atm_skew_arr = np.array([atm_skew(params, T) for T in maturities])
    return {
        "maturities": maturities,
        "atm_vol": atm_vol_arr,
        "atm_skew": atm_skew_arr,
    }
