"""
Forward variance curve extraction from implied-volatility surfaces.

In rough-vol models the initial forward variance curve ξ(t) = E[V_t | F₀]
is an input.  For rBergomi the flat choice ξ(t) = ξ₀ is standard; for
rfHeston ξ depends on (λ, θ, V₀, H).  Given a market IV surface we can
extract model-free forward variances using total-variance differences.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray


def atm_total_variance(
    surface,
) -> tuple[ndarray, ndarray]:
    """ATM total variance w(T) = σ_ATM(T)² · T for each maturity.

    Parameters
    ----------
    surface : ImpliedVolSurface

    Returns
    -------
    mats : ndarray, shape (n_T,)
        Sorted maturities.
    w_atm : ndarray, shape (n_T,)
        ATM total variance.  NaN where the ATM vol cannot be interpolated.
    """
    sigma_atm = surface.atm_vols()          # (n_T,)
    w_atm = sigma_atm ** 2 * surface.maturities
    return surface.maturities.copy(), w_atm


def forward_variance_curve(
    surface,
) -> tuple[ndarray, ndarray]:
    """Local forward variance between adjacent maturities.

    Defines the forward variance over [T_i, T_{i+1}] as

        f(T_i, T_{i+1}) = (w(T_{i+1}) − w(T_i)) / (T_{i+1} − T_i)

    where w(T) is the ATM total variance.  This is the model-free analogue
    of the instantaneous forward variance ξ(t) sampled at mid-points.

    Parameters
    ----------
    surface : ImpliedVolSurface

    Returns
    -------
    t_mid : ndarray, shape (n_T − 1,)
        Mid-points (T_i + T_{i+1}) / 2.
    fv : ndarray, shape (n_T − 1,)
        Forward variance over each bucket.
    """
    mats, w = atm_total_variance(surface)
    if len(mats) < 2:
        return np.array([]), np.array([])
    dw = np.diff(w)
    dT = np.diff(mats)
    fv = dw / dT
    t_mid = 0.5 * (mats[:-1] + mats[1:])
    return t_mid, fv


def instantaneous_forward_variance_rbergomi(params, t: float | ndarray) -> ndarray:
    """Forward variance curve for the rough Bergomi model.

    Under the flat initialisation (constant forward variance curve),

        ξ(t) = E[V_t | F₀] = ξ₀  for all t ≥ 0.

    Parameters
    ----------
    params : RBergomiParams
    t : float or ndarray
        Query times.

    Returns
    -------
    ndarray
        Forward variance ξ₀ evaluated at each t (constant).
    """
    t_arr = np.asarray(t, dtype=float)
    return np.full_like(t_arr, params.xi0)


def forward_variance_rfheston(
    params,
    t: float | ndarray,
    n_steps: int = 200,
    n_quad: int = 128,
) -> ndarray:
    """Forward variance curve for the rough Heston model.

    Uses the relationship E[V_t] = V₀ E_α(−λ tᵐ) + θ (1 − E_α(−λ tᵐ))
    where E_α is the Mittag-Leffler function (for H < 1/2, α = H + 1/2).

    A simple numerical approximation via the moment formula:
        E[V_t] ≈ θ + (V₀ − θ) · exp(−λ t)   (classical Heston limit)

    For rough Heston (H < 0.5) the mean-reversion speed is modified.
    This function returns the classical approximation as a quick proxy.

    Parameters
    ----------
    params : RHestonParams
    t : float or ndarray

    Returns
    -------
    ndarray
        Approximate forward variance E[V_t].
    """
    t_arr = np.asarray(t, dtype=float)
    lam = params.lambda_
    theta = params.theta
    V0 = params.V0
    # Classical Heston mean-reversion (proxy for rough Heston)
    return theta + (V0 - theta) * np.exp(-lam * t_arr)


def term_structure_of_variance(
    surface,
    use_atm_approx: bool = True,
) -> tuple[ndarray, ndarray]:
    """Extract the term structure of implied variance from a surface.

    Returns the ATM implied variance σ_ATM(T)² as a function of maturity,
    which approximates the term structure of expected integrated variance.

    Parameters
    ----------
    surface : ImpliedVolSurface
    use_atm_approx : bool
        When True, use ATM IV as the variance proxy.

    Returns
    -------
    mats : ndarray, shape (n_T,)
    atm_var : ndarray, shape (n_T,)
        σ_ATM(T)² per maturity.
    """
    sigma_atm = surface.atm_vols()
    return surface.maturities.copy(), sigma_atm ** 2
