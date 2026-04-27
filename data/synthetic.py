"""
Synthetic market data generation for rough-volatility research and testing.

Provides factory functions for creating :class:`ImpliedVolSurface` objects
from model parameters or analytical formulas, as well as CSV I/O helpers.

Quick reference
---------------
>>> from data.synthetic import (
...     make_rfheston_surface,
...     make_rbergomi_surface,
...     make_flat_surface,
...     make_skew_surface,
...     standard_strike_grid,
...     standard_maturity_grid,
... )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rough_vol.calibration.surface import ImpliedVolSurface
from rough_vol.models.rbergomi import RBergomiParams
from rough_vol.models.rbergomi import implied_vol_surface as _rbergomi_ivs
from rough_vol.models.rfheston import RHestonParams
from rough_vol.models.rfheston import implied_vol_surface as _rfheston_ivs


# ---------------------------------------------------------------------------
# Standard grids
# ---------------------------------------------------------------------------

def standard_strike_grid(
    moneyness_lo: float = 0.80,
    moneyness_hi: float = 1.20,
    n_strikes: int = 9,
    spot: float = 1.0,
) -> np.ndarray:
    """Linearly-spaced absolute strike grid.

    Parameters
    ----------
    moneyness_lo, moneyness_hi : float
        Lowest and highest strike as a fraction of spot.
    n_strikes : int
        Number of strikes.
    spot : float
        Spot price.

    Returns
    -------
    ndarray, shape (n_strikes,)
    """
    mono = np.linspace(moneyness_lo, moneyness_hi, n_strikes)
    return mono * spot


def log_strike_grid(
    k_lo: float = -0.20,
    k_hi: float = 0.20,
    n_strikes: int = 9,
    spot: float = 1.0,
) -> np.ndarray:
    """Log-moneyness spaced absolute strike grid K = spot · exp(k).

    Parameters
    ----------
    k_lo, k_hi : float
        Log-moneyness bounds.
    n_strikes : int
    spot : float

    Returns
    -------
    ndarray, shape (n_strikes,)
    """
    k = np.linspace(k_lo, k_hi, n_strikes)
    return spot * np.exp(k)


def standard_maturity_grid(
    short: float = 0.083,   # 1 month
    long: float = 2.0,
    n_mats: int = 6,
    log_spacing: bool = True,
) -> np.ndarray:
    """Standard maturity grid (log-spaced by default for denser short end).

    Parameters
    ----------
    short, long : float
        Shortest and longest maturity in years.
    n_mats : int
    log_spacing : bool
        When True, use log spacing for denser coverage at short maturities.

    Returns
    -------
    ndarray, shape (n_mats,)
    """
    if log_spacing:
        return np.exp(np.linspace(np.log(short), np.log(long), n_mats))
    return np.linspace(short, long, n_mats)


# ---------------------------------------------------------------------------
# Analytical surfaces
# ---------------------------------------------------------------------------

def make_flat_surface(
    sigma: float = 0.20,
    strikes: np.ndarray | None = None,
    maturities: np.ndarray | None = None,
    spot: float = 1.0,
) -> ImpliedVolSurface:
    """Flat (constant) implied-volatility surface.

    Useful as a simple baseline / sanity-check surface.

    Parameters
    ----------
    sigma : float
        Constant implied volatility (0.20 = 20 %).
    strikes : ndarray or None
        Absolute strikes.  Defaults to 9-point log grid ±20 % around spot.
    maturities : ndarray or None
        Maturities in years.  Defaults to standard 6-point log grid.
    spot : float

    Returns
    -------
    ImpliedVolSurface
    """
    if strikes is None:
        strikes = log_strike_grid(spot=spot)
    if maturities is None:
        maturities = standard_maturity_grid()
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    ivols = np.full((len(maturities), len(strikes)), sigma)
    return ImpliedVolSurface(strikes, maturities, ivols, spot=spot)


def make_skew_surface(
    sigma_atm: float = 0.20,
    skew: float = -0.10,
    curvature: float = 0.05,
    strikes: np.ndarray | None = None,
    maturities: np.ndarray | None = None,
    spot: float = 1.0,
    skew_decay: float = 0.5,
) -> ImpliedVolSurface:
    """Parametric skew surface with power-law maturity decay.

    Implied vol is modelled as

        σ(k, T) = σ_ATM + skew · T^{skew_decay − 0.5} · k + curvature · k²

    where k = log(K/F) is log-moneyness.

    Parameters
    ----------
    sigma_atm : float
        ATM implied volatility.
    skew : float
        ATM skew slope (∂σ/∂k at k=0, T=1).  Typically negative.
    curvature : float
        Smile curvature (∂²σ/∂k² at k=0).
    strikes : ndarray or None
    maturities : ndarray or None
    spot : float
    skew_decay : float
        Controls how quickly skew decays with maturity.  Set to H + 0.5
        to mimic rough-vol power law (H = 0.1 → skew_decay = 0.6).

    Returns
    -------
    ImpliedVolSurface
    """
    if strikes is None:
        strikes = log_strike_grid(spot=spot)
    if maturities is None:
        maturities = standard_maturity_grid()
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    n_T, n_K = len(maturities), len(strikes)

    # log-moneyness grid
    k = np.log(strikes / spot)   # (n_K,); forward = spot for r=0

    ivols = np.empty((n_T, n_K))
    for i, T in enumerate(maturities):
        skew_T = skew * T ** (skew_decay - 0.5)
        ivols[i] = sigma_atm + skew_T * k + curvature * k ** 2

    ivols = np.clip(ivols, 1e-4, None)
    return ImpliedVolSurface(strikes, maturities, ivols, spot=spot)


# ---------------------------------------------------------------------------
# Model-generated surfaces
# ---------------------------------------------------------------------------

def make_rbergomi_surface(
    params: RBergomiParams | None = None,
    strikes: np.ndarray | None = None,
    maturities: np.ndarray | None = None,
    spot: float = 1.0,
    n_paths: int = 10_000,
    n_steps_per_year: int = 52,
    rng: int = 42,
) -> ImpliedVolSurface:
    """Implied-vol surface generated by the rough Bergomi model.

    Parameters
    ----------
    params : RBergomiParams or None
        Model parameters.  Defaults to H=0.1, η=1.9, ρ=−0.9, ξ₀=0.04.
    strikes : ndarray or None
        Absolute strikes.  rBergomi treats strikes as moneyness fractions
        (K/spot), so spot=1.0 means strikes ARE moneyness.
    maturities : ndarray or None
    spot : float
    n_paths : int
    n_steps_per_year : int
    rng : int
        Random seed for reproducibility.

    Returns
    -------
    ImpliedVolSurface
    """
    if params is None:
        params = RBergomiParams(H=0.1, eta=1.9, rho=-0.9, xi0=0.04)
    if strikes is None:
        strikes = log_strike_grid(k_lo=-0.15, k_hi=0.15, n_strikes=9, spot=spot)
    if maturities is None:
        maturities = standard_maturity_grid(short=0.25, long=2.0, n_mats=4)
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)

    # rBergomi ivs expects moneyness fractions
    ivols = _rbergomi_ivs(
        params,
        strikes / spot,
        maturities,
        n_paths=n_paths,
        n_steps_per_year=n_steps_per_year,
        rng=rng,
    )
    return ImpliedVolSurface(strikes, maturities, ivols, spot=spot)


def make_rfheston_surface(
    params: RHestonParams | None = None,
    strikes: np.ndarray | None = None,
    maturities: np.ndarray | None = None,
    spot: float = 1.0,
    n_steps: int = 100,
    n_quad: int = 64,
) -> ImpliedVolSurface:
    """Implied-vol surface generated by the rough Heston model.

    The rough Heston pricer is deterministic (no Monte Carlo).

    Parameters
    ----------
    params : RHestonParams or None
        Defaults to H=0.1, λ=1.0, θ=0.04, ρ=−0.7, ν=0.3, V₀=0.04.
    strikes : ndarray or None
    maturities : ndarray or None
    spot : float
    n_steps : int
        Adams-scheme steps for the fractional Riccati solver.
    n_quad : int
        Gauss-Laguerre quadrature nodes.

    Returns
    -------
    ImpliedVolSurface
    """
    if params is None:
        params = RHestonParams(H=0.1, lambda_=1.0, theta=0.04, rho=-0.7, nu=0.3, V0=0.04)
    if strikes is None:
        strikes = log_strike_grid(k_lo=-0.15, k_hi=0.15, n_strikes=9, spot=spot)
    if maturities is None:
        maturities = standard_maturity_grid(short=0.25, long=2.0, n_mats=4)
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)

    ivols = _rfheston_ivs(
        params,
        strikes,
        maturities,
        S0=spot,
        n_steps=n_steps,
        n_quad=n_quad,
    )
    return ImpliedVolSurface(strikes, maturities, ivols, spot=spot)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def surface_to_dataframe(surface: ImpliedVolSurface) -> pd.DataFrame:
    """Convert an ImpliedVolSurface to a tidy long-form DataFrame.

    Returns
    -------
    DataFrame with columns: maturity, strike, implied_vol, log_moneyness
    """
    rows = []
    k_grid = surface.to_log_moneyness()  # (n_T, n_K)
    for i, T in enumerate(surface.maturities):
        for j, K in enumerate(surface.strikes):
            rows.append({
                "maturity": T,
                "strike": K,
                "implied_vol": surface.implied_vols[i, j],
                "log_moneyness": k_grid[i, j],
                "forward": surface.forwards[i],
            })
    return pd.DataFrame(rows)


def save_surface_csv(surface: ImpliedVolSurface, path: str | Path) -> None:
    """Save an ImpliedVolSurface as a tidy CSV file.

    The CSV includes a ``spot`` column so the surface can be round-tripped
    via :func:`load_surface_csv`.

    Parameters
    ----------
    surface : ImpliedVolSurface
    path : str or Path
    """
    df = surface_to_dataframe(surface)
    df["spot"] = surface.spot
    df.to_csv(path, index=False)


def load_surface_csv(path: str | Path, spot: float = 1.0) -> ImpliedVolSurface:
    """Load a surface from a tidy CSV file.

    Expected columns: ``maturity``, ``strike``, ``implied_vol``.
    Optional: ``forward``, ``spot``.

    Parameters
    ----------
    path : str or Path
    spot : float
        Default spot if not in CSV.

    Returns
    -------
    ImpliedVolSurface
    """
    return ImpliedVolSurface.from_csv(path, spot=spot)


def make_benchmark_surfaces() -> dict[str, ImpliedVolSurface]:
    """Generate a small collection of reference surfaces for benchmarking.

    Returns
    -------
    dict mapping name → ImpliedVolSurface:
        ``'flat'``        : flat σ=0.20
        ``'skew'``        : downward-sloping smile
        ``'rfheston_rough'`` : rough Heston H=0.10
        ``'rfheston_smooth'``: rough Heston H=0.40 (near-classical)
    """
    strikes = log_strike_grid(k_lo=-0.15, k_hi=0.15, n_strikes=7)
    mats = standard_maturity_grid(short=0.25, long=1.0, n_mats=3)

    return {
        "flat": make_flat_surface(sigma=0.20, strikes=strikes, maturities=mats),
        "skew": make_skew_surface(
            sigma_atm=0.20, skew=-0.10, curvature=0.03,
            strikes=strikes, maturities=mats,
        ),
        "rfheston_rough": make_rfheston_surface(
            RHestonParams(H=0.10, lambda_=1.0, theta=0.04, rho=-0.7, nu=0.3, V0=0.04),
            strikes=strikes, maturities=mats, n_steps=50, n_quad=32,
        ),
        "rfheston_smooth": make_rfheston_surface(
            RHestonParams(H=0.40, lambda_=2.0, theta=0.04, rho=-0.5, nu=0.3, V0=0.04),
            strikes=strikes, maturities=mats, n_steps=50, n_quad=32,
        ),
    }
