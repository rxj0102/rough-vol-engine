"""
Calibration objective functions for rough-volatility models.

Provides:
  ``compute_weights``      — build calibration weight matrix from a surface
  ``RBergomiObjective``    — WSSE objective for the rough Bergomi model (MC)
  ``RHestonObjective``     — WSSE objective for the rough Heston model (det.)

For backward compatibility the legacy ``weighted_rmse`` / ``weighted_mae``
helper functions are preserved.

Weight schemes
--------------
``'uniform'``
    All valid points weighted equally (w = 1).
``'vega'``
    w = forward-Black vega = F · N′(d₁) · √T.  Down-weights deep OTM
    options that are hard to fit and liquid near-ATM options dominate.
``'relative'``
    w = 1 / σ²(K, T).  Penalises *relative* implied-vol errors uniformly;
    low-vol points receive more weight.

Thread / multiprocessing safety
---------------------------------
Each objective instance carries mutable state (``n_evals``, ``history``).
When used with ``scipy.optimize.differential_evolution(workers > 1)`` the
state is isolated to each worker process and will NOT reflect in the parent.
Always use ``workers=1`` if you need accurate ``n_evals`` or ``history``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy import ndarray
from scipy.stats import norm

from rough_vol.calibration.surface import ImpliedVolSurface
from rough_vol.models.rbergomi import RBergomiParams
from rough_vol.models.rbergomi import implied_vol_surface as rbergomi_ivs
from rough_vol.models.rfheston import RHestonParams
from rough_vol.models.rfheston import implied_vol_surface as rfheston_ivs

_LARGE_LOSS: float = 1e6


# ---------------------------------------------------------------------------
# Backward-compatible loss helpers
# ---------------------------------------------------------------------------

def weighted_rmse(
    model_vols: ndarray,
    market_vols: ndarray,
    weights: ndarray,
    mask: ndarray | None = None,
) -> float:
    """Weighted root-mean-square error between model and market implied vols."""
    m = np.asarray(model_vols, dtype=float)
    v = np.asarray(market_vols, dtype=float)
    w = np.asarray(weights, dtype=float)

    if mask is None:
        mask = np.isfinite(m) & np.isfinite(v) & (w > 0)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(m) & np.isfinite(v) & (w > 0)

    w_sel, err = w[mask], (m - v)[mask]
    total_w = w_sel.sum()
    if total_w <= 0.0:
        return float("inf")
    return float(np.sqrt((w_sel * err ** 2).sum() / total_w))


def weighted_mae(
    model_vols: ndarray,
    market_vols: ndarray,
    weights: ndarray,
    mask: ndarray | None = None,
) -> float:
    """Weighted mean-absolute error between model and market implied vols."""
    m = np.asarray(model_vols, dtype=float)
    v = np.asarray(market_vols, dtype=float)
    w = np.asarray(weights, dtype=float)

    if mask is None:
        mask = np.isfinite(m) & np.isfinite(v) & (w > 0)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(m) & np.isfinite(v) & (w > 0)

    w_sel, err = w[mask], np.abs((m - v)[mask])
    total_w = w_sel.sum()
    if total_w <= 0.0:
        return float("inf")
    return float((w_sel * err).sum() / total_w)


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def compute_weights(
    surface: ImpliedVolSurface,
    scheme: Literal["uniform", "vega", "relative"],
) -> ndarray:
    """Compute a calibration weight matrix from a surface.

    Parameters
    ----------
    surface : ImpliedVolSurface
    scheme : {'uniform', 'vega', 'relative'}

    Returns
    -------
    ndarray, shape (n_T, n_K)
        Non-negative weights; 0 where the IV is NaN or non-positive.
    """
    ivols = surface.implied_vols  # (n_T, n_K)
    valid = np.isfinite(ivols) & (ivols > 0)

    if scheme == "uniform":
        return np.where(valid, 1.0, 0.0)

    if scheme == "vega":
        # forward-Black vega = F * N'(d1) * sqrt(T)
        n_T, n_K = ivols.shape
        weights = np.zeros((n_T, n_K))
        for i in range(n_T):
            T = surface.maturities[i]
            F = surface.forwards[i]
            sqrtT = np.sqrt(T)
            for j in range(n_K):
                if not valid[i, j]:
                    continue
                sigma = ivols[i, j]
                K = surface.strikes[j]
                d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
                weights[i, j] = F * float(norm.pdf(d1)) * sqrtT
        return weights

    if scheme == "relative":
        # w = 1/sigma^2
        return np.where(valid, 1.0 / ivols ** 2, 0.0)

    raise ValueError(f"Unknown weight scheme '{scheme}'; choose 'uniform', 'vega', or 'relative'")


# ---------------------------------------------------------------------------
# Core loss: weighted sum of squared errors
# ---------------------------------------------------------------------------

def _wsse(
    model_vols: ndarray,
    market_vols: ndarray,
    weights: ndarray,
) -> float:
    """Weighted sum of squared errors Σ w·(σ_model − σ_market)²."""
    m = np.asarray(model_vols, dtype=float)
    v = np.asarray(market_vols, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(m) & np.isfinite(v) & (w > 0)
    if not valid.any():
        return float("inf")
    err = (m - v)[valid]
    return float((w[valid] * err ** 2).sum())


# ---------------------------------------------------------------------------
# Parameter bounds (shared constants)
# ---------------------------------------------------------------------------

_RBERGOMI_BOUNDS: list[tuple[float, float]] = [
    (0.01, 0.49),   # H
    (0.05, 5.00),   # eta
    (-0.99, 0.99),  # rho
    (1e-4, 2.00),   # xi0
]
_RBERGOMI_NAMES = ("H", "eta", "rho", "xi0")

_RHESTON_BOUNDS: list[tuple[float, float]] = [
    (0.01, 0.49),   # H
    (0.05, 10.0),   # lambda_
    (1e-4, 0.50),   # theta
    (-0.99, 0.99),  # rho
    (0.05, 2.00),   # nu
    (1e-4, 0.50),   # V0
]
_RHESTON_NAMES = ("H", "lambda_", "theta", "rho", "nu", "V0")


# ---------------------------------------------------------------------------
# rBergomi objective
# ---------------------------------------------------------------------------

class RBergomiObjective:
    """WSSE calibration objective for the rough Bergomi model.

    Evaluates Σ w·(σ_model − σ_market)² where σ_model comes from an MC
    simulation seeded with a fixed ``rng_seed``, making the objective
    deterministic.

    Parameters
    ----------
    market : ImpliedVolSurface
        Market implied-volatility surface.
    weight_scheme : {'uniform', 'vega', 'relative'}
        How to assign calibration weights.
    n_paths : int
        Monte Carlo paths per evaluation.
    n_steps_per_year : int
        Time-discretisation steps per year.
    rng_seed : int
        Fixed RNG seed — same seed → same paths → deterministic loss.
    """

    bounds: list[tuple[float, float]] = _RBERGOMI_BOUNDS
    param_names: tuple[str, ...] = _RBERGOMI_NAMES

    def __init__(
        self,
        market: ImpliedVolSurface,
        weight_scheme: Literal["uniform", "vega", "relative"] = "uniform",
        n_paths: int = 5_000,
        n_steps_per_year: int = 52,
        rng_seed: int = 42,
    ) -> None:
        self.market = market
        self.weight_scheme = weight_scheme
        self.n_paths = n_paths
        self.n_steps_per_year = n_steps_per_year
        self.rng_seed = rng_seed

        self._weights = compute_weights(market, weight_scheme)
        self._market_vols = market.implied_vols
        self._maturities = market.maturities
        # rBergomi treats S0=1; strikes must be expressed as moneyness
        self._strikes_mono = market.strikes / market.spot

        self.n_evals: int = 0
        self._best: float = _LARGE_LOSS
        self.history: list[float] = []

    # ------------------------------------------------------------------
    # Parameter encoding
    # ------------------------------------------------------------------

    @staticmethod
    def to_params(theta: ndarray) -> RBergomiParams:
        """Convert vector [H, eta, rho, xi0] → ``RBergomiParams``."""
        H, eta, rho, xi0 = (float(theta[i]) for i in range(4))
        return RBergomiParams(H=H, eta=eta, rho=rho, xi0=xi0)

    @staticmethod
    def from_params(params: RBergomiParams) -> ndarray:
        """Convert ``RBergomiParams`` → parameter vector."""
        return np.array([params.H, params.eta, params.rho, params.xi0])

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _in_bounds(self, theta: ndarray) -> bool:
        return all(lo <= float(theta[i]) <= hi for i, (lo, hi) in enumerate(self.bounds))

    def __call__(self, theta: ndarray) -> float:
        """Evaluate the WSSE loss at parameter vector *theta*.

        Parameters
        ----------
        theta : array-like, shape (4,)
            [H, eta, rho, xi0]

        Returns
        -------
        float
            WSSE in implied-vol squared units.  Returns ``_LARGE_LOSS`` for
            invalid parameters or failed evaluations.
        """
        theta = np.asarray(theta, dtype=float)
        if not self._in_bounds(theta):
            return _LARGE_LOSS

        try:
            params = self.to_params(theta)
        except ValueError:
            return _LARGE_LOSS

        try:
            model_vols = rbergomi_ivs(
                params,
                self._strikes_mono,
                self._maturities,
                n_paths=self.n_paths,
                n_steps_per_year=self.n_steps_per_year,
                rng=self.rng_seed,
            )
        except Exception:
            return _LARGE_LOSS

        loss = _wsse(model_vols, self._market_vols, self._weights)

        self.n_evals += 1
        if loss < self._best:
            self._best = loss
        self.history.append(self._best)
        return loss


# ---------------------------------------------------------------------------
# rough Heston objective
# ---------------------------------------------------------------------------

class RHestonObjective:
    """WSSE calibration objective for the rough Heston model.

    The rough Heston pricer is fully deterministic (Adams predictor-corrector
    + Fourier inversion), so no MC noise is introduced.

    Parameters
    ----------
    market : ImpliedVolSurface
        Market implied-volatility surface.
    weight_scheme : {'uniform', 'vega', 'relative'}
        How to assign calibration weights.
    n_steps : int
        Adams-scheme steps for the fractional Riccati solver.
    n_quad : int
        Gauss-Laguerre quadrature nodes for Fourier inversion.
    """

    bounds: list[tuple[float, float]] = _RHESTON_BOUNDS
    param_names: tuple[str, ...] = _RHESTON_NAMES

    def __init__(
        self,
        market: ImpliedVolSurface,
        weight_scheme: Literal["uniform", "vega", "relative"] = "uniform",
        n_steps: int = 100,
        n_quad: int = 64,
    ) -> None:
        self.market = market
        self.weight_scheme = weight_scheme
        self.n_steps = n_steps
        self.n_quad = n_quad

        self._weights = compute_weights(market, weight_scheme)
        self._market_vols = market.implied_vols
        self._maturities = market.maturities
        self._strikes = market.strikes
        self._spot = market.spot

        self.n_evals: int = 0
        self._best: float = _LARGE_LOSS
        self.history: list[float] = []

    @staticmethod
    def to_params(theta: ndarray) -> RHestonParams:
        """Convert [H, lambda_, theta_lr, rho, nu, V0] → ``RHestonParams``."""
        H, lam, th, rho, nu, V0 = (float(theta[i]) for i in range(6))
        return RHestonParams(H=H, lambda_=lam, theta=th, rho=rho, nu=nu, V0=V0)

    @staticmethod
    def from_params(params: RHestonParams) -> ndarray:
        return np.array([
            params.H, params.lambda_, params.theta,
            params.rho, params.nu, params.V0,
        ])

    def _in_bounds(self, theta: ndarray) -> bool:
        return all(lo <= float(theta[i]) <= hi for i, (lo, hi) in enumerate(self.bounds))

    def __call__(self, theta: ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        if not self._in_bounds(theta):
            return _LARGE_LOSS

        try:
            params = self.to_params(theta)
        except ValueError:
            return _LARGE_LOSS

        try:
            model_vols = rfheston_ivs(
                params,
                self._strikes,
                self._maturities,
                S0=self._spot,
                n_steps=self.n_steps,
                n_quad=self.n_quad,
            )
        except Exception:
            return _LARGE_LOSS

        loss = _wsse(model_vols, self._market_vols, self._weights)

        self.n_evals += 1
        if loss < self._best:
            self._best = loss
        self.history.append(self._best)
        return loss
