"""
Calibration objective functions for rough-volatility models.

Provides ``weighted_rmse`` / ``weighted_mae`` loss functions and two
callable objective classes — ``RBergomiObjective`` and ``RHestonObjective``
— that wrap the respective model pricers and expose a scalar loss suitable
for consumption by scipy optimisers.

Thread / multiprocessing safety
--------------------------------
Each objective instance carries mutable state (``n_evals``, ``history``).
When used with ``scipy.optimize.differential_evolution(workers > 1)`` the
state is isolated to each worker process and will NOT reflect in the parent.
Always use ``workers=1`` (the default in this package) if you need accurate
``n_evals`` or ``history`` tracking.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy import ndarray

from rough_vol.calibration.surface import VolSurface
from rough_vol.models.rbergomi import RBergomiParams
from rough_vol.models.rbergomi import implied_vol_surface as rbergomi_ivs
from rough_vol.models.rfheston import RHestonParams
from rough_vol.models.rfheston import implied_vol_surface as rfheston_ivs

# Returned when the model evaluation fails or parameters are out of bounds.
_LARGE_LOSS: float = 10.0


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def weighted_rmse(
    model_vols: ndarray,
    market_vols: ndarray,
    weights: ndarray,
    mask: ndarray | None = None,
) -> float:
    """Weighted root-mean-square error between model and market implied vols.

    Parameters
    ----------
    model_vols, market_vols : ndarray, same shape
        Model and market implied volatilities (decimals, e.g. 0.20 = 20 %).
    weights : ndarray, same shape
        Non-negative calibration weights.
    mask : bool ndarray or None
        Additional validity mask.  When *None*, points where either surface
        is non-finite or the weight is zero are excluded automatically.

    Returns
    -------
    float
        WRMS error, or ``inf`` when no valid points remain.
    """
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
# rBergomi objective
# ---------------------------------------------------------------------------

# Parameter order: H, eta, rho, xi0
_RBERGOMI_BOUNDS: list[tuple[float, float]] = [
    (0.01, 0.49),   # H
    (0.05, 5.00),   # eta
    (-0.99, 0.99),  # rho
    (1e-4, 2.00),   # xi0
]
_RBERGOMI_NAMES = ("H", "eta", "rho", "xi0")


class RBergomiObjective:
    """Callable objective for rough Bergomi model calibration.

    Evaluates the weighted-RMSE (or MAE) between the model-implied vol
    surface and the market surface stored in ``surface``.

    Parameters
    ----------
    surface : VolSurface
        Market implied-volatility surface.
    n_paths : int
        Monte Carlo paths per evaluation.  More paths → less noise but
        slower evaluations.  Recommended ≥ 2 000 for production.
    n_steps_per_year : int
        Time-discretisation steps per year for path simulation.
    rng_seed : int
        Fixed RNG seed.  The same seed is used on every call, making the
        objective deterministic (same θ → same loss).
    metric : {'rmse', 'mae'}
        Loss function.
    """

    bounds: list[tuple[float, float]] = _RBERGOMI_BOUNDS
    param_names: tuple[str, ...] = _RBERGOMI_NAMES

    def __init__(
        self,
        surface: VolSurface,
        n_paths: int = 5_000,
        n_steps_per_year: int = 52,
        rng_seed: int = 42,
        metric: Literal["rmse", "mae"] = "rmse",
    ) -> None:
        self.surface = surface
        self.n_paths = n_paths
        self.n_steps_per_year = n_steps_per_year
        self.rng_seed = rng_seed
        self.metric = metric

        mats, strikes, ivols, weights = surface.to_arrays()
        self._maturities = mats
        self._strikes = strikes
        self._market_vols = ivols
        self._weights = weights
        self._mask = np.isfinite(ivols) & (weights > 0)

        # Mutable tracking (single-process only)
        self.n_evals: int = 0
        self._best: float = _LARGE_LOSS
        self.history: list[float] = []  # running-minimum trace

    # ------------------------------------------------------------------
    # Parameter encoding / decoding
    # ------------------------------------------------------------------

    @staticmethod
    def to_params(theta: ndarray) -> RBergomiParams:
        """Convert parameter vector [H, eta, rho, xi0] to ``RBergomiParams``."""
        H, eta, rho, xi0 = float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3])
        return RBergomiParams(H=H, eta=eta, rho=rho, xi0=xi0)

    @staticmethod
    def from_params(params: RBergomiParams) -> ndarray:
        """Convert ``RBergomiParams`` to parameter vector."""
        return np.array([params.H, params.eta, params.rho, params.xi0])

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _in_bounds(self, theta: ndarray) -> bool:
        return all(lo <= float(theta[i]) <= hi for i, (lo, hi) in enumerate(self.bounds))

    def __call__(self, theta: ndarray) -> float:
        """Evaluate the calibration loss at parameter vector *theta*.

        Parameters
        ----------
        theta : array-like, shape (4,)
            [H, eta, rho, xi0]

        Returns
        -------
        float
            Weighted RMSE (or MAE) in implied-vol units.  Returns
            ``_LARGE_LOSS`` for invalid parameters.
        """
        theta = np.asarray(theta, dtype=float)
        if not self._in_bounds(theta):
            return _LARGE_LOSS

        try:
            params = self.to_params(theta)
        except ValueError:
            return _LARGE_LOSS

        model_vols = rbergomi_ivs(
            params,
            self._strikes,
            self._maturities,
            n_paths=self.n_paths,
            n_steps_per_year=self.n_steps_per_year,
            rng=self.rng_seed,
        )

        if self.metric == "rmse":
            loss = weighted_rmse(model_vols, self._market_vols, self._weights, self._mask)
        else:
            loss = weighted_mae(model_vols, self._market_vols, self._weights, self._mask)

        self.n_evals += 1
        if loss < self._best:
            self._best = loss
        self.history.append(self._best)
        return loss


# ---------------------------------------------------------------------------
# rough Heston objective
# ---------------------------------------------------------------------------

# Parameter order: H, lambda_, theta, rho, nu, V0
_RHESTON_BOUNDS: list[tuple[float, float]] = [
    (0.01, 0.49),   # H
    (0.05, 10.0),   # lambda_
    (1e-4, 0.50),   # theta  (long-run variance)
    (-0.99, 0.99),  # rho
    (0.05, 2.00),   # nu
    (1e-4, 0.50),   # V0
]
_RHESTON_NAMES = ("H", "lambda_", "theta", "rho", "nu", "V0")


class RHestonObjective:
    """Callable objective for rough Heston model calibration.

    The rough Heston pricer is deterministic (Adams predictor-corrector +
    Fourier inversion), so this objective is noise-free and converges faster
    than the MC-based rBergomi counterpart.

    Parameters
    ----------
    surface : VolSurface
        Market implied-volatility surface.
    n_steps : int
        Adams-scheme steps for the fractional Riccati solver.
    n_quad : int
        Gauss-Laguerre quadrature nodes for Fourier inversion.
    metric : {'rmse', 'mae'}
        Loss function.
    """

    bounds: list[tuple[float, float]] = _RHESTON_BOUNDS
    param_names: tuple[str, ...] = _RHESTON_NAMES

    def __init__(
        self,
        surface: VolSurface,
        n_steps: int = 100,
        n_quad: int = 64,
        metric: Literal["rmse", "mae"] = "rmse",
    ) -> None:
        self.surface = surface
        self.n_steps = n_steps
        self.n_quad = n_quad
        self.metric = metric

        mats, strikes, ivols, weights = surface.to_arrays()
        self._maturities = mats
        self._strikes = strikes
        self._market_vols = ivols
        self._weights = weights
        self._mask = np.isfinite(ivols) & (weights > 0)

        self.n_evals: int = 0
        self._best: float = _LARGE_LOSS
        self.history: list[float] = []

    @staticmethod
    def to_params(theta: ndarray) -> RHestonParams:
        """Convert [H, lambda_, theta_lr, rho, nu, V0] to ``RHestonParams``."""
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
                n_steps=self.n_steps,
                n_quad=self.n_quad,
            )
        except Exception:
            return _LARGE_LOSS

        if self.metric == "rmse":
            loss = weighted_rmse(model_vols, self._market_vols, self._weights, self._mask)
        else:
            loss = weighted_mae(model_vols, self._market_vols, self._weights, self._mask)

        self.n_evals += 1
        if loss < self._best:
            self._best = loss
        self.history.append(self._best)
        return loss
