"""
Two-stage calibration optimiser for rough-volatility models.

Stage 1 — Differential Evolution (DE)
    A population-based global search that explores the full parameter space
    without gradient information.  Robust to local minima in the objective.

Stage 2 — Nelder-Mead polish
    A simplex-based local search seeded from DE's best solution.  Tightens
    the solution at low additional cost after the global search has found a
    good basin.

Entry points
------------
``calibrate_rbergomi``  — fits rBergomi (H, η, ρ, ξ₀)
``calibrate_rfheston``  — fits rough Heston (H, λ, θ, ρ, ν, V₀)

References
----------
Storn, R., & Price, K. (1997). Differential evolution — a simple and
    efficient heuristic for global optimization over continuous spaces.
    Journal of Global Optimization, 11(4), 341–359.
Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization.
    Computer Journal, 7(4), 308–313.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution, minimize

from rough_vol.calibration.objective import RBergomiObjective, RHestonObjective
from rough_vol.calibration.surface import VolSurface
from rough_vol.models.rbergomi import RBergomiParams
from rough_vol.models.rfheston import RHestonParams


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Result of a two-stage calibration run.

    Attributes
    ----------
    params : RBergomiParams or RHestonParams
        Best-fit model parameters.
    loss : float
        Objective value (weighted RMSE or MAE) at the best-fit parameters.
    n_evals : int
        Total number of objective evaluations (DE + polish combined; only
        accurate when ``workers=1``).
    success : bool
        ``True`` when DE reported convergence within its tolerance.
    method : str
        Description of the optimisation stages applied.
    message : str
        Termination message from the DE solver.
    history : list[float]
        Running-minimum loss trace across all evaluations.  Useful for
        convergence diagnostics.  Empty when ``workers > 1``.
    de_result : OptimizeResult
        Raw scipy DE result object (for advanced diagnostics).
    """

    params: RBergomiParams | RHestonParams
    loss: float
    n_evals: int
    success: bool
    method: str
    message: str
    history: list[float] = field(default_factory=list)
    de_result: OptimizeResult | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"CalibrationResult(\n"
            f"  params={self.params},\n"
            f"  loss={self.loss:.6f},\n"
            f"  n_evals={self.n_evals},\n"
            f"  success={self.success},\n"
            f"  method='{self.method}'\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clip_to_bounds(
    x: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return np.clip(x, lo, hi)


def _nelder_mead_polish(
    obj: RBergomiObjective | RHestonObjective,
    x0: np.ndarray,
    maxiter: int,
    fatol: float,
    xatol: float,
) -> OptimizeResult:
    return minimize(
        obj,
        x0=x0,
        method="Nelder-Mead",
        bounds=obj.bounds,
        options={
            "maxiter": maxiter,
            "fatol": fatol,
            "xatol": xatol,
            "adaptive": True,
        },
    )


# ---------------------------------------------------------------------------
# rBergomi calibration
# ---------------------------------------------------------------------------

def calibrate_rbergomi(
    surface: VolSurface,
    *,
    n_paths: int = 5_000,
    n_steps_per_year: int = 52,
    rng_seed: int = 42,
    metric: Literal["rmse", "mae"] = "rmse",
    # Differential Evolution
    de_popsize: int = 10,
    de_maxiter: int = 300,
    de_seed: int = 0,
    de_tol: float = 1e-6,
    de_mutation: float | tuple[float, float] = (0.5, 1.0),
    de_recombination: float = 0.7,
    # Nelder-Mead polish
    polish: bool = True,
    polish_maxiter: int = 500,
    polish_fatol: float = 1e-7,
    polish_xatol: float = 1e-6,
    # Misc
    workers: int = 1,
    callback: Callable | None = None,
) -> CalibrationResult:
    """Calibrate the rough Bergomi model to a market implied-vol surface.

    Uses a two-stage approach:

    1. **Differential Evolution** — global search over
       ``H ∈ (0.01, 0.49)``, ``η ∈ (0.05, 5)``,
       ``ρ ∈ (−0.99, 0.99)``, ``ξ₀ ∈ (0.0001, 2)``.
    2. **Nelder-Mead polish** — local refinement from DE's best solution.

    Parameters
    ----------
    surface : VolSurface
        Market implied-volatility surface (strikes as moneyness, IVs as
        decimals).
    n_paths : int
        MC paths per objective evaluation.
    n_steps_per_year : int
        Path-simulation time steps per year.
    rng_seed : int
        Fixed seed making the MC objective deterministic.
    metric : {'rmse', 'mae'}
        Calibration loss function.
    de_popsize : int
        DE population multiplier (actual size = ``de_popsize * n_params``).
    de_maxiter : int
        Maximum DE generations.
    de_seed : int
        RNG seed for DE's mutation/selection.
    de_tol : float
        Relative convergence tolerance for DE.
    de_mutation : float or (float, float)
        DE mutation constant F, or dithering range.
    de_recombination : float
        DE crossover probability CR ∈ (0, 1].
    polish : bool
        Whether to apply Nelder-Mead after DE.
    polish_maxiter : int
        Maximum Nelder-Mead iterations.
    polish_fatol, polish_xatol : float
        Nelder-Mead absolute tolerances on function value and parameter step.
    workers : int
        Parallel workers for DE (``-1`` = all CPUs). State tracking
        (``n_evals``, ``history``) is only accurate for ``workers=1``.
    callback : callable or None
        Called after each DE generation as ``callback(xk, convergence)``.

    Returns
    -------
    CalibrationResult
    """
    obj = RBergomiObjective(
        surface,
        n_paths=n_paths,
        n_steps_per_year=n_steps_per_year,
        rng_seed=rng_seed,
        metric=metric,
    )

    de_res = differential_evolution(
        obj,
        bounds=obj.bounds,
        maxiter=de_maxiter,
        popsize=de_popsize,
        seed=de_seed,
        tol=de_tol,
        mutation=de_mutation,
        recombination=de_recombination,
        init="latinhypercube",
        workers=workers,
        callback=callback,
        polish=False,
    )

    best_x = de_res.x.copy()
    best_loss = float(de_res.fun)

    method_str = "differential_evolution"
    if polish:
        nm_res = _nelder_mead_polish(obj, best_x, polish_maxiter, polish_fatol, polish_xatol)
        if nm_res.fun < best_loss:
            best_x = nm_res.x.copy()
            best_loss = float(nm_res.fun)
        method_str += "+nelder-mead"

    best_x_clipped = _clip_to_bounds(best_x, obj.bounds)
    try:
        params = obj.to_params(best_x_clipped)
    except ValueError:
        params = obj.to_params(_clip_to_bounds(best_x, obj.bounds))

    return CalibrationResult(
        params=params,
        loss=best_loss,
        n_evals=obj.n_evals,
        success=bool(de_res.success),
        method=method_str,
        message=str(de_res.message),
        history=obj.history,
        de_result=de_res,
    )


# ---------------------------------------------------------------------------
# rough Heston calibration
# ---------------------------------------------------------------------------

def calibrate_rfheston(
    surface: VolSurface,
    *,
    metric: Literal["rmse", "mae"] = "rmse",
    n_steps: int = 100,
    n_quad: int = 64,
    # Differential Evolution
    de_popsize: int = 10,
    de_maxiter: int = 300,
    de_seed: int = 0,
    de_tol: float = 1e-6,
    de_mutation: float | tuple[float, float] = (0.5, 1.0),
    de_recombination: float = 0.7,
    # Nelder-Mead polish
    polish: bool = True,
    polish_maxiter: int = 500,
    polish_fatol: float = 1e-7,
    polish_xatol: float = 1e-6,
    # Misc
    workers: int = 1,
    callback: Callable | None = None,
) -> CalibrationResult:
    """Calibrate the rough Heston model to a market implied-vol surface.

    Fits six parameters:
    ``H ∈ (0.01, 0.49)``, ``λ ∈ (0.05, 10)``, ``θ ∈ (0.0001, 0.5)``,
    ``ρ ∈ (−0.99, 0.99)``, ``ν ∈ (0.05, 2)``, ``V₀ ∈ (0.0001, 0.5)``.

    The rough Heston pricer is deterministic (no MC), so the objective is
    noise-free and typically converges faster than the rBergomi version.

    Parameters
    ----------
    surface : VolSurface
        Market implied-volatility surface.
    metric : {'rmse', 'mae'}
        Loss function.
    n_steps : int
        Adams-scheme time steps for the fractional Riccati solver.
    n_quad : int
        Gauss-Laguerre quadrature nodes for Fourier inversion.
    de_popsize, de_maxiter, de_seed, de_tol : int / float
        Differential Evolution hyper-parameters.
    de_mutation, de_recombination : float
        DE mutation and crossover constants.
    polish : bool
        Apply Nelder-Mead after DE.
    polish_maxiter, polish_fatol, polish_xatol : int / float
        Nelder-Mead stopping criteria.
    workers : int
        Parallel workers for DE.
    callback : callable or None
        Per-generation callback.

    Returns
    -------
    CalibrationResult
    """
    obj = RHestonObjective(
        surface,
        n_steps=n_steps,
        n_quad=n_quad,
        metric=metric,
    )

    de_res = differential_evolution(
        obj,
        bounds=obj.bounds,
        maxiter=de_maxiter,
        popsize=de_popsize,
        seed=de_seed,
        tol=de_tol,
        mutation=de_mutation,
        recombination=de_recombination,
        init="latinhypercube",
        workers=workers,
        callback=callback,
        polish=False,
    )

    best_x = de_res.x.copy()
    best_loss = float(de_res.fun)

    method_str = "differential_evolution"
    if polish:
        nm_res = _nelder_mead_polish(obj, best_x, polish_maxiter, polish_fatol, polish_xatol)
        if nm_res.fun < best_loss:
            best_x = nm_res.x.copy()
            best_loss = float(nm_res.fun)
        method_str += "+nelder-mead"

    best_x_clipped = _clip_to_bounds(best_x, obj.bounds)
    try:
        params = obj.to_params(best_x_clipped)
    except ValueError:
        params = obj.to_params(_clip_to_bounds(best_x, obj.bounds))

    return CalibrationResult(
        params=params,
        loss=best_loss,
        n_evals=obj.n_evals,
        success=bool(de_res.success),
        method=method_str,
        message=str(de_res.message),
        history=obj.history,
        de_result=de_res,
    )
