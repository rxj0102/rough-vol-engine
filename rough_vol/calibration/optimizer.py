"""
Calibration optimisers for rough-volatility models.

``RoughVolCalibrator``
    Unified class supporting three calibration strategies (differential
    evolution, Nelder-Mead, and the two-stage DE → NM combination) plus
    a static H-estimation utility from the ATM skew power law.

``calibrate_rbergomi`` / ``calibrate_rfheston``
    Convenience wrappers that perform two-stage calibration directly.

ATM skew power law
------------------
In rough-vol models the ATM skew satisfies

    |∂σ_ATM / ∂k| ∝ T^{H − 1/2}   as T → 0

so a log–log regression of observed skews on maturities gives slope
H − 1/2, from which H can be read off.

References
----------
Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is rough.
    Quantitative Finance, 18(6), 933–949.
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
from rough_vol.calibration.surface import ImpliedVolSurface
from rough_vol.models.rbergomi import RBergomiParams
from rough_vol.models.rfheston import RHestonParams


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Result of a calibration run.

    Attributes
    ----------
    params : RBergomiParams or RHestonParams
        Best-fit model parameters.
    loss : float
        Objective value (WSSE) at the best-fit parameters.
    n_evals : int
        Total number of objective evaluations (only accurate for
        ``workers=1``).
    success : bool
        ``True`` when the primary solver reported convergence.
    method : str
        Name of the optimisation strategy used.
    message : str
        Termination message from the primary solver.
    history : list[float]
        Running-minimum loss trace across all evaluations.  Useful for
        convergence diagnostics.  Empty when ``workers > 1``.
    de_result : OptimizeResult or None
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
# rBergomi calibration (convenience wrapper)
# ---------------------------------------------------------------------------

def calibrate_rbergomi(
    surface: ImpliedVolSurface,
    *,
    n_paths: int = 5_000,
    n_steps_per_year: int = 52,
    rng_seed: int = 42,
    weight_scheme: Literal["uniform", "vega", "relative"] = "uniform",
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
    surface : ImpliedVolSurface
        Market implied-volatility surface.
    n_paths : int
        MC paths per objective evaluation.
    n_steps_per_year : int
        Path-simulation time steps per year.
    rng_seed : int
        Fixed seed making the MC objective deterministic.
    weight_scheme : {'uniform', 'vega', 'relative'}
        Calibration weight scheme.
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
        weight_scheme=weight_scheme,
        n_paths=n_paths,
        n_steps_per_year=n_steps_per_year,
        rng_seed=rng_seed,
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
# rough Heston calibration (convenience wrapper)
# ---------------------------------------------------------------------------

def calibrate_rfheston(
    surface: ImpliedVolSurface,
    *,
    weight_scheme: Literal["uniform", "vega", "relative"] = "uniform",
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
    surface : ImpliedVolSurface
        Market implied-volatility surface.
    weight_scheme : {'uniform', 'vega', 'relative'}
        Calibration weight scheme.
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
        weight_scheme=weight_scheme,
        n_steps=n_steps,
        n_quad=n_quad,
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
# RoughVolCalibrator — unified class API
# ---------------------------------------------------------------------------

class RoughVolCalibrator:
    """Calibrate a rough-volatility model to a market implied-vol surface.

    Wraps :class:`RBergomiObjective` / :class:`RHestonObjective` and exposes
    three search strategies through a single :meth:`calibrate` entry point.
    A static :meth:`estimate_H_from_skew` utility provides a data-driven
    initial guess for the Hurst exponent.

    Parameters
    ----------
    surface : ImpliedVolSurface
        Market implied-volatility surface.
    model : {'rbergomi', 'rfheston'}
        Which rough-vol model to calibrate.
    weight_scheme : {'uniform', 'vega', 'relative'}
        Calibration weight scheme; forwarded to the objective.
    n_paths : int
        MC paths per evaluation (rBergomi only).
    n_steps_per_year : int
        Time-discretisation steps per year (rBergomi only).
    rng_seed : int
        Fixed RNG seed for deterministic MC (rBergomi only).
    n_steps : int
        Adams-scheme steps for the fractional Riccati solver (rfHeston only).
    n_quad : int
        Gauss-Laguerre quadrature nodes for Fourier inversion (rfHeston only).

    Attributes
    ----------
    objective : RBergomiObjective or RHestonObjective
        The underlying callable objective (read-only).
    """

    def __init__(
        self,
        surface: ImpliedVolSurface,
        model: Literal["rbergomi", "rfheston"] = "rfheston",
        weight_scheme: Literal["uniform", "vega", "relative"] = "uniform",
        # rBergomi-specific
        n_paths: int = 5_000,
        n_steps_per_year: int = 52,
        rng_seed: int = 42,
        # rfHeston-specific
        n_steps: int = 100,
        n_quad: int = 64,
    ) -> None:
        self.surface = surface
        self.model = model
        self.weight_scheme = weight_scheme

        if model == "rbergomi":
            self._obj: RBergomiObjective | RHestonObjective = RBergomiObjective(
                surface,
                weight_scheme=weight_scheme,
                n_paths=n_paths,
                n_steps_per_year=n_steps_per_year,
                rng_seed=rng_seed,
            )
        elif model == "rfheston":
            self._obj = RHestonObjective(
                surface,
                weight_scheme=weight_scheme,
                n_steps=n_steps,
                n_quad=n_quad,
            )
        else:
            raise ValueError(
                f"Unknown model '{model}'; choose 'rbergomi' or 'rfheston'"
            )

    @property
    def objective(self) -> RBergomiObjective | RHestonObjective:
        return self._obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset objective evaluation counters before a fresh calibration."""
        obj = self._obj
        obj.n_evals = 0
        obj._best = float("inf")
        obj.history = []

    def _midpoint_x0(self) -> np.ndarray:
        """Parameter-space midpoint — fallback starting point for NM."""
        return np.array([0.5 * (lo + hi) for lo, hi in self._obj.bounds])

    # ------------------------------------------------------------------
    # calibrate
    # ------------------------------------------------------------------

    def calibrate(
        self,
        method: Literal[
            "differential_evolution", "nelder_mead", "two_stage"
        ] = "two_stage",
        x0: np.ndarray | None = None,
        *,
        # Differential Evolution
        de_popsize: int = 10,
        de_maxiter: int = 300,
        de_seed: int = 0,
        de_tol: float = 1e-6,
        de_mutation: float | tuple[float, float] = (0.5, 1.0),
        de_recombination: float = 0.7,
        # Nelder-Mead
        nm_maxiter: int = 500,
        nm_fatol: float = 1e-7,
        nm_xatol: float = 1e-6,
        # Misc
        workers: int = 1,
        callback: Callable | None = None,
    ) -> CalibrationResult:
        """Run calibration with the chosen search strategy.

        Parameters
        ----------
        method : {'differential_evolution', 'nelder_mead', 'two_stage'}
            Optimisation strategy.

            ``'differential_evolution'``
                Global stochastic search; robust to local minima.  No
                initial guess required.
            ``'nelder_mead'``
                Local simplex search from ``x0``.  Fast when the basin of
                the global minimum is known.  Defaults to the
                parameter-space midpoint when ``x0`` is *None*.
            ``'two_stage'``
                DE global search followed by Nelder-Mead polish of the
                best DE solution.  Recommended for production use.

        x0 : array-like, shape (n_params,) or None
            Starting point for ``'nelder_mead'``.  Ignored by
            ``'differential_evolution'`` and ``'two_stage'``.
        de_popsize : int
            DE population multiplier (actual size = ``de_popsize × n_params``).
        de_maxiter : int
            Maximum DE generations.
        de_seed : int
            RNG seed for DE mutation / selection.
        de_tol : float
            Relative convergence tolerance for DE.
        de_mutation : float or (float, float)
            DE mutation constant F, or dithering range [F_lo, F_hi].
        de_recombination : float
            DE crossover probability CR ∈ (0, 1].
        nm_maxiter : int
            Maximum Nelder-Mead iterations.
        nm_fatol, nm_xatol : float
            NM absolute tolerances on function value and parameter step.
        workers : int
            Parallel workers for DE (``-1`` = all CPUs).  History tracking
            is only accurate with ``workers=1``.
        callback : callable or None
            Per-generation callback for DE: ``callback(xk, convergence)``.

        Returns
        -------
        CalibrationResult
        """
        valid_methods = {"differential_evolution", "nelder_mead", "two_stage"}
        if method not in valid_methods:
            raise ValueError(
                f"Unknown method '{method}'; choose one of {sorted(valid_methods)}"
            )

        self._reset_state()
        obj = self._obj

        if method == "differential_evolution":
            return self._run_de(
                obj, de_popsize, de_maxiter, de_seed, de_tol,
                de_mutation, de_recombination, workers, callback,
            )

        if method == "nelder_mead":
            if x0 is None:
                x0 = self._midpoint_x0()
            x0_arr = _clip_to_bounds(np.asarray(x0, dtype=float), obj.bounds)
            return self._run_nm(obj, x0_arr, nm_maxiter, nm_fatol, nm_xatol)

        # two_stage: DE global search → NM polish
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

        nm_res = _nelder_mead_polish(obj, best_x, nm_maxiter, nm_fatol, nm_xatol)
        if nm_res.fun < best_loss:
            best_x = nm_res.x.copy()
            best_loss = float(nm_res.fun)

        best_x = _clip_to_bounds(best_x, obj.bounds)
        params = obj.to_params(best_x)
        return CalibrationResult(
            params=params,
            loss=best_loss,
            n_evals=obj.n_evals,
            success=bool(de_res.success),
            method="two_stage",
            message=str(de_res.message),
            history=list(obj.history),
            de_result=de_res,
        )

    def _run_de(
        self,
        obj: RBergomiObjective | RHestonObjective,
        popsize: int,
        maxiter: int,
        seed: int,
        tol: float,
        mutation: float | tuple[float, float],
        recombination: float,
        workers: int,
        callback: Callable | None,
    ) -> CalibrationResult:
        de_res = differential_evolution(
            obj,
            bounds=obj.bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            tol=tol,
            mutation=mutation,
            recombination=recombination,
            init="latinhypercube",
            workers=workers,
            callback=callback,
            polish=False,
        )
        best_x = _clip_to_bounds(de_res.x.copy(), obj.bounds)
        params = obj.to_params(best_x)
        return CalibrationResult(
            params=params,
            loss=float(de_res.fun),
            n_evals=obj.n_evals,
            success=bool(de_res.success),
            method="differential_evolution",
            message=str(de_res.message),
            history=list(obj.history),
            de_result=de_res,
        )

    def _run_nm(
        self,
        obj: RBergomiObjective | RHestonObjective,
        x0: np.ndarray,
        maxiter: int,
        fatol: float,
        xatol: float,
    ) -> CalibrationResult:
        nm_res = _nelder_mead_polish(obj, x0, maxiter, fatol, xatol)
        best_x = _clip_to_bounds(nm_res.x.copy(), obj.bounds)
        params = obj.to_params(best_x)
        return CalibrationResult(
            params=params,
            loss=float(nm_res.fun),
            n_evals=obj.n_evals,
            success=bool(nm_res.success),
            method="nelder_mead",
            message=str(nm_res.message),
            history=list(obj.history),
            de_result=None,
        )

    # ------------------------------------------------------------------
    # H estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_H_from_skew(
        surface: ImpliedVolSurface,
        dk: float = 0.05,
        min_points: int = 2,
    ) -> float | None:
        """Estimate H from the power-law decay of the ATM implied-vol skew.

        Uses the asymptotic relation

            |∂σ_ATM / ∂k|(T) ≈ C · T^{H − 1/2}

        to fit H via an OLS regression of ``log|skew|`` on ``log(T)``.

        Parameters
        ----------
        surface : ImpliedVolSurface
            Surface with at least ``min_points`` maturities that have
            computable ATM skews (non-zero, finite).
        dk : float
            Finite-difference half-width passed to
            :meth:`~ImpliedVolSurface.atm_skews`.
        min_points : int
            Minimum number of valid skew data points required; returns
            *None* when fewer are found.

        Returns
        -------
        float or None
            Estimated Hurst exponent clipped to (0.01, 0.49), or *None*
            when the surface lacks sufficient skew information.
        """
        skews = surface.atm_skews(dk=dk)
        mats = surface.maturities

        valid = np.isfinite(skews) & (np.abs(skews) > 1e-12)
        if int(valid.sum()) < min_points:
            return None

        log_T = np.log(mats[valid])
        log_abs_skew = np.log(np.abs(skews[valid]))

        # slope = H − 1/2  →  H = slope + 0.5
        slope, _ = np.polyfit(log_T, log_abs_skew, 1)
        return float(np.clip(slope + 0.5, 0.01, 0.49))
