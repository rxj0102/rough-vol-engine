"""
Market implied-volatility surface data structures.

Two classes are provided:

``ImpliedVolSurface``
    Rich surface object that stores absolute strikes, maturities, implied vols,
    spot and forwards.  Offers transforms (log-moneyness, total variance),
    ATM diagnostics, no-arbitrage checks, and CSV I/O.

``VolSurface`` / ``MarketSlice``
    Lightweight containers used by the calibration pipeline
    (``RBergomiObjective``, ``RHestonObjective``).  Strikes stored as
    moneyness fractions of spot; NaN-aware weights for illiquid quotes.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy import ndarray
from scipy.stats import norm


# ---------------------------------------------------------------------------
# MarketSlice — one expiry
# ---------------------------------------------------------------------------

@dataclass
class MarketSlice:
    """Implied-volatility data for a single expiry.

    Parameters
    ----------
    maturity : float
        Time to expiry in years (> 0).
    strikes : ndarray, shape (n_K,)
        Strike levels (moneyness fractions of spot, sorted ascending).
    mid_vols : ndarray, shape (n_K,)
        Mid implied volatilities.  NaN where the quote is unavailable.
    weights : ndarray, shape (n_K,)
        Non-negative calibration weights (0 = excluded).
        Defaults to 1 for finite-vol strikes and 0 for NaN strikes.
    """

    maturity: float
    strikes: ndarray
    mid_vols: ndarray
    weights: ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.strikes = np.asarray(self.strikes, dtype=float)
        self.mid_vols = np.asarray(self.mid_vols, dtype=float)
        n = len(self.strikes)
        if len(self.mid_vols) != n:
            raise ValueError(
                f"strikes length {n} != mid_vols length {len(self.mid_vols)}"
            )
        if self.maturity <= 0.0:
            raise ValueError(f"maturity must be > 0, got {self.maturity}")
        if self.weights is None:
            self.weights = np.where(np.isfinite(self.mid_vols), 1.0, 0.0)
        else:
            self.weights = np.asarray(self.weights, dtype=float)
            if len(self.weights) != n:
                raise ValueError(
                    f"strikes length {n} != weights length {len(self.weights)}"
                )
            if np.any(self.weights < 0):
                raise ValueError("weights must be non-negative")

    @property
    def valid(self) -> ndarray:
        """Boolean mask: True where the IV is finite and weight > 0."""
        return np.isfinite(self.mid_vols) & (self.weights > 0)

    @property
    def n_strikes(self) -> int:
        return len(self.strikes)

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())


# ---------------------------------------------------------------------------
# VolSurface — full surface
# ---------------------------------------------------------------------------

@dataclass
class VolSurface:
    """Implied-volatility surface as a list of per-expiry slices.

    Parameters
    ----------
    slices : list[MarketSlice]
        One slice per expiry.  Will be sorted by ascending maturity on
        construction.
    """

    slices: list[MarketSlice]

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValueError("VolSurface must contain at least one MarketSlice")
        self.slices = sorted(self.slices, key=lambda s: s.maturity)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        strikes: ndarray,
        maturities: ndarray,
        ivols: ndarray,
        weights: ndarray | None = None,
    ) -> "VolSurface":
        """Construct from rectangular numpy arrays.

        Parameters
        ----------
        strikes : array, shape (n_K,)
            Common strike grid shared by all maturities.
        maturities : array, shape (n_T,)
            Expiry times in years.
        ivols : array, shape (n_T, n_K)
            Implied volatilities. Use NaN for missing quotes.
        weights : array, shape (n_T, n_K) or None
            Per-point calibration weights.  Defaults to 1 for finite IVs
            and 0 for NaN entries.

        Returns
        -------
        VolSurface
        """
        strikes = np.asarray(strikes, dtype=float)
        maturities = np.asarray(maturities, dtype=float)
        ivols = np.asarray(ivols, dtype=float)

        if ivols.shape != (len(maturities), len(strikes)):
            raise ValueError(
                f"ivols shape {ivols.shape} inconsistent with "
                f"maturities {maturities.shape} × strikes {strikes.shape}"
            )

        if weights is None:
            weights_arr = np.where(np.isfinite(ivols), 1.0, 0.0)
        else:
            weights_arr = np.asarray(weights, dtype=float)
            if weights_arr.shape != ivols.shape:
                raise ValueError(
                    f"weights shape {weights_arr.shape} != ivols shape {ivols.shape}"
                )

        slices = [
            MarketSlice(
                maturity=float(T),
                strikes=strikes.copy(),
                mid_vols=ivols[i].copy(),
                weights=weights_arr[i].copy(),
            )
            for i, T in enumerate(maturities)
        ]
        return cls(slices=slices)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def maturities(self) -> ndarray:
        """Sorted array of expiry times."""
        return np.array([s.maturity for s in self.slices])

    @property
    def n_slices(self) -> int:
        return len(self.slices)

    @property
    def n_total_points(self) -> int:
        """Total number of (maturity, strike) grid points."""
        return sum(s.n_strikes for s in self.slices)

    @property
    def n_valid_points(self) -> int:
        """Number of points with finite IV and positive weight."""
        return sum(s.n_valid for s in self.slices)

    # ------------------------------------------------------------------
    # Array conversion
    # ------------------------------------------------------------------

    def to_arrays(self) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """Export as rectangular numpy arrays.

        Requires all slices to share the same strike grid.  For ragged
        surfaces, iterate over ``self.slices`` directly.

        Returns
        -------
        maturities : ndarray, shape (n_T,)
        strikes : ndarray, shape (n_K,)
        ivols : ndarray, shape (n_T, n_K)
        weights : ndarray, shape (n_T, n_K)

        Raises
        ------
        ValueError
            If slices have different strike grids.
        """
        ref = self.slices[0].strikes
        for s in self.slices[1:]:
            if not np.array_equal(s.strikes, ref):
                raise ValueError(
                    "Slices have different strike grids; use slice-by-slice access."
                )
        ivols = np.stack([s.mid_vols for s in self.slices])
        weights = np.stack([s.weights for s in self.slices])
        return self.maturities, ref.copy(), ivols, weights

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def valid_mask(self) -> ndarray:
        """Boolean array of shape (n_T, n_K) indicating valid surface points."""
        return np.stack([s.valid for s in self.slices])

    def filter_weights(self, min_weight: float = 1e-8) -> "VolSurface":
        """Return a new surface with sub-threshold weights zeroed out."""
        new_slices = []
        for s in self.slices:
            w = s.weights.copy()
            w[w < min_weight] = 0.0
            new_slices.append(
                MarketSlice(s.maturity, s.strikes.copy(), s.mid_vols.copy(), w)
            )
        return VolSurface(slices=new_slices)

    def __repr__(self) -> str:
        T_str = ", ".join(f"{T:.2f}" for T in self.maturities)
        return (
            f"VolSurface(n_slices={self.n_slices}, "
            f"maturities=[{T_str}], "
            f"valid_points={self.n_valid_points}/{self.n_total_points})"
        )


# ---------------------------------------------------------------------------
# ImpliedVolSurface — rich surface with transforms and diagnostics
# ---------------------------------------------------------------------------

def _bs_call_scalar(F: float, K: float, sigma: float, T: float) -> float:
    """Black-Scholes call price (r=0, i.e. in forward measure)."""
    if sigma <= 0.0 or T <= 0.0:
        return max(F - K, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return float(F * norm.cdf(d1) - K * norm.cdf(d2))


class ImpliedVolSurface:
    """Implied-volatility surface with transforms and no-arbitrage diagnostics.

    Stores absolute strikes, maturities, Black-Scholes implied vols, the
    current spot price and per-maturity forward prices.

    Parameters
    ----------
    strikes : array-like, shape (n_K,)
        Absolute strike prices, sorted ascending.
    maturities : array-like, shape (n_T,)
        Times to expiry in years, sorted ascending.
    implied_vols : array-like, shape (n_T, n_K)
        Black-Scholes implied volatilities (decimals, 0.20 = 20 %).
        NaN for missing quotes.
    spot : float
        Current spot price (default 1.0).
    forwards : array-like, shape (n_T,) or None
        Forward price for each maturity.  Defaults to ``spot`` for all
        maturities (i.e. risk-free rate = 0).

    Attributes
    ----------
    strikes, maturities, implied_vols, spot, forwards : as above (immutable).
    n_strikes, n_maturities : int
    """

    def __init__(
        self,
        strikes: ndarray,
        maturities: ndarray,
        implied_vols: ndarray,
        spot: float = 1.0,
        forwards: ndarray | None = None,
    ) -> None:
        self.strikes = np.asarray(strikes, dtype=float)
        self.maturities = np.asarray(maturities, dtype=float)
        self.implied_vols = np.asarray(implied_vols, dtype=float)
        self.spot = float(spot)

        if forwards is None:
            self.forwards = np.full(len(self.maturities), self.spot)
        else:
            self.forwards = np.asarray(forwards, dtype=float)

        # Validate
        n_T, n_K = len(self.maturities), len(self.strikes)
        if self.implied_vols.shape != (n_T, n_K):
            raise ValueError(
                f"implied_vols shape {self.implied_vols.shape} != "
                f"({n_T}, {n_K})"
            )
        if len(self.forwards) != n_T:
            raise ValueError(
                f"forwards length {len(self.forwards)} != n_maturities {n_T}"
            )
        if self.spot <= 0.0:
            raise ValueError(f"spot must be > 0, got {self.spot}")
        if np.any(self.maturities <= 0):
            raise ValueError("all maturities must be > 0")
        if np.any(self.strikes <= 0):
            raise ValueError("all strikes must be > 0")

        # Sort by maturity
        order = np.argsort(self.maturities)
        self.maturities = self.maturities[order]
        self.implied_vols = self.implied_vols[order]
        self.forwards = self.forwards[order]

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def n_strikes(self) -> int:
        return len(self.strikes)

    @property
    def n_maturities(self) -> int:
        return len(self.maturities)

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def to_log_moneyness(self) -> ndarray:
        """Log-moneyness grid k = log(K / F(T)).

        Returns
        -------
        ndarray, shape (n_T, n_K)
            k[i, j] = log(strikes[j] / forwards[i]).
            k < 0 for in-the-money puts / out-of-the-money calls.
            k > 0 for out-of-the-money puts / in-the-money calls.
        """
        # broadcasts (n_K,) strikes against (n_T, 1) forwards
        return np.log(self.strikes[np.newaxis, :] / self.forwards[:, np.newaxis])

    def to_total_variance(self) -> ndarray:
        """Total implied variance w = σ²(K, T) · T.

        Returns
        -------
        ndarray, shape (n_T, n_K)
            w[i, j] = implied_vols[i, j]² × maturities[i].
        """
        return self.implied_vols ** 2 * self.maturities[:, np.newaxis]

    # ------------------------------------------------------------------
    # ATM diagnostics
    # ------------------------------------------------------------------

    def atm_vols(self) -> ndarray:
        """ATM implied volatility for each maturity.

        Interpolates the IV smile at log-moneyness k = 0 (i.e. K = F)
        using piecewise-linear interpolation in k-space.

        Returns
        -------
        ndarray, shape (n_T,)
            ATM implied vol per maturity.  NaN if k=0 is outside the
            strike range for that maturity.
        """
        k = self.to_log_moneyness()  # (n_T, n_K)
        result = np.empty(self.n_maturities)
        for i in range(self.n_maturities):
            ki = k[i]
            vi = self.implied_vols[i]
            valid = np.isfinite(vi)
            if valid.sum() < 2 or not (ki[valid].min() <= 0.0 <= ki[valid].max()):
                result[i] = np.nan
            else:
                result[i] = np.interp(0.0, ki[valid], vi[valid])
        return result

    def atm_skews(self, dk: float = 0.05) -> ndarray:
        """ATM implied-vol skew ∂σ/∂k|_{k=0} by central finite difference.

        Evaluates the implied vol smile at k = +dk and k = −dk via linear
        interpolation, then returns (σ(+dk) − σ(−dk)) / (2 dk).

        Parameters
        ----------
        dk : float
            Half-width of the finite-difference stencil in log-moneyness
            units (default 0.05, approximately ±5 % moneyness).

        Returns
        -------
        ndarray, shape (n_T,)
            ATM skew per maturity.  NaN when the smile does not cover
            ±dk from ATM.
        """
        if dk <= 0.0:
            raise ValueError(f"dk must be > 0, got {dk}")
        k = self.to_log_moneyness()  # (n_T, n_K)
        result = np.empty(self.n_maturities)
        for i in range(self.n_maturities):
            ki = k[i]
            vi = self.implied_vols[i]
            valid = np.isfinite(vi)
            ki_v, vi_v = ki[valid], vi[valid]
            if ki_v.size < 2 or ki_v.min() > -dk or ki_v.max() < dk:
                result[i] = np.nan
            else:
                sigma_plus = float(np.interp(dk, ki_v, vi_v))
                sigma_minus = float(np.interp(-dk, ki_v, vi_v))
                result[i] = (sigma_plus - sigma_minus) / (2.0 * dk)
        return result

    # ------------------------------------------------------------------
    # No-arbitrage checks
    # ------------------------------------------------------------------

    def check_no_arbitrage(self, tol: float = 1e-8) -> dict:
        """Check the surface for static-arbitrage violations.

        Two conditions are tested:

        **Calendar-spread**: total variance must be non-decreasing in
        maturity at every fixed (absolute) strike.  A violation means a
        short calendar spread has negative cost, generating risk-free profit.

        **Butterfly**: call prices (computed via Black-Scholes from the
        implied vols) must be convex in strike.  Equivalently, the
        risk-neutral density implied by the surface must be non-negative.
        Checked at each interior strike via the finite-difference approximation
        C(K−) + C(K+) − 2C(K) ≥ 0.

        Parameters
        ----------
        tol : float
            Numerical tolerance for declaring a violation.

        Returns
        -------
        dict with keys:
            ``no_calendar_arbitrage`` : bool
            ``no_butterfly_arbitrage`` : bool
            ``is_arbitrage_free``      : bool
            ``calendar_violations``    : list[dict]
            ``butterfly_violations``   : list[dict]
        """
        tv = self.to_total_variance()  # (n_T, n_K)

        # --- Calendar spread ---
        cal_ok = True
        cal_violations: list[dict] = []
        for j in range(self.n_strikes):
            for i in range(self.n_maturities - 1):
                w1, w2 = tv[i, j], tv[i + 1, j]
                if not (np.isnan(w1) or np.isnan(w2)) and w2 < w1 - tol:
                    cal_ok = False
                    cal_violations.append({
                        "T1": float(self.maturities[i]),
                        "T2": float(self.maturities[i + 1]),
                        "K": float(self.strikes[j]),
                        "w1": float(w1),
                        "w2": float(w2),
                    })

        # --- Butterfly ---
        fly_ok = True
        fly_violations: list[dict] = []
        for i, (T, F) in enumerate(zip(self.maturities, self.forwards)):
            for j in range(1, self.n_strikes - 1):
                s0 = self.implied_vols[i, j - 1]
                s1 = self.implied_vols[i, j]
                s2 = self.implied_vols[i, j + 1]
                if np.isnan(s0) or np.isnan(s1) or np.isnan(s2):
                    continue
                C0 = _bs_call_scalar(F, self.strikes[j - 1], s0, T)
                C1 = _bs_call_scalar(F, self.strikes[j],     s1, T)
                C2 = _bs_call_scalar(F, self.strikes[j + 1], s2, T)
                butterfly = C0 + C2 - 2.0 * C1
                if butterfly < -tol:
                    fly_ok = False
                    fly_violations.append({
                        "T": float(T),
                        "K": float(self.strikes[j]),
                        "butterfly": float(butterfly),
                    })

        return {
            "no_calendar_arbitrage": cal_ok,
            "no_butterfly_arbitrage": fly_ok,
            "is_arbitrage_free": cal_ok and fly_ok,
            "calendar_violations": cal_violations,
            "butterfly_violations": fly_violations,
        }

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        spot: float = 1.0,
    ) -> "ImpliedVolSurface":
        """Load a surface from a CSV file in long (tidy) format.

        Required columns
        ----------------
        ``maturity``    : float — years to expiry
        ``strike``      : float — absolute strike price
        ``implied_vol`` : float — Black-Scholes IV (decimal, e.g. 0.20)

        Optional columns
        ----------------
        ``forward`` : float — forward price for that row's maturity.
            When present, one forward per maturity is extracted (the first
            value seen for each maturity group).  When absent, ``spot`` is
            used for all maturities.
        ``spot``    : float — overrides the ``spot`` argument; the first
            value in the column is used.

        Parameters
        ----------
        path : str or Path
            Path to the CSV file.
        spot : float
            Default spot price if no ``spot`` column is present.

        Returns
        -------
        ImpliedVolSurface

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        import pandas as pd

        df = pd.read_csv(path)
        required = {"maturity", "strike", "implied_vol"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # Spot override
        if "spot" in df.columns:
            spot = float(df["spot"].iloc[0])

        # Pivot to rectangular grid; sorts both axes automatically
        pivot = df.pivot_table(
            index="maturity", columns="strike", values="implied_vol", aggfunc="first"
        )
        maturities = pivot.index.to_numpy(dtype=float)
        strikes = pivot.columns.to_numpy(dtype=float)
        ivols = pivot.to_numpy(dtype=float)

        # Forwards
        if "forward" in df.columns:
            fwd = (
                df.groupby("maturity")["forward"]
                .first()
                .reindex(pivot.index)
                .to_numpy(dtype=float)
            )
        else:
            fwd = np.full(len(maturities), spot)

        return cls(strikes, maturities, ivols, spot=spot, forwards=fwd)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ImpliedVolSurface("
            f"n_maturities={self.n_maturities}, "
            f"n_strikes={self.n_strikes}, "
            f"maturities={np.round(self.maturities, 3).tolist()}, "
            f"spot={self.spot})"
        )
