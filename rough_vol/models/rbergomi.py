"""
Rough Bergomi model (Bayer, Friz & Gatheral, 2016).

Implements the rBergomi stochastic volatility model in which the log-variance
process is driven by a Riemann-Liouville fractional Brownian motion with
Hurst exponent H < 1/2, producing the power-law explosion of the at-the-money
volatility skew observed in equity markets.

References
----------
Bayer, C., Friz, P., & Gatheral, J. (2016). Pricing under rough volatility.
    Quantitative Finance, 16(6), 887-904.
"""

from __future__ import annotations

from dataclasses import dataclass

from functools import lru_cache

import numpy as np
from scipy.special import hyp2f1
from numpy.random import Generator, default_rng
from scipy.stats import norm

from rough_vol.simulation.fbm import hybrid_sim


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RBergomiParams:
    """Parameters for the rough Bergomi model.

    Attributes
    ----------
    H : float
        Hurst exponent in (0, 0.5).  Controls roughness of the vol path.
    eta : float
        Vol-of-vol (> 0).
    rho : float
        Spot-vol correlation in (-1, 1).
    xi0 : float
        Initial forward variance (> 0).  Flat forward variance curve assumed.
    """
    H: float
    eta: float
    rho: float
    xi0: float

    def __post_init__(self) -> None:
        if not (0.0 < self.H < 0.5):
            raise ValueError(f"H must be in (0, 0.5) for rough vol, got {self.H}")
        if self.eta <= 0.0:
            raise ValueError(f"eta must be > 0, got {self.eta}")
        if not (-1.0 < self.rho < 1.0):
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if self.xi0 <= 0.0:
            raise ValueError(f"xi0 must be > 0, got {self.xi0}")


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _bs_call_price(F: float, K: float, sigma: float, T: float, r: float = 0.0) -> float:
    """Black-Scholes call price for forward F, strike K, vol sigma, maturity T."""
    if sigma <= 0.0 or T <= 0.0:
        return max(F * np.exp(-r * T) - K * np.exp(-r * T), 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def _bs_implied_vol(
    price: float,
    F: float,
    K: float,
    T: float,
    r: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Invert Black-Scholes formula to find implied volatility.

    Uses Newton-Raphson with bisection fallback.  Returns NaN when the price
    is outside the no-arbitrage bounds.
    """
    intrinsic = max(np.exp(-r * T) * (F - K), 0.0)
    upper_bound = np.exp(-r * T) * F
    if price <= intrinsic or price >= upper_bound:
        return float("nan")

    lo, hi = 1e-6, 10.0

    # Bracket: ensure lo and hi straddle the solution
    while _bs_call_price(F, K, hi, T, r) < price:
        hi *= 2.0
        if hi > 100.0:
            return float("nan")

    # Initial guess: mid-point of log-scale bracket
    sigma = np.sqrt(lo * hi)

    sqrtT = np.sqrt(T)
    for _ in range(max_iter):
        c = _bs_call_price(F, K, sigma, T, r)
        diff = c - price
        if abs(diff) < tol * (1.0 + price):
            return sigma
        # Update bracket
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        # Newton step
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
        vega = np.exp(-r * T) * F * norm.pdf(d1) * sqrtT
        if vega < 1e-14:
            sigma = 0.5 * (lo + hi)
        else:
            sigma_new = sigma - diff / vega
            # Fall back to bisection if Newton leaves the bracket
            if lo < sigma_new < hi:
                sigma = sigma_new
            else:
                sigma = 0.5 * (lo + hi)
    return sigma


# ---------------------------------------------------------------------------
# Volterra correlation weights
# ---------------------------------------------------------------------------

def volterra_weights(n_steps: int, H: float, T: float) -> np.ndarray:
    """Compute the Volterra kernel quadrature weights for the rBergomi scheme.

    These are the BFG (2017) weights used in the FFT convolution representation
    of the Riemann-Liouville fBm:

        W^H(t_n) ≈ Σ_{k=1}^{n} b_k * ΔW_{n-k+1}

    where

        b_k = [ (k·dt)^{H+1/2} − ((k-1)·dt)^{H+1/2} ] / (H + 1/2)

    The weights are positive and decreasing: the most recent Brownian increment
    gets the largest weight, reflecting the power-law singularity of the
    Volterra kernel near zero.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    H : float
        Hurst exponent in (0, 0.5).
    T : float
        Terminal time.

    Returns
    -------
    np.ndarray, shape (n_steps,)
        Volterra weights b_1, b_2, …, b_{n_steps}.
    """
    dt = T / n_steps
    k = np.arange(1, n_steps + 1, dtype=float)
    alpha = H + 0.5
    return (k ** alpha - (k - 1) ** alpha) * dt ** (alpha - 1) / alpha


# ---------------------------------------------------------------------------
# rBergomi path simulator
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _joint_cholesky(H: float, n_steps: int, T: float) -> np.ndarray:
    """Cholesky factor of Cov([Wv(t_1..t_n), W(t_1..t_n)]) on the grid t_i = i T / n.

    Cov(Wv(u), Wv(v)) = 2H/(H+1/2) u^(H+1/2) v^(H-1/2) 2F1(1/2-H, 1; 3/2+H; u/v) for u <= v
    Cov(Wv(v), W(u))  = sqrt(2H)/(H+1/2) (v^(H+1/2) - (v - min(u, v))^(H+1/2))
    Cov(W(u), W(v))   = min(u, v)
    """
    t = np.linspace(T / n_steps, T, n_steps)
    u, v = np.meshgrid(t, t, indexing="ij")
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    g = H + 0.5
    cov_vv = 2 * H / g * lo ** g * hi ** (H - 0.5) * hyp2f1(0.5 - H, 1.0, 1.5 + H, lo / hi)
    cov_vw = np.sqrt(2 * H) / g * (u ** g - (u - np.minimum(u, v)) ** g)   # rows Wv(u), cols W(v)
    cov_ww = lo
    C = np.block([[cov_vv, cov_vw], [cov_vw.T, cov_ww]])
    jitter = 1e-12 * np.trace(C) / C.shape[0]
    return np.linalg.cholesky(C + jitter * np.eye(C.shape[0]))


def simulate_paths(
    params: RBergomiParams,
    n_paths: int,
    n_steps: int,
    T: float = 1.0,
    rng: int | Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate variance and asset paths under the rough Bergomi model.

    The model is

        V(t) = ξ₀ · exp(η · W^H(t) − ½ η² t^{2H})
        dS/S = √V(t) dB(t)

    where W^H is a standard Riemann-Liouville fBm (variance t^{2H} at time t)
    and B is a Brownian correlated with the driving Brownian of W^H:

        dB = ρ dW + √(1−ρ²) dW^perp

    Both W and W^perp are independent standard Brownians.

    Parameters
    ----------
    params : RBergomiParams
        Model parameters (H, eta, rho, xi0).
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of time steps per path.
    T : float
        Terminal time.
    rng : int, Generator, or None
        Random seed or numpy Generator.

    Returns
    -------
    V : np.ndarray, shape (n_paths, n_steps + 1)
        Variance process V(t) at times 0, dt, 2dt, ..., T.
    S : np.ndarray, shape (n_paths, n_steps + 1)
        Asset price S(t) normalised so S(0) = 1.
    """
    if isinstance(rng, Generator):
        rng_ = rng
    else:
        rng_ = default_rng(rng)

    H, eta, rho, xi0 = params.H, params.eta, params.rho, params.xi0
    dt = T / n_steps

    # --- Exact joint simulation of the Volterra driver and its Brownian motion ---
    # Bayer, Friz and Gatheral (2016): V is driven by the Riemann-Liouville process
    #   Wv(t) = sqrt(2H) * int_0^t (t-s)^(H-1/2) dW(s),   Var Wv(t) = t^(2H),
    # and the spot is driven by dB = rho dW + sqrt(1-rho^2) dW_perp, with the SAME W.
    # (Wv, W) is jointly Gaussian with known covariance, so it is sampled exactly with
    # a Cholesky factor. W has independent increments, so sqrt(V(t_i)) is independent
    # of dB_i and the discretised price is a martingale.
    L = _joint_cholesky(H, n_steps, T)                     # (2n, 2n)
    Z = rng_.standard_normal((n_paths, 2 * n_steps))
    X = Z @ L.T                                            # rows: [Wv(t_1..t_n), W(t_1..t_n)]
    WV = np.zeros((n_paths, n_steps + 1))
    WV[:, 1:] = X[:, :n_steps]
    W = np.zeros((n_paths, n_steps + 1))
    W[:, 1:] = X[:, n_steps:]
    dW = np.diff(W, axis=1)                                # N(0, dt), independent across steps
    dWperp = rng_.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    dB = rho * dW + np.sqrt(1.0 - rho ** 2) * dWperp

    # --- Variance process ---
    # V(t) = xi0 * exp(eta * Wv(t) - 0.5 * eta^2 * t^{2H})
    times = np.linspace(0.0, T, n_steps + 1)
    drift = -0.5 * eta ** 2 * times ** (2 * H)
    V = xi0 * np.exp(eta * WV + drift[np.newaxis, :])

    # --- Asset process (Euler-Maruyama on log price) ---
    # d(log S) = -V/2 dt + sqrt(V) dB
    log_S = np.zeros((n_paths, n_steps + 1))
    for i in range(n_steps):
        v_i = V[:, i]
        log_S[:, i + 1] = log_S[:, i] - 0.5 * v_i * dt + np.sqrt(v_i) * dB[:, i]

    S = np.exp(log_S)
    return V, S


# ---------------------------------------------------------------------------
# European option pricing
# ---------------------------------------------------------------------------

def price_european(
    params: RBergomiParams,
    K: float | np.ndarray,
    T: float,
    r: float = 0.0,
    n_paths: int = 20_000,
    n_steps: int = 252,
    rng: int | Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Price European call options by Monte Carlo under the rBergomi model.

    Parameters
    ----------
    params : RBergomiParams
        Model parameters.
    K : float or array-like
        Strike price(s).
    T : float
        Maturity.
    r : float
        Risk-free rate (continuous compounding).
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of time steps.
    rng : int, Generator, or None
        Random seed.

    Returns
    -------
    prices : np.ndarray
        Monte Carlo call prices, shape matching K.
    stderr : np.ndarray
        Standard errors of the MC estimates.
    """
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    scalar = K_arr.ndim == 0 or (isinstance(K, (int, float)))

    _, S = simulate_paths(params, n_paths, n_steps, T, rng=rng)
    S_T = S[:, -1]  # terminal asset prices, shape (n_paths,)

    discount = np.exp(-r * T)
    prices = np.empty(K_arr.shape)
    stderr = np.empty(K_arr.shape)

    for idx, k in np.ndenumerate(K_arr):
        payoffs = np.maximum(S_T - k, 0.0)
        prices[idx] = discount * payoffs.mean()
        stderr[idx] = discount * payoffs.std() / np.sqrt(n_paths)

    return prices, stderr


# ---------------------------------------------------------------------------
# Implied vol surface
# ---------------------------------------------------------------------------

def implied_vol_surface(
    params: RBergomiParams,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float = 0.0,
    n_paths: int = 20_000,
    n_steps_per_year: int = 252,
    rng: int | Generator | None = None,
) -> np.ndarray:
    """Compute the Black-Scholes implied volatility surface.

    Parameters
    ----------
    params : RBergomiParams
        Model parameters.
    strikes : array-like, shape (n_K,)
        Strike prices (as fractions of spot, i.e. moneyness levels).
    maturities : array-like, shape (n_T,)
        Option maturities.
    r : float
        Risk-free rate.
    n_paths : int
        Number of MC paths (shared across all maturities).
    n_steps_per_year : int
        Time steps per unit time.
    rng : int, Generator, or None
        Random seed.

    Returns
    -------
    np.ndarray, shape (n_T, n_K)
        Implied volatility surface.  NaN where inversion fails.
    """
    if isinstance(rng, Generator):
        rng_ = rng
    else:
        rng_ = default_rng(rng)

    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    n_T, n_K = len(maturities), len(strikes)
    ivol = np.full((n_T, n_K), np.nan)

    T_max = float(maturities.max())
    n_steps = max(int(np.ceil(n_steps_per_year * T_max)), 1)
    dt = T_max / n_steps

    # Simulate paths once up to T_max
    _, S = simulate_paths(params, n_paths, n_steps, T_max, rng=rng_)
    # S shape: (n_paths, n_steps + 1)

    for i, T in enumerate(maturities):
        step_idx = min(int(round(T / dt)), n_steps)
        S_T = S[:, step_idx]
        F = S_T.mean()  # forward (r=0 assumed in simulation; add drift for r≠0)
        F_bs = np.exp(r * T)  # forward price (S0=1)

        for j, k in enumerate(strikes):
            payoffs = np.maximum(S_T - k, 0.0)
            price = np.exp(-r * T) * payoffs.mean()
            iv = _bs_implied_vol(price, F_bs, k, T, r)
            ivol[i, j] = iv

    return ivol
