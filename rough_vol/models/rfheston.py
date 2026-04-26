"""
Rough Heston model (El Euch & Rosenbaum, 2019).

Implements the rough Heston model where the instantaneous variance follows a
fractional Riccati equation.  The model inherits the mean-reversion and
leverage structure of the classical Heston model while reproducing the rough
behaviour of realised volatility at short time scales.

References
----------
El Euch, O., & Rosenbaum, M. (2019). The characteristic function of rough
    Heston models. Mathematical Finance, 29(1), 3-38.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as sc_gamma
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RHestonParams:
    """Parameters for the rough Heston model.

    Attributes
    ----------
    H : float
        Hurst exponent in (0, 0.5).
    lambda_ : float
        Mean-reversion speed (> 0).
    theta : float
        Long-run variance level (> 0).
    rho : float
        Spot-vol correlation in (-1, 1).
    nu : float
        Vol-of-vol (> 0).
    V0 : float
        Initial variance (> 0).
    """
    H: float
    lambda_: float
    theta: float
    rho: float
    nu: float
    V0: float

    def __post_init__(self) -> None:
        if not (0.0 < self.H < 0.5):
            raise ValueError(f"H must be in (0, 0.5), got {self.H}")
        if self.lambda_ <= 0.0:
            raise ValueError(f"lambda_ must be > 0, got {self.lambda_}")
        if self.theta <= 0.0:
            raise ValueError(f"theta must be > 0, got {self.theta}")
        if not (-1.0 < self.rho < 1.0):
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if self.nu <= 0.0:
            raise ValueError(f"nu must be > 0, got {self.nu}")
        if self.V0 <= 0.0:
            raise ValueError(f"V0 must be > 0, got {self.V0}")


# ---------------------------------------------------------------------------
# Fractional Riccati equation solver
# ---------------------------------------------------------------------------

def _fractional_riccati(
    u: complex,
    T: float,
    params: RHestonParams,
    n_steps: int = 200,
) -> np.ndarray:
    """Solve the fractional Riccati equation for rough Heston.

    The characteristic function of log S_T under the rough Heston model is

        E[exp(i·u·log S_T)] = exp(i·u·log S_0 + h(u, T))

    where h satisfies the Volterra integral equation

        h(u, T) = ∫_0^T [ (λθ - ½ν²u²) ψ(u, s) + (iuρν - λ) ψ²(u,s)/2 ] ds / Γ(H+1/2)...

    More precisely, the function ψ(u, t) solves

        ψ(u, t) = ∫_0^t (t-s)^{H-1/2} / Γ(H+1/2) * F(u, ψ(u, s)) ds

    with F(u, ψ) = -λψ + iuρνψ - ½ν²ψ² + iuρν/...

    The standard form (El Euch & Rosenbaum 2019, eq. 2.8-2.9):

        ψ(u, t) = ∫_0^t K_{H}(t-s) [ F(u) + (λ(H+1/2) + iuρν)ψ(u,s) - ½ν²ψ²(u,s) ] ds

    where K_H(x) = x^{H-1/2} / Γ(H+1/2).

    We use the Adams predictor-corrector scheme for the fractional ODE:

        D^α ψ = f(ψ)   where α = H + 1/2 ∈ (1/2, 1)

    with f(ψ) = F_1 + F_2·ψ - ½ν²ψ²,  F_1 = -iu(iu+1)/2, F_2 = iuρν - λ.

    Returns
    -------
    np.ndarray, shape (n_steps + 1,), complex
        ψ(u, t_k) for k = 0, 1, …, n_steps.
    """
    H = params.H
    lambda_ = params.lambda_
    theta = params.theta
    rho = params.rho
    nu = params.nu

    alpha = H + 0.5  # fractional order; α ∈ (0.5, 1.0)
    dt = T / n_steps
    Ga = sc_gamma(alpha)
    Ga1 = sc_gamma(alpha + 1.0)
    Ga2 = sc_gamma(2.0 * alpha + 1.0)

    # Coefficients of the fractional Riccati RHS
    # F(u, ψ) = F1 + F2·ψ − ½ν²ψ²
    # F1 = -½ u(u+i) = -u²/2 - iu/2  (the log-price quadratic).
    # Note: -½*(iu)*(iu+1) = u²/2 - iu/2 has the WRONG real-part sign.
    F1 = -0.5 * u * (u + 1j)              # = -u²/2 - iu/2
    F2 = 1j * u * rho * nu - lambda_

    def rhs(psi: complex) -> complex:
        return F1 + F2 * psi - 0.5 * nu ** 2 * psi ** 2

    # Adams predictor-corrector for Caputo fractional ODE of order α
    # ψ(t_{n+1}) = sum_{j=0}^{n} a_{j,n+1} * f(ψ_j) + predictor term
    psi = np.zeros(n_steps + 1, dtype=complex)
    # ψ(0) = 0

    # Precompute Adams weights (one-step kernel quadrature)
    # a_{j, n+1} for the corrector: (Diethelm 2002, eq. 3.2)
    # b_{j, n+1} for the predictor
    for n in range(n_steps):
        # Predictor: explicit Adams–Bashforth
        j = np.arange(n + 1)
        # b_{j,n+1} = dt^alpha / Gamma(alpha+1) * ((n-j+1)^alpha - (n-j)^alpha)
        b = (dt ** alpha / Ga1) * ((n - j + 1) ** alpha - (n - j) ** alpha)
        psi_pred = np.dot(b, np.array([rhs(psi[jj]) for jj in j]))

        # Corrector: implicit Adams–Moulton
        # a_{j,n+1} for j=0,1,...,n and j=n+1 (using predictor)
        # a_{0,n+1} = dt^alpha/Gamma(alpha+2) * (n^{alpha+1} - (n-alpha)(n+1)^alpha)
        # a_{j,n+1} = dt^alpha/Gamma(alpha+2) * ((n-j+2)^{alpha+1}
        #              + (n-j)^{alpha+1} - 2(n-j+1)^{alpha+1})  for 1<=j<=n
        # a_{n+1, n+1} = dt^alpha / Gamma(alpha+2)
        n_ = float(n)
        a = np.zeros(n + 2, dtype=float)
        # j = 0
        a[0] = (dt ** alpha / Ga2) * (
            n_ ** (alpha + 1) - (n_ - alpha) * (n_ + 1.0) ** alpha
        )
        # j = 1 .. n
        jj = np.arange(1, n + 1, dtype=float)
        a[1 : n + 1] = (dt ** alpha / Ga2) * (
            (n_ - jj + 2.0) ** (alpha + 1.0)
            + (n_ - jj) ** (alpha + 1.0)
            - 2.0 * (n_ - jj + 1.0) ** (alpha + 1.0)
        )
        # j = n+1 (using predictor value)
        a[n + 1] = dt ** alpha / Ga2

        rhs_vals = np.array([rhs(psi[jj]) for jj in range(n + 1)])
        psi[n + 1] = np.dot(a[:n + 1], rhs_vals) + a[n + 1] * rhs(psi_pred)

        # Abort if the solution has blown up (unphysical for reasonable parameters).
        if not np.isfinite(psi[n + 1].real) or abs(psi[n + 1]) > 1e8:
            psi[n + 1:] = np.nan
            break

    return psi


def characteristic_function(
    u: float | complex,
    T: float,
    params: RHestonParams,
    n_steps: int = 200,
) -> complex:
    """Log-characteristic function of log S_T under rough Heston.

    Returns log E[exp(i·u·log S_T)] (excluding the trivial i·u·log S_0 term).

    The formula is (El Euch & Rosenbaum 2019):

        log CF = λθ / Γ(H+3/2) ∫_0^T (T-s)^{H+1/2} ψ(u,s) ds
                 + V_0 / Γ(H+1/2) ∫_0^T (T-s)^{H-1/2} ψ(u,s) ds

    Both integrals are approximated by trapezoidal quadrature over the ψ grid.

    Parameters
    ----------
    u : float or complex
        Fourier argument.
    T : float
        Maturity.
    params : RHestonParams
        Model parameters.
    n_steps : int
        Number of steps for the fractional ODE solver.

    Returns
    -------
    complex
        log E[exp(i·u·log(S_T/S_0))].
    """
    H = params.H
    lambda_ = params.lambda_
    theta = params.theta
    V0 = params.V0

    psi = _fractional_riccati(u, T, params, n_steps)  # shape (n_steps+1,)

    if not np.all(np.isfinite(psi)):
        return complex(float("nan"), float("nan"))

    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)
    tau = T - t  # (T-s) for each grid point

    # First term: λθ ∫_0^T ψ(u, s) ds  — plain integral, no kernel weight.
    int1 = np.trapezoid(psi, dx=dt)

    # Second term: V_0 ∫_0^T K_H(T-s) ψ(u, s) ds
    # K_H(x) = x^{H-1/2} / Γ(H+1/2); guard 0^negative at s=T (tau=0).
    safe_tau = np.where(tau > 0, tau, 1.0)
    w2 = np.where(tau > 0, safe_tau ** (H - 0.5), 0.0)
    int2 = np.trapezoid(w2 * psi, dx=dt)

    log_cf = lambda_ * theta * int1 + V0 / sc_gamma(H + 0.5) * int2
    return complex(log_cf)


# ---------------------------------------------------------------------------
# Fourier pricing (Gil-Pelaez)
# ---------------------------------------------------------------------------

def price_european(
    params: RHestonParams,
    K: float | np.ndarray,
    T: float,
    r: float = 0.0,
    S0: float = 1.0,
    n_steps: int = 200,
    n_quad: int = 128,
    u_max: float = 30.0,
) -> np.ndarray:
    """Price European calls via the Heston-style Gil-Pelaez Fourier inversion.

    Uses the standard two-probability decomposition:

        C = e^{-rT} [F · P₁ − K · P₂]

    where F = S₀ e^{rT} and:

        P₂ = 1/2 + (1/π) ∫₀^∞ Im[e^{−iuk} φ(u)] / u  du
        P₁ = 1/2 + (1/π) ∫₀^∞ Im[e^{−iuk} φ(u − i)] / u  du

    with k = log(K/F) and φ(u) = exp(log_cf(u)) the characteristic function of
    log(S_T/S₀).  Only real-u evaluations are needed for P₂; for P₁ the shift
    u − i stays within the strip of analyticity because E[S_T/F] = 1.

    Parameters
    ----------
    params : RHestonParams
    K : float or array-like
        Strike prices.
    T : float
        Maturity.
    r : float
        Risk-free rate.
    S0 : float
        Initial spot price.
    n_steps : int
        ODE solver steps for the fractional Riccati equation.
    n_quad : int
        Number of Gaussian quadrature points for the Fourier integral.
    u_max : float
        Upper limit for numerical Fourier integration.

    Returns
    -------
    np.ndarray
        Call prices, same shape as K.
    """
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    F = S0 * np.exp(r * T)  # forward price
    k_arr = np.log(K_arr / F)  # log-moneyness

    # Gauss-Legendre quadrature on [eps, u_max] (avoid singularity at u=0)
    xi, wi = np.polynomial.legendre.leggauss(n_quad)
    eps = 1e-4
    u_nodes = 0.5 * (u_max - eps) * (xi + 1.0) + eps   # map to [eps, u_max]
    w_nodes = 0.5 * (u_max - eps) * wi

    # Pre-compute CF at each quadrature node for both P1 and P2
    phi_real = np.zeros(n_quad, dtype=complex)   # φ(u) for P2
    phi_shift = np.zeros(n_quad, dtype=complex)  # φ(u-i) for P1
    for j, u in enumerate(u_nodes):
        lc2 = characteristic_function(float(u), T, params, n_steps)
        lc1 = characteristic_function(u - 1j, T, params, n_steps)
        phi_real[j] = np.exp(lc2) if np.isfinite(lc2.real) else 0.0
        phi_shift[j] = np.exp(lc1) if np.isfinite(lc1.real) else 0.0

    prices = np.empty(K_arr.shape)
    for idx, (k, K_i) in enumerate(zip(k_arr.flat, K_arr.flat)):
        e_ik = np.exp(-1j * u_nodes * k)  # e^{-iuk}, shape (n_quad,)

        # P2: Im[e^{-iuk} φ(u)] / u
        integrand2 = np.imag(e_ik * phi_real) / u_nodes
        I2 = float(np.dot(w_nodes, integrand2))
        P2 = 0.5 + I2 / np.pi

        # P1: Im[e^{-iuk} φ(u-i)] / u
        integrand1 = np.imag(e_ik * phi_shift) / u_nodes
        I1 = float(np.dot(w_nodes, integrand1))
        P1 = 0.5 + I1 / np.pi

        prices.flat[idx] = np.exp(-r * T) * (F * P1 - K_i * P2)

    return prices


# ---------------------------------------------------------------------------
# Implied vol surface
# ---------------------------------------------------------------------------

def _bs_call_price(F: float, K: float, sigma: float, T: float, r: float = 0.0) -> float:
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
    intrinsic = max(np.exp(-r * T) * (F - K), 0.0)
    upper_bound = np.exp(-r * T) * F
    if price <= intrinsic or price >= upper_bound:
        return float("nan")

    lo, hi = 1e-6, 10.0
    while _bs_call_price(F, K, hi, T, r) < price:
        hi *= 2.0
        if hi > 100.0:
            return float("nan")

    sigma = np.sqrt(lo * hi)
    sqrtT = np.sqrt(T)

    for _ in range(max_iter):
        c = _bs_call_price(F, K, sigma, T, r)
        diff = c - price
        if abs(diff) < tol * (1.0 + price):
            return sigma
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
        vega = np.exp(-r * T) * F * norm.pdf(d1) * sqrtT
        if vega < 1e-14:
            sigma = 0.5 * (lo + hi)
        else:
            sigma_new = sigma - diff / vega
            if lo < sigma_new < hi:
                sigma = sigma_new
            else:
                sigma = 0.5 * (lo + hi)
    return sigma


def implied_vol_surface(
    params: RHestonParams,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float = 0.0,
    S0: float = 1.0,
    n_steps: int = 200,
    n_quad: int = 128,
    u_max: float = 30.0,
) -> np.ndarray:
    """Compute Black-Scholes implied volatility surface for rough Heston.

    Parameters
    ----------
    params : RHestonParams
    strikes : array-like, shape (n_K,)
    maturities : array-like, shape (n_T,)
    r : float
    S0 : float
    n_steps, n_quad, u_max : solver / quadrature controls.

    Returns
    -------
    np.ndarray, shape (n_T, n_K)
        Implied volatility surface.  NaN where inversion fails.
    """
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    n_T, n_K = len(maturities), len(strikes)
    ivol = np.full((n_T, n_K), np.nan)

    for i, T in enumerate(maturities):
        F = S0 * np.exp(r * T)
        prices = price_european(params, strikes, T, r, S0, n_steps, n_quad, u_max)
        for j, (K_j, p) in enumerate(zip(strikes, prices)):
            ivol[i, j] = _bs_implied_vol(float(p), F, float(K_j), T, r)

    return ivol
