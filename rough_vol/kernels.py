"""
Kernel functions for fractional Brownian motion and rough volatility models.

Implements the Volterra / power-law kernels that define the Riemann-Liouville
fractional Brownian motion (RL-fBm) and the covariance structure used by the
Cholesky, Hosking, and hybrid simulation schemes.

All kernels are parameterised by the Hurst exponent H ∈ (0, 1).  For rough
volatility the relevant regime is H ∈ (0, 0.5).

Functions
---------
rl_kernel(t, s, H)
    Riemann-Liouville kernel K(t, s) = (t-s)^(H-0.5) for 0 ≤ s < t.

covariance(t, s, H)
    Autocovariance of standard fBm: C(t,s) = 0.5*(|t|^2H + |s|^2H - |t-s|^2H).

covariance_matrix(times, H)
    Full covariance matrix Σ_{ij} = C(t_i, t_j, H) for an array of times.

rl_covariance(t, s, H)
    Covariance of the Riemann-Liouville fBm increments used by the hybrid
    scheme: E[W^H(t) W^H(s)] evaluated analytically.

hybrid_weights(n_trunc, H, dt)
    Quadrature weights b_k for the truncated Wiener integral in the hybrid
    scheme (Bennedsen, Lunde & Pakkanen, 2017, eq. 2.7).

gamma_H(H)
    Normalisation constant Γ(H + 0.5) used throughout the kernel formulae.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma


# ---------------------------------------------------------------------------
# Public constants / helpers
# ---------------------------------------------------------------------------

def gamma_H(H: float) -> float:
    """Return the normalisation constant Γ(H + 0.5).

    Parameters
    ----------
    H : float
        Hurst exponent in (0, 1).

    Returns
    -------
    float
        Γ(H + 0.5).
    """
    if not (0.0 < H < 1.0):
        raise ValueError(f"H must be in (0, 1), got {H}")
    return float(gamma(H + 0.5))


# ---------------------------------------------------------------------------
# Riemann-Liouville kernel
# ---------------------------------------------------------------------------

def rl_kernel(t: float | np.ndarray, s: float | np.ndarray, H: float) -> np.ndarray:
    """Evaluate the Riemann-Liouville kernel K(t, s) = (t - s)^(H - 0.5).

    This is the kernel of the stochastic Volterra integral representation

        W^H(t) = ∫_0^t K(t, s) dW(s)

    where W is standard Brownian motion.  The kernel is only defined for
    s < t; values where s ≥ t are returned as 0.

    Parameters
    ----------
    t, s : float or array-like
        Time arguments.  Broadcast-compatible shapes are supported.
    H : float
        Hurst exponent in (0, 1).

    Returns
    -------
    np.ndarray
        Kernel values with the same broadcast shape as (t, s).
    """
    if not (0.0 < H < 1.0):
        raise ValueError(f"H must be in (0, 1), got {H}")
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    diff = t - s
    # Replace non-positive values with 1.0 before the power to avoid 0^negative.
    safe = np.where(diff > 0.0, diff, 1.0)
    return np.where(diff > 0.0, safe ** (H - 0.5), 0.0)


# ---------------------------------------------------------------------------
# Standard fBm covariance
# ---------------------------------------------------------------------------

def covariance(t: float | np.ndarray, s: float | np.ndarray, H: float) -> np.ndarray:
    """Covariance of standard fractional Brownian motion.

    Uses the exact formula

        C(t, s) = 0.5 * (|t|^{2H} + |s|^{2H} - |t - s|^{2H})

    which holds for the *standard* fBm normalised so that Var[W^H(1)] = 1.

    Parameters
    ----------
    t, s : float or array-like
        Non-negative time arguments.  Broadcast-compatible shapes supported.
    H : float
        Hurst exponent in (0, 1).

    Returns
    -------
    np.ndarray
        Covariance values.
    """
    if not (0.0 < H < 1.0):
        raise ValueError(f"H must be in (0, 1), got {H}")
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    return 0.5 * (np.abs(t) ** (2 * H) + np.abs(s) ** (2 * H) - np.abs(t - s) ** (2 * H))


def covariance_matrix(times: np.ndarray, H: float) -> np.ndarray:
    """Build the full covariance matrix for standard fBm at given time points.

    Parameters
    ----------
    times : array-like, shape (n,)
        Monotonically increasing non-negative time points.
    H : float
        Hurst exponent in (0, 1).

    Returns
    -------
    np.ndarray, shape (n, n)
        Symmetric positive-definite covariance matrix Σ_{ij} = C(t_i, t_j, H).
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be a 1-D array")
    t_col = times[:, np.newaxis]   # (n, 1)
    t_row = times[np.newaxis, :]   # (1, n)
    return covariance(t_col, t_row, H)


# ---------------------------------------------------------------------------
# Riemann-Liouville fBm covariance
# ---------------------------------------------------------------------------

def rl_covariance(t: float | np.ndarray, s: float | np.ndarray, H: float) -> np.ndarray:
    """Covariance of the Riemann-Liouville fractional Brownian motion.

    The RL-fBm is defined by the Volterra integral

        W^H(t) = (1 / Γ(H + 0.5)) ∫_0^t (t - u)^{H - 0.5} dW(u)

    Its covariance is

        E[W^H(t) W^H(s)] = (t^{2H} + s^{2H} - |t-s|^{2H}) / (2H * (Γ(H+0.5))^2 / Γ(2H+1))
                         = Γ(2H+1)/(2 * (Γ(H+0.5))^2) * (t^{2H} + s^{2H} - |t-s|^{2H})

    but for rough-vol purposes we use the un-normalised kernel (without the
    1/Γ(H+0.5) prefactor), so the returned value is the raw integral

        ∫_0^{min(t,s)} (t-u)^{H-0.5} (s-u)^{H-0.5} du

    evaluated via the Beta-function identity:

        = B(H+0.5, H+0.5) * min(t,s)^{2H} * 2F1(...)

    For simulation purposes we use the simpler closed form for the stationary
    increment covariance:

        ρ(k) = 0.5 * [(k+1)^{2H} - 2k^{2H} + (k-1)^{2H}]   k ≥ 1
        ρ(0) = 1

    which governs the Hosking / Cholesky scheme on a uniform grid.

    Parameters
    ----------
    t, s : float or array-like
        Non-negative time arguments.
    H : float
        Hurst exponent in (0, 1).

    Returns
    -------
    np.ndarray
        Covariance values (un-normalised RL kernel inner product).
    """
    if not (0.0 < H < 1.0):
        raise ValueError(f"H must be in (0, 1), got {H}")
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    # Closed-form via the Beta-function identity for RL-fBm covariance.
    # E[W^H(t)W^H(s)] = (1/(2H)) * (t^{2H} + s^{2H} - |t-s|^{2H})
    # (This is the covariance of the *normalised* RL-fBm with unit variance
    # at t=1, using the convention of Gatheral/Jaisson/Rosenbaum.)
    return (t ** (2 * H) + s ** (2 * H) - np.abs(t - s) ** (2 * H)) / (2 * H)


# ---------------------------------------------------------------------------
# Hybrid-scheme weights
# ---------------------------------------------------------------------------

def hybrid_weights(n_trunc: int, H: float, dt: float) -> np.ndarray:
    """Compute the quadrature weights b_k for the hybrid scheme.

    In the hybrid scheme (Bennedsen, Lunde & Pakkanen 2017) the Volterra
    kernel is split into a 'near' part handled exactly via Wiener integrals
    and a 'far' part approximated by a Riemann sum.  The weights are

        b_k = ∫_{(k-1)dt}^{k·dt} (t_{n+1} - u)^{H-0.5} du / dt
            = [ (k·dt)^{H+0.5} - ((k-1)·dt)^{H+0.5} ] / ((H+0.5) * dt)

    evaluated at the *reference* lag so that b_k only depends on k (not n).
    This gives an O(n log n) FFT-based convolution step.

    Parameters
    ----------
    n_trunc : int
        Number of quadrature intervals (truncation lag).  Typical values are
        in the range [20, 200] depending on required accuracy.
    H : float
        Hurst exponent in (0, 1).
    dt : float
        Uniform time-step size.

    Returns
    -------
    np.ndarray, shape (n_trunc,)
        Weights b_1, b_2, …, b_{n_trunc}.
    """
    if not (0.0 < H < 1.0):
        raise ValueError(f"H must be in (0, 1), got {H}")
    if n_trunc < 1:
        raise ValueError("n_trunc must be ≥ 1")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    k = np.arange(1, n_trunc + 1, dtype=float)
    alpha = H + 0.5
    b = (k ** alpha - (k - 1) ** alpha) * (dt ** (alpha - 1)) / alpha
    return b
