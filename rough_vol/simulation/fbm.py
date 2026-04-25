"""
Fractional Brownian motion (fBm) path generators.

Provides three simulation schemes for the Riemann-Liouville fBm used in rough
volatility models:

- **Cholesky scheme** — exact but O(n²) in memory; suited for low path counts.
- **Hosking scheme** — exact method via recursive conditional Gaussians; O(n²)
  but with lower memory footprint than Cholesky.
- **Hybrid / circulant-embedding scheme** (Wood & Chan, 1994) — exact and
  O(n log n) via FFT; the default for production Monte Carlo.

All three return arrays of shape (n_paths, n_steps) representing the fBm
*increments* W^H(t_{i+1}) - W^H(t_i) on a uniform grid [0, T].

Functions
---------
cholesky_sim(n_paths, n_steps, H, T, rng)
    Exact simulation via Cholesky factorisation of the covariance matrix.

hosking_sim(n_paths, n_steps, H, T, rng)
    Exact simulation via the Hosking (1984) recursive algorithm.

hybrid_sim(n_paths, n_steps, H, T, n_trunc, rng)
    Fast approximate simulation via the Bennedsen-Lunde-Pakkanen hybrid
    scheme using FFT-based convolution.

simulate(n_paths, n_steps, H, T, method, **kwargs)
    Unified dispatcher that delegates to one of the three schemes above.

fbm_paths(increments, T)
    Convert an increment array to cumulative fBm paths (prepends t=0).

References
----------
Hosking, J. R. M. (1984). Modeling persistence in hydrological time series
    using fractional differencing. Water Resources Research, 20(12), 1898-1908.

Bennedsen, M., Lunde, A., & Pakkanen, M. S. (2017). Hybrid scheme for
    Brownian semistationary processes. Finance and Stochastics, 21(4), 931-965.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, default_rng

from rough_vol.kernels import covariance_matrix, hybrid_weights, rl_covariance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_rng(rng: int | Generator | None) -> Generator:
    if isinstance(rng, Generator):
        return rng
    return default_rng(rng)


def _increment_cov_row(n_steps: int, H: float, dt: float) -> np.ndarray:
    """First row of the Toeplitz covariance matrix of fBm increments.

    The stationary covariance of increments ΔW^H_i = W^H(i·dt) - W^H((i-1)·dt)
    is

        γ(k) = dt^{2H} * 0.5 * [(k+1)^{2H} - 2k^{2H} + |k-1|^{2H}]

    with γ(0) = dt^{2H}.
    """
    k = np.arange(n_steps, dtype=float)
    # Clamp to ≥ 1 before the power to avoid (-1)^{2H} (complex) at k=0.
    safe_k = np.maximum(k, 1.0)
    cov = np.where(
        k == 0,
        1.0,
        0.5 * ((safe_k + 1) ** (2 * H) - 2 * safe_k ** (2 * H) + (safe_k - 1) ** (2 * H)),
    )
    return cov * (dt ** (2 * H))


# ---------------------------------------------------------------------------
# Cholesky scheme
# ---------------------------------------------------------------------------

def cholesky_sim(
    n_paths: int,
    n_steps: int,
    H: float,
    T: float = 1.0,
    rng: int | Generator | None = None,
) -> np.ndarray:
    """Simulate fBm increments via Cholesky factorisation (exact).

    Builds the full (n_steps × n_steps) covariance matrix of the fBm
    increments, computes its lower-triangular Cholesky factor L, and then
    returns L @ z for standard normal vectors z.

    Complexity: O(n_steps² ) memory, O(n_steps³ ) factorisation — suitable
    only for small n_steps (≲ 500).

    Parameters
    ----------
    n_paths : int
        Number of independent paths to simulate.
    n_steps : int
        Number of time steps (length of each increment sequence).
    H : float
        Hurst exponent in (0, 1).
    T : float
        Terminal time.  The step size is dt = T / n_steps.
    rng : int, Generator, or None
        Random seed or numpy Generator.  None → unseeded.

    Returns
    -------
    np.ndarray, shape (n_paths, n_steps)
        fBm increments on the uniform grid [0, T].
    """
    if not (0.0 < H < 1.0):
        raise ValueError(f"H must be in (0, 1), got {H}")
    rng = _make_rng(rng)
    dt = T / n_steps

    # Time grid for the *levels* W^H(0), W^H(dt), ..., W^H(T)
    times = np.linspace(0.0, T, n_steps + 1)  # shape (n_steps+1,)

    # Covariance matrix of the levels
    Sigma = covariance_matrix(times, H)        # (n_steps+1, n_steps+1)

    # Difference operator D such that increments = D @ levels
    D = np.eye(n_steps, n_steps + 1, k=1) - np.eye(n_steps, n_steps + 1)

    # Covariance of increments
    Sigma_inc = D @ Sigma @ D.T               # (n_steps, n_steps)

    # Regularise for numerical stability
    Sigma_inc += 1e-12 * np.eye(n_steps)

    L = np.linalg.cholesky(Sigma_inc)         # (n_steps, n_steps)
    z = rng.standard_normal((n_steps, n_paths))
    return (L @ z).T  # (n_paths, n_steps)


# ---------------------------------------------------------------------------
# Hosking scheme
# ---------------------------------------------------------------------------

def hosking_sim(
    n_paths: int,
    n_steps: int,
    H: float,
    T: float = 1.0,
    rng: int | Generator | None = None,
) -> np.ndarray:
    """Simulate fBm increments via the Hosking (1984) recursive algorithm (exact).

    The Hosking method generates a stationary Gaussian series with a given
    autocovariance sequence γ(k) by computing conditional means and variances
    recursively using the Durbin-Levinson algorithm.  Each new increment is
    drawn from its conditional Gaussian given all previous increments.

    Complexity: O(n_steps²) time, O(n_steps) memory — lower memory footprint
    than Cholesky but the same asymptotic cost.

    Parameters
    ----------
    n_paths : int
        Number of independent paths to simulate.
    n_steps : int
        Number of time steps.
    H : float
        Hurst exponent in (0, 1).
    T : float
        Terminal time.
    rng : int, Generator, or None
        Random seed or numpy Generator.

    Returns
    -------
    np.ndarray, shape (n_paths, n_steps)
        fBm increments on the uniform grid [0, T].
    """
    if not (0.0 < H < 1.0):
        raise ValueError(f"H must be in (0, 1), got {H}")
    rng = _make_rng(rng)
    dt = T / n_steps

    gamma = _increment_cov_row(n_steps, H, dt)  # shape (n_steps,)

    # Durbin-Levinson coefficients — shared across all paths
    phi = np.zeros(n_steps)
    # Conditional variances (scalar per step)
    v = np.zeros(n_steps)
    v[0] = gamma[0]

    # Output array: (n_paths, n_steps)
    out = np.empty((n_paths, n_steps))

    # First increment is just N(0, sqrt(gamma[0]))
    out[:, 0] = rng.standard_normal(n_paths) * np.sqrt(v[0])

    phi_prev = np.zeros(1)
    phi_prev[0] = gamma[1] / v[0] if n_steps > 1 else 0.0
    v_prev = v[0] * (1.0 - phi_prev[0] ** 2) if n_steps > 1 else v[0]

    for m in range(1, n_steps):
        # phi_prev holds the AR coefficients for lag m (length m)
        phi_m = phi_prev  # length m

        # Conditional mean  μ_{m+1} = φ_1 X_m + φ_2 X_{m-1} + ... + φ_m X_1
        # (Durbin-Levinson: phi_m[0] is the coefficient of the most recent obs)
        cond_mean = out[:, :m] @ phi_m[::-1]  # (n_paths,)
        cond_std = np.sqrt(max(v_prev, 1e-14))

        out[:, m] = cond_mean + rng.standard_normal(n_paths) * cond_std

        if m < n_steps - 1:
            # Update Durbin-Levinson for step m+1
            # New reflection coefficient
            num = gamma[m + 1] - np.dot(phi_m, gamma[1 : m + 1][::-1])
            phi_new = num / v_prev
            phi_next = np.empty(m + 1)
            phi_next[:m] = phi_m - phi_new * phi_m[::-1]
            phi_next[m] = phi_new
            v_next = v_prev * (1.0 - phi_new ** 2)
            phi_prev = phi_next
            v_prev = max(v_next, 1e-14)

    return out


# ---------------------------------------------------------------------------
# Hybrid scheme
# ---------------------------------------------------------------------------

def hybrid_sim(
    n_paths: int,
    n_steps: int,
    H: float,
    T: float = 1.0,
    n_trunc: int = 50,
    rng: int | Generator | None = None,
) -> np.ndarray:
    """Simulate fBm increments via circulant embedding (Wood-Chan / Davies-Harte).

    Embeds the Toeplitz autocovariance matrix of the fBm increments into a
    larger *circulant* matrix, diagonalises it with a single FFT, and generates
    exact samples in O(n_steps log n_steps) time.  This is the fast method
    suitable for large-scale Monte Carlo.

    The method is exact (up to floating-point arithmetic) for all H ∈ (0, 1),
    unlike the original BLP flat-quadrature hybrid scheme which has significant
    discretisation bias for rough H (H ≪ 0.5).  ``n_trunc`` is accepted for
    API compatibility but has no effect.

    Algorithm (Davies & Harte 1987, Wood & Chan 1994):

    1. Compute the autocovariance γ(k) of fBm increments.
    2. Form a circulant c of length m = 2·n_steps with rows
       [γ(0), …, γ(n-1), γ(n), γ(n-1), …, γ(1)].
    3. Compute eigenvalues λ = FFT(c) (all non-negative for fBm).
    4. Draw z ~ N(0, I_m); compute X = IRFFT(√λ · RFFT(z), m)[:n_steps].

    Parameters
    ----------
    n_paths : int
        Number of independent paths.
    n_steps : int
        Number of time steps.
    H : float
        Hurst exponent in (0, 1).
    T : float
        Terminal time.
    n_trunc : int
        Accepted for API compatibility; unused in this implementation.
    rng : int, Generator, or None
        Random seed or numpy Generator.

    Returns
    -------
    np.ndarray, shape (n_paths, n_steps)
        fBm increments on the uniform grid [0, T].

    References
    ----------
    Davies, R. B., & Harte, D. S. (1987). Tests for Hurst effect.
        Biometrika, 74(1), 95-101.
    Wood, A. T. A., & Chan, G. (1994). Simulation of stationary Gaussian
        processes in [0, 1]^d. Journal of Computational and Graphical
        Statistics, 3(4), 409-432.
    """
    if not (0.0 < H < 1.0):
        raise ValueError(f"H must be in (0, 1), got {H}")
    if n_trunc < 1:
        raise ValueError("n_trunc must be ≥ 1")
    rng = _make_rng(rng)
    dt = T / n_steps

    # Autocovariance of fBm increments at lags 0 … n_steps.
    # We need one extra lag (n_steps) to build the circulant embedding.
    k = np.arange(n_steps + 1, dtype=float)
    safe_k = np.maximum(k, 1.0)  # avoid (-1)^{2H} at k=0
    gamma = np.where(
        k == 0,
        1.0,
        0.5 * ((safe_k + 1) ** (2 * H) - 2 * safe_k ** (2 * H) + (safe_k - 1) ** (2 * H)),
    ) * (dt ** (2 * H))  # shape (n_steps+1,)

    # Circulant embedding of length m = 2*n_steps:
    #   c = [γ(0), γ(1), …, γ(n-1), γ(n), γ(n-1), …, γ(1)]
    m = 2 * n_steps
    c = np.empty(m)
    c[: n_steps + 1] = gamma                    # γ(0) … γ(n)
    c[n_steps + 1 :] = gamma[n_steps - 1: 0: -1]  # γ(n-1) … γ(1)

    # Eigenvalues of the circulant (real; non-negative for fBm).
    lam = np.fft.rfft(c).real                   # shape (n_steps+1,)
    lam = np.maximum(lam, 0.0)                  # guard against tiny negatives
    sqrt_lam = np.sqrt(lam)

    # Generate n_paths samples: draw z ~ N(0, I_m), scale in frequency domain.
    z = rng.standard_normal((n_paths, m))
    Z = np.fft.rfft(z, axis=1)                  # (n_paths, n_steps+1)
    X = np.fft.irfft(sqrt_lam[np.newaxis, :] * Z, n=m, axis=1)  # (n_paths, m)

    return X[:, :n_steps]


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def simulate(
    n_paths: int,
    n_steps: int,
    H: float,
    T: float = 1.0,
    method: str = "hybrid",
    **kwargs,
) -> np.ndarray:
    """Simulate fBm increments using a chosen scheme.

    Parameters
    ----------
    n_paths : int
        Number of independent paths.
    n_steps : int
        Number of time steps.
    H : float
        Hurst exponent in (0, 1).
    T : float
        Terminal time.
    method : {"hybrid", "cholesky", "hosking"}
        Simulation scheme.  Defaults to "hybrid".
    **kwargs
        Extra keyword arguments forwarded to the chosen scheme (e.g.
        ``n_trunc`` for the hybrid method, ``rng`` for all methods).

    Returns
    -------
    np.ndarray, shape (n_paths, n_steps)
        fBm increments.
    """
    methods = {
        "cholesky": cholesky_sim,
        "hosking": hosking_sim,
        "hybrid": hybrid_sim,
    }
    key = method.lower()
    if key not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(methods)}")
    return methods[key](n_paths, n_steps, H, T, **kwargs)


# ---------------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------------

def fbm_paths(increments: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Convert an fBm increment array to cumulative paths.

    Prepends a column of zeros (the value at t = 0) and returns cumulative
    sums so that the output represents W^H(t_0), W^H(t_1), …, W^H(t_n).

    Parameters
    ----------
    increments : np.ndarray, shape (n_paths, n_steps)
        Increment array as returned by any of the simulation functions.
    T : float
        Terminal time (unused in computation; kept for API consistency).

    Returns
    -------
    np.ndarray, shape (n_paths, n_steps + 1)
        Cumulative fBm paths starting from 0.
    """
    n_paths, n_steps = increments.shape
    zeros = np.zeros((n_paths, 1))
    return np.concatenate([zeros, np.cumsum(increments, axis=1)], axis=1)
