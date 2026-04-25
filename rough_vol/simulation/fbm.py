"""
Fractional Brownian motion (fBm) path generators.

Provides two simulation schemes for the Riemann-Liouville fBm used in rough
volatility models:

- **Cholesky scheme** — exact but O(n²) in memory; suited for low path counts.
- **Hybrid scheme** (Bennedsen, Lunde & Pakkanen, 2017) — approximate but
  O(n log n) via FFT convolution; the default for production Monte Carlo.

References
----------
Bennedsen, M., Lunde, A., & Pakkanen, M. S. (2017). Hybrid scheme for
    Brownian semistationary processes. Finance and Stochastics, 21(4),
    931-965.
"""
