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
