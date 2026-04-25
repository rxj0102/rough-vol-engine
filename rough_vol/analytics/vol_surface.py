"""
Volatility surface construction and interpolation.

Builds implied volatility surfaces from model prices over a (strike, maturity)
grid, applies arbitrage-free smoothing, and provides strike / maturity
interpolation.  Also exposes term-structure and skew summary statistics.
"""
