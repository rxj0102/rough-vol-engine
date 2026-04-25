"""
Greeks and price sensitivities.

Computes option Greeks (delta, gamma, vega, theta, rho) via finite-difference
bump-and-reprice and pathwise (likelihood-ratio) estimators using Monte Carlo
paths from rough_vol.simulation.
"""
