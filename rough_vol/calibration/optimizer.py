"""
Parameter calibration optimiser.

Fits model parameters (e.g. H, η, ρ for rBergomi) to an observed implied
volatility surface by minimising a weighted root-mean-square error objective.
Wraps scipy.optimize solvers and supports both local (L-BFGS-B, Nelder-Mead)
and global (differential evolution) search strategies.
"""
