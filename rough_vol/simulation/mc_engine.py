"""
Monte Carlo simulation engine.

Orchestrates end-to-end Monte Carlo pricing: draws correlated Brownian
increments, delegates variance process simulation to a model instance, evolves
the asset price, and hands the resulting paths to a pricer.  Supports
variance-reduction techniques including antithetic variates and control
variates.
"""
