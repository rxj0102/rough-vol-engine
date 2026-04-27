"""
rough_vol.variance — variance swap pricing and forward variance analysis.

Sub-modules
-----------
``integrated``
    Trapezoidal integrated variance and realized quadratic variation from
    simulated paths; variance risk premium estimation.
``forward``
    Forward variance curve extraction from implied-vol surfaces; model
    forward variance for rBergomi and rough Heston.
``swap``
    Variance swap fair-strike computation via Monte Carlo (rBergomi) and
    model-free log-contract replication from a surface.

Quick reference
---------------
>>> from rough_vol.variance import (
...     integrated_variance,
...     annualized_integrated_variance,
...     realized_variance,
...     atm_total_variance,
...     forward_variance_curve,
...     variance_swap_strike_mc,
...     variance_swap_strike_atm_approx,
...     variance_swap_pnl,
... )
"""

from rough_vol.variance.integrated import (
    annualized_integrated_variance,
    annualized_realized_variance,
    integrated_variance,
    realized_variance,
    variance_path_stats,
    variance_risk_premium,
)
from rough_vol.variance.forward import (
    atm_total_variance,
    forward_variance_curve,
    forward_variance_rfheston,
    instantaneous_forward_variance_rbergomi,
    term_structure_of_variance,
)
from rough_vol.variance.swap import (
    variance_swap_breakeven,
    variance_swap_delta,
    variance_swap_pnl,
    variance_swap_strike_atm_approx,
    variance_swap_strike_log_contract,
    variance_swap_strike_mc,
)

__all__ = [
    # integrated
    "integrated_variance",
    "annualized_integrated_variance",
    "realized_variance",
    "annualized_realized_variance",
    "variance_risk_premium",
    "variance_path_stats",
    # forward
    "atm_total_variance",
    "forward_variance_curve",
    "instantaneous_forward_variance_rbergomi",
    "forward_variance_rfheston",
    "term_structure_of_variance",
    # swap
    "variance_swap_strike_mc",
    "variance_swap_strike_atm_approx",
    "variance_swap_strike_log_contract",
    "variance_swap_pnl",
    "variance_swap_breakeven",
    "variance_swap_delta",
]
