"""
rough_vol.calibration — model parameter calibration.

Provides routines for fitting rough volatility model parameters to
observed implied volatility surfaces via gradient-free and gradient-based
optimisers.
"""

from rough_vol.calibration.surface import ImpliedVolSurface, MarketSlice, VolSurface
from rough_vol.calibration.objective import (
    RBergomiObjective,
    RHestonObjective,
    weighted_mae,
    weighted_rmse,
)
from rough_vol.calibration.optimizer import (
    CalibrationResult,
    calibrate_rbergomi,
    calibrate_rfheston,
)

__all__ = [
    "ImpliedVolSurface",
    "MarketSlice",
    "VolSurface",
    "RBergomiObjective",
    "RHestonObjective",
    "weighted_rmse",
    "weighted_mae",
    "CalibrationResult",
    "calibrate_rbergomi",
    "calibrate_rfheston",
]
