"""
Rough Heston model (El Euch & Rosenbaum, 2019).

Implements the rough Heston model where the instantaneous variance follows a
fractional Riccati equation.  The model inherits the mean-reversion and
leverage structure of the classical Heston model while reproducing the rough
behaviour of realised volatility at short time scales.

References
----------
El Euch, O., & Rosenbaum, M. (2019). The characteristic function of rough
    Heston models. Mathematical Finance, 29(1), 3-38.
"""
