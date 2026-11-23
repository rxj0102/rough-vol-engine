"""Model-free variance swap strike (Demeterfi, Derman, Kamal and Zou 1999)."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from rough_vol.calibration.surface import ImpliedVolSurface
from rough_vol.variance.swap import variance_swap_strike_log_contract


def _surface(strikes, vols, T=0.5, F=100.0):
    return ImpliedVolSurface(
        strikes=np.asarray(strikes, dtype=float),
        maturities=np.array([T]),
        implied_vols=np.asarray(vols, dtype=float)[np.newaxis, :],
        spot=F,
        forwards=np.array([F]),
    )


def test_flat_smile_returns_the_variance():
    strikes = 100.0 * np.exp(np.linspace(-2.0, 2.0, 81))
    s = _surface(strikes, np.full(81, 0.2))
    assert variance_swap_strike_log_contract(s, 0.5, n_quad=2001) == pytest.approx(0.04, rel=2e-3)


def test_flat_smile_does_not_depend_on_the_forward_level():
    strikes = np.exp(np.linspace(-2.0, 2.0, 81))
    a = variance_swap_strike_log_contract(_surface(strikes, np.full(81, 0.3), F=1.0), 0.5, 2001)
    b = variance_swap_strike_log_contract(
        _surface(250.0 * strikes, np.full(81, 0.3), F=250.0), 0.5, 2001
    )
    assert a == pytest.approx(b, rel=1e-9)


def test_matches_direct_strike_integral_on_a_skewed_smile():
    T, F = 0.25, 100.0
    k = np.linspace(-1.0, 0.6, 41)
    vols = 0.2 - 0.15 * k + 0.1 * k**2
    s = _surface(F * np.exp(k), vols, T=T, F=F)

    def otm(K):
        v = np.interp(np.log(K / F), k, vols)
        sd = v * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sd**2) / sd
        d2 = d1 - sd
        if K < F:
            return K * norm.cdf(-d2) - F * norm.cdf(-d1)
        return F * norm.cdf(d1) - K * norm.cdf(d2)

    lo, hi = F * np.exp(k[0]), F * np.exp(k[-1])
    direct = 2.0 / T * (
        quad(lambda K: otm(K) / K**2, lo, F, limit=200)[0]
        + quad(lambda K: otm(K) / K**2, F, hi, limit=200)[0]
    )
    got = variance_swap_strike_log_contract(s, T, n_quad=4001)
    assert got == pytest.approx(direct, rel=1e-4)


def test_negative_skew_puts_the_strike_above_atm_variance():
    k = np.linspace(-1.5, 1.0, 51)
    vols = np.clip(0.2 - 0.2 * k, 0.08, None)
    s = _surface(100.0 * np.exp(k), vols)
    assert variance_swap_strike_log_contract(s, 0.5, 2001) > 0.2**2
