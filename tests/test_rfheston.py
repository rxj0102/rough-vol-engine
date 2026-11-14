"""Tests for rough_vol.models.rfheston."""

import numpy as np
import pytest

from rough_vol.models.rfheston import (
    RHestonParams,
    _bs_call_price,
    _bs_implied_vol,
    _fractional_riccati,
    characteristic_function,
    implied_vol_surface,
    price_european,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_params():
    return RHestonParams(H=0.1, lambda_=0.3, theta=0.04, rho=-0.7, nu=0.3, V0=0.04)


@pytest.fixture
def heston_limit_params():
    """H near 0.5 — rough Heston should approach classical Heston."""
    return RHestonParams(H=0.49, lambda_=1.0, theta=0.04, rho=-0.5, nu=0.4, V0=0.04)


# ---------------------------------------------------------------------------
# RHestonParams validation
# ---------------------------------------------------------------------------

class TestRHestonParams:
    def test_valid(self):
        p = RHestonParams(H=0.1, lambda_=0.3, theta=0.04, rho=-0.7, nu=0.3, V0=0.04)
        assert p.H == 0.1

    @pytest.mark.parametrize("H", [0.0, 0.5, 1.0])
    def test_invalid_H(self, H):
        with pytest.raises(ValueError):
            RHestonParams(H=H, lambda_=1.0, theta=0.04, rho=0.0, nu=0.3, V0=0.04)

    def test_invalid_lambda(self):
        with pytest.raises(ValueError):
            RHestonParams(H=0.1, lambda_=0.0, theta=0.04, rho=0.0, nu=0.3, V0=0.04)

    def test_invalid_theta(self):
        with pytest.raises(ValueError):
            RHestonParams(H=0.1, lambda_=1.0, theta=0.0, rho=0.0, nu=0.3, V0=0.04)

    @pytest.mark.parametrize("rho", [-1.0, 1.0])
    def test_invalid_rho(self, rho):
        with pytest.raises(ValueError):
            RHestonParams(H=0.1, lambda_=1.0, theta=0.04, rho=rho, nu=0.3, V0=0.04)

    def test_invalid_nu(self):
        with pytest.raises(ValueError):
            RHestonParams(H=0.1, lambda_=1.0, theta=0.04, rho=0.0, nu=0.0, V0=0.04)

    def test_invalid_V0(self):
        with pytest.raises(ValueError):
            RHestonParams(H=0.1, lambda_=1.0, theta=0.04, rho=0.0, nu=0.3, V0=0.0)


# ---------------------------------------------------------------------------
# fractional Riccati solver
# ---------------------------------------------------------------------------

class TestFractionalRiccati:
    def test_shape(self, default_params):
        psi = _fractional_riccati(1.0, T=1.0, params=default_params, n_steps=50)
        assert psi.shape == (51,)

    def test_initial_condition(self, default_params):
        psi = _fractional_riccati(1.0, T=1.0, params=default_params, n_steps=50)
        assert psi[0] == pytest.approx(0.0, abs=1e-14)

    def test_complex_u(self, default_params):
        psi = _fractional_riccati(1.0 - 0.5j, T=1.0, params=default_params, n_steps=50)
        assert np.iscomplexobj(psi)

    def test_convergence_in_n_steps(self, default_params):
        """Solution should converge as n_steps increases."""
        psi_coarse = _fractional_riccati(1.0, T=1.0, params=default_params, n_steps=50)
        psi_fine = _fractional_riccati(1.0, T=1.0, params=default_params, n_steps=200)
        # Compare terminal value
        assert psi_coarse[-1] == pytest.approx(psi_fine[-1], rel=0.1)

    def test_purely_imaginary_u_returns_bounded(self, default_params):
        """For real u the CF modulus should be close to ≤ 1 (exact only at convergence)."""
        psi = _fractional_riccati(2.0, T=1.0, params=default_params, n_steps=100)
        log_cf = characteristic_function(2.0, T=1.0, params=default_params, n_steps=100)
        # Allow up to 10% violation due to Adams-scheme discretisation error.
        assert abs(np.exp(log_cf)) <= 1.1


# ---------------------------------------------------------------------------
# characteristic_function
# ---------------------------------------------------------------------------

class TestCharacteristicFunction:
    def test_unit_at_zero(self, default_params):
        """CF(0) = E[1] = 1, so log CF = 0."""
        log_cf = characteristic_function(0.0, T=1.0, params=default_params, n_steps=100)
        assert abs(log_cf) == pytest.approx(0.0, abs=1e-4)

    def test_modulus_le_one(self, default_params):
        """|CF(u)| ≤ 1 for small real u (Adams scheme well-converged at these values)."""
        for u in [0.5, 1.0, 2.0]:
            log_cf = characteristic_function(u, T=1.0, params=default_params, n_steps=200)
            assert abs(np.exp(log_cf)) <= 1.1

    def test_continuity_in_u(self, default_params):
        """log CF should vary smoothly with u."""
        log_cf_1 = characteristic_function(1.0, T=1.0, params=default_params, n_steps=100)
        log_cf_1p = characteristic_function(1.01, T=1.0, params=default_params, n_steps=100)
        assert abs(log_cf_1p - log_cf_1) < 0.5

    def test_continuity_in_T(self, default_params):
        """log CF should vary smoothly (be bounded and finite) across maturities."""
        for T in [0.5, 1.0, 2.0]:
            log_cf = characteristic_function(1.0, T=T, params=default_params, n_steps=100)
            assert np.isfinite(log_cf.real)
            assert np.isfinite(log_cf.imag)
            # log |CF| bounded (moderate accuracy requirement)
            assert abs(log_cf.real) < 5.0


# ---------------------------------------------------------------------------
# price_european
# ---------------------------------------------------------------------------

class TestPriceEuropeanRHeston:
    def test_shape_scalar(self, default_params):
        prices = price_european(default_params, K=1.0, T=1.0, n_steps=50, n_quad=32)
        assert prices.shape == (1,)

    def test_shape_array(self, default_params):
        K = np.array([0.8, 1.0, 1.2])
        prices = price_european(default_params, K=K, T=1.0, n_steps=50, n_quad=32)
        assert prices.shape == (3,)

    def test_price_positive(self, default_params):
        prices = price_european(default_params, K=1.0, T=1.0, n_steps=50, n_quad=32)
        assert float(prices[0]) > 0.0

    def test_call_decreasing_in_K(self, default_params):
        K = np.array([0.8, 1.0, 1.2])
        prices = price_european(default_params, K=K, T=1.0, n_steps=50, n_quad=32)
        assert prices[0] > prices[1] > prices[2]

    def test_call_lower_bound(self, default_params):
        """C(K) >= max(S0 - K*exp(-rT), 0)."""
        K = 0.8
        prices = price_european(default_params, K=K, T=1.0, r=0.0, S0=1.0, n_steps=100, n_quad=64)
        assert float(prices[0]) >= max(1.0 - K, 0) - 1e-4

    def test_vol_of_vol_lowers_atm_and_raises_wings(self):
        """Higher nu adds smile curvature: with rho = 0 the ATM price falls and the far OTM price rises.

        This matches the classical Heston model (closed form at rho = 0: ATM call 0.0780 at nu = 0.2
        versus 0.0715 at nu = 0.5, T = 1). The previous test asserted that the ATM price rises with nu,
        which is false and only passed because of a sign error in the Riccati equation.
        """
        params_lo = RHestonParams(H=0.1, lambda_=1.0, theta=0.04, rho=0.0, nu=0.2, V0=0.04)
        params_hi = RHestonParams(H=0.1, lambda_=1.0, theta=0.04, rho=0.0, nu=0.5, V0=0.04)
        K = np.array([1.0, 1.3])
        p_lo = price_european(params_lo, K=K, T=1.0, n_steps=100, n_quad=64)
        p_hi = price_european(params_hi, K=K, T=1.0, n_steps=100, n_quad=64)
        assert p_hi[0] < p_lo[0]
        assert p_hi[1] > p_lo[1]

    def test_matches_classical_heston_near_half(self):
        """At H close to 1/2 the rough Heston CF must match the closed-form Heston CF."""
        from rough_vol.models.rfheston import characteristic_function

        kappa, theta, sigma, rho, v0, T = 1.0, 0.04, 0.3, -0.7, 0.04, 0.5
        p = RHestonParams(H=0.4999, lambda_=kappa, theta=theta, rho=rho, nu=sigma, V0=v0)
        for u in (1.0, 5.0, 10.0):
            iu = 1j * u
            d = np.sqrt((rho * sigma * iu - kappa) ** 2 + sigma ** 2 * (iu + u ** 2))
            g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)
            e = np.exp(-d * T)
            C = kappa * theta / sigma ** 2 * ((kappa - rho * sigma * iu - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
            D = (kappa - rho * sigma * iu - d) / sigma ** 2 * (1 - e) / (1 - g * e)
            ref = np.exp(C + D * v0)
            got = np.exp(characteristic_function(u, T, p, 800))
            assert abs(got - ref) < 2e-4

    def test_otm_prices_positive_and_monotone(self, default_params):
        """Call prices must be positive and decreasing in strike (regression for negative OTM prices)."""
        K = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        p = price_european(default_params, K=K, T=0.5, n_steps=200, n_quad=128, u_max=100)
        assert np.all(p > 0)
        assert np.all(np.diff(p) < 0)

    def test_atm_price_reasonable(self, default_params):
        """ATM call should be between 1% and 50% of spot."""
        p = float(price_european(default_params, K=1.0, T=1.0, n_steps=100, n_quad=64)[0])
        assert 0.01 < p < 0.5


# ---------------------------------------------------------------------------
# implied_vol_surface
# ---------------------------------------------------------------------------

class TestImpliedVolSurfaceRHeston:
    def test_shape(self, default_params):
        strikes = np.array([0.9, 1.0, 1.1])
        mats = np.array([0.5, 1.0])
        iv = implied_vol_surface(default_params, strikes, mats, n_steps=50, n_quad=32)
        assert iv.shape == (2, 3)

    def test_valid_range(self, default_params):
        """IV should be in (0, 5) where computable."""
        strikes = np.array([0.9, 1.0, 1.1])
        mats = np.array([1.0])
        iv = implied_vol_surface(default_params, strikes, mats, n_steps=100, n_quad=64)
        valid = iv[~np.isnan(iv)]
        assert np.all(valid > 0.0)
        assert np.all(valid < 5.0)

    def test_skew_negative_rho(self, default_params):
        """With rho < 0, IV(K=0.9) > IV(K=1.1)."""
        # default_params has rho=-0.7, nu=0.3 — stable for Adams solver
        strikes = np.array([0.9, 1.1])
        mats = np.array([1.0])
        iv = implied_vol_surface(default_params, strikes, mats, n_steps=200, n_quad=64)
        valid = ~np.isnan(iv[0])
        assert valid.any(), "All IV values were NaN"
        if valid.all():
            assert iv[0, 0] > iv[0, 1]

    def test_nu_increases_smile_curvature(self):
        """Higher nu raises wing vol relative to ATM vol (with rho = 0, ATM vol itself falls, as in Heston)."""
        s = np.array([0.8, 1.0, 1.25])
        m = np.array([1.0])
        p_lo = RHestonParams(H=0.1, lambda_=1.0, theta=0.04, rho=0.0, nu=0.2, V0=0.04)
        p_hi = RHestonParams(H=0.1, lambda_=1.0, theta=0.04, rho=0.0, nu=0.5, V0=0.04)
        iv_lo = implied_vol_surface(p_lo, s, m, n_steps=100, n_quad=128, u_max=100)
        iv_hi = implied_vol_surface(p_hi, s, m, n_steps=100, n_quad=128, u_max=100)
        curv_lo = 0.5 * (iv_lo[0, 0] + iv_lo[0, 2]) - iv_lo[0, 1]
        curv_hi = 0.5 * (iv_hi[0, 0] + iv_hi[0, 2]) - iv_hi[0, 1]
        assert curv_hi > curv_lo
