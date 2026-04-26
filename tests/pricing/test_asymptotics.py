"""Tests for rough_vol.pricing.asymptotics."""

import math

import numpy as np
import pytest
from scipy.special import gamma

from rough_vol.models.rbergomi import RBergomiParams
from rough_vol.pricing.asymptotics import (
    atm_implied_vol,
    atm_skew,
    atm_skew_coefficient,
    fukasawa_coefficient,
    rough_vol_term_structure,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_params():
    return RBergomiParams(H=0.1, eta=1.9, rho=-0.9, xi0=0.04)


@pytest.fixture
def zero_skew_params():
    """rho=0 → skew = 0 for any eta, H, xi0."""
    return RBergomiParams(H=0.1, eta=1.9, rho=0.0, xi0=0.04)


# ---------------------------------------------------------------------------
# fukasawa_coefficient
# ---------------------------------------------------------------------------

class TestFukasawaCoefficient:
    @pytest.mark.parametrize("H", [0.1, 0.2, 0.3, 0.4])
    def test_positive(self, H):
        assert fukasawa_coefficient(H) > 0.0

    def test_formula(self):
        H = 0.1
        expected = 2.0 / ((2 * H + 1) ** 2 * gamma(H + 0.5) ** 2)
        assert fukasawa_coefficient(H) == pytest.approx(expected, rel=1e-10)

    def test_classical_limit(self):
        """At H=0.5, C_H = 2/((1+1)^2 * Γ(1)^2) = 2/4 = 0.5."""
        # H=0.5 is outside valid range but we test the limit analytically
        H = 0.499
        C = fukasawa_coefficient(H)
        expected_limit = 0.5
        assert C == pytest.approx(expected_limit, rel=0.01)

    def test_decreasing_in_H(self):
        """C_H decreases as H increases toward 0.5."""
        C_values = [fukasawa_coefficient(H) for H in [0.1, 0.2, 0.3, 0.4]]
        assert all(C_values[i] > C_values[i + 1] for i in range(len(C_values) - 1))

    def test_invalid_H_zero(self):
        with pytest.raises(ValueError):
            fukasawa_coefficient(0.0)

    def test_invalid_H_half(self):
        with pytest.raises(ValueError):
            fukasawa_coefficient(0.5)

    def test_invalid_H_negative(self):
        with pytest.raises(ValueError):
            fukasawa_coefficient(-0.1)


# ---------------------------------------------------------------------------
# atm_implied_vol
# ---------------------------------------------------------------------------

class TestAtmImpliedVol:
    def test_equals_sqrt_xi0(self, default_params):
        assert atm_implied_vol(default_params, T=1.0) == pytest.approx(math.sqrt(default_params.xi0))

    def test_independent_of_T(self, default_params):
        iv1 = atm_implied_vol(default_params, T=0.5)
        iv2 = atm_implied_vol(default_params, T=2.0)
        assert iv1 == pytest.approx(iv2, rel=1e-10)

    def test_correct_value(self):
        # xi0=0.04 → sqrt(xi0)=0.2
        params = RBergomiParams(H=0.1, eta=1.0, rho=0.0, xi0=0.04)
        assert atm_implied_vol(params, T=1.0) == pytest.approx(0.2, rel=1e-10)

    @pytest.mark.parametrize("xi0", [0.01, 0.04, 0.09, 0.16])
    def test_various_xi0(self, xi0):
        params = RBergomiParams(H=0.1, eta=1.0, rho=0.0, xi0=xi0)
        assert atm_implied_vol(params, T=1.0) == pytest.approx(math.sqrt(xi0), rel=1e-10)


# ---------------------------------------------------------------------------
# atm_skew
# ---------------------------------------------------------------------------

class TestAtmSkew:
    def test_negative_rho_gives_negative_skew(self, default_params):
        assert atm_skew(default_params, T=1.0) < 0.0

    def test_positive_rho_gives_positive_skew(self):
        params = RBergomiParams(H=0.1, eta=1.0, rho=0.5, xi0=0.04)
        assert atm_skew(params, T=1.0) > 0.0

    def test_zero_rho_gives_zero_skew(self, zero_skew_params):
        assert atm_skew(zero_skew_params, T=1.0) == pytest.approx(0.0, abs=1e-15)

    def test_power_law_scaling(self, default_params):
        """Skew should scale as T^{H - 0.5}."""
        T1, T2 = 1.0, 4.0
        s1 = atm_skew(default_params, T1)
        s2 = atm_skew(default_params, T2)
        H = default_params.H
        expected_ratio = (T2 / T1) ** (H - 0.5)
        assert s2 / s1 == pytest.approx(expected_ratio, rel=1e-8)

    def test_formula_value(self, default_params):
        T = 1.0
        H, eta, rho = default_params.H, default_params.eta, default_params.rho
        C_H = fukasawa_coefficient(H)
        expected = rho * eta * C_H * T ** (H - 0.5)
        assert atm_skew(default_params, T) == pytest.approx(expected, rel=1e-10)

    def test_invalid_T_zero(self, default_params):
        with pytest.raises(ValueError):
            atm_skew(default_params, T=0.0)

    def test_invalid_T_negative(self, default_params):
        with pytest.raises(ValueError):
            atm_skew(default_params, T=-1.0)

    def test_skew_blows_up_as_T_to_zero(self, default_params):
        """Short-maturity skew should diverge as T → 0 (H < 0.5)."""
        s_short = abs(atm_skew(default_params, T=0.01))
        s_long = abs(atm_skew(default_params, T=1.0))
        assert s_short > s_long

    def test_increases_with_eta(self):
        """Larger eta → larger |skew|."""
        p_lo = RBergomiParams(H=0.1, eta=0.5, rho=-0.9, xi0=0.04)
        p_hi = RBergomiParams(H=0.1, eta=2.0, rho=-0.9, xi0=0.04)
        assert abs(atm_skew(p_hi, T=1.0)) > abs(atm_skew(p_lo, T=1.0))


# ---------------------------------------------------------------------------
# atm_skew_coefficient
# ---------------------------------------------------------------------------

class TestAtmSkewCoefficient:
    def test_equals_rho_eta_CH(self, default_params):
        H, eta, rho = default_params.H, default_params.eta, default_params.rho
        expected = rho * eta * fukasawa_coefficient(H)
        assert atm_skew_coefficient(default_params) == pytest.approx(expected, rel=1e-10)

    def test_coefficient_times_T_power_equals_skew(self, default_params):
        T = 0.5
        H = default_params.H
        c = atm_skew_coefficient(default_params)
        skew = atm_skew(default_params, T)
        assert c * T ** (H - 0.5) == pytest.approx(skew, rel=1e-10)

    def test_zero_rho(self, zero_skew_params):
        assert atm_skew_coefficient(zero_skew_params) == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# rough_vol_term_structure
# ---------------------------------------------------------------------------

class TestRoughVolTermStructure:
    def test_keys(self, default_params):
        ts = rough_vol_term_structure(default_params, np.array([0.5, 1.0, 2.0]))
        assert "maturities" in ts
        assert "atm_vol" in ts
        assert "atm_skew" in ts

    def test_shapes(self, default_params):
        mats = np.array([0.25, 0.5, 1.0, 2.0])
        ts = rough_vol_term_structure(default_params, mats)
        assert ts["maturities"].shape == (4,)
        assert ts["atm_vol"].shape == (4,)
        assert ts["atm_skew"].shape == (4,)

    def test_vol_constant(self, default_params):
        """ATM vol is constant (√ξ₀) across maturities."""
        mats = np.array([0.5, 1.0, 2.0])
        ts = rough_vol_term_structure(default_params, mats)
        np.testing.assert_allclose(ts["atm_vol"], math.sqrt(default_params.xi0))

    def test_skew_power_law(self, default_params):
        """Skew should follow power-law T^{H-0.5}."""
        mats = np.array([0.5, 1.0, 2.0])
        ts = rough_vol_term_structure(default_params, mats)
        skews = ts["atm_skew"]
        H = default_params.H
        ratios = skews[1:] / skews[:-1]
        T_ratios = (mats[1:] / mats[:-1]) ** (H - 0.5)
        np.testing.assert_allclose(ratios, T_ratios, rtol=1e-8)

    def test_skew_negative_for_negative_rho(self, default_params):
        mats = np.array([0.5, 1.0, 2.0])
        ts = rough_vol_term_structure(default_params, mats)
        assert np.all(ts["atm_skew"] < 0.0)

    def test_single_maturity(self, default_params):
        ts = rough_vol_term_structure(default_params, np.array([1.0]))
        assert ts["atm_vol"].shape == (1,)
        assert ts["atm_skew"].shape == (1,)
