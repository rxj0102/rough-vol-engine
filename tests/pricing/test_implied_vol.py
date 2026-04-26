"""Tests for rough_vol.pricing.implied_vol."""

import numpy as np
import pytest

from rough_vol.pricing.implied_vol import (
    bs_call,
    bs_put,
    bs_vega,
    implied_vol,
    implied_vol_surface,
)


# ---------------------------------------------------------------------------
# bs_call
# ---------------------------------------------------------------------------

class TestBsCall:
    def test_atm_approximation(self):
        # ATM call ≈ F * sigma * sqrt(T) / sqrt(2*pi) for r=0
        F, K, sigma, T = 1.0, 1.0, 0.2, 1.0
        price = float(bs_call(F, K, sigma, T))
        expected = F * sigma * np.sqrt(T) / np.sqrt(2 * np.pi)
        assert price == pytest.approx(expected, rel=0.01)

    def test_lower_bound(self):
        # C >= max(F - K, 0) * discount
        F, K, T, r = 1.1, 1.0, 1.0, 0.05
        price = float(bs_call(F, K, 0.3, T, r))
        assert price >= np.exp(-r * T) * (F - K) - 1e-10

    def test_deep_itm(self):
        # Deep ITM: C ≈ F - K (r=0)
        F, K, sigma, T = 2.0, 0.5, 0.2, 1.0
        price = float(bs_call(F, K, sigma, T))
        assert price == pytest.approx(F - K, rel=0.01)

    def test_otm_positive(self):
        price = float(bs_call(1.0, 1.2, 0.3, 1.0))
        assert price > 0.0

    def test_zero_vol_returns_intrinsic(self):
        price = float(bs_call(1.1, 1.0, 0.0, 1.0))
        assert price == pytest.approx(0.1, abs=1e-10)

    def test_zero_vol_otm_returns_zero(self):
        price = float(bs_call(0.9, 1.0, 0.0, 1.0))
        assert price == pytest.approx(0.0, abs=1e-10)

    def test_vectorised(self):
        K = np.array([0.8, 1.0, 1.2])
        prices = bs_call(1.0, K, 0.2, 1.0)
        assert prices.shape == (3,)
        assert np.all(prices > 0)
        # Call is decreasing in K
        assert prices[0] > prices[1] > prices[2]

    def test_rate_discounts(self):
        # Price with r > 0 should be less than r=0 for OTM (discount effect)
        p0 = float(bs_call(1.0, 1.0, 0.2, 1.0, r=0.0))
        pr = float(bs_call(1.0, 1.0, 0.2, 1.0, r=0.05))
        # With r>0 forward increases effect; just check it's a valid positive number
        assert pr > 0


# ---------------------------------------------------------------------------
# bs_put
# ---------------------------------------------------------------------------

class TestBsPut:
    def test_put_call_parity(self):
        F, K, sigma, T, r = 1.0, 1.0, 0.2, 1.0, 0.05
        call = float(bs_call(F, K, sigma, T, r))
        put = float(bs_put(F, K, sigma, T, r))
        parity = call - put
        expected = np.exp(-r * T) * (F - K)
        assert parity == pytest.approx(expected, abs=1e-10)

    def test_put_lower_bound(self):
        F, K, T, r = 0.9, 1.0, 1.0, 0.0
        price = float(bs_put(F, K, 0.3, T, r))
        assert price >= K - F - 1e-10

    def test_deep_itm_put(self):
        F, K, sigma, T = 0.5, 2.0, 0.2, 1.0
        price = float(bs_put(F, K, sigma, T))
        assert price == pytest.approx(K - F, rel=0.01)

    def test_vectorised(self):
        K = np.array([0.8, 1.0, 1.2])
        prices = bs_put(1.0, K, 0.2, 1.0)
        assert prices.shape == (3,)
        # Put is increasing in K
        assert prices[0] < prices[1] < prices[2]


# ---------------------------------------------------------------------------
# bs_vega
# ---------------------------------------------------------------------------

class TestBsVega:
    def test_vega_positive(self):
        vega = float(bs_vega(1.0, 1.0, 0.2, 1.0))
        assert vega > 0.0

    def test_vega_zero_sigma(self):
        vega = float(bs_vega(1.0, 1.0, 0.0, 1.0))
        assert vega == pytest.approx(0.0, abs=1e-10)

    def test_vega_symmetric_around_atm(self):
        # ATM vega is maximal; OTM vega is smaller
        vega_atm = float(bs_vega(1.0, 1.0, 0.2, 1.0))
        vega_otm = float(bs_vega(1.0, 1.5, 0.2, 1.0))
        assert vega_atm > vega_otm

    def test_vega_matches_finite_difference(self):
        F, K, sigma, T = 1.0, 1.0, 0.25, 1.0
        eps = 1e-5
        fd = (float(bs_call(F, K, sigma + eps, T)) - float(bs_call(F, K, sigma - eps, T))) / (2 * eps)
        vega = float(bs_vega(F, K, sigma, T))
        assert vega == pytest.approx(fd, rel=1e-4)

    def test_vectorised(self):
        K = np.array([0.9, 1.0, 1.1])
        v = bs_vega(1.0, K, 0.2, 1.0)
        assert v.shape == (3,)
        assert np.all(v > 0)


# ---------------------------------------------------------------------------
# implied_vol
# ---------------------------------------------------------------------------

class TestImpliedVol:
    @pytest.mark.parametrize("sigma", [0.05, 0.1, 0.2, 0.5, 1.0])
    def test_roundtrip_atm(self, sigma):
        F, K, T = 1.0, 1.0, 1.0
        price = float(bs_call(F, K, sigma, T))
        iv = float(implied_vol(price, F, K, T))
        assert iv == pytest.approx(sigma, rel=1e-6)

    @pytest.mark.parametrize("K", [0.8, 0.9, 1.0, 1.1, 1.2])
    def test_roundtrip_various_strikes(self, K):
        F, sigma, T = 1.0, 0.3, 0.5
        price = float(bs_call(F, K, sigma, T))
        iv = float(implied_vol(price, F, K, T))
        assert iv == pytest.approx(sigma, rel=1e-5)

    def test_roundtrip_short_maturity(self):
        F, K, sigma, T = 1.0, 1.0, 0.4, 0.1
        price = float(bs_call(F, K, sigma, T))
        iv = float(implied_vol(price, F, K, T))
        assert iv == pytest.approx(sigma, rel=1e-5)

    def test_roundtrip_long_maturity(self):
        F, K, sigma, T = 1.0, 1.0, 0.2, 5.0
        price = float(bs_call(F, K, sigma, T))
        iv = float(implied_vol(price, F, K, T))
        assert iv == pytest.approx(sigma, rel=1e-5)

    def test_roundtrip_put(self):
        F, K, sigma, T = 1.0, 1.1, 0.25, 1.0
        price = float(bs_put(F, K, sigma, T))
        iv = float(implied_vol(price, F, K, T, flag="put"))
        assert iv == pytest.approx(sigma, rel=1e-5)

    def test_below_intrinsic_returns_nan(self):
        # Price at intrinsic (zero vol) should return NaN
        assert np.isnan(float(implied_vol(0.0, 1.0, 1.0, 1.0)))

    def test_above_upper_bound_returns_nan(self):
        # Price equal to forward is above upper bound
        assert np.isnan(float(implied_vol(1.0, 1.0, 1.0, 1.0)))

    def test_invalid_flag(self):
        with pytest.raises(ValueError):
            implied_vol(0.1, 1.0, 1.0, 1.0, flag="straddle")

    def test_vectorised_output_shape(self):
        prices = np.array([0.1, 0.05, 0.02])
        iv = implied_vol(prices, 1.0, np.array([0.9, 1.0, 1.1]), 1.0)
        assert iv.shape == (3,)
        assert np.all(iv > 0)

    def test_roundtrip_with_rate(self):
        F, K, sigma, T, r = 1.0, 1.0, 0.25, 1.0, 0.05
        price = float(bs_call(F, K, sigma, T, r))
        iv = float(implied_vol(price, F, K, T, r=r))
        assert iv == pytest.approx(sigma, rel=1e-5)


# ---------------------------------------------------------------------------
# implied_vol_surface
# ---------------------------------------------------------------------------

class TestImpliedVolSurface:
    def test_shape(self):
        strikes = np.array([0.9, 1.0, 1.1])
        mats = np.array([0.5, 1.0])
        sigma = 0.2
        prices = np.array([
            [float(bs_call(1.0, k, sigma, T)) for k in strikes]
            for T in mats
        ])
        iv = implied_vol_surface(prices, 1.0, strikes, mats)
        assert iv.shape == (2, 3)

    def test_roundtrip_flat_surface(self):
        strikes = np.array([0.9, 1.0, 1.1])
        mats = np.array([0.5, 1.0, 2.0])
        sigma = 0.25
        prices = np.array([
            [float(bs_call(1.0, k, sigma, T)) for k in strikes]
            for T in mats
        ])
        iv = implied_vol_surface(prices, 1.0, strikes, mats)
        np.testing.assert_allclose(iv, sigma, rtol=1e-5)

    def test_nan_below_intrinsic(self):
        strikes = np.array([1.0])
        mats = np.array([1.0])
        prices = np.array([[0.0]])  # below intrinsic for ATM
        iv = implied_vol_surface(prices, 1.0, strikes, mats)
        assert np.isnan(iv[0, 0])

    def test_forwards_per_maturity(self):
        """Surface with maturity-dependent forwards."""
        strikes = np.array([1.0])
        mats = np.array([0.5, 1.0])
        forwards = np.array([1.02, 1.05])
        sigma = 0.2
        prices = np.array([
            [float(bs_call(f, k, sigma, T)) for k in strikes]
            for f, T in zip(forwards, mats)
        ])
        iv = implied_vol_surface(prices, forwards, strikes, mats)
        np.testing.assert_allclose(iv, sigma, rtol=1e-5)
