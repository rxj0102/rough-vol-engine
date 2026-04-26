"""Tests for rough_vol.models.rbergomi."""

import numpy as np
import pytest
from numpy.random import default_rng

from rough_vol.models.rbergomi import (
    RBergomiParams,
    _bs_call_price,
    _bs_implied_vol,
    implied_vol_surface,
    price_european,
    simulate_paths,
    volterra_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_params():
    return RBergomiParams(H=0.1, eta=1.9, rho=-0.9, xi0=0.04)


@pytest.fixture
def bm_params():
    """H close to 0.5 — vol process behaves like a log-normal SV model."""
    return RBergomiParams(H=0.49, eta=1.0, rho=0.0, xi0=0.04)


# ---------------------------------------------------------------------------
# RBergomiParams validation
# ---------------------------------------------------------------------------

class TestRBergomiParams:
    def test_valid(self):
        p = RBergomiParams(H=0.1, eta=1.9, rho=-0.9, xi0=0.04)
        assert p.H == 0.1

    @pytest.mark.parametrize("H", [0.0, 0.5, 0.6, 1.0])
    def test_invalid_H(self, H):
        with pytest.raises(ValueError):
            RBergomiParams(H=H, eta=1.0, rho=0.0, xi0=0.04)

    @pytest.mark.parametrize("eta", [0.0, -1.0])
    def test_invalid_eta(self, eta):
        with pytest.raises(ValueError):
            RBergomiParams(H=0.1, eta=eta, rho=0.0, xi0=0.04)

    @pytest.mark.parametrize("rho", [-1.0, 1.0, 1.5])
    def test_invalid_rho(self, rho):
        with pytest.raises(ValueError):
            RBergomiParams(H=0.1, eta=1.0, rho=rho, xi0=0.04)

    def test_invalid_xi0(self):
        with pytest.raises(ValueError):
            RBergomiParams(H=0.1, eta=1.0, rho=0.0, xi0=0.0)


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

class TestBSHelpers:
    def test_call_price_atm(self):
        # At-the-money call: C ≈ F * sigma * sqrt(T) / sqrt(2*pi) for r=0
        F, K, sigma, T = 1.0, 1.0, 0.2, 1.0
        price = _bs_call_price(F, K, sigma, T)
        expected = F * sigma * np.sqrt(T) / np.sqrt(2 * np.pi)
        assert price == pytest.approx(expected, rel=0.01)

    def test_call_lower_bound(self):
        # Call >= max(F - K, 0) for r=0
        F, K = 1.1, 1.0
        price = _bs_call_price(F, K, 0.2, 1.0)
        assert price >= F - K

    def test_call_deep_itm(self):
        # Deep ITM call ≈ F - K (discounted)
        F, K, sigma, T = 2.0, 0.5, 0.2, 1.0
        price = _bs_call_price(F, K, sigma, T)
        assert price == pytest.approx(F - K, rel=0.01)

    def test_implied_vol_roundtrip(self):
        F, K, sigma, T = 1.0, 1.0, 0.25, 1.0
        price = _bs_call_price(F, K, sigma, T)
        iv = _bs_implied_vol(price, F, K, T)
        assert iv == pytest.approx(sigma, rel=1e-5)

    @pytest.mark.parametrize("K", [0.8, 0.9, 1.0, 1.1, 1.2])
    def test_implied_vol_various_strikes(self, K):
        F, sigma, T = 1.0, 0.3, 0.5
        price = _bs_call_price(F, K, sigma, T)
        iv = _bs_implied_vol(price, F, K, T)
        assert iv == pytest.approx(sigma, rel=1e-4)

    def test_implied_vol_below_intrinsic_returns_nan(self):
        assert np.isnan(_bs_implied_vol(0.0, 1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# volterra_weights
# ---------------------------------------------------------------------------

class TestVolterraWeights:
    def test_shape(self):
        w = volterra_weights(50, H=0.1, T=1.0)
        assert w.shape == (50,)

    def test_positive(self):
        w = volterra_weights(50, H=0.1, T=1.0)
        assert np.all(w > 0)

    def test_decreasing(self):
        # Far-away increments get smaller weight (power-law decay)
        w = volterra_weights(50, H=0.1, T=1.0)
        assert np.all(np.diff(w) < 0)

    def test_scaling_with_dt(self):
        # b_k ∝ dt^{H-0.5}: doubling dt scales all weights by 2^{H-0.5}
        H = 0.2
        w1 = volterra_weights(10, H=H, T=1.0)
        w2 = volterra_weights(10, H=H, T=2.0)  # same n_steps, double T → double dt
        ratio = w2 / w1
        expected = 2.0 ** (H - 0.5)
        np.testing.assert_allclose(ratio, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# simulate_paths
# ---------------------------------------------------------------------------

class TestSimulatePaths:
    N_PATHS = 2000
    N_STEPS = 64

    def test_shapes(self, default_params):
        V, S = simulate_paths(default_params, self.N_PATHS, self.N_STEPS, T=1.0, rng=0)
        assert V.shape == (self.N_PATHS, self.N_STEPS + 1)
        assert S.shape == (self.N_PATHS, self.N_STEPS + 1)

    def test_initial_conditions(self, default_params):
        V, S = simulate_paths(default_params, 10, self.N_STEPS, T=1.0, rng=1)
        # S(0) = 1
        np.testing.assert_array_equal(S[:, 0], 1.0)
        # V(0) = xi0
        np.testing.assert_allclose(V[:, 0], default_params.xi0, rtol=1e-10)

    def test_variance_positive(self, default_params):
        V, _ = simulate_paths(default_params, 50, self.N_STEPS, T=1.0, rng=2)
        assert np.all(V > 0)

    def test_asset_positive(self, default_params):
        _, S = simulate_paths(default_params, 50, self.N_STEPS, T=1.0, rng=3)
        assert np.all(S > 0)

    def test_martingale_property(self, bm_params):
        """E[S_T] ≈ S_0 = 1 (risk-neutral measure, r=0)."""
        _, S = simulate_paths(bm_params, 5000, self.N_STEPS, T=1.0, rng=42)
        mean_terminal = S[:, -1].mean()
        assert mean_terminal == pytest.approx(1.0, abs=0.05)

    def test_reproducibility(self, default_params):
        V1, S1 = simulate_paths(default_params, 8, 16, T=1.0, rng=7)
        V2, S2 = simulate_paths(default_params, 8, 16, T=1.0, rng=7)
        np.testing.assert_array_equal(V1, V2)
        np.testing.assert_array_equal(S1, S2)

    def test_accepts_rng_object(self, default_params):
        rng = default_rng(10)
        V, S = simulate_paths(default_params, 4, 16, T=1.0, rng=rng)
        assert S.shape == (4, 17)

    def test_initial_variance_level(self, default_params):
        """V(0) must equal xi0."""
        V, _ = simulate_paths(default_params, 100, 32, T=1.0, rng=5)
        np.testing.assert_allclose(V[:, 0], default_params.xi0, rtol=1e-10)


# ---------------------------------------------------------------------------
# price_european
# ---------------------------------------------------------------------------

class TestPriceEuropean:
    def test_shapes_scalar(self, default_params):
        prices, stderr = price_european(default_params, K=1.0, T=1.0, rng=0, n_paths=500, n_steps=32)
        assert prices.shape == (1,)
        assert stderr.shape == (1,)

    def test_shapes_array(self, default_params):
        K = np.array([0.8, 1.0, 1.2])
        prices, stderr = price_european(default_params, K=K, T=1.0, rng=0, n_paths=500, n_steps=32)
        assert prices.shape == (3,)
        assert stderr.shape == (3,)

    def test_price_positive(self, default_params):
        prices, _ = price_european(default_params, K=1.0, T=1.0, rng=1, n_paths=500, n_steps=32)
        assert float(prices[0]) > 0.0

    def test_call_spread_ordering(self, default_params):
        """C(K1) >= C(K2) for K1 < K2 (call is decreasing in K)."""
        K = np.array([0.8, 1.0, 1.2])
        prices, _ = price_european(default_params, K=K, T=1.0, rng=2, n_paths=2000, n_steps=32)
        assert prices[0] > prices[1] > prices[2]

    def test_lower_bound(self, bm_params):
        """MC call price >= intrinsic (S0=1, r=0, K=0.8)."""
        prices, _ = price_european(bm_params, K=0.8, T=1.0, r=0.0, rng=3, n_paths=2000, n_steps=32)
        assert float(prices[0]) >= 0.2 - 1e-3

    def test_stderr_positive(self, default_params):
        _, stderr = price_european(default_params, K=1.0, T=1.0, rng=4, n_paths=500, n_steps=32)
        assert float(stderr[0]) > 0.0

    def test_stderr_decreases_with_paths(self, default_params):
        _, se1 = price_european(default_params, K=1.0, T=1.0, rng=5, n_paths=200, n_steps=32)
        _, se2 = price_european(default_params, K=1.0, T=1.0, rng=5, n_paths=2000, n_steps=32)
        assert float(se2[0]) < float(se1[0])

    def test_atm_price_reasonable(self, default_params):
        """ATM call price should be between 1% and 50% of spot."""
        prices, _ = price_european(default_params, K=1.0, T=1.0, rng=0, n_paths=5000, n_steps=64)
        p = float(prices[0])
        assert 0.01 < p < 0.5

    def test_convergence_with_paths(self, bm_params):
        """Price estimate should be stable with increasing paths."""
        p1, _ = price_european(bm_params, K=1.0, T=1.0, rng=0, n_paths=1000, n_steps=32)
        p2, _ = price_european(bm_params, K=1.0, T=1.0, rng=0, n_paths=5000, n_steps=32)
        assert float(p1[0]) == pytest.approx(float(p2[0]), rel=0.2)


# ---------------------------------------------------------------------------
# implied_vol_surface
# ---------------------------------------------------------------------------

class TestImpliedVolSurface:
    def test_shape(self, default_params):
        strikes = np.array([0.9, 1.0, 1.1])
        maturities = np.array([0.5, 1.0])
        iv = implied_vol_surface(default_params, strikes, maturities, n_paths=500, rng=0)
        assert iv.shape == (2, 3)

    def test_no_arbitrage_range(self, default_params):
        """Implied vols should be in (0, 5.0) where computable."""
        strikes = np.array([0.9, 1.0, 1.1])
        maturities = np.array([1.0])
        iv = implied_vol_surface(default_params, strikes, maturities, n_paths=1000, rng=1)
        valid = iv[~np.isnan(iv)]
        assert np.all(valid > 0.0)
        assert np.all(valid < 5.0)

    def test_returns_ndarray(self, default_params):
        strikes = np.array([1.0])
        mats = np.array([1.0])
        iv = implied_vol_surface(default_params, strikes, mats, n_paths=200, rng=2)
        assert isinstance(iv, np.ndarray)

    def test_skew_negative_rho(self, default_params):
        """With rho < 0, IV(K<1) > IV(K>1) for same maturity (skew)."""
        strikes = np.array([0.9, 1.1])
        mats = np.array([1.0])
        iv = implied_vol_surface(default_params, strikes, mats, n_paths=3000,
                                  n_steps_per_year=64, rng=3)
        # IV at K=0.9 should be higher than at K=1.1 for negative rho
        assert iv[0, 0] > iv[0, 1]
