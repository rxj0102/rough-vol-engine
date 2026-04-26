"""Tests for rough_vol.calibration.objective."""

import numpy as np
import pytest

from rough_vol.calibration.objective import (
    RBergomiObjective,
    RHestonObjective,
    _wsse,
    compute_weights,
    weighted_mae,
    weighted_rmse,
)
from rough_vol.calibration.surface import ImpliedVolSurface
from rough_vol.models.rbergomi import RBergomiParams
from rough_vol.models.rbergomi import implied_vol_surface as rbergomi_ivs
from rough_vol.models.rfheston import RHestonParams
from rough_vol.models.rfheston import implied_vol_surface as rfheston_ivs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_surface(sigma: float = 0.20, spot: float = 1.0) -> ImpliedVolSurface:
    """Simple 2×3 flat surface for lightweight tests."""
    strikes = np.array([0.9, 1.0, 1.1])
    mats = np.array([0.5, 1.0])
    ivols = np.full((2, 3), sigma)
    return ImpliedVolSurface(strikes, mats, ivols, spot=spot)


# ---------------------------------------------------------------------------
# _wsse
# ---------------------------------------------------------------------------

class TestWSSE:
    def test_zero_when_identical(self):
        m = np.array([[0.2, 0.2], [0.2, 0.2]])
        w = np.ones_like(m)
        assert _wsse(m, m.copy(), w) == 0.0

    def test_known_value(self):
        model = np.array([0.22])
        market = np.array([0.20])
        weights = np.array([1.0])
        expected = (0.22 - 0.20) ** 2
        assert abs(_wsse(model, market, weights) - expected) < 1e-15

    def test_weighted_sum(self):
        model = np.array([0.21, 0.22])
        market = np.array([0.20, 0.20])
        weights = np.array([2.0, 1.0])
        expected = 2.0 * 0.01 ** 2 + 1.0 * 0.02 ** 2
        assert abs(_wsse(model, market, weights) - expected) < 1e-15

    def test_nan_model_ignored(self):
        model = np.array([np.nan, 0.22])
        market = np.array([0.20, 0.20])
        weights = np.array([1.0, 1.0])
        # only second point valid
        expected = 0.02 ** 2
        assert abs(_wsse(model, market, weights) - expected) < 1e-15

    def test_zero_weight_ignored(self):
        model = np.array([0.30, 0.22])
        market = np.array([0.20, 0.20])
        weights = np.array([0.0, 1.0])
        expected = 0.02 ** 2
        assert abs(_wsse(model, market, weights) - expected) < 1e-15

    def test_all_invalid_returns_inf(self):
        m = np.array([np.nan, np.nan])
        w = np.ones(2)
        assert _wsse(m, m.copy(), w) == float("inf")


# ---------------------------------------------------------------------------
# compute_weights
# ---------------------------------------------------------------------------

class TestComputeWeights:
    def test_uniform_all_ones(self):
        surf = _flat_surface()
        w = compute_weights(surf, "uniform")
        assert w.shape == (2, 3)
        np.testing.assert_array_equal(w, 1.0)

    def test_uniform_nan_is_zero(self):
        strikes = np.array([0.9, 1.0, 1.1])
        mats = np.array([1.0])
        ivols = np.array([[0.2, np.nan, 0.2]])
        surf = ImpliedVolSurface(strikes, mats, ivols)
        w = compute_weights(surf, "uniform")
        assert w[0, 1] == 0.0
        assert w[0, 0] == 1.0

    def test_vega_positive_near_atm(self):
        surf = _flat_surface()
        w = compute_weights(surf, "vega")
        assert w.shape == (2, 3)
        # all valid points → positive weights
        assert np.all(w > 0)

    def test_vega_nan_is_zero(self):
        strikes = np.array([0.9, 1.0, 1.1])
        mats = np.array([1.0])
        ivols = np.array([[0.2, np.nan, 0.2]])
        surf = ImpliedVolSurface(strikes, mats, ivols)
        w = compute_weights(surf, "vega")
        assert w[0, 1] == 0.0

    def test_relative_formula(self):
        sigma = 0.25
        surf = _flat_surface(sigma=sigma)
        w = compute_weights(surf, "relative")
        expected = 1.0 / sigma ** 2
        np.testing.assert_allclose(w, expected)

    def test_relative_nan_is_zero(self):
        strikes = np.array([0.9, 1.0])
        mats = np.array([1.0])
        ivols = np.array([[np.nan, 0.2]])
        surf = ImpliedVolSurface(strikes, mats, ivols)
        w = compute_weights(surf, "relative")
        assert w[0, 0] == 0.0
        assert abs(w[0, 1] - 1.0 / 0.04) < 1e-12

    def test_invalid_scheme_raises(self):
        surf = _flat_surface()
        with pytest.raises(ValueError, match="Unknown weight scheme"):
            compute_weights(surf, "bad_scheme")


# ---------------------------------------------------------------------------
# RBergomiObjective — zero-loss when model == market (same seed)
# ---------------------------------------------------------------------------

class TestRBergomiObjective:
    @pytest.fixture(scope="class")
    def params(self):
        return RBergomiParams(H=0.1, eta=1.9, rho=-0.7, xi0=0.04)

    @pytest.fixture(scope="class")
    def market_surface(self, params):
        strikes = np.array([0.9, 1.0, 1.1])
        mats = np.array([0.5])
        ivols = rbergomi_ivs(
            params, strikes, mats,
            n_paths=500, n_steps_per_year=32, rng=42,
        )
        return ImpliedVolSurface(strikes, mats, ivols, spot=1.0)

    def test_zero_loss_identical_surfaces(self, params, market_surface):
        """WSSE must be exactly 0 when the same seed re-generates identical IVs."""
        obj = RBergomiObjective(
            market_surface,
            n_paths=500,
            n_steps_per_year=32,
            rng_seed=42,
        )
        theta = RBergomiObjective.from_params(params)
        loss = obj(theta)
        assert loss == 0.0, f"Expected 0.0, got {loss}"

    def test_positive_loss_wrong_params(self, market_surface):
        """Different parameters must produce a non-zero loss."""
        wrong_params = RBergomiParams(H=0.4, eta=0.5, rho=0.5, xi0=0.5)
        obj = RBergomiObjective(
            market_surface,
            n_paths=500,
            n_steps_per_year=32,
            rng_seed=42,
        )
        theta = RBergomiObjective.from_params(wrong_params)
        loss = obj(theta)
        assert loss > 0.0

    def test_out_of_bounds_returns_large_loss(self, market_surface):
        obj = RBergomiObjective(market_surface, n_paths=100, n_steps_per_year=16, rng_seed=0)
        # H out of range
        theta = np.array([0.6, 1.0, -0.5, 0.04])
        from rough_vol.calibration.objective import _LARGE_LOSS
        assert obj(theta) == _LARGE_LOSS

    def test_n_evals_increments(self, params, market_surface):
        obj = RBergomiObjective(market_surface, n_paths=100, n_steps_per_year=16, rng_seed=0)
        theta = RBergomiObjective.from_params(params)
        assert obj.n_evals == 0
        obj(theta)
        assert obj.n_evals == 1
        obj(theta)
        assert obj.n_evals == 2

    def test_history_tracks_running_min(self, params, market_surface):
        obj = RBergomiObjective(market_surface, n_paths=100, n_steps_per_year=16, rng_seed=0)
        theta = RBergomiObjective.from_params(params)
        obj(theta)
        obj(theta)
        assert len(obj.history) == 2
        # history is running minimum: second entry ≤ first
        assert obj.history[1] <= obj.history[0]

    def test_weight_scheme_vega(self, params, market_surface):
        obj = RBergomiObjective(
            market_surface,
            weight_scheme="vega",
            n_paths=500,
            n_steps_per_year=32,
            rng_seed=42,
        )
        theta = RBergomiObjective.from_params(params)
        loss = obj(theta)
        assert loss == 0.0

    def test_weight_scheme_relative(self, params, market_surface):
        obj = RBergomiObjective(
            market_surface,
            weight_scheme="relative",
            n_paths=500,
            n_steps_per_year=32,
            rng_seed=42,
        )
        theta = RBergomiObjective.from_params(params)
        loss = obj(theta)
        assert loss == 0.0

    def test_to_params_from_params_roundtrip(self, params):
        theta = RBergomiObjective.from_params(params)
        recovered = RBergomiObjective.to_params(theta)
        assert abs(recovered.H - params.H) < 1e-12
        assert abs(recovered.eta - params.eta) < 1e-12
        assert abs(recovered.rho - params.rho) < 1e-12
        assert abs(recovered.xi0 - params.xi0) < 1e-12


# ---------------------------------------------------------------------------
# RHestonObjective — zero-loss when model == market (deterministic)
# ---------------------------------------------------------------------------

class TestRHestonObjective:
    @pytest.fixture(scope="class")
    def params(self):
        return RHestonParams(H=0.1, lambda_=0.3, theta=0.04, rho=-0.7, nu=0.3, V0=0.04)

    @pytest.fixture(scope="class")
    def market_surface(self, params):
        strikes = np.array([0.9, 1.0, 1.1])
        mats = np.array([0.5])
        ivols = rfheston_ivs(params, strikes, mats, n_steps=50, n_quad=32)
        return ImpliedVolSurface(strikes, mats, ivols, spot=1.0)

    def test_zero_loss_identical_surfaces(self, params, market_surface):
        """Deterministic pricer → WSSE must be exactly 0 at true params."""
        obj = RHestonObjective(market_surface, n_steps=50, n_quad=32)
        theta = RHestonObjective.from_params(params)
        loss = obj(theta)
        assert loss == 0.0, f"Expected 0.0, got {loss}"

    def test_positive_loss_wrong_params(self, market_surface):
        wrong_params = RHestonParams(H=0.4, lambda_=5.0, theta=0.2, rho=0.5, nu=1.5, V0=0.2)
        obj = RHestonObjective(market_surface, n_steps=50, n_quad=32)
        theta = RHestonObjective.from_params(wrong_params)
        loss = obj(theta)
        assert loss > 0.0

    def test_out_of_bounds_returns_large_loss(self, market_surface):
        obj = RHestonObjective(market_surface, n_steps=50, n_quad=32)
        theta = np.array([0.6, 0.3, 0.04, -0.7, 0.3, 0.04])  # H out of range
        from rough_vol.calibration.objective import _LARGE_LOSS
        assert obj(theta) == _LARGE_LOSS

    def test_n_evals_increments(self, params, market_surface):
        obj = RHestonObjective(market_surface, n_steps=50, n_quad=32)
        theta = RHestonObjective.from_params(params)
        assert obj.n_evals == 0
        obj(theta)
        assert obj.n_evals == 1

    def test_weight_scheme_vega_zero_loss(self, params, market_surface):
        obj = RHestonObjective(market_surface, weight_scheme="vega", n_steps=50, n_quad=32)
        theta = RHestonObjective.from_params(params)
        assert obj(theta) == 0.0

    def test_weight_scheme_relative_zero_loss(self, params, market_surface):
        obj = RHestonObjective(market_surface, weight_scheme="relative", n_steps=50, n_quad=32)
        theta = RHestonObjective.from_params(params)
        assert obj(theta) == 0.0

    def test_to_params_from_params_roundtrip(self, params):
        theta = RHestonObjective.from_params(params)
        recovered = RHestonObjective.to_params(theta)
        assert abs(recovered.H - params.H) < 1e-12
        assert abs(recovered.lambda_ - params.lambda_) < 1e-12
        assert abs(recovered.theta - params.theta) < 1e-12
        assert abs(recovered.rho - params.rho) < 1e-12
        assert abs(recovered.nu - params.nu) < 1e-12
        assert abs(recovered.V0 - params.V0) < 1e-12


# ---------------------------------------------------------------------------
# Backward-compatible helpers
# ---------------------------------------------------------------------------

class TestWeightedRmse:
    def test_zero_for_identical(self):
        v = np.array([0.2, 0.3])
        w = np.ones(2)
        assert weighted_rmse(v, v.copy(), w) == 0.0

    def test_known_value(self):
        m = np.array([0.22])
        v = np.array([0.20])
        w = np.array([1.0])
        assert abs(weighted_rmse(m, v, w) - 0.02) < 1e-12

    def test_nan_excluded(self):
        m = np.array([np.nan, 0.22])
        v = np.array([0.20, 0.20])
        w = np.ones(2)
        assert abs(weighted_rmse(m, v, w) - 0.02) < 1e-12


class TestWeightedMae:
    def test_zero_for_identical(self):
        v = np.array([0.2, 0.3])
        w = np.ones(2)
        assert weighted_mae(v, v.copy(), w) == 0.0

    def test_known_value(self):
        m = np.array([0.22])
        v = np.array([0.20])
        w = np.array([1.0])
        assert abs(weighted_mae(m, v, w) - 0.02) < 1e-12
