"""Tests for rough_vol.simulation.fbm."""

import numpy as np
import pytest
from numpy.random import default_rng

from rough_vol.simulation.fbm import (
    cholesky_sim,
    fbm_paths,
    hosking_sim,
    hybrid_sim,
    simulate,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

N_PATHS = 4000   # enough for variance/covariance statistics
N_PATHS_SMALL = 8
N_STEPS = 64
H_VALUES = [0.1, 0.3, 0.49]
METHODS = ["cholesky", "hosking", "hybrid"]


def _theoretical_variance(H: float, dt: float) -> float:
    """Var[W^H(dt)] = dt^{2H}."""
    return dt ** (2 * H)


def _theoretical_cov_lag1(H: float, dt: float) -> float:
    """Cov[ΔW^H_1, ΔW^H_2] for uniform grid with step dt."""
    return 0.5 * dt ** (2 * H) * (2 ** (2 * H) - 2)


# ---------------------------------------------------------------------------
# cholesky_sim
# ---------------------------------------------------------------------------

class TestCholeskySim:
    def test_output_shape(self):
        inc = cholesky_sim(N_PATHS_SMALL, N_STEPS, H=0.3, T=1.0, rng=0)
        assert inc.shape == (N_PATHS_SMALL, N_STEPS)

    def test_zero_mean(self):
        inc = cholesky_sim(N_PATHS, N_STEPS, H=0.3, T=1.0, rng=42)
        np.testing.assert_allclose(inc.mean(axis=0), 0.0, atol=0.05)

    @pytest.mark.parametrize("H", H_VALUES)
    def test_variance(self, H):
        dt = 1.0 / N_STEPS
        inc = cholesky_sim(N_PATHS, N_STEPS, H=H, T=1.0, rng=1)
        empirical_var = inc.var(axis=0).mean()
        expected_var = _theoretical_variance(H, dt)
        assert empirical_var == pytest.approx(expected_var, rel=0.12)

    def test_reproducibility(self):
        a = cholesky_sim(5, 10, H=0.2, T=1.0, rng=99)
        b = cholesky_sim(5, 10, H=0.2, T=1.0, rng=99)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = cholesky_sim(5, 10, H=0.2, T=1.0, rng=1)
        b = cholesky_sim(5, 10, H=0.2, T=1.0, rng=2)
        assert not np.allclose(a, b)

    def test_accepts_rng_object(self):
        rng = default_rng(7)
        inc = cholesky_sim(3, 8, H=0.3, T=1.0, rng=rng)
        assert inc.shape == (3, 8)

    def test_invalid_H(self):
        with pytest.raises(ValueError):
            cholesky_sim(4, 8, H=0.0)
        with pytest.raises(ValueError):
            cholesky_sim(4, 8, H=1.0)

    def test_terminal_variance_of_path(self):
        # Var[W^H(T)] = T^{2H}
        H, T = 0.3, 1.0
        inc = cholesky_sim(N_PATHS, N_STEPS, H=H, T=T, rng=5)
        terminal = inc.sum(axis=1)
        assert terminal.var() == pytest.approx(T ** (2 * H), rel=0.12)


# ---------------------------------------------------------------------------
# hosking_sim
# ---------------------------------------------------------------------------

class TestHoskingSim:
    def test_output_shape(self):
        inc = hosking_sim(N_PATHS_SMALL, N_STEPS, H=0.3, T=1.0, rng=0)
        assert inc.shape == (N_PATHS_SMALL, N_STEPS)

    def test_zero_mean(self):
        inc = hosking_sim(N_PATHS, N_STEPS, H=0.3, T=1.0, rng=42)
        np.testing.assert_allclose(inc.mean(axis=0), 0.0, atol=0.05)

    @pytest.mark.parametrize("H", H_VALUES)
    def test_variance(self, H):
        dt = 1.0 / N_STEPS
        inc = hosking_sim(N_PATHS, N_STEPS, H=H, T=1.0, rng=7)
        empirical_var = inc.var(axis=0).mean()
        expected_var = _theoretical_variance(H, dt)
        assert empirical_var == pytest.approx(expected_var, rel=0.12)

    def test_reproducibility(self):
        a = hosking_sim(5, 10, H=0.2, T=1.0, rng=99)
        b = hosking_sim(5, 10, H=0.2, T=1.0, rng=99)
        np.testing.assert_array_equal(a, b)

    def test_terminal_variance_of_path(self):
        H, T = 0.3, 1.0
        inc = hosking_sim(N_PATHS, N_STEPS, H=H, T=T, rng=11)
        terminal = inc.sum(axis=1)
        assert terminal.var() == pytest.approx(T ** (2 * H), rel=0.15)

    def test_lag1_autocorrelation(self):
        # For rough H < 0.5, lag-1 autocorr of increments should be negative
        H = 0.1
        inc = hosking_sim(N_PATHS, N_STEPS, H=H, T=1.0, rng=3)
        # Compute lag-1 autocorr across paths for the middle step
        x = inc[:, N_STEPS // 2]
        y = inc[:, N_STEPS // 2 + 1]
        corr = np.corrcoef(x, y)[0, 1]
        assert corr < 0.0, "increments should be negatively correlated for H<0.5"

    def test_invalid_H(self):
        with pytest.raises(ValueError):
            hosking_sim(4, 8, H=0.0)


# ---------------------------------------------------------------------------
# hybrid_sim
# ---------------------------------------------------------------------------

class TestHybridSim:
    def test_output_shape(self):
        inc = hybrid_sim(N_PATHS_SMALL, N_STEPS, H=0.3, T=1.0, rng=0)
        assert inc.shape == (N_PATHS_SMALL, N_STEPS)

    def test_zero_mean(self):
        inc = hybrid_sim(N_PATHS, N_STEPS, H=0.3, T=1.0, rng=42)
        np.testing.assert_allclose(inc.mean(axis=0), 0.0, atol=0.05)

    @pytest.mark.parametrize("H", H_VALUES)
    def test_variance_order_of_magnitude(self, H):
        # Hybrid is approximate; just check the variance is in the right ballpark
        dt = 1.0 / N_STEPS
        inc = hybrid_sim(N_PATHS, N_STEPS, H=H, T=1.0, rng=9, n_trunc=100)
        empirical_var = inc.var(axis=0).mean()
        expected_var = _theoretical_variance(H, dt)
        assert empirical_var == pytest.approx(expected_var, rel=0.25)

    def test_reproducibility(self):
        a = hybrid_sim(5, 16, H=0.2, T=1.0, rng=99)
        b = hybrid_sim(5, 16, H=0.2, T=1.0, rng=99)
        np.testing.assert_array_equal(a, b)

    def test_n_trunc_parameter(self):
        # Higher n_trunc shouldn't crash and should give non-identical results
        a = hybrid_sim(4, 32, H=0.3, T=1.0, rng=0, n_trunc=10)
        b = hybrid_sim(4, 32, H=0.3, T=1.0, rng=0, n_trunc=50)
        assert a.shape == b.shape

    def test_invalid_H(self):
        with pytest.raises(ValueError):
            hybrid_sim(4, 8, H=0.0)

    def test_invalid_n_trunc(self):
        with pytest.raises(ValueError):
            hybrid_sim(4, 8, H=0.3, n_trunc=0)


# ---------------------------------------------------------------------------
# simulate (dispatcher)
# ---------------------------------------------------------------------------

class TestSimulate:
    @pytest.mark.parametrize("method", METHODS)
    def test_shape(self, method):
        inc = simulate(4, 16, H=0.3, T=1.0, method=method, rng=0)
        assert inc.shape == (4, 16)

    def test_default_method_is_hybrid(self):
        a = simulate(4, 16, H=0.3, T=1.0, rng=7)
        b = hybrid_sim(4, 16, H=0.3, T=1.0, rng=7)
        np.testing.assert_array_equal(a, b)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            simulate(4, 8, H=0.3, method="invalid")

    def test_case_insensitive(self):
        a = simulate(4, 16, H=0.3, method="Cholesky", rng=1)
        b = simulate(4, 16, H=0.3, method="cholesky", rng=1)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# fbm_paths
# ---------------------------------------------------------------------------

class TestFbmPaths:
    def test_shape(self):
        inc = np.ones((5, 10))
        paths = fbm_paths(inc)
        assert paths.shape == (5, 11)

    def test_starts_at_zero(self):
        inc = cholesky_sim(8, 20, H=0.3, rng=0)
        paths = fbm_paths(inc)
        np.testing.assert_array_equal(paths[:, 0], 0.0)

    def test_cumsum_property(self):
        rng = default_rng(0)
        inc = rng.standard_normal((5, 10))
        paths = fbm_paths(inc)
        np.testing.assert_allclose(np.diff(paths, axis=1), inc, atol=1e-14)

    def test_terminal_value(self):
        inc = np.ones((3, 4))
        paths = fbm_paths(inc)
        np.testing.assert_array_equal(paths[:, -1], 4.0)


# ---------------------------------------------------------------------------
# Cross-method consistency
# ---------------------------------------------------------------------------

class TestCrossMethodConsistency:
    """Statistical properties should be consistent across all three methods."""

    @pytest.mark.parametrize("H", [0.1, 0.3])
    def test_terminal_variance_matches_across_methods(self, H):
        """Terminal W^H(T) variance should be ≈ T^{2H} for all methods."""
        T, n_steps = 1.0, 32
        kwargs = dict(n_paths=N_PATHS, n_steps=n_steps, H=H, T=T, rng=0)
        expected = T ** (2 * H)
        for method in METHODS:
            inc = simulate(**kwargs, method=method)
            tv = inc.sum(axis=1).var()
            assert tv == pytest.approx(expected, rel=0.2), (
                f"method={method}, H={H}: got {tv:.4f}, expected {expected:.4f}"
            )
