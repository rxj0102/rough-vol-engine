"""Tests for rough_vol.kernels."""

import numpy as np
import pytest
from scipy.special import gamma as scipy_gamma

from rough_vol.kernels import (
    covariance,
    covariance_matrix,
    gamma_H,
    hybrid_weights,
    rl_covariance,
    rl_kernel,
)


# ---------------------------------------------------------------------------
# gamma_H
# ---------------------------------------------------------------------------

class TestGammaH:
    def test_known_value(self):
        H = 0.1
        expected = scipy_gamma(H + 0.5)
        assert gamma_H(H) == pytest.approx(expected, rel=1e-10)

    def test_half(self):
        # H=0.5 → Γ(1.0) = 1
        assert gamma_H(0.5) == pytest.approx(1.0, rel=1e-10)

    @pytest.mark.parametrize("bad", [-0.1, 0.0, 1.0, 1.5])
    def test_invalid_H(self, bad):
        with pytest.raises(ValueError):
            gamma_H(bad)


# ---------------------------------------------------------------------------
# rl_kernel
# ---------------------------------------------------------------------------

class TestRLKernel:
    def test_positive_diff(self):
        val = rl_kernel(1.0, 0.5, H=0.3)
        assert float(val) == pytest.approx(0.5 ** (0.3 - 0.5), rel=1e-10)

    def test_zero_when_s_geq_t(self):
        assert float(rl_kernel(0.5, 0.5, H=0.3)) == 0.0
        assert float(rl_kernel(0.3, 0.5, H=0.3)) == 0.0

    def test_vectorised(self):
        t = np.array([1.0, 2.0, 3.0])
        s = np.array([0.5, 2.0, 1.0])
        result = rl_kernel(t, s, H=0.2)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(0.5 ** (0.2 - 0.5), rel=1e-10)
        assert result[1] == 0.0  # s == t
        assert result[2] == pytest.approx(2.0 ** (0.2 - 0.5), rel=1e-10)

    def test_broadcast_2d(self):
        t = np.array([[1.0], [2.0]])
        s = np.array([[0.5, 1.5]])
        result = rl_kernel(t, s, H=0.3)
        assert result.shape == (2, 2)

    @pytest.mark.parametrize("bad_H", [0.0, 1.0, -0.5])
    def test_invalid_H(self, bad_H):
        with pytest.raises(ValueError):
            rl_kernel(1.0, 0.5, H=bad_H)

    def test_scaling(self):
        # K(2t, 2s, H) = 2^{H-0.5} * K(t, s, H) by power-law
        H = 0.3
        scale = 2.0
        t, s = 1.0, 0.4
        assert float(rl_kernel(scale * t, scale * s, H)) == pytest.approx(
            scale ** (H - 0.5) * float(rl_kernel(t, s, H)), rel=1e-10
        )


# ---------------------------------------------------------------------------
# covariance
# ---------------------------------------------------------------------------

class TestCovariance:
    def test_variance_at_one(self):
        # Var[W^H(1)] = C(1, 1) = 1 for standard fBm
        assert covariance(1.0, 1.0, H=0.3) == pytest.approx(1.0, rel=1e-10)
        assert covariance(1.0, 1.0, H=0.5) == pytest.approx(1.0, rel=1e-10)

    def test_symmetry(self):
        for H in [0.1, 0.3, 0.7]:
            t, s = 0.7, 0.3
            assert covariance(t, s, H) == pytest.approx(covariance(s, t, H), rel=1e-10)

    def test_zero_at_origin(self):
        assert covariance(0.0, 0.5, H=0.3) == pytest.approx(0.0, abs=1e-14)
        assert covariance(0.0, 0.0, H=0.3) == pytest.approx(0.0, abs=1e-14)

    def test_bm_case(self):
        # H=0.5 should reduce to min(t, s)
        t, s = 0.4, 0.7
        assert covariance(t, s, H=0.5) == pytest.approx(min(t, s), rel=1e-10)

    def test_positive_definite(self):
        times = np.linspace(0.01, 1.0, 6)
        t_col = times[:, None]
        t_row = times[None, :]
        C = covariance(t_col, t_row, H=0.3)
        eigs = np.linalg.eigvalsh(C)
        assert np.all(eigs > -1e-10)

    @pytest.mark.parametrize("bad_H", [0.0, 1.0])
    def test_invalid_H(self, bad_H):
        with pytest.raises(ValueError):
            covariance(1.0, 1.0, H=bad_H)


# ---------------------------------------------------------------------------
# covariance_matrix
# ---------------------------------------------------------------------------

class TestCovarianceMatrix:
    def test_shape(self):
        times = np.linspace(0.1, 1.0, 5)
        C = covariance_matrix(times, H=0.3)
        assert C.shape == (5, 5)

    def test_symmetric(self):
        times = np.linspace(0.1, 1.0, 8)
        C = covariance_matrix(times, H=0.2)
        np.testing.assert_allclose(C, C.T, atol=1e-14)

    def test_positive_semidefinite(self):
        times = np.linspace(0.1, 1.0, 10)
        C = covariance_matrix(times, H=0.1)
        eigs = np.linalg.eigvalsh(C)
        assert np.all(eigs > -1e-8)

    def test_diagonal_equals_variance(self):
        times = np.array([0.5, 1.0, 2.0])
        C = covariance_matrix(times, H=0.3)
        expected_diag = times ** (2 * 0.3)
        np.testing.assert_allclose(np.diag(C), expected_diag, rtol=1e-10)

    def test_1d_requirement(self):
        with pytest.raises(ValueError):
            covariance_matrix(np.ones((3, 3)), H=0.3)


# ---------------------------------------------------------------------------
# rl_covariance
# ---------------------------------------------------------------------------

class TestRLCovariance:
    def test_symmetry(self):
        for H in [0.1, 0.3, 0.4]:
            assert rl_covariance(0.6, 0.9, H) == pytest.approx(
                rl_covariance(0.9, 0.6, H), rel=1e-10
            )

    def test_variance_positive(self):
        for H in [0.1, 0.3, 0.49]:
            assert float(rl_covariance(1.0, 1.0, H)) > 0.0

    def test_zero_origin(self):
        assert float(rl_covariance(0.0, 1.0, H=0.3)) == pytest.approx(0.0, abs=1e-14)

    def test_vectorised(self):
        t = np.array([0.5, 1.0])
        s = np.array([0.5, 1.0])
        result = rl_covariance(t, s, H=0.2)
        assert result.shape == (2,)
        assert np.all(result > 0)

    @pytest.mark.parametrize("bad_H", [0.0, 1.0])
    def test_invalid_H(self, bad_H):
        with pytest.raises(ValueError):
            rl_covariance(1.0, 1.0, H=bad_H)


# ---------------------------------------------------------------------------
# hybrid_weights
# ---------------------------------------------------------------------------

class TestHybridWeights:
    def test_shape(self):
        b = hybrid_weights(20, H=0.1, dt=0.01)
        assert b.shape == (20,)

    def test_positive(self):
        b = hybrid_weights(30, H=0.3, dt=0.004)
        assert np.all(b > 0)

    def test_decreasing(self):
        # Weights should be decreasing (far-away increments get smaller weight)
        b = hybrid_weights(50, H=0.1, dt=0.01)
        # The first weight is largest (closest to the evaluation point)
        assert np.all(np.diff(b) < 0)

    def test_scaling_with_dt(self):
        # b_k = dt^{H-0.5} * [...], so doubling dt scales by 2^{H-0.5}.
        H = 0.2
        dt1, dt2 = 0.01, 0.02
        b1 = hybrid_weights(10, H=H, dt=dt1)
        b2 = hybrid_weights(10, H=H, dt=dt2)
        ratio = b2 / b1
        expected = (dt2 / dt1) ** (H - 0.5)
        np.testing.assert_allclose(ratio, expected, rtol=1e-8)

    @pytest.mark.parametrize("bad", [("n_trunc", 0), ("dt", -0.01), ("dt", 0.0)])
    def test_invalid_args(self, bad):
        key, val = bad
        kwargs = {"n_trunc": 10, "H": 0.3, "dt": 0.01}
        kwargs[key] = val
        with pytest.raises(ValueError):
            hybrid_weights(**kwargs)
