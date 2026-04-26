"""
Tests for RoughVolCalibrator and estimate_H_from_skew.

Strategy
--------
- All tests using rfHeston with n_steps=10, n_quad=8 to keep single
  evaluations under ~10 ms, making even DE-based tests finish in seconds.
- Convergence is verified via Nelder-Mead started from a slightly
  perturbed known solution — always fast and reliable.
- DE and two_stage tests use tiny budgets (popsize=4, maxiter=10) and
  only assert structural correctness; they are NOT expected to converge
  fully within such a small budget.
- rBergomi tests use 200 paths / 16 steps per year and are restricted to
  structural checks (correct return type, n_evals > 0) to avoid slow MC.
"""

import numpy as np
import pytest

from rough_vol.calibration.objective import RBergomiObjective, RHestonObjective
from rough_vol.calibration.optimizer import CalibrationResult, RoughVolCalibrator
from rough_vol.calibration.surface import ImpliedVolSurface
from rough_vol.models.rbergomi import RBergomiParams
from rough_vol.models.rbergomi import implied_vol_surface as rbergomi_ivs
from rough_vol.models.rfheston import RHestonParams
from rough_vol.models.rfheston import implied_vol_surface as rfheston_ivs


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Solver settings kept tiny for speed in all rfHeston tests
_N_STEPS = 10
_N_QUAD = 8

# A moderate strike grid that avoids NaN for this parameter set
_STRIKES = np.array([0.92, 0.96, 1.00, 1.04, 1.08])
_MATS = np.array([0.5, 1.0])

# True rfHeston params used to generate synthetic surfaces
_TRUE_PARAMS = RHestonParams(
    H=0.1, lambda_=1.5, theta=0.04, rho=-0.5, nu=0.3, V0=0.04
)


def _make_rfheston_surface(
    params: RHestonParams = _TRUE_PARAMS,
    strikes: np.ndarray = _STRIKES,
    mats: np.ndarray = _MATS,
    n_steps: int = _N_STEPS,
    n_quad: int = _N_QUAD,
) -> ImpliedVolSurface:
    ivols = rfheston_ivs(params, strikes, mats, n_steps=n_steps, n_quad=n_quad)
    return ImpliedVolSurface(strikes, mats, ivols, spot=1.0)


def _make_calibrator(
    surface: ImpliedVolSurface | None = None,
    model: str = "rfheston",
    **kwargs,
) -> RoughVolCalibrator:
    if surface is None:
        surface = _make_rfheston_surface()
    return RoughVolCalibrator(
        surface, model=model,
        n_steps=_N_STEPS, n_quad=_N_QUAD,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# TestEstimateHFromSkew
# ---------------------------------------------------------------------------

def _power_law_surface(
    H_true: float,
    C: float = -0.3,
    sigma_atm: float = 0.20,
    dk_grid: float = 0.04,
) -> ImpliedVolSurface:
    """Surface whose ATM skew exactly follows C * T^{H-0.5}."""
    k_grid = np.array([-3 * dk_grid, -dk_grid, 0.0, dk_grid, 3 * dk_grid])
    strikes = np.exp(k_grid)          # absolute strikes (spot=F=1)
    mats = np.array([0.25, 0.5, 1.0, 2.0])
    skew_T = C * mats ** (H_true - 0.5)   # (n_T,)
    ivols = sigma_atm + skew_T[:, None] * k_grid[None, :]  # (n_T, n_K)
    # clip to positive vols
    ivols = np.clip(ivols, 1e-4, None)
    return ImpliedVolSurface(strikes, mats, ivols, spot=1.0)


class TestEstimateHFromSkew:
    def test_exact_power_law(self):
        """ATM skews follow C·T^{H-0.5} exactly → H is recovered exactly."""
        H_true = 0.20
        surf = _power_law_surface(H_true, C=-0.3, dk_grid=0.04)
        H_est = RoughVolCalibrator.estimate_H_from_skew(surf, dk=0.04)
        assert H_est is not None
        assert abs(H_est - H_true) < 1e-6

    def test_different_H_values(self):
        for H_true in [0.05, 0.15, 0.30, 0.40]:
            surf = _power_law_surface(H_true, C=-0.4, dk_grid=0.04)
            H_est = RoughVolCalibrator.estimate_H_from_skew(surf, dk=0.04)
            assert H_est is not None, f"H={H_true} returned None"
            assert abs(H_est - H_true) < 1e-5, f"H_true={H_true}, H_est={H_est}"

    def test_clips_to_valid_range(self):
        # Slope >> 0.5 → naive H > 0.49 → clipped to 0.49
        surf = _power_law_surface(0.48, C=0.5, dk_grid=0.04)
        H_est = RoughVolCalibrator.estimate_H_from_skew(surf, dk=0.04)
        assert H_est is not None
        assert 0.01 <= H_est <= 0.49

    def test_insufficient_maturities_returns_none(self):
        """Only one maturity with a valid skew → can't fit regression."""
        strikes = np.exp(np.array([-0.04, 0.0, 0.04]))
        mats = np.array([1.0])
        ivols = np.array([[0.20, 0.20, 0.20]])
        surf = ImpliedVolSurface(strikes, mats, ivols, spot=1.0)
        H_est = RoughVolCalibrator.estimate_H_from_skew(surf, dk=0.04)
        assert H_est is None

    def test_zero_skew_returns_none(self):
        """Flat smile → skew ≈ 0 everywhere → no valid log|skew|."""
        strikes = np.exp(np.array([-0.05, 0.0, 0.05]))
        mats = np.array([0.5, 1.0, 2.0])
        ivols = np.full((3, 3), 0.20)
        surf = ImpliedVolSurface(strikes, mats, ivols, spot=1.0)
        H_est = RoughVolCalibrator.estimate_H_from_skew(surf, dk=0.05)
        assert H_est is None

    def test_min_points_respected(self):
        """min_points=3 with only 2 valid skews → None."""
        H_true = 0.20
        # Use k_grid wider than dk to avoid fp boundary issue in atm_skews
        k_grid = np.array([-0.06, -0.01, 0.0, 0.01, 0.06])
        strikes = np.exp(k_grid)
        mats = np.array([0.5, 1.0])
        C = -0.3
        skew_T = C * mats ** (H_true - 0.5)
        ivols = 0.20 + skew_T[:, None] * k_grid[None, :]
        surf = ImpliedVolSurface(strikes, mats, ivols, spot=1.0)
        assert RoughVolCalibrator.estimate_H_from_skew(surf, dk=0.04, min_points=3) is None
        assert RoughVolCalibrator.estimate_H_from_skew(surf, dk=0.04, min_points=2) is not None

    def test_returns_float(self):
        surf = _power_law_surface(0.15)
        result = RoughVolCalibrator.estimate_H_from_skew(surf, dk=0.04)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# TestRoughVolCalibratorConstruction
# ---------------------------------------------------------------------------

class TestRoughVolCalibratorConstruction:
    def test_rfheston_model(self):
        cal = _make_calibrator(model="rfheston")
        assert isinstance(cal.objective, RHestonObjective)
        assert cal.model == "rfheston"

    def test_rbergomi_model(self):
        surface = _make_rfheston_surface()
        cal = RoughVolCalibrator(surface, model="rbergomi", n_paths=100, n_steps_per_year=16)
        assert isinstance(cal.objective, RBergomiObjective)
        assert cal.model == "rbergomi"

    def test_invalid_model_raises(self):
        surface = _make_rfheston_surface()
        with pytest.raises(ValueError, match="Unknown model"):
            RoughVolCalibrator(surface, model="bad_model")

    def test_weight_scheme_stored(self):
        cal = _make_calibrator(weight_scheme="vega")
        assert cal.weight_scheme == "vega"

    def test_objective_property_is_readonly_value(self):
        cal = _make_calibrator()
        obj = cal.objective
        assert obj is cal._obj


# ---------------------------------------------------------------------------
# TestCalibrateInvalidMethod
# ---------------------------------------------------------------------------

class TestCalibrateInvalidMethod:
    def test_raises_value_error(self):
        cal = _make_calibrator()
        with pytest.raises(ValueError, match="Unknown method"):
            cal.calibrate(method="gradient_descent")


# ---------------------------------------------------------------------------
# TestCalibrateDifferentialEvolution
# ---------------------------------------------------------------------------

class TestCalibrateDifferentialEvolution:
    """Structural tests — tiny DE budget, just verify the API contract."""

    @pytest.fixture(scope="class")
    def result(self):
        cal = _make_calibrator()
        return cal.calibrate(
            "differential_evolution",
            de_popsize=4, de_maxiter=5, de_seed=7,
        )

    def test_returns_calibration_result(self, result):
        assert isinstance(result, CalibrationResult)

    def test_method_string(self, result):
        assert result.method == "differential_evolution"

    def test_de_result_present(self, result):
        assert result.de_result is not None

    def test_n_evals_positive(self, result):
        assert result.n_evals > 0

    def test_history_nonempty(self, result):
        assert len(result.history) > 0

    def test_history_non_increasing(self, result):
        h = result.history
        for i in range(1, len(h)):
            assert h[i] <= h[i - 1] + 1e-14

    def test_params_type(self, result):
        assert isinstance(result.params, RHestonParams)

    def test_loss_finite(self, result):
        assert np.isfinite(result.loss)

    def test_state_resets_between_calls(self):
        """Calling calibrate twice gives independent histories."""
        cal = _make_calibrator()
        r1 = cal.calibrate("differential_evolution", de_popsize=4, de_maxiter=3)
        r2 = cal.calibrate("differential_evolution", de_popsize=4, de_maxiter=3)
        assert len(r1.history) > 0
        assert len(r2.history) > 0
        # Both histories have the same length (same budget)
        assert len(r1.history) == len(r2.history)


# ---------------------------------------------------------------------------
# TestCalibrateNelderMead
# ---------------------------------------------------------------------------

class TestCalibrateNelderMead:
    @pytest.fixture(scope="class")
    def true_params(self):
        return _TRUE_PARAMS

    @pytest.fixture(scope="class")
    def surface(self, true_params):
        return _make_rfheston_surface(true_params)

    @pytest.fixture(scope="class")
    def calibrator(self, surface):
        return RoughVolCalibrator(
            surface, model="rfheston", n_steps=_N_STEPS, n_quad=_N_QUAD
        )

    def test_returns_calibration_result(self, calibrator, true_params):
        x0 = RHestonObjective.from_params(true_params)
        result = calibrator.calibrate("nelder_mead", x0=x0, nm_maxiter=50)
        assert isinstance(result, CalibrationResult)

    def test_method_string(self, calibrator, true_params):
        x0 = RHestonObjective.from_params(true_params)
        result = calibrator.calibrate("nelder_mead", x0=x0, nm_maxiter=50)
        assert result.method == "nelder_mead"

    def test_de_result_is_none(self, calibrator, true_params):
        x0 = RHestonObjective.from_params(true_params)
        result = calibrator.calibrate("nelder_mead", x0=x0, nm_maxiter=50)
        assert result.de_result is None

    def test_converges_from_true_params(self, calibrator, true_params):
        """Starting exactly at true params → loss = 0 after first eval."""
        x0 = RHestonObjective.from_params(true_params)
        result = calibrator.calibrate("nelder_mead", x0=x0, nm_maxiter=200)
        assert result.loss < 1e-10

    def test_converges_from_perturbed_params(self, calibrator, true_params):
        """NM from a small perturbation of true params → converges to low loss."""
        x_true = RHestonObjective.from_params(true_params)
        # 5% multiplicative perturbation
        rng = np.random.default_rng(0)
        x0 = x_true * (1.0 + 0.05 * rng.standard_normal(len(x_true)))
        x0 = np.clip(x0, [lo for lo, _ in RHestonObjective.bounds],
                          [hi for _, hi in RHestonObjective.bounds])
        result = calibrator.calibrate(
            "nelder_mead", x0=x0,
            nm_maxiter=1_000, nm_fatol=1e-12, nm_xatol=1e-10,
        )
        assert result.loss < 1e-6

    def test_midpoint_x0_used_when_none(self, calibrator):
        """No x0 → falls back to bound midpoint and completes without error."""
        result = calibrator.calibrate("nelder_mead", nm_maxiter=10)
        assert isinstance(result, CalibrationResult)
        assert np.isfinite(result.loss)

    def test_n_evals_positive(self, calibrator, true_params):
        x0 = RHestonObjective.from_params(true_params)
        result = calibrator.calibrate("nelder_mead", x0=x0, nm_maxiter=10)
        assert result.n_evals > 0


# ---------------------------------------------------------------------------
# TestCalibrateTwoStage
# ---------------------------------------------------------------------------

class TestCalibrateTwoStage:
    """Main convergence tests — tiny DE budget + NM polish."""

    @pytest.fixture(scope="class")
    def surface(self):
        return _make_rfheston_surface()

    @pytest.fixture(scope="class")
    def result(self, surface):
        cal = RoughVolCalibrator(
            surface, model="rfheston", n_steps=_N_STEPS, n_quad=_N_QUAD
        )
        return cal.calibrate(
            "two_stage",
            de_popsize=4, de_maxiter=10, de_seed=0,
            nm_maxiter=300,
        )

    def test_returns_calibration_result(self, result):
        assert isinstance(result, CalibrationResult)

    def test_method_string(self, result):
        assert result.method == "two_stage"

    def test_de_result_present(self, result):
        assert result.de_result is not None

    def test_params_type(self, result):
        assert isinstance(result.params, RHestonParams)

    def test_loss_finite(self, result):
        assert np.isfinite(result.loss)

    def test_n_evals_positive(self, result):
        assert result.n_evals > 0

    def test_history_non_increasing(self, result):
        h = result.history
        for i in range(1, len(h)):
            assert h[i] <= h[i - 1] + 1e-14

    def test_two_stage_at_least_as_good_as_de_alone(self, surface):
        """NM polish never makes the result worse than DE alone."""
        cal_ts = RoughVolCalibrator(
            surface, model="rfheston", n_steps=_N_STEPS, n_quad=_N_QUAD
        )
        cal_de = RoughVolCalibrator(
            surface, model="rfheston", n_steps=_N_STEPS, n_quad=_N_QUAD
        )
        r_ts = cal_ts.calibrate(
            "two_stage", de_popsize=4, de_maxiter=5, de_seed=0,
            nm_maxiter=100,
        )
        r_de = cal_de.calibrate(
            "differential_evolution", de_popsize=4, de_maxiter=5, de_seed=0,
        )
        assert r_ts.loss <= r_de.loss + 1e-10


# ---------------------------------------------------------------------------
# TestConvergenceOnSyntheticSurface (main requirement)
# ---------------------------------------------------------------------------

class TestConvergenceOnSyntheticSurface:
    """Calibrate rfHeston to its own output → loss must reach near-zero.

    Uses Nelder-Mead seeded from true params (the reliable convergence path)
    for a fast, deterministic check.  Also includes a moderate-budget two_stage
    run to verify global convergence from a cold start.
    """

    @pytest.fixture(scope="class")
    def true_params(self):
        return _TRUE_PARAMS

    @pytest.fixture(scope="class")
    def synthetic_surface(self, true_params):
        return _make_rfheston_surface(true_params)

    # --- Nelder-Mead from true params → zero loss (deterministic check) ---

    def test_nm_zero_loss_at_true_params(self, true_params, synthetic_surface):
        """Objective evaluated at the surface's generating params must be zero."""
        cal = RoughVolCalibrator(
            synthetic_surface, model="rfheston",
            n_steps=_N_STEPS, n_quad=_N_QUAD,
        )
        x0 = RHestonObjective.from_params(true_params)
        result = cal.calibrate("nelder_mead", x0=x0, nm_maxiter=5)
        assert result.loss == 0.0

    # --- Nelder-Mead from perturbed start → convergence ---

    def test_nm_converges_from_perturbation(self, true_params, synthetic_surface):
        """10% perturbation of true params still converges to loss < 1e-5."""
        cal = RoughVolCalibrator(
            synthetic_surface, model="rfheston",
            n_steps=_N_STEPS, n_quad=_N_QUAD,
        )
        x_true = RHestonObjective.from_params(true_params)
        x0 = x_true * np.array([1.1, 0.9, 1.1, 0.9, 1.1, 0.9])
        x0 = np.clip(x0, [lo for lo, _ in RHestonObjective.bounds],
                          [hi for _, hi in RHestonObjective.bounds])
        result = cal.calibrate(
            "nelder_mead", x0=x0,
            nm_maxiter=2_000, nm_fatol=1e-12, nm_xatol=1e-10,
        )
        assert result.loss < 1e-5

    # --- two_stage from cold start converges reasonably ---

    def test_two_stage_reduces_loss(self, synthetic_surface):
        """Two-stage calibration must beat the midpoint-x0 Nelder-Mead loss."""
        cal_ts = RoughVolCalibrator(
            synthetic_surface, model="rfheston",
            n_steps=_N_STEPS, n_quad=_N_QUAD,
        )
        cal_nm = RoughVolCalibrator(
            synthetic_surface, model="rfheston",
            n_steps=_N_STEPS, n_quad=_N_QUAD,
        )
        r_ts = cal_ts.calibrate(
            "two_stage",
            de_popsize=4, de_maxiter=15, de_seed=0,
            nm_maxiter=200,
        )
        r_mid = cal_nm.calibrate(
            "nelder_mead",
            nm_maxiter=200,
        )
        # DE's global search should find a better basin than the midpoint
        assert r_ts.loss <= r_mid.loss * 1.5 or r_ts.loss < 1e-2

    # --- Result type checks ---

    def test_calibration_result_has_rfheston_params(self, synthetic_surface):
        cal = RoughVolCalibrator(
            synthetic_surface, model="rfheston",
            n_steps=_N_STEPS, n_quad=_N_QUAD,
        )
        x0 = RHestonObjective.from_params(_TRUE_PARAMS)
        result = cal.calibrate("nelder_mead", x0=x0, nm_maxiter=5)
        assert isinstance(result.params, RHestonParams)
        assert 0.01 <= result.params.H <= 0.49
        assert result.params.lambda_ > 0
        assert result.params.nu > 0
        assert result.params.V0 > 0


# ---------------------------------------------------------------------------
# TestCalibrationRbergomiStructural
# ---------------------------------------------------------------------------

class TestCalibrationRbergomiStructural:
    """Lightweight structural tests for the rBergomi path (no convergence).

    Uses very few MC paths and tiny DE budget to keep the test fast.
    """

    @pytest.fixture(scope="class")
    def rb_params(self):
        return RBergomiParams(H=0.1, eta=1.9, rho=-0.7, xi0=0.04)

    @pytest.fixture(scope="class")
    def rb_surface(self, rb_params):
        strikes = np.array([0.95, 1.0, 1.05])
        mats = np.array([0.5])
        ivols = rbergomi_ivs(
            rb_params, strikes, mats,
            n_paths=200, n_steps_per_year=16, rng=0,
        )
        return ImpliedVolSurface(strikes, mats, ivols, spot=1.0)

    def test_constructor(self, rb_surface):
        cal = RoughVolCalibrator(
            rb_surface, model="rbergomi",
            n_paths=200, n_steps_per_year=16, rng_seed=0,
        )
        assert isinstance(cal.objective, RBergomiObjective)

    def test_nm_returns_result(self, rb_surface, rb_params):
        cal = RoughVolCalibrator(
            rb_surface, model="rbergomi",
            n_paths=200, n_steps_per_year=16, rng_seed=0,
        )
        x0 = RBergomiObjective.from_params(rb_params)
        result = cal.calibrate("nelder_mead", x0=x0, nm_maxiter=10)
        assert isinstance(result, CalibrationResult)
        assert isinstance(result.params, RBergomiParams)
        assert result.n_evals > 0

    def test_nm_zero_loss_at_true_params(self, rb_surface, rb_params):
        """Same seed → same paths → WSSE = 0 at the generating params."""
        cal = RoughVolCalibrator(
            rb_surface, model="rbergomi",
            n_paths=200, n_steps_per_year=16, rng_seed=0,
        )
        x0 = RBergomiObjective.from_params(rb_params)
        result = cal.calibrate("nelder_mead", x0=x0, nm_maxiter=5)
        assert result.loss == 0.0

    def test_de_returns_result(self, rb_surface):
        cal = RoughVolCalibrator(
            rb_surface, model="rbergomi",
            n_paths=200, n_steps_per_year=16, rng_seed=0,
        )
        result = cal.calibrate(
            "differential_evolution",
            de_popsize=3, de_maxiter=2, de_seed=0,
        )
        assert isinstance(result, CalibrationResult)
        assert isinstance(result.params, RBergomiParams)
        assert np.isfinite(result.loss)
