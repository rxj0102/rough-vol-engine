"""Tests for rough_vol.calibration.surface.ImpliedVolSurface."""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rough_vol.calibration.surface import ImpliedVolSurface


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat():
    """Flat IV surface: σ = 0.20 everywhere, spot = 100, r = 0."""
    spot = 100.0
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    maturities = np.array([0.5, 1.0, 2.0])
    implied_vols = np.full((3, 5), 0.2)
    forwards = np.full(3, spot)
    return ImpliedVolSurface(strikes, maturities, implied_vols,
                             spot=spot, forwards=forwards)


@pytest.fixture
def flat_csv(tmp_path):
    """CSV file encoding the same flat surface."""
    lines = ["maturity,strike,implied_vol,forward,spot"]
    spot, fwd = 100.0, 100.0
    for T in [0.5, 1.0, 2.0]:
        for K in [90.0, 95.0, 100.0, 105.0, 110.0]:
            lines.append(f"{T},{K},0.2,{fwd},{spot}")
    p = tmp_path / "flat.csv"
    p.write_text("\n".join(lines))
    return p


# ---------------------------------------------------------------------------
# Construction and basic attributes
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic(self, flat):
        assert flat.n_maturities == 3
        assert flat.n_strikes == 5
        assert flat.spot == 100.0

    def test_maturities_sorted_on_input(self):
        """Constructor must sort slices by ascending maturity."""
        ivols = np.full((3, 3), 0.2)
        surf = ImpliedVolSurface(
            [90.0, 100.0, 110.0],
            [2.0, 0.5, 1.0],   # unsorted
            ivols,
        )
        np.testing.assert_array_equal(surf.maturities, [0.5, 1.0, 2.0])

    def test_default_forwards_equal_spot(self):
        surf = ImpliedVolSurface([90.0, 100.0, 110.0], [1.0], np.full((1, 3), 0.2), spot=50.0)
        np.testing.assert_array_equal(surf.forwards, [50.0])

    def test_custom_forwards(self, flat):
        np.testing.assert_array_equal(flat.forwards, [100.0, 100.0, 100.0])

    def test_invalid_spot(self):
        with pytest.raises(ValueError, match="spot"):
            ImpliedVolSurface([100.0], [1.0], np.full((1, 1), 0.2), spot=0.0)

    def test_invalid_maturity(self):
        with pytest.raises(ValueError, match="maturities"):
            ImpliedVolSurface([100.0], [0.0], np.full((1, 1), 0.2))

    def test_invalid_strike(self):
        with pytest.raises(ValueError, match="strikes"):
            ImpliedVolSurface([0.0], [1.0], np.full((1, 1), 0.2))

    def test_shape_mismatch_ivols(self):
        with pytest.raises(ValueError):
            ImpliedVolSurface([90.0, 100.0], [0.5, 1.0], np.full((3, 2), 0.2))

    def test_shape_mismatch_forwards(self):
        with pytest.raises(ValueError, match="forwards"):
            ImpliedVolSurface(
                [90.0, 100.0], [0.5, 1.0], np.full((2, 2), 0.2),
                forwards=np.array([100.0]),   # length 1, not 2
            )

    def test_repr(self, flat):
        r = repr(flat)
        assert "ImpliedVolSurface" in r
        assert "n_maturities=3" in r
        assert "n_strikes=5" in r


# ---------------------------------------------------------------------------
# to_log_moneyness
# ---------------------------------------------------------------------------

class TestToLogMoneyness:
    def test_shape(self, flat):
        k = flat.to_log_moneyness()
        assert k.shape == (3, 5)

    def test_atm_strike_is_zero(self, flat):
        """K = F = 100 → k = 0."""
        k = flat.to_log_moneyness()
        for i in range(3):
            assert k[i, 2] == pytest.approx(0.0, abs=1e-12)

    def test_otm_call_positive(self, flat):
        """K > F → k > 0."""
        k = flat.to_log_moneyness()
        assert np.all(k[:, 3] > 0)
        assert np.all(k[:, 4] > 0)

    def test_otm_put_negative(self, flat):
        """K < F → k < 0."""
        k = flat.to_log_moneyness()
        assert np.all(k[:, 0] < 0)
        assert np.all(k[:, 1] < 0)

    def test_same_across_maturities_when_forwards_equal(self, flat):
        """Flat forward curve → identical k-grid for all maturities."""
        k = flat.to_log_moneyness()
        np.testing.assert_allclose(k[0], k[1])
        np.testing.assert_allclose(k[0], k[2])

    def test_values(self, flat):
        k = flat.to_log_moneyness()
        expected = np.log(np.array([90, 95, 100, 105, 110]) / 100.0)
        np.testing.assert_allclose(k[0], expected, rtol=1e-12)

    def test_forward_dependent(self):
        """Different forwards per maturity shift the k-grid accordingly."""
        surf = ImpliedVolSurface(
            [100.0], [0.5, 1.0], np.full((2, 1), 0.2),
            spot=100.0, forwards=np.array([100.0, 110.0]),
        )
        k = surf.to_log_moneyness()
        assert k[0, 0] == pytest.approx(np.log(100.0 / 100.0))
        assert k[1, 0] == pytest.approx(np.log(100.0 / 110.0))


# ---------------------------------------------------------------------------
# to_total_variance
# ---------------------------------------------------------------------------

class TestToTotalVariance:
    def test_shape(self, flat):
        assert flat.to_total_variance().shape == (3, 5)

    def test_flat_surface_values(self, flat):
        """w = σ² T = 0.04 T for each maturity."""
        tv = flat.to_total_variance()
        np.testing.assert_allclose(tv[0], 0.04 * 0.5)   # T = 0.5
        np.testing.assert_allclose(tv[1], 0.04 * 1.0)   # T = 1.0
        np.testing.assert_allclose(tv[2], 0.04 * 2.0)   # T = 2.0

    def test_constant_across_strikes_for_flat(self, flat):
        """Flat vol → TV constant across strikes at each maturity."""
        tv = flat.to_total_variance()
        for i in range(3):
            assert np.allclose(tv[i], tv[i, 0])

    def test_nondecreasing_in_T_for_flat(self, flat):
        """Calendar condition: TV(T2, K) ≥ TV(T1, K) for flat surface."""
        tv = flat.to_total_variance()
        for j in range(5):
            assert tv[0, j] < tv[1, j] < tv[2, j]

    def test_nan_propagation(self):
        ivols = np.array([[0.2, np.nan, 0.2]])
        surf = ImpliedVolSurface([90., 100., 110.], [1.0], ivols)
        tv = surf.to_total_variance()
        assert np.isnan(tv[0, 1])
        assert not np.isnan(tv[0, 0])


# ---------------------------------------------------------------------------
# atm_vols
# ---------------------------------------------------------------------------

class TestAtmVols:
    def test_shape(self, flat):
        assert flat.atm_vols().shape == (3,)

    def test_flat_surface_returns_sigma(self, flat):
        """All ATM vols should equal 0.20 for a flat surface."""
        np.testing.assert_allclose(flat.atm_vols(), 0.2)

    def test_interpolates_between_strikes(self):
        """When K=F falls between two strikes, ATM vol is linearly interpolated."""
        # Forward at 97.5 (between K=95 and K=100)
        surf = ImpliedVolSurface(
            [90., 95., 100., 105., 110.],
            [1.0],
            np.full((1, 5), 0.2),
            spot=97.5,
            forwards=np.array([97.5]),
        )
        atm = surf.atm_vols()
        assert atm[0] == pytest.approx(0.2, rel=1e-10)

    def test_skewed_surface_atm_interpolation(self):
        """ATM vol picks the correct interpolated value for a linearly skewed smile."""
        # Linear smile: σ(k) = 0.20 - 0.10 * k
        strikes = np.array([90., 95., 100., 105., 110.])
        F = 100.0
        k = np.log(strikes / F)                 # [-0.105, -0.051, 0, 0.049, 0.095]
        vols = np.array([[0.2 - 0.10 * ki for ki in k]])  # interpolated at k=0 → 0.20
        surf = ImpliedVolSurface(strikes, [1.0], vols, spot=F, forwards=np.array([F]))
        assert surf.atm_vols()[0] == pytest.approx(0.2, rel=1e-10)

    def test_nan_when_forward_outside_strike_range(self):
        """ATM vol is NaN when F is outside the strike grid."""
        surf = ImpliedVolSurface(
            [90., 95., 100.],
            [1.0],
            np.full((1, 3), 0.2),
            spot=110.0,
            forwards=np.array([110.0]),  # F=110 > max strike 100
        )
        assert np.isnan(surf.atm_vols()[0])


# ---------------------------------------------------------------------------
# atm_skews
# ---------------------------------------------------------------------------

class TestAtmSkews:
    def test_shape(self, flat):
        assert flat.atm_skews().shape == (3,)

    def test_flat_surface_zero_skew(self, flat):
        """Flat IV → zero skew for any dk."""
        np.testing.assert_allclose(flat.atm_skews(dk=0.05), 0.0, atol=1e-12)
        np.testing.assert_allclose(flat.atm_skews(dk=0.03), 0.0, atol=1e-12)

    def test_skew_sign_for_downward_slope(self):
        """σ decreasing in K → negative skew (equity convention)."""
        strikes = np.array([80., 90., 100., 110., 120.])
        F = 100.0
        k = np.log(strikes / F)
        # Linearly decreasing smile: σ(k) = 0.20 - 0.20*k  → skew ≈ -0.20
        vols = np.array([[0.20 - 0.20 * ki for ki in k]])
        surf = ImpliedVolSurface(strikes, [1.0], vols, spot=F, forwards=np.array([F]))
        skew = surf.atm_skews(dk=0.05)
        assert skew[0] < 0.0
        # Exact: (σ(+0.05) - σ(-0.05))/(0.10) = (-0.20*0.05 - 0.20*(-0.05))/0.10
        # = (-0.01 + 0.01)/0.10 = 0 ... wait, let me recalculate
        # σ(k) = 0.20 - 0.20*k
        # σ(+0.05) = 0.20 - 0.20*0.05 = 0.19
        # σ(-0.05) = 0.20 - 0.20*(-0.05) = 0.21
        # skew = (0.19 - 0.21)/(2*0.05) = -0.02/0.10 = -0.20
        assert skew[0] == pytest.approx(-0.20, rel=1e-6)

    def test_skew_sign_for_upward_slope(self):
        """σ increasing in K → positive skew."""
        strikes = np.array([80., 90., 100., 110., 120.])
        F = 100.0
        k = np.log(strikes / F)
        vols = np.array([[0.20 + 0.20 * ki for ki in k]])
        surf = ImpliedVolSurface(strikes, [1.0], vols, spot=F, forwards=np.array([F]))
        assert surf.atm_skews(dk=0.05)[0] > 0.0

    def test_invalid_dk(self, flat):
        with pytest.raises(ValueError, match="dk"):
            flat.atm_skews(dk=0.0)

    def test_nan_when_dk_exceeds_range(self):
        """dk larger than the smile coverage → NaN."""
        surf = ImpliedVolSurface(
            [99.0, 100.0, 101.0],   # very narrow grid
            [1.0],
            np.full((1, 3), 0.2),
            spot=100.0,
            forwards=np.array([100.0]),
        )
        # k range ≈ [-0.01, 0, +0.01]; dk=0.05 > 0.01 → NaN
        assert np.isnan(surf.atm_skews(dk=0.05)[0])

    def test_multiple_maturities(self, flat):
        """Skew computed independently per maturity."""
        skews = flat.atm_skews(dk=0.04)
        assert skews.shape == (3,)
        np.testing.assert_allclose(skews, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# check_no_arbitrage
# ---------------------------------------------------------------------------

class TestCheckNoArbitrage:
    def test_flat_surface_passes_all(self, flat):
        result = flat.check_no_arbitrage()
        assert result["no_calendar_arbitrage"] is True
        assert result["no_butterfly_arbitrage"] is True
        assert result["is_arbitrage_free"] is True
        assert result["calendar_violations"] == []
        assert result["butterfly_violations"] == []

    def test_calendar_violation_detected(self):
        """Short-maturity vol > long-maturity vol → TV decreasing → violation."""
        spot = 100.0
        strikes = np.array([95.0, 100.0, 105.0])
        # T=0.5 at σ=0.30: TV=0.045;  T=1.0 at σ=0.10: TV=0.010  <  0.045
        ivols = np.array([[0.30, 0.30, 0.30],
                          [0.10, 0.10, 0.10]])
        surf = ImpliedVolSurface(strikes, [0.5, 1.0], ivols,
                                 spot=spot, forwards=np.full(2, spot))
        result = surf.check_no_arbitrage()
        assert result["no_calendar_arbitrage"] is False
        assert result["is_arbitrage_free"] is False
        assert len(result["calendar_violations"]) > 0

    def test_calendar_violation_fields(self):
        """Each calendar violation dict has expected keys."""
        spot = 100.0
        ivols = np.array([[0.30], [0.10]])
        surf = ImpliedVolSurface([100.0], [0.5, 1.0], ivols,
                                 spot=spot, forwards=np.full(2, spot))
        v = surf.check_no_arbitrage()["calendar_violations"][0]
        assert "T1" in v and "T2" in v and "K" in v
        assert "w1" in v and "w2" in v
        assert v["T1"] < v["T2"]
        assert v["w2"] < v["w1"]

    def test_butterfly_violation_detected(self):
        """Reverse smile (high ATM vol, low wing vols) → concave calls → violation."""
        spot = 100.0
        # σ=[0.10, 0.40, 0.10] at [90, 100, 110]: BS butterfly < 0  (verified analytically)
        ivols = np.array([[0.10, 0.40, 0.10]])
        surf = ImpliedVolSurface(
            [90.0, 100.0, 110.0], [1.0], ivols,
            spot=spot, forwards=np.array([spot]),
        )
        result = surf.check_no_arbitrage()
        assert result["no_butterfly_arbitrage"] is False
        assert result["is_arbitrage_free"] is False
        assert len(result["butterfly_violations"]) > 0

    def test_butterfly_violation_fields(self):
        ivols = np.array([[0.10, 0.40, 0.10]])
        surf = ImpliedVolSurface(
            [90.0, 100.0, 110.0], [1.0], ivols,
            spot=100.0, forwards=np.array([100.0]),
        )
        v = surf.check_no_arbitrage()["butterfly_violations"][0]
        assert "T" in v and "K" in v and "butterfly" in v
        assert v["butterfly"] < 0.0

    def test_normal_smile_passes_butterfly(self):
        """Standard (convex) smile passes butterfly check."""
        spot = 100.0
        # Small normal smile: σ increases away from ATM (no violation expected)
        k = np.log(np.array([90., 95., 100., 105., 110.]) / spot)
        vols = np.array([[0.20 + 0.05 * ki ** 2 for ki in k]])
        surf = ImpliedVolSurface(
            [90., 95., 100., 105., 110.], [1.0], vols,
            spot=spot, forwards=np.array([spot]),
        )
        result = surf.check_no_arbitrage()
        assert result["no_butterfly_arbitrage"] is True

    def test_nan_vols_skipped_in_butterfly(self):
        """NaN entries in the vol grid must not crash the butterfly check."""
        ivols = np.array([[0.2, np.nan, 0.2, 0.2, 0.2]])
        surf = ImpliedVolSurface(
            [90., 95., 100., 105., 110.], [1.0], ivols,
            spot=100.0, forwards=np.array([100.0]),
        )
        result = surf.check_no_arbitrage()
        assert isinstance(result["no_butterfly_arbitrage"], bool)

    def test_single_maturity_no_calendar_check(self):
        """A single maturity cannot have a calendar violation."""
        surf = ImpliedVolSurface(
            [90., 100., 110.], [1.0], np.full((1, 3), 0.2),
            spot=100.0,
        )
        result = surf.check_no_arbitrage()
        assert result["no_calendar_arbitrage"] is True
        assert result["calendar_violations"] == []


# ---------------------------------------------------------------------------
# from_csv
# ---------------------------------------------------------------------------

class TestFromCsv:
    def test_shape(self, flat_csv):
        surf = ImpliedVolSurface.from_csv(flat_csv)
        assert surf.n_maturities == 3
        assert surf.n_strikes == 5

    def test_ivols_correct(self, flat_csv):
        surf = ImpliedVolSurface.from_csv(flat_csv)
        np.testing.assert_allclose(surf.implied_vols, 0.2)

    def test_maturities_correct(self, flat_csv):
        surf = ImpliedVolSurface.from_csv(flat_csv)
        np.testing.assert_allclose(surf.maturities, [0.5, 1.0, 2.0])

    def test_strikes_correct(self, flat_csv):
        surf = ImpliedVolSurface.from_csv(flat_csv)
        np.testing.assert_allclose(surf.strikes, [90., 95., 100., 105., 110.])

    def test_spot_read_from_csv(self, flat_csv):
        surf = ImpliedVolSurface.from_csv(flat_csv)
        assert surf.spot == 100.0

    def test_forwards_read_from_csv(self, flat_csv):
        surf = ImpliedVolSurface.from_csv(flat_csv)
        np.testing.assert_allclose(surf.forwards, 100.0)

    def test_spot_default_when_column_absent(self, tmp_path):
        """Without a 'spot' column, the caller-supplied default is used."""
        lines = ["maturity,strike,implied_vol"]
        for T in [0.5, 1.0]:
            for K in [90., 100., 110.]:
                lines.append(f"{T},{K},0.2")
        p = tmp_path / "no_spot.csv"
        p.write_text("\n".join(lines))
        surf = ImpliedVolSurface.from_csv(p, spot=50.0)
        assert surf.spot == 50.0

    def test_forwards_default_to_spot_when_column_absent(self, tmp_path):
        lines = ["maturity,strike,implied_vol"]
        for T in [1.0]:
            for K in [90., 100.]:
                lines.append(f"{T},{K},0.2")
        p = tmp_path / "no_fwd.csv"
        p.write_text("\n".join(lines))
        surf = ImpliedVolSurface.from_csv(p, spot=80.0)
        np.testing.assert_allclose(surf.forwards, 80.0)

    def test_missing_required_column_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("maturity,strike\n1.0,100.0\n")
        with pytest.raises(ValueError, match="implied_vol"):
            ImpliedVolSurface.from_csv(p)

    def test_path_as_string(self, flat_csv):
        surf = ImpliedVolSurface.from_csv(str(flat_csv))
        assert surf.n_maturities == 3

    def test_roundtrip_atm_vols(self, flat_csv):
        """ATM vols on a surface loaded from CSV should equal σ=0.20."""
        surf = ImpliedVolSurface.from_csv(flat_csv)
        np.testing.assert_allclose(surf.atm_vols(), 0.2)

    def test_roundtrip_no_arbitrage(self, flat_csv):
        surf = ImpliedVolSurface.from_csv(flat_csv)
        assert surf.check_no_arbitrage()["is_arbitrage_free"] is True
