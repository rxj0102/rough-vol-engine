"""Tests for rough_vol.pricing.payoffs."""

import numpy as np
import pytest

from rough_vol.pricing.payoffs import (
    asian_call_payoff,
    asian_put_payoff,
    barrier_call_payoff,
    call_payoff,
    digital_call_payoff,
    digital_put_payoff,
    put_payoff,
)


# ---------------------------------------------------------------------------
# call / put
# ---------------------------------------------------------------------------

class TestCallPayoff:
    def test_itm(self):
        np.testing.assert_allclose(call_payoff(np.array([1.2]), 1.0), [0.2])

    def test_otm(self):
        np.testing.assert_allclose(call_payoff(np.array([0.8]), 1.0), [0.0])

    def test_atm(self):
        np.testing.assert_allclose(call_payoff(np.array([1.0]), 1.0), [0.0])

    def test_vectorised(self):
        S = np.array([0.8, 1.0, 1.2, 1.5])
        payoffs = call_payoff(S, 1.0)
        np.testing.assert_allclose(payoffs, [0.0, 0.0, 0.2, 0.5])

    def test_non_negative(self):
        S = np.linspace(0.5, 2.0, 100)
        assert np.all(call_payoff(S, 1.0) >= 0)


class TestPutPayoff:
    def test_itm(self):
        np.testing.assert_allclose(put_payoff(np.array([0.8]), 1.0), [0.2])

    def test_otm(self):
        np.testing.assert_allclose(put_payoff(np.array([1.2]), 1.0), [0.0])

    def test_vectorised(self):
        S = np.array([0.8, 1.0, 1.2])
        np.testing.assert_allclose(put_payoff(S, 1.0), [0.2, 0.0, 0.0])

    def test_put_call_parity_payoff(self):
        """C - P = S_T - K for each path."""
        S = np.linspace(0.5, 2.0, 50)
        K = 1.0
        diff = call_payoff(S, K) - put_payoff(S, K)
        np.testing.assert_allclose(diff, S - K)


# ---------------------------------------------------------------------------
# digital call / put
# ---------------------------------------------------------------------------

class TestDigitalPayoffs:
    def test_digital_call_above(self):
        np.testing.assert_allclose(digital_call_payoff(np.array([1.1]), 1.0), [1.0])

    def test_digital_call_below(self):
        np.testing.assert_allclose(digital_call_payoff(np.array([0.9]), 1.0), [0.0])

    def test_digital_call_at_strike(self):
        # Exactly at strike: 1_{S>K} = 0
        np.testing.assert_allclose(digital_call_payoff(np.array([1.0]), 1.0), [0.0])

    def test_digital_put_above(self):
        np.testing.assert_allclose(digital_put_payoff(np.array([1.1]), 1.0), [0.0])

    def test_digital_put_below(self):
        np.testing.assert_allclose(digital_put_payoff(np.array([0.9]), 1.0), [1.0])

    def test_digital_call_plus_put_le_one_at_strike(self):
        S = np.array([1.0])
        # At strike: call=0, put=0 (strict inequalities)
        assert digital_call_payoff(S, 1.0)[0] + digital_put_payoff(S, 1.0)[0] == 0.0

    def test_digital_call_plus_put_is_one_away_from_strike(self):
        S = np.linspace(0.5, 1.5, 11)
        non_at_strike = S[S != 1.0]
        total = digital_call_payoff(non_at_strike, 1.0) + digital_put_payoff(non_at_strike, 1.0)
        np.testing.assert_allclose(total, 1.0)


# ---------------------------------------------------------------------------
# Asian
# ---------------------------------------------------------------------------

class TestAsianPayoffs:
    def _make_paths(self, terminal_val):
        # Paths where average = terminal_val (constant paths)
        return np.full((5, 11), terminal_val)

    def test_asian_call_itm(self):
        paths = self._make_paths(1.2)
        payoffs = asian_call_payoff(paths, 1.0)
        np.testing.assert_allclose(payoffs, 0.2)

    def test_asian_call_otm(self):
        paths = self._make_paths(0.8)
        payoffs = asian_call_payoff(paths, 1.0)
        np.testing.assert_allclose(payoffs, 0.0)

    def test_asian_put_itm(self):
        paths = self._make_paths(0.8)
        payoffs = asian_put_payoff(paths, 1.0)
        np.testing.assert_allclose(payoffs, 0.2)

    def test_asian_put_otm(self):
        paths = self._make_paths(1.2)
        payoffs = asian_put_payoff(paths, 1.0)
        np.testing.assert_allclose(payoffs, 0.0)

    def test_asian_uses_average(self):
        # Path that starts low and ends high: average < terminal
        paths = np.array([[0.5, 1.0, 1.5]])  # avg=1.0, terminal=1.5
        K = 1.0
        asian = float(asian_call_payoff(paths, K)[0])
        vanilla = float(paths[0, -1] - K)  # 0.5
        assert asian < vanilla

    def test_asian_put_call_parity(self):
        # Asian C - Asian P = avg - K
        paths = np.random.default_rng(42).uniform(0.8, 1.2, (20, 50))
        K = 1.0
        avgs = paths.mean(axis=1)
        diff = asian_call_payoff(paths, K) - asian_put_payoff(paths, K)
        np.testing.assert_allclose(diff, avgs - K)


# ---------------------------------------------------------------------------
# Barrier
# ---------------------------------------------------------------------------

class TestBarrierCallPayoff:
    def _flat_paths(self, level, n_paths=4, n_steps=10):
        return np.full((n_paths, n_steps + 1), level)

    def test_up_out_knocked(self):
        # Paths exceed barrier → zero payoff
        paths = self._flat_paths(1.5)
        payoffs = barrier_call_payoff(paths, K=1.0, barrier=1.2, barrier_type="up-out")
        np.testing.assert_allclose(payoffs, 0.0)

    def test_up_out_alive(self):
        # Paths never reach barrier → vanilla payoff
        paths = self._flat_paths(1.1)
        payoffs = barrier_call_payoff(paths, K=1.0, barrier=1.5, barrier_type="up-out")
        np.testing.assert_allclose(payoffs, 0.1)

    def test_up_in_knocked(self):
        paths = self._flat_paths(1.5)
        payoffs = barrier_call_payoff(paths, K=1.0, barrier=1.2, barrier_type="up-in")
        np.testing.assert_allclose(payoffs, 0.5)

    def test_up_in_not_knocked(self):
        paths = self._flat_paths(1.1)
        payoffs = barrier_call_payoff(paths, K=1.0, barrier=1.5, barrier_type="up-in")
        np.testing.assert_allclose(payoffs, 0.0)

    def test_down_out_knocked(self):
        paths = self._flat_paths(0.5)
        payoffs = barrier_call_payoff(paths, K=1.0, barrier=0.8, barrier_type="down-out")
        # Path goes through barrier but terminal < K → payoff = 0 regardless
        np.testing.assert_allclose(payoffs, 0.0)

    def test_down_out_alive(self):
        # Path stays above barrier, terminal > K
        paths = self._flat_paths(1.2)
        payoffs = barrier_call_payoff(paths, K=1.0, barrier=0.8, barrier_type="down-out")
        np.testing.assert_allclose(payoffs, 0.2)

    def test_down_in_knocked(self):
        paths = self._flat_paths(0.5)
        # Path touches barrier; terminal=0.5, K=1.0 → OTM → payoff=0
        payoffs = barrier_call_payoff(paths, K=1.0, barrier=0.8, barrier_type="down-in")
        np.testing.assert_allclose(payoffs, 0.0)

    def test_up_out_plus_up_in_equals_vanilla(self):
        """Up-out + up-in = vanilla call."""
        rng = np.random.default_rng(0)
        paths = rng.lognormal(mean=0.0, sigma=0.2, size=(100, 21))
        K, barrier = 1.0, 1.5
        out = barrier_call_payoff(paths, K, barrier, "up-out")
        inp = barrier_call_payoff(paths, K, barrier, "up-in")
        vanilla = np.maximum(paths[:, -1] - K, 0.0)
        np.testing.assert_allclose(out + inp, vanilla, atol=1e-12)

    def test_down_out_plus_down_in_equals_vanilla(self):
        rng = np.random.default_rng(1)
        paths = rng.lognormal(mean=0.0, sigma=0.2, size=(100, 21))
        K, barrier = 1.0, 0.7
        out = barrier_call_payoff(paths, K, barrier, "down-out")
        inp = barrier_call_payoff(paths, K, barrier, "down-in")
        vanilla = np.maximum(paths[:, -1] - K, 0.0)
        np.testing.assert_allclose(out + inp, vanilla, atol=1e-12)

    def test_invalid_barrier_type(self):
        with pytest.raises(ValueError):
            barrier_call_payoff(np.ones((5, 11)), K=1.0, barrier=1.5, barrier_type="left-in")
