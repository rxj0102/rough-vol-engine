"""Tests for rough_vol.pricing.monte_carlo."""

import numpy as np
import pytest
from numpy.random import default_rng

from rough_vol.pricing.monte_carlo import (
    MCResult,
    mc_asian_call,
    mc_call,
    mc_digital_call,
    mc_price,
    mc_put,
)
from rough_vol.pricing.payoffs import call_payoff, put_payoff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _constant_paths(value: float, n_paths: int = 1000, n_steps: int = 50) -> np.ndarray:
    """Paths that are constant at `value` — deterministic payoffs."""
    return np.full((n_paths, n_steps + 1), value)


def _bs_paths(S0: float = 1.0, sigma: float = 0.2, T: float = 1.0,
              n_paths: int = 5000, n_steps: int = 100, seed: int = 0) -> np.ndarray:
    """Exact geometric Brownian motion paths for BS benchmarking."""
    rng = default_rng(seed)
    dt = T / n_steps
    increments = rng.standard_normal((n_paths, n_steps))
    log_increments = (- 0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * increments)
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


# ---------------------------------------------------------------------------
# MCResult
# ---------------------------------------------------------------------------

class TestMCResult:
    def test_fields(self):
        res = MCResult(price=0.1, stderr=0.002, n_paths=1000, ci_lower=0.096, ci_upper=0.104)
        assert res.price == 0.1
        assert res.stderr == 0.002
        assert res.n_paths == 1000
        assert res.ci_lower < res.price < res.ci_upper

    def test_repr(self):
        res = MCResult(price=0.1, stderr=0.002, n_paths=1000, ci_lower=0.096, ci_upper=0.104)
        r = repr(res)
        assert "MCResult" in r
        assert "price" in r

    def test_ci_width(self):
        """CI width should be ≈ 2 * 1.96 * stderr."""
        res = MCResult(price=0.1, stderr=0.01, n_paths=1000, ci_lower=0.0804, ci_upper=0.1196)
        width = res.ci_upper - res.ci_lower
        assert width == pytest.approx(2 * 1.959964 * res.stderr, rel=1e-4)


# ---------------------------------------------------------------------------
# mc_price
# ---------------------------------------------------------------------------

class TestMcPrice:
    def test_deterministic_call_itm(self):
        """Constant paths at 1.2, K=1.0 → payoff 0.2 exactly."""
        paths = _constant_paths(1.2)
        res = mc_price(paths, lambda p: call_payoff(p[:, -1], 1.0), T=1.0, r=0.0)
        assert res.price == pytest.approx(0.2, abs=1e-10)
        assert res.stderr == pytest.approx(0.0, abs=1e-10)

    def test_deterministic_call_otm(self):
        paths = _constant_paths(0.8)
        res = mc_price(paths, lambda p: call_payoff(p[:, -1], 1.0), T=1.0, r=0.0)
        assert res.price == pytest.approx(0.0, abs=1e-10)

    def test_discounting(self):
        """Discount factor applied correctly."""
        paths = _constant_paths(1.2)
        r = 0.1
        T = 1.0
        res = mc_price(paths, lambda p: call_payoff(p[:, -1], 1.0), T=T, r=r)
        assert res.price == pytest.approx(0.2 * np.exp(-r * T), rel=1e-8)

    def test_stderr_positive_for_random_paths(self):
        paths = _bs_paths(n_paths=500, seed=0)
        res = mc_price(paths, lambda p: call_payoff(p[:, -1], 1.0), T=1.0)
        assert res.stderr > 0.0

    def test_n_paths_field(self):
        paths = _constant_paths(1.0, n_paths=200)
        res = mc_price(paths, lambda p: call_payoff(p[:, -1], 1.0), T=1.0)
        assert res.n_paths == 200

    def test_ci_contains_mean(self):
        paths = _bs_paths(n_paths=1000, seed=1)
        res = mc_price(paths, lambda p: call_payoff(p[:, -1], 1.0), T=1.0)
        assert res.ci_lower < res.price < res.ci_upper

    def test_stderr_decreases_with_paths(self):
        res_small = mc_price(_bs_paths(n_paths=200, seed=2), lambda p: call_payoff(p[:, -1], 1.0), T=1.0)
        res_large = mc_price(_bs_paths(n_paths=2000, seed=2), lambda p: call_payoff(p[:, -1], 1.0), T=1.0)
        assert res_large.stderr < res_small.stderr

    def test_custom_payoff(self):
        """Pass a custom payoff: digital call."""
        paths = _constant_paths(1.1)
        res = mc_price(paths, lambda p: (p[:, -1] > 1.0).astype(float), T=1.0)
        assert res.price == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# mc_call / mc_put
# ---------------------------------------------------------------------------

class TestMcCallPut:
    def test_mc_call_positive(self):
        paths = _bs_paths(n_paths=2000, seed=3)
        res = mc_call(paths, K=1.0, T=1.0)
        assert res.price > 0.0

    def test_mc_put_positive(self):
        paths = _bs_paths(n_paths=2000, seed=4)
        res = mc_put(paths, K=1.0, T=1.0)
        assert res.price > 0.0

    def test_put_call_parity_mc(self):
        """C - P ≈ S₀ - K * exp(-rT) (MC version, r=0 → C - P ≈ S₀ - K)."""
        paths = _bs_paths(n_paths=10000, seed=5)
        K = 1.0
        call = mc_call(paths, K, T=1.0)
        put = mc_put(paths, K, T=1.0)
        S0_mean = paths[:, 0].mean()
        diff = call.price - put.price
        # Should be close to S0 - K = 0
        assert diff == pytest.approx(S0_mean - K, abs=0.02)

    def test_call_decreasing_in_K(self):
        paths = _bs_paths(n_paths=2000, seed=6)
        c1 = mc_call(paths, K=0.9, T=1.0).price
        c2 = mc_call(paths, K=1.0, T=1.0).price
        c3 = mc_call(paths, K=1.1, T=1.0).price
        assert c1 > c2 > c3

    def test_put_increasing_in_K(self):
        paths = _bs_paths(n_paths=2000, seed=7)
        p1 = mc_put(paths, K=0.9, T=1.0).price
        p2 = mc_put(paths, K=1.0, T=1.0).price
        p3 = mc_put(paths, K=1.1, T=1.0).price
        assert p1 < p2 < p3

    def test_call_convergence_to_bs(self):
        """MC call price should approach BS price for large n_paths."""
        from rough_vol.pricing.implied_vol import bs_call
        sigma, K, T = 0.2, 1.0, 1.0
        paths = _bs_paths(sigma=sigma, n_paths=20000, n_steps=252, seed=42)
        res = mc_call(paths, K=K, T=T)
        bs_price = float(bs_call(1.0, K, sigma, T))
        assert res.price == pytest.approx(bs_price, rel=0.05)


# ---------------------------------------------------------------------------
# mc_asian_call
# ---------------------------------------------------------------------------

class TestMcAsianCall:
    def test_asian_le_vanilla(self):
        """Asian call price ≤ vanilla call (by Jensen's inequality: E[avg] ≤ avg[E])."""
        paths = _bs_paths(n_paths=5000, seed=10)
        asian = mc_asian_call(paths, K=1.0, T=1.0).price
        vanilla = mc_call(paths, K=1.0, T=1.0).price
        # Asian averaging reduces variance, so price ≤ vanilla
        assert asian <= vanilla + 0.01  # small tolerance

    def test_asian_positive(self):
        paths = _bs_paths(n_paths=1000, seed=11)
        res = mc_asian_call(paths, K=1.0, T=1.0)
        assert res.price > 0.0

    def test_asian_constant_paths(self):
        """Constant paths: Asian = vanilla."""
        paths = _constant_paths(1.2)
        asian = mc_asian_call(paths, K=1.0, T=1.0).price
        assert asian == pytest.approx(0.2, abs=1e-10)


# ---------------------------------------------------------------------------
# mc_digital_call
# ---------------------------------------------------------------------------

class TestMcDigitalCall:
    def test_digital_probability(self):
        """Digital call price ≈ N(d2) = N(-sigma*sqrt(T)/2) for ATM, r=0."""
        from scipy.stats import norm
        sigma, T = 0.2, 1.0
        paths = _bs_paths(sigma=sigma, n_paths=20000, seed=12)
        res = mc_digital_call(paths, K=1.0, T=T)
        # P(S_T > S_0) = N(-sigma*sqrt(T)/2) ≈ 0.460 for sigma=0.2
        expected = norm.cdf(-sigma * np.sqrt(T) / 2)
        assert res.price == pytest.approx(expected, abs=0.02)

    def test_digital_decreasing_in_K(self):
        paths = _bs_paths(n_paths=5000, seed=13)
        d1 = mc_digital_call(paths, K=0.8, T=1.0).price
        d2 = mc_digital_call(paths, K=1.0, T=1.0).price
        d3 = mc_digital_call(paths, K=1.2, T=1.0).price
        assert d1 > d2 > d3

    def test_digital_between_zero_and_one(self):
        paths = _bs_paths(n_paths=1000, seed=14)
        res = mc_digital_call(paths, K=1.0, T=1.0)
        assert 0.0 < res.price < 1.0
