# rough-vol-engine

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

A Python library for pricing and calibrating **rough stochastic volatility models**.

---

## Overview

`rough-vol-engine` implements the two principal rough-volatility model families,
a Monte Carlo simulation engine, option pricing routines, and a two-stage
parameter calibration framework built on fractional Brownian motion (fBm).

| Model | Reference | Characteristic function | MC simulation |
|---|---|---|---|
| **Rough Bergomi** (rBergomi) | Bayer, Friz & Gatheral (2016) | — | Hybrid fBm scheme |
| **Rough Heston** (rfHeston) | El Euch & Rosenbaum (2019) | Adams fractional Riccati | — |

Both models reproduce the empirically observed power-law explosion of the
short-maturity ATM implied-volatility skew,
`|∂σ/∂k| ~ C·T^{H-1/2}`, with Hurst exponents H ≈ 0.10–0.14 for SPX.

---

## Repository layout

```
rough-vol-engine/
├── rough_vol/                  # Main library package
│   ├── models/
│   │   ├── rbergomi.py         # rBergomi params + simulation + IV surface
│   │   └── rfheston.py         # Rough Heston params + Riccati solver + pricing
│   ├── calibration/
│   │   ├── surface.py          # ImpliedVolSurface (no-arb checks, skew, CSV I/O)
│   │   ├── objective.py        # WSSE loss functions (uniform / vega / relative)
│   │   └── optimizer.py        # RoughVolCalibrator (DE + Nelder-Mead)
│   └── variance/
│       ├── integrated.py       # Integrated and realized variance from paths
│       ├── forward.py          # ATM total variance, forward variance curve
│       └── swap.py             # Variance swap fair strike (MC + surface approx)
├── data/
│   └── synthetic.py            # Surface factory functions for tests and demos
├── scripts/
│   ├── run_simulation.py       # Simulate rBergomi paths, print variance stats
│   ├── run_calibration.py      # Calibrate rBergomi or rfHeston to a surface
│   ├── compare_models.py       # Side-by-side rBergomi vs rfHeston surface comparison
│   └── variance_analysis.py    # MC variance swaps, forward var curve, VRP
├── notebooks/
│   ├── 01_rough_vol_intro.ipynb      # fBm paths, smile surfaces, skew power law
│   ├── 02_calibration_demo.ipynb     # Two-stage calibration workflow
│   └── 03_variance_swaps.ipynb       # Variance swaps, forward var, VRP
├── tests/                      # pytest test suite
├── docs/
│   ├── math_background.md      # Full mathematical treatment with LaTeX
│   └── numerical_methods.md    # Algorithm details (Adams, Gil-Pelaez, Brent, DE)
├── configs/                    # YAML configuration files
├── pyproject.toml
└── LICENSE
```

---

## Installation

```bash
# Development install (editable, with test dependencies)
pip install -e ".[dev]"
```

Requirements: Python 3.10+, NumPy, SciPy. Optional: matplotlib (plots), Jupyter (notebooks).

---

## Quick start

### 1. Simulate rBergomi paths and price an option

```python
from rough_vol.models.rbergomi import RBergomiParams, simulate_paths, implied_vol_surface
import numpy as np

params = RBergomiParams(H=0.10, eta=1.9, rho=-0.90, xi0=0.04)
V, S = simulate_paths(params, n_paths=10_000, n_steps=52, T=1.0, rng=42)

strikes = np.array([0.90, 0.95, 1.00, 1.05, 1.10])
surface = implied_vol_surface(params, strikes=strikes, maturities=[0.25, 0.5, 1.0],
                               n_paths=10_000, rng=42)
print(surface)           # ATM vols, no-arb checks
print(surface.atm_vols())
```

### 2. Price with the rough Heston characteristic function

```python
from rough_vol.models.rfheston import RHestonParams, rfheston_ivs
import numpy as np

params = RHestonParams(H=0.10, lambda_=1.5, theta=0.04, rho=-0.60, nu=0.35, V0=0.04)
log_strikes = np.linspace(-0.20, 0.20, 11)
ivs = rfheston_ivs(params, log_strikes=log_strikes, T=1.0, n_steps=200, n_quad=64)
```

### 3. Calibrate to a surface

```python
from data.synthetic import make_rfheston_surface
from rough_vol.calibration import RoughVolCalibrator

surface = make_rfheston_surface()          # synthetic target surface

cal = RoughVolCalibrator(surface, model='rfheston', n_steps=100, n_quad=64)
result = cal.calibrate(method='two_stage')

print(result.params)    # calibrated RHestonParams
print(f"Loss: {result.loss:.6f}  evals: {result.n_evals}")
```

### 4. Compute a variance swap fair strike

```python
from rough_vol.models.rbergomi import RBergomiParams
from rough_vol.variance import variance_swap_strike_mc

params = RBergomiParams(H=0.10, eta=1.9, rho=-0.90, xi0=0.04)
k_var, std_err = variance_swap_strike_mc(params, T=1.0, n_paths=50_000)
print(f"K_var = {k_var:.4f} ± {std_err:.4f}  (theory: ξ₀ = {params.xi0:.4f})")
```

---

## Module reference

### `rough_vol.models`

| Symbol | Description |
|---|---|
| `RBergomiParams` | Dataclass: H, eta, rho, xi0 |
| `simulate_paths(params, n_paths, n_steps, T, rng)` | Returns `(V, S)` arrays, shape `(n_paths, n_steps+1)` |
| `implied_vol_surface(params, strikes, maturities, ...)` | Returns `ImpliedVolSurface` from MC |
| `RHestonParams` | Dataclass: H, lambda_, theta, rho, nu, V0 |
| `rfheston_ivs(params, log_strikes, T, n_steps, n_quad)` | Fourier-inversion implied vols |
| `rfheston_surface(params, strikes, maturities, ...)` | Returns `ImpliedVolSurface` |

### `rough_vol.calibration`

| Symbol | Description |
|---|---|
| `ImpliedVolSurface` | Surface container: vols, strikes, maturities; ATM, skew, CSV I/O |
| `RBergomiObjective` | WSSE loss for rBergomi |
| `RHestonObjective` | WSSE loss for rough Heston |
| `RoughVolCalibrator` | Two-stage calibration engine |
| `CalibrationResult` | Dataclass: params, loss, n_evals, success, history |

**`RoughVolCalibrator`**

```python
cal = RoughVolCalibrator(
    surface,
    model='rfheston',          # or 'rbergomi'
    weight_scheme='uniform',   # 'uniform' | 'vega' | 'relative'
    n_paths=5_000,             # rBergomi MC paths
    n_steps=100,               # rfHeston Riccati steps
    n_quad=64,                 # Gauss-Legendre nodes
)

result = cal.calibrate(
    method='two_stage',        # 'differential_evolution' | 'nelder_mead' | 'two_stage'
    de_popsize=10,             # DE population multiplier
    de_maxiter=300,
    nm_maxiter=500,
)

H_est = RoughVolCalibrator.estimate_H_from_skew(surface)
```

### `rough_vol.variance`

| Symbol | Description |
|---|---|
| `integrated_variance(V, dt)` | Per-path ∫₀ᵀ Vt dt via trapezoid rule |
| `annualized_integrated_variance(V, dt)` | ∫V dt / T |
| `realized_variance(S)` | Σ(log-returns)² |
| `annualized_realized_variance(S, T)` | Σ(log-returns)² / T |
| `variance_risk_premium(iv, rv)` | Mean and std-err of RV − IV |
| `variance_path_stats(V, dt)` | Dict: mean/std/skew/kurtosis of IV |
| `atm_total_variance(surface)` | σ²_ATM · T for each maturity |
| `forward_variance_curve(surface)` | Δw/ΔT between consecutive maturities |
| `variance_swap_strike_mc(params, T, ...)` | MC fair strike + std-err (rBergomi) |
| `variance_swap_strike_atm_approx(surface, T)` | ATM proxy for K_var |
| `variance_swap_strike_log_contract(surface, T)` | Model-free log-contract replication |
| `variance_swap_pnl(K_var, RV, notional, long)` | P&L at expiry |
| `variance_swap_delta(K_var, V_current, T_rem, T_tot)` | MtM value of running swap |

### `data.synthetic`

| Symbol | Description |
|---|---|
| `make_rfheston_surface(params, ...)` | rfHeston IV surface (Fourier pricing) |
| `make_rbergomi_surface(params, ...)` | rBergomi IV surface (Monte Carlo) |
| `make_skew_surface(sigma_atm, skew, curvature, ...)` | Analytic skew surface |
| `make_benchmark_surfaces()` | Dict of named benchmark surfaces |
| `log_strike_grid(k_lo, k_hi, n)` | Log-moneyness grid helper |
| `standard_maturity_grid()` | Default [0.25, 0.5, 1.0, 2.0] grid |

---

## Scripts

```bash
# Simulate rBergomi paths and print variance statistics
python scripts/run_simulation.py --H 0.10 --xi0 0.04 --eta 1.9 --rho -0.9 \
    --n-paths 10000 --T 1.0 --plot

# Calibrate rough Heston (two-stage: DE + Nelder-Mead) to a synthetic surface
python scripts/run_calibration.py --model rfheston --method two_stage

# Calibrate to your own CSV surface
python scripts/run_calibration.py --surface data/my_surface.csv --model rfheston --out result.json

# Compare rBergomi and rfHeston at matched ATM vol
python scripts/compare_models.py --H 0.10 --sigma-atm 0.20 --plot

# Variance swap analysis: MC strikes, forward var curve, VRP distribution
python scripts/variance_analysis.py --H 0.10 --xi0 0.04 --n-paths 5000 --plot
```

---

## Mathematical background

The key relationships are summarised here; full derivations are in `docs/math_background.md`.

**fBm** with Hurst exponent H satisfies E[(W^H_t)²] = t^{2H}.
For H < 1/2 (rough regime), increments are negatively correlated.

**Rough Bergomi**: V(t) = ξ₀ · exp(η·W^H(t) − ½η²t^{2H}), so E[V(t)] = ξ₀.
Variance swap fair strike = ξ₀ (exact for flat forward-var curve).

**Rough Heston**: fractional CIR with characteristic function
E[e^{iu log S_T}] = exp(g(u,T) + V₀·ψ(u,T)),
where g, ψ satisfy a fractional Riccati ODE (Adams predictor–corrector).

**ATM skew power law**: |∂σ_ATM/∂k| ~ C·T^{H-1/2} as T→0.
Log–log regression of observed skew vs maturity recovers H:
H_est = slope + 0.5.  Empirical SPX: H ≈ 0.10 ± 0.02.

---

## Running tests

```bash
pytest                              # all tests (~2 min)
pytest tests/test_calibration.py -v # calibration tests only
pytest -x --tb=short                # stop on first failure
```

The test suite has 45+ tests covering:
- rBergomi path simulation and moments
- rfHeston characteristic function and IV computation
- Two-stage calibration convergence (round-trip on synthetic surface)
- No-arbitrage surface checks
- Variance swap strike accuracy (E[IV] = ξ₀ for rBergomi)
- Forward variance curve extraction

---

## Notebooks

| Notebook | Content |
|---|---|
| `01_rough_vol_intro.ipynb` | fBm path visualisation, rBergomi/rfHeston smile surfaces, H sensitivity, skew power-law log–log plots |
| `02_calibration_demo.ipynb` | Nelder-Mead warm-start, two-stage DE→NM calibration, weight scheme comparison, convergence plots |
| `03_variance_swaps.ipynb` | MC variance swap strikes, forward variance curve, VRP distribution, effect of H on K_var |

---

## References

- Bayer, C., Friz, P., & Gatheral, J. (2016). Pricing under rough volatility.
  *Quantitative Finance*, 16(6), 887–904.
- El Euch, O., & Rosenbaum, M. (2019). The characteristic function of rough Heston models.
  *Mathematical Finance*, 29(1), 3–38.
- Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is rough.
  *Quantitative Finance*, 18(6), 933–949.
- Demeterfi, K., Derman, E., Kamal, M., & Zou, J. (1999). More than you ever wanted
  to know about volatility swaps. Goldman Sachs Quantitative Strategies.

---

## License

MIT — see [LICENSE](LICENSE).
