# Mathematical Background

This document gives a self-contained treatment of the mathematical objects underlying `rough-vol-engine`. It is aimed at practitioners who are comfortable with stochastic calculus but may be encountering rough-volatility models for the first time. Every section maps to one or more library modules; cross-references are collected in the **See Also** section at the end.

---

## 1. Fractional Brownian Motion

A **fractional Brownian motion** (fBm) $W^H = (W^H_t)_{t \ge 0}$ with **Hurst exponent** $H \in (0, 1)$ is the unique (up to indistinguishability) continuous centred Gaussian process satisfying

$$
\mathbb{E}\!\left[(W^H_t - W^H_s)^2\right] = |t - s|^{2H}, \qquad s, t \ge 0,
$$

and in particular $\mathbb{E}[(W^H_t)^2] = t^{2H}$.

### Riemann–Liouville representation

The most computationally useful representation is the **moving-average** (Riemann–Liouville) form:

$$
W^H_t = \int_0^t (t - s)^{H - 1/2}\, dW_s,
$$

where $W$ is an ordinary Brownian motion. This representation makes the "memory kernel" $(t-s)^{H-1/2}$ explicit and is the starting point for both Monte Carlo simulation and asymptotic analysis.

### Role of the Hurst exponent

| Regime | Increments | Path roughness |
|---|---|---|
| $H = 1/2$ | Independent (standard Brownian motion) | Continuous, nowhere differentiable |
| $H > 1/2$ | Positively correlated (persistent) | Smoother than BM |
| $H < 1/2$ | Negatively correlated (anti-persistent, *rough*) | More irregular than BM |

For $H < 1/2$ — the empirically relevant regime for equity volatility — successive increments are negatively correlated. Concretely, for $h > 0$,

$$
\text{Cov}\!\left(W^H_{t+h} - W^H_t,\; W^H_t - W^H_{t-h}\right) = h^{2H}\!\left(2^{2H-1} - 1\right) < 0 \quad \text{when } H < \tfrac{1}{2}.
$$

This anti-persistence is what generates the steep short-maturity ATM skew observed in equity option markets.

---

## 2. Rough Volatility — Motivation

### Empirical evidence

Gatheral, Jaisson, and Rosenbaum (2018) estimated the Hurst exponent of log-realised-variance time series across a broad cross-section of equities and indices. Their central finding is that log-volatility behaves like a fractional Brownian motion with

$$
H \approx 0.10\text{--}0.14 \quad (\text{SPX}).
$$

The diagnostic is the variance scaling of log-volatility increments:

$$
\operatorname{Var}\!\left[\log \sigma(t + \Delta) - \log \sigma(t)\right] \sim \Delta^{2H}.
$$

A log–log regression of the left-hand side against $\Delta$ gives a slope of approximately $2H \approx 0.20$, far below the $2H = 1$ that classical diffusion models would predict.

### Short-maturity ATM skew

In every semi-martingale stochastic volatility model, the at-the-money (ATM) implied-volatility skew decays as

$$
\left|\frac{\partial \sigma_{\text{BS}}}{\partial k}\right|_{k=0} \sim C \cdot T^{H - 1/2} \quad \text{as } T \to 0.
$$

| Model | $H$ | Skew behaviour |
|---|---|---|
| Heston, SABR (classical) | $1/2$ | $T^0 = \text{const}$ (too flat empirically) |
| Rough Bergomi / Rough Heston | $\approx 0.10$ | $T^{-0.40}$ (steep power-law explosion) |

The inability of classical models to reproduce this power-law explosion is the primary motivation for rough-volatility modelling.

---

## 3. Rough Bergomi Model

The **rough Bergomi** (rBergomi) model, introduced by Bayer, Friz, and Gatheral (2016), defines instantaneous variance as

$$
V(t) = \xi_0 \cdot \exp\!\left(\eta\, W^H(t) - \tfrac{1}{2}\eta^2 t^{2H}\right),
$$

where the exponential martingale correction $-\tfrac{1}{2}\eta^2 t^{2H}$ ensures $\mathbb{E}[V(t)] = \xi_0$ for all $t$, so $\xi_0$ is simultaneously the initial spot variance and the flat initial forward variance curve.

The asset price satisfies

$$
\frac{dS_t}{S_t} = \sqrt{V(t)}\, dB_t, \qquad \text{Corr}(dW, dB) = \rho.
$$

### Parameters

| Symbol | Interpretation | Typical SPX range |
|---|---|---|
| $H$ | Roughness of volatility | $0.05$–$0.20$ |
| $\eta$ | Volatility of volatility | $1.0$–$3.0$ |
| $\rho$ | Spot–vol correlation (skew driver) | $-0.90$–$-0.50$ |
| $\xi_0$ | Initial variance level | $0.01$–$0.10$ |

### ATM skew formula

Under rBergomi, the leading-order ATM skew is

$$
\frac{\partial \sigma_{\text{BS}}}{\partial k}\bigg|_{k=0} \approx -\frac{\rho\,\eta}{2\sqrt{2\pi}} \cdot T^{H - 1/2}.
$$

This closed-form expression makes the role of each parameter transparent: $\rho$ controls the sign, $\eta$ the magnitude, and $H$ the maturity decay exponent.

---

## 4. Rough Heston Model

The **rough Heston** model (El Euch and Rosenbaum 2019) replaces the integer-order CIR mean reversion with a fractional integral:

$$
V(t) = V_0 + \frac{1}{\Gamma(H + \tfrac{1}{2})} \int_0^t (t - s)^{H - 1/2} \!\left[\lambda(\theta - V(s))\, ds + \nu\sqrt{V(s)}\, dW_s\right],
$$

where $\Gamma$ denotes the Euler gamma function and $W$ is a standard Brownian motion with $\operatorname{Corr}(dW, dB) = \rho$.

### Characteristic function

The log-price characteristic function admits the semi-closed form

$$
\mathbb{E}\!\left[e^{iu \log S_T}\right] = \exp\!\left(g(u, T) + V_0 \cdot \psi(u, T)\right),
$$

where $g$ and $\psi$ are determined by the **fractional Riccati equation**

$$
D^\alpha \psi(u, t) = f\!\left(u, \psi(u, t)\right), \qquad \alpha = H + \tfrac{1}{2},
$$

with $D^\alpha$ the Caputo fractional derivative and $f$ a quadratic function of $\psi$. Because no closed-form solution exists for $\alpha \ne 1$, numerical integration of this fractional ODE is required (see `numerical_methods.md`, Section 3).

### Parameters

| Symbol | Interpretation |
|---|---|
| $H$ | Roughness exponent ($< 1/2$ for rough regime) |
| $\lambda$ | Mean-reversion speed |
| $\theta$ | Long-run variance |
| $\nu$ | Volatility of volatility |
| $\rho$ | Correlation |
| $V_0$ | Initial variance |

---

## 5. Implied Volatility Surface

Given a European call price $C(K, T)$ with forward $F = S_0 e^{rT}$, the **Black–Scholes implied volatility** $\sigma(k, T)$ is the unique positive root of

$$
C_{\text{BS}}(F, K, \sigma, T) = C(K, T),
$$

where $k = \log(K/F)$ is log-moneyness. The **total implied variance** is

$$
w(k, T) = \sigma^2(k, T) \cdot T.
$$

Working in $(k, w)$ coordinates simplifies no-arbitrage conditions significantly.

### No-arbitrage conditions

1. **Calendar spread**: $w(k, T)$ must be non-decreasing in $T$ for every fixed $k$.
2. **Butterfly**: $w(k, T)$ must be convex in $k$ for every fixed $T$, specifically

$$
\left(1 - \frac{k\,\partial_k w}{2w}\right)^2 - \frac{(\partial_k w)^2}{4}\!\left(\frac{1}{w} + \frac{1}{4}\right) + \frac{\partial_{kk} w}{2} \ge 0.
$$

### SVI parameterisation

For diagnostic purposes and as a calibration target, the **Stochastic Volatility Inspired** (SVI) slice is

$$
w(k) = a + b\!\left(\tilde\rho\,(k - m) + \sqrt{(k - m)^2 + \tilde\sigma^2}\right),
$$

with parameters $\{a, b, \tilde\rho, m, \tilde\sigma\}$. The library uses market SVI surfaces as calibration benchmarks.

---

## 6. Variance Swaps

A **variance swap** pays at maturity $T$

$$
N \cdot \left(\mathrm{RV}_T - K_{\text{var}}\right), \qquad \mathrm{RV}_T = \frac{1}{T}\int_0^T V_t\, dt,
$$

where $N$ is notional and $K_{\text{var}}$ is the fair strike agreed at inception.

### Fair strike

By definition, $K_{\text{var}} = \mathbb{E}[\mathrm{RV}_T]$. For rBergomi with a flat initial forward variance curve, $\mathbb{E}[V(t)] = \xi_0$ for all $t$, giving

$$
K_{\text{var}} = \xi_0.
$$

### Model-free replication

The fair variance-swap strike can be computed without specifying a model from the observed option prices:

$$
K_{\text{var}} = \frac{2}{T}\left[\int_F^\infty \frac{C(K)}{K^2}\, dK + \int_0^F \frac{P(K)}{K^2}\, dK\right].
$$

This follows from the log-contract replication identity: shorting $\tfrac{2}{T}\log(F/S_T)$ replicates the realised variance payoff in a continuous diffusion.

### Variance risk premium

The **variance risk premium** (VRP) is defined as

$$
\mathrm{VRP} = \mathbb{E}^{\mathbb{P}}[\mathrm{RV}_T] - K_{\text{var}}.
$$

Empirically, VRP is negative for equity indices (investors pay to hedge variance), which is an important stylised fact that rough-volatility models can accommodate through the risk-neutral vs physical change of measure.

---

## 7. ATM Skew Power Law and H Estimation

The power-law relationship

$$
\left|\frac{\partial \sigma_{\text{ATM}}}{\partial k}\right| \sim C \cdot T^{H - 1/2}
$$

implies that a log–log regression of observed ATM skew against maturity recovers $H$:

$$
\log\left|\text{skew}(T)\right| = \underbrace{(H - \tfrac{1}{2})}_{\beta} \cdot \log T + \text{const}.
$$

An OLS fit of $\log|\text{skew}|$ against $\log T$ gives slope $\hat\beta$, from which

$$
\hat{H} = \hat\beta + \tfrac{1}{2}.
$$

For SPX, the empirical estimate is $\hat{H} \approx 0.10 \pm 0.02$ (Gatheral–Jaisson–Rosenbaum 2018), consistent with the direct time-series estimation of Section 2.

The estimate is most reliable when:
- The maturity range spans at least one decade (e.g. 1 week to 2 years).
- The implied-volatility surface is liquid enough to compute reliable finite-difference skew estimates.
- Very short maturities (below $\sim$1 week) are excluded due to microstructure noise.

---

## References

- Bayer, C., Friz, P., and Gatheral, J. (2016). *Pricing under rough volatility.* Quantitative Finance, 16(6), 887–904.
- El Euch, O. and Rosenbaum, M. (2019). *The characteristic function of rough Heston models.* Mathematical Finance, 29(1), 3–38.
- Gatheral, J., Jaisson, T., and Rosenbaum, M. (2018). *Volatility is rough.* Quantitative Finance, 18(6), 933–949.

---

## See Also

| Topic | Library module |
|---|---|
| fBm simulation | `rough_vol.simulation.fbm` |
| rBergomi pricing | `rough_vol.models.rough_bergomi` |
| Rough Heston pricing | `rough_vol.models.rough_heston` |
| Implied volatility surface | `rough_vol.surface.implied_vol` |
| Variance swap utilities | `rough_vol.products.variance_swap` |
| H estimation | `rough_vol.estimation.hurst` |
| Numerical methods detail | `docs/numerical_methods.md` |
