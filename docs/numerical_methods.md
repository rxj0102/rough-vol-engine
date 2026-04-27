# Numerical Methods

This document describes every numerical algorithm used in `rough-vol-engine`, with enough detail to understand convergence properties, implementation choices, and tuning parameters. For the underlying mathematics, see `docs/math_background.md`.

---

## 1. Fractional Brownian Motion Simulation

Three algorithms are available. The library uses the **hybrid/circulant-embedding** method by default because it achieves the optimal $O(N \log N)$ cost while remaining exact.

### 1.1 Cholesky method (exact, $O(N^2)$)

The joint distribution of $(W^H_{t_1}, \ldots, W^H_{t_N})$ is multivariate Gaussian with covariance

$$
\Sigma_{ij} = \tfrac{1}{2}\!\left(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H}\right).
$$

**Algorithm:**
1. Build $\Sigma \in \mathbb{R}^{N \times N}$ using the formula above.
2. Compute the Cholesky factor $L$ such that $\Sigma = L L^\top$.
3. Sample $z \sim \mathcal{N}(0, I_N)$.
4. Return $W^H = Lz$.

This is exact for any time grid but becomes prohibitive for $N \gtrsim 10^3$ due to the $O(N^2)$ memory and $O(N^3)$ Cholesky factorisation. It is used in tests to validate the faster methods.

### 1.2 Hosking method (exact, $O(N^2)$)

The Hosking algorithm generates stationary Gaussian increments recursively via the Durbin–Levinson recursion. At each step $n$, it computes the conditional distribution of the next increment given all previous increments:

$$
\Delta W^H_{n+1} \mid \Delta W^H_1, \ldots, \Delta W^H_n \sim \mathcal{N}(\mu_n,\, \sigma_n^2),
$$

where $\mu_n$ and $\sigma_n^2$ are obtained from the autocovariance sequence of fBm increments. The cost is $O(N^2)$ but with a smaller constant than Cholesky; it is useful for moderate $N$ and non-uniform grids.

### 1.3 Hybrid/Circulant-embedding method (exact, $O(N \log N)$) — **default**

For a uniform grid with $N$ steps and step size $h = T/N$, the fBm increments form a stationary sequence. The circulant-embedding method exploits this stationarity:

**Algorithm:**
1. Compute the autocovariance vector $c_k = \tfrac{1}{2}(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})\cdot h^{2H}$ for $k = 0, 1, \ldots, N-1$.
2. Embed in a circulant matrix of size $M \ge 2N$ (smallest power of 2 works).
3. Compute the circulant eigenvalues $\lambda = \operatorname{FFT}([c_0, c_1, \ldots, c_{N-1}, c_{N-2}, \ldots, c_1])$.
4. Verify $\lambda \ge 0$ (guaranteed for $H \in (0,1)$ on uniform grids).
5. Sample $z_1, z_2 \sim \mathcal{N}(0, I_M)$ and form $w = z_1 + iz_2$.
6. Compute $\tilde{w} = \operatorname{FFT}\!\left(\sqrt{\lambda} \odot w\right)$.
7. Return the first $N$ elements of $\operatorname{Re}(\tilde{w}) / \sqrt{M}$ as the fBm increments.

The "hybrid" qualifier refers to combining circulant embedding for the stationary-increment part with an exact Cholesky correction near $t=0$ to handle the non-stationarity of fBm itself (as opposed to its increments). This is the approach of Bennedsen, Lunde, and Pakkanen (2017).

**Complexity:** $O(N \log N)$ time, $O(N)$ memory. Exact for stationary increments on uniform grids.

```python
# Pseudocode sketch — see rough_vol/simulation/fbm.py for full implementation
def simulate_fbm_hybrid(H, N, T, n_paths):
    h = T / N
    autocov = build_autocov(H, N, h)           # O(N)
    eigenvalues = np.fft.rfft(embed_circulant(autocov))  # O(N log N)
    sqrt_eig = np.sqrt(np.maximum(eigenvalues, 0))
    z = rng.standard_normal((2, M, n_paths))
    w = np.fft.irfft(sqrt_eig[:, None] * (z[0] + 1j * z[1]), n=M, axis=0)
    return np.cumsum(w[:N], axis=0) * h**H      # integrate to get fBm
```

---

## 2. Rough Bergomi Monte Carlo

The rBergomi pricer combines fBm simulation with a log-Euler scheme for the asset price.

### Steps

1. **Simulate** $W^H_{t_1}, \ldots, W^H_{t_N}$ via the hybrid scheme (Section 1.3).
2. **Construct variance process** (no numerical integration — closed form):
$$
V(t_i) = \xi_0 \cdot \exp\!\left(\eta\, W^H_{t_i} - \tfrac{1}{2}\eta^2 t_i^{2H}\right).
$$
3. **Correlated Brownian for the asset**: let $W_\perp$ be an independent standard BM. Then
$$
dB = \rho\,\frac{dW^H}{\|dW^H\|} + \sqrt{1 - \rho^2}\, dW_\perp,
$$
with the normalisation ensuring unit diffusion coefficient.
4. **Log-Euler scheme**:
$$
\log S_{t_{i+1}} = \log S_{t_i} - \tfrac{1}{2} V(t_i)\,\Delta t + \sqrt{V(t_i)}\,\Delta B_i.
$$
5. **Payoff evaluation**: compute $\max(S_T - K, 0)$ per path, average, and discount.

### Computational cost

For $P$ paths and $N$ time steps: $O(N P)$ total work (plus $O(N \log N)$ per batch for fBm). Typical production settings are $N = 364$ (daily) or $N = 52$ (weekly), $P = 10^5$–$10^6$.

### Variance reduction

- **Antithetic variates**: simulate $(W^H, -W^H)$ in pairs.
- **Control variate**: the Black–Scholes price under $V \equiv \xi_0$ is available in closed form and can be used as a control.

---

## 3. Rough Heston — Fractional Riccati Solver

The rough Heston characteristic function requires solving the fractional ODE

$$
D^\alpha \psi(u, t) = f\!\left(u, \psi(u, t)\right), \qquad \alpha = H + \tfrac{1}{2} \in \left(\tfrac{1}{2}, 1\right),
$$

where $D^\alpha$ is the Caputo derivative and $f(u, h) = \tfrac{1}{2}(-u^2 - iu) + (\rho\nu iu - \lambda)h + \tfrac{\nu^2}{2}h^2$.

### Adams predictor–corrector scheme

Let $h^n \approx \psi(u, t_n)$, $t_n = n\,\delta t$, and $\Gamma$ denote the gamma function.

**Predictor step:**

$$
h^{n+1}_P = \sum_{j=0}^{n} a_{n+1,j}^{(\alpha)}\, h^j + \frac{\delta t^\alpha}{\Gamma(\alpha + 2)} f(h^n, t_n),
$$

**Corrector step:**

$$
h^{n+1} = \sum_{j=0}^{n} a_{n+1,j}^{(\alpha)}\, h^j + \frac{\delta t^\alpha}{\Gamma(\alpha + 2)}\!\left[f\!\left(h^{n+1}_P, t_{n+1}\right) + \left((n+1)^{\alpha+1} - (n - \alpha)(n+1)^\alpha\right) f\!\left(h^0, t_0\right)\right],
$$

where the weights $a_{n+1,j}^{(\alpha)}$ encode the memory of the Caputo derivative:

$$
a_{n+1,j}^{(\alpha)} = \frac{\delta t^\alpha}{\Gamma(\alpha + 2)} \times \begin{cases} (n-\alpha)(n+1)^\alpha - n^{\alpha+1} + (\alpha+1)(n+1)^{\alpha} & j = 0 \\ (n-j+2)^{\alpha+1} - 2(n-j+1)^{\alpha+1} + (n-j)^{\alpha+1} & 1 \le j \le n \\ 1 & j = n+1 \end{cases}
$$

### Convergence

The scheme is $O(\delta t^{1+\alpha})$ for $\alpha \in (0,1)$, which for typical $H = 0.10$ gives $\alpha = 0.60$ and effective order $\approx 1.6$ — substantially better than the naive Euler $O(\delta t^\alpha)$ scheme. In practice, $N = 200$–$400$ steps on $[0, T_{\max}]$ is sufficient for sub-basis-point accuracy in implied volatilities.

---

## 4. Gil-Pelaez Fourier Inversion

Given the log-price characteristic function $\varphi(u) = \mathbb{E}[e^{iu \log S_T}]$, the **Gil-Pelaez** inversion formula expresses the exercise probabilities as

$$
P_2 = \frac{1}{2} + \frac{1}{\pi} \int_0^\infty \frac{\operatorname{Im}\!\left[e^{-iuk}\,\varphi(u)\right]}{u}\, du,
$$

$$
P_1 = \frac{1}{2} + \frac{1}{\pi} \int_0^\infty \frac{\operatorname{Im}\!\left[e^{-iuk}\,\varphi(u - i)\right]}{u}\, du,
$$

where $k = \log(K/F)$ is log-moneyness. The undiscounted call price is then $C = F\,P_1 - K\,P_2$.

### Numerical integration

The integral is evaluated with **Gauss-Legendre quadrature** on the interval $[\varepsilon, u_{\max}]$, typically with $\varepsilon = 10^{-6}$ and $u_{\max} = 200$:

1. Map $[\varepsilon, u_{\max}]$ to $[-1, 1]$ via an affine change of variables.
2. Evaluate $\varphi$ at $n_q = 64$–$256$ Gauss-Legendre nodes.
3. Form the weighted sum.

The integrand decays rapidly for large $u$ due to the Gaussian moment of $\log S_T$; $u_{\max} = 200$ is conservative for all standard parameter sets. The small lower limit $\varepsilon > 0$ is used to avoid the integrable singularity at $u = 0$.

```python
# Pseudocode — see rough_vol/pricing/fourier.py
def call_price_gil_pelaez(log_cf, k, nodes, weights, u_max):
    u = 0.5 * u_max * (nodes + 1) + eps  # map [-1,1] -> [eps, u_max]
    phi   = log_cf(u)
    phi_i = log_cf(u - 1j)
    p2 = 0.5 + (u_max / (2 * np.pi)) * np.dot(weights, np.imag(np.exp(-1j*u*k) * phi)   / u)
    p1 = 0.5 + (u_max / (2 * np.pi)) * np.dot(weights, np.imag(np.exp(-1j*u*k) * phi_i) / u)
    return np.exp(-r*T) * (F * p1 - K * p2)
```

---

## 5. Implied Volatility Inversion — Brent's Method

Given a model price $C^*$, the library recovers the Black–Scholes implied volatility $\hat\sigma$ by solving $C_{\text{BS}}(\hat\sigma) = C^*$ with **Brent's method**.

### Algorithm

1. **Bracket**: use the initial interval $[\sigma_{\text{lo}}, \sigma_{\text{hi}}] = [10^{-8},\ 10.0]$. Both endpoints are evaluated; $C_{\text{BS}}$ is monotone increasing in $\sigma$, so a bracket is guaranteed whenever $C^* \in (C_{\text{intrinsic}},\ C_{\text{upper}})$.
2. **Fallback**: if $|C^* - C_{\text{intrinsic}}| < \varepsilon_{\text{machine}}$, return `NaN` — the option is at the no-arbitrage boundary and the implied volatility is undefined.
3. **Brent iteration**: at each step the algorithm selects among three moves:
   - **Bisection**: guaranteed progress, used when other moves are rejected.
   - **Secant**: linear interpolation of the two most recent function values.
   - **Inverse quadratic interpolation**: fits a quadratic through three points and inverts; superlinearly convergent near the root.
4. **Termination**: converges to machine precision in $\le 50$ iterations for all practical inputs.

No derivatives of $C_{\text{BS}}$ are required, avoiding the division by vega that causes Newton–Raphson to fail for deep in-the-money or very short-maturity options.

---

## 6. Two-Stage Calibration

Calibration minimises the weighted root-mean-square error between model and market implied volatilities:

$$
\mathcal{L}(\theta) = \sqrt{\frac{1}{|S|}\sum_{(k_i, T_i)\in S} w_i\,\left(\sigma_{\text{model}}(k_i, T_i;\theta) - \sigma_{\text{mkt}}(k_i, T_i)\right)^2}.
$$

Because $\mathcal{L}$ is non-convex with multiple local minima, a **two-stage** strategy is used.

### Stage 1 — Differential Evolution (global search)

Differential Evolution (DE) is a population-based global optimiser that requires only function evaluations.

**Algorithm (DE/rand/1/bin):**

1. **Initialise** a population of $P$ candidate parameter vectors using Latin hypercube sampling within the parameter bounds. $P = 15n$ where $n$ is the parameter dimension ($n=4$ for rBergomi, $n=6$ for rough Heston).
2. **For each generation:**
   - **Mutation**: for each individual $x$, select three distinct random individuals $r_1, r_2, r_3$ and form the donor $v = x_{r_1} + F \cdot (x_{r_2} - x_{r_3})$ with scale factor $F \in [0.5, 1.0]$.
   - **Crossover**: form the trial vector $u$ by taking component $i$ from $v$ if $\operatorname{rand}() < \mathrm{CR}$, otherwise from $x$. At least one component is taken from $v$.
   - **Selection**: replace $x$ with $u$ if $\mathcal{L}(u) \le \mathcal{L}(x)$.
3. **Convergence criterion**: stop when

$$
\frac{\max_p \mathcal{L}(\theta^p) - \operatorname{mean}_p \mathcal{L}(\theta^p)}{|\operatorname{mean}_p \mathcal{L}(\theta^p)|} < \tau,
$$

with $\tau = 10^{-5}$ by default.

**Hyperparameters**: $F = 0.8$, $\mathrm{CR} = 0.9$ (default). Latin hypercube initialisation ensures better coverage than pure uniform random sampling, which is important for a 4–6 dimensional parameter space.

### Stage 2 — Nelder-Mead simplex (local refinement)

The best solution from DE seeds a **Nelder-Mead simplex** optimiser for fast local convergence.

**Algorithm:**

1. Construct an initial simplex of $n+1$ vertices by perturbing the DE solution along each coordinate direction by a small step $\delta$.
2. At each iteration, evaluate $\mathcal{L}$ at the worst vertex $x_w$ and attempt:
   - **Reflection** ($\alpha = 1$): reflect $x_w$ through the centroid of the remaining vertices.
   - **Expansion** ($\gamma = 2$): extend the reflection if it gives the best point yet.
   - **Contraction** ($\beta = 0.5$): contract toward the centroid if reflection is not better.
   - **Shrink**: if all else fails, shrink all vertices toward the best vertex by factor $0.5$.
3. Use the **adaptive variant** (Gao and Han 2012) that scales $\alpha, \gamma, \beta$ with $n$, which outperforms the fixed-coefficient version for $n \ge 4$.

Because Nelder-Mead is seeded from DE's global best, it typically converges in $O(n^2)$ function evaluations rather than the $O(10^3)$ required without a good starting point.

---

## 7. H Estimation from ATM Skew

Given a calibrated or observed implied-volatility surface, the Hurst exponent $H$ is estimated from the power-law decay of the ATM skew.

### Algorithm

1. **Compute ATM skew** at each available maturity $T_j$ using centred finite differences:
$$
\widehat{\text{skew}}(T_j) = \frac{\sigma(k_{+}, T_j) - \sigma(k_{-}, T_j)}{k_{+} - k_{-}},
$$
with $k_{\pm} = \pm 0.01$ (1 vol-point in log-moneyness).

2. **Filter**: retain only maturities where $|\widehat{\text{skew}}(T_j)| > 10^{-12}$ to avoid taking the logarithm of zero or near-zero values.

3. **OLS regression**: fit

$$
\log\left|\widehat{\text{skew}}(T_j)\right| = \beta \cdot \log T_j + c + \varepsilon_j
$$

by ordinary least squares over the retained maturities.

4. **Extract H**: $\hat{H} = \hat\beta + 0.5$, clipped to the interval $(0.01, 0.49)$ to remain in the rough regime.

### Accuracy and limitations

Typical accuracy is $\hat{H} \pm 0.02$–$0.05$, depending on:
- **Maturity range**: a wider range (e.g. 1 week to 2 years) gives a more reliable slope estimate; a narrow range inflates the standard error.
- **Smile quality**: noisy or sparsely sampled smiles introduce finite-difference error in the skew.
- **Model error**: the power law is exact under rBergomi/rough Heston; deviations from power-law behaviour indicate jumps or other non-rough components.

```python
# Pseudocode — see rough_vol/estimation/hurst.py
def estimate_H_from_skew(maturities, atm_skews):
    mask = np.abs(atm_skews) > 1e-12
    log_T     = np.log(maturities[mask])
    log_skew  = np.log(np.abs(atm_skews[mask]))
    beta, _   = np.polyfit(log_T, log_skew, deg=1)
    return float(np.clip(beta + 0.5, 0.01, 0.49))
```

---

## See Also

| Topic | Library module |
|---|---|
| fBm simulation (all three methods) | `rough_vol/simulation/fbm.py` |
| rBergomi Monte Carlo engine | `rough_vol/models/rough_bergomi.py` |
| Fractional Riccati solver | `rough_vol/models/rough_heston.py` |
| Gil-Pelaez Fourier pricer | `rough_vol/pricing/fourier.py` |
| Implied vol inversion (Brent) | `rough_vol/pricing/implied_vol.py` |
| Two-stage calibration | `rough_vol/calibration/calibrator.py` |
| H estimation | `rough_vol/estimation/hurst.py` |
| Mathematical background | `docs/math_background.md` |
